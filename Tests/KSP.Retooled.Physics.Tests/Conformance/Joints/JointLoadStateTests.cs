using PhysicsScene = KSP.Retooled.Physics.Tests.Scenes.PhysicsScene;
using KSP.Retooled.Physics.Stepping;
using NUnit.Framework;
using Unity.Mathematics;
using UnityEngine;
using Unity.Physics.Store.Bodies;
using Unity.Physics.Store.Joints;
using Unity.Physics.Store.World;
using Unity.Physics.Stepping;
using Unity.Physics.Dynamics.Joint.Loads;
using Unity.Physics.Configuration;
using Unity.Physics.Dynamics.Solver.Islands.Data;

namespace KSP.Retooled.Physics.Tests.Conformance.Joints
{
    /// <summary>
    /// Test dense joint load slots through creation, removal, break limit changes and managed detach.
    /// </summary>
    public unsafe class JointLoadStateTests : PhysicsScene
    {
        [SetUp]
        public void RequireRetooled()
        {
            if (!BackendInfo.IsRetooled)
            {
                Assert.Ignore("The dense array of joint loads is part of the Unity Physics backend.");
            }
        }

        private static WorldData* Store => StepEntryPoints.Current->Store;

        /// <summary>A body that hangs from the world by one locked joint at its centre.</summary>
        private ConfigurableJoint Hang(Vector3 position, float mass)
        {
            Rigidbody body = DynamicBox(position, mass);
            ConfigurableJoint joint = body.gameObject.AddComponent<ConfigurableJoint>();
            joint.autoConfigureConnectedAnchor = false;
            joint.anchor = Vector3.zero;
            joint.connectedAnchor = position;
            joint.xMotion = joint.yMotion = joint.zMotion = ConfigurableJointMotion.Locked;
            joint.angularXMotion = joint.angularYMotion = joint.angularZMotion = ConfigurableJointMotion.Locked;
            return joint;
        }

        /// <summary>
        /// The store's record of the joint that holds a body, found by the position of its body. The bodies in these
        /// tests hang at least 4 m apart.
        /// </summary>
        private static JointData* RecordOf(WorldData* store, ConfigurableJoint joint, out int index)
        {
            Vector3 position = joint.GetComponent<Rigidbody>().position;
            for (int i = 0; i < (int)store->jointCount; i++)
            {
                JointData* record = store->joints[i];
                Body* body = record->body0 != null ? record->body0 : record->body1;
                if (body != null && math.distance(store->bodies[body->index].pose->Position, position) < 0.5f)
                {
                    index = i;
                    return record;
                }
            }

            Assert.Fail("the joint has no record");
            index = -1;
            return null;
        }

        private static void AssertSame(JointLoadState expected, JointLoadState actual, string what)
        {
            Assert.That(actual.Force.Equals(expected.Force) && actual.Torque.Equals(expected.Torque), Is.True,
                $"{what}: the loads {actual.Force} {actual.Torque} instead of {expected.Force} {expected.Torque}");
            Assert.That(actual.BreakingForce, Is.EqualTo(expected.BreakingForce), $"{what}: the smoothed force");
            Assert.That(actual.BreakingTorque, Is.EqualTo(expected.BreakingTorque), $"{what}: the smoothed torque");
            Assert.That(actual.FullForce.Equals(expected.FullForce) && actual.FullTorque.Equals(expected.FullTorque), Is.True,
                $"{what}: the full loads");
            Assert.That(actual.SmoothedFullForce, Is.EqualTo(expected.SmoothedFullForce), $"{what}: the smoothed full force");
            Assert.That(actual.SmoothedFullTorque, Is.EqualTo(expected.SmoothedFullTorque), $"{what}: the smoothed full torque");
            Assert.That(actual.Serial, Is.EqualTo(expected.Serial), $"{what}: the serial number");
        }

        [Test]
        public void AJointKeepsItsLoadStateWhenAnotherJointIsDestroyedOrMade()
        {
            // The first joint is destroyed later, so that the last joint takes its place in the store's list.
            ConfigurableJoint victim = Hang(new Vector3(-4.0f, 5.0f, 0.0f), 7.0f);
            ConfigurableJoint light = Hang(new Vector3(0.0f, 5.0f, 0.0f), 1.0f);
            ConfigurableJoint heavy = Hang(new Vector3(4.0f, 5.0f, 0.0f), 3.0f);
            Simulate(3);

            WorldData* store = Store;
            Assert.That(store->jointCount, Is.EqualTo(3u), "the three joints");
            RecordOf(store, heavy, out int heavyIndex);
            Assert.That(heavyIndex, Is.EqualTo(2), "the heavy body's joint is last");
            JointLoadState lightBefore = *JointLoadStates.Find(RecordOf(store, light, out _));
            JointLoadState heavyBefore = *JointLoadStates.Find(RecordOf(store, heavy, out _));
            Assert.That(heavyBefore.BreakingForce, Is.GreaterThan(0.0f), "the smoothed force has started to rise");
            Assert.That(heavyBefore.BreakingForce, Is.LessThan(0.9f * 3.0f * 9.81f), "the smoothed force is still rising");

            Object.DestroyImmediate(victim);
            Assert.That(store->jointCount, Is.EqualTo(2u), "the victim is gone");
            JointData* heavyRecord = RecordOf(store, heavy, out heavyIndex);
            Assert.That(heavyIndex, Is.EqualTo(0), "the last joint took the victim's place");
            AssertSame(heavyBefore, *JointLoadStates.Find(heavyRecord), "the moved joint");
            AssertSame(lightBefore, *JointLoadStates.Find(RecordOf(store, light, out _)), "the joint that stayed");
            Assert.That((float3)heavy.currentForce, Is.EqualTo(heavyBefore.Force), "GetForces finds the moved joint's slot");

            // A new joint starts with no load, and the other joints keep theirs.
            ConfigurableJoint added = Hang(new Vector3(8.0f, 5.0f, 0.0f), 2.0f);
            Assert.That(store->jointCount, Is.EqualTo(3u), "the new joint");
            JointLoadState fresh = *JointLoadStates.Find(RecordOf(store, added, out int addedIndex));
            Assert.That(addedIndex, Is.EqualTo(2), "the new joint is last");
            Assert.That(fresh.Force.Equals(float3.zero) && fresh.BreakingForce == 0.0f && float.IsPositiveInfinity(fresh.BreakForce), Is.True,
                "the new joint's slot");
            AssertSame(heavyBefore, *JointLoadStates.Find(heavyRecord), "the moved joint after the new one");

            // In the next step, the moved joint's smoothed force goes on from where it was.
            Simulate(1);
            JointLoadState heavyAfter = *JointLoadStates.Find(heavyRecord);
            float follow = Step / (0.1f + Step);
            float expected = heavyBefore.BreakingForce + (math.length(heavyAfter.Force) - heavyBefore.BreakingForce) * follow;
            Assert.That(heavyAfter.BreakingForce, Is.EqualTo(expected).Within(1e-5f * expected), "the smoothed force goes on");
            Simulate(100);
            AssertVector(new Vector3(0.0f, 3.0f * 9.81f, 0.0f), heavy.currentForce, 0.5f, "the heavy body's weight");
            AssertVector(new Vector3(0.0f, 2.0f * 9.81f, 0.0f), added.currentForce, 0.5f, "the new body's weight");
        }

        [Test]
        public void ALowerBreakForceSetBetweenStepsBreaksTheJointInTheNextStep()
        {
            ConfigurableJoint joint = Hang(new Vector3(0.0f, 5.0f, 0.0f), 3.0f);
            joint.breakForce = 1e6f;

            // The body falls asleep after about twenty steps at rest, and a joint whose bodies sleep doesn't break.
            // After twelve steps, the smoothed force is about 26 N.
            Simulate(12);
            WorldData* store = Store;
            JointLoadState held = *JointLoadStates.Find(RecordOf(store, joint, out _));
            Assert.That(held.BreakingForce, Is.InRange(22.0f, 3.0f * 9.81f), "the smoothed weight");
            Assert.That(held.BreakForce, Is.EqualTo(1e6f), "the first limit");

            joint.breakForce = 20.0f;
            Assert.That(JointLoadStates.Find(RecordOf(store, joint, out _))->BreakForce, Is.EqualTo(20.0f), "the slot has the new limit");
            Simulate(1);
            Assert.That(store->jointBreakCount, Is.EqualTo(1u), "the joint broke in the first step under the new limit");
        }

        [Test]
        public void AHigherBreakForceSetBetweenStepsKeepsTheJoint()
        {
            // The weight is about 29 N. Under a limit of 25 N, the smoothed force would pass the limit after about ten steps.
            ConfigurableJoint joint = Hang(new Vector3(0.0f, 5.0f, 0.0f), 3.0f);
            joint.breakForce = 25.0f;
            Simulate(2);
            WorldData* store = Store;
            Assert.That(store->jointBreakCount, Is.Zero, "the smoothed force is still under the limit");

            joint.breakForce = 1e6f;
            Simulate(20);
            Assert.That(store->jointBreakCount, Is.Zero, "nothing broke under the new limit");
            AssertVector(new Vector3(0.0f, 3.0f * 9.81f, 0.0f), joint.currentForce, 0.5f, "the joint still holds the body");
        }

        [TestCase(true, true)]
        [TestCase(true, false)]
        [TestCase(false, false)]
        public void PackingClearsThePreviousBreakLoad(bool vessels, bool fused)
        {
            EnvironmentSwitches.Read();
            bool previousVessels = VesselSwitches.Enabled;
            bool previousFused = SolverSwitches.VesselFused;
            try
            {
                VesselSwitches.Enabled = vessels;
                SolverSwitches.VesselFused = fused;
                UnityEngine.Physics.gravity = Vector3.zero;
                Ground();
                Vector3 footPosition = 0.5f * Vector3.up;
                Vector3 tankPosition = 2.0f * Vector3.up;
                Rigidbody foot = WeightlessBox(footPosition, 0.05f);
                Rigidbody tank = WeightlessBox(tankPosition, 81.0f);
                foot.sleepThreshold = tank.sleepThreshold = 0.0f;
                ConfigurableJoint joint = tank.gameObject.AddComponent<ConfigurableJoint>();
                joint.connectedBody = foot;
                joint.autoConfigureConnectedAnchor = false;
                joint.anchor = Vector3.up - tankPosition;
                joint.connectedAnchor = Vector3.up - footPosition;
                joint.xMotion = joint.yMotion = joint.zMotion = ConfigurableJointMotion.Locked;
                joint.angularXMotion = joint.angularYMotion = joint.angularZMotion = ConfigurableJointMotion.Locked;
                joint.breakForce = joint.breakTorque = float.PositiveInfinity;

                for (int step = 0; step < 40; step++)
                {
                    foot.AddForce(Vector3.up * Gravity, ForceMode.Acceleration);
                    tank.AddForce(Vector3.up * Gravity, ForceMode.Acceleration);
                    Simulate();
                }

                Assert.That(joint.currentForce.magnitude, Is.EqualTo(-tank.mass * Gravity).Within(30.0f));
                foot.isKinematic = tank.isKinematic = true;
                foot.position = footPosition;
                tank.position = tankPosition;
                Simulate(3);
                Vector3 packedForce = joint.currentForce;
                Vector3 packedTorque = joint.currentTorque;

                foot.isKinematic = tank.isKinematic = false;
                foot.linearVelocity = tank.linearVelocity = Vector3.zero;
                joint.breakForce = joint.breakTorque = 200.0f;
                float acceleration = Gravity / 16.0f;
                foot.AddForce(Vector3.up * acceleration, ForceMode.Acceleration);
                tank.AddForce(Vector3.up * acceleration, ForceMode.Acceleration);
                long fusedBefore = IslandWorkCounters.FusedIslandSolves;
                Simulate();

                Assert.That(StepEntryPoints.Current->VesselSet.VesselCount, Is.EqualTo(vessels ? 1 : 0));
                Assert.That(IslandWorkCounters.FusedIslandSolves - fusedBefore, Is.EqualTo(fused ? 1 : 0));
                Assert.That(Store->jointBreakCount, Is.Zero, $"packed force was {packedForce.magnitude:F3} kN");
                Assert.That(packedForce, Is.EqualTo(Vector3.zero));
                Assert.That(packedTorque, Is.EqualTo(Vector3.zero));
                Assert.That(joint.currentForce.magnitude, Is.EqualTo(-tank.mass * acceleration).Within(15.0f));
            }
            finally
            {
                VesselSwitches.Enabled = previousVessels;
                SolverSwitches.VesselFused = previousFused;
            }
        }

        [Test]
        public void GetForcesRightAfterAStepReadsTheStepsLoads()
        {
            ConfigurableJoint joint = Hang(new Vector3(0.0f, 5.0f, 0.0f), 3.0f);
            Simulate(1);
            WorldData* store = Store;
            for (int step = 0; step < 5; step++)
            {
                Simulate(1);
                JointLoadState state = *JointLoadStates.Find(RecordOf(store, joint, out _));
                Assert.That((float3)joint.currentForce, Is.EqualTo(state.Force), $"the force after step {step}");
                Assert.That((float3)joint.currentTorque, Is.EqualTo(state.Torque), $"the torque after step {step}");
            }

            AssertVector(new Vector3(0.0f, 3.0f * 9.81f, 0.0f), joint.currentForce, 0.5f, "the body's weight");
        }

        [Test]
        public void DetachingWritesTheLoadsBackAndAttachingReadsThem()
        {
            ConfigurableJoint joint = Hang(new Vector3(0.0f, 5.0f, 0.0f), 3.0f);
            Simulate(3);
            WorldData* store = Store;
            JointData* record = RecordOf(store, joint, out _);
            JointLoadState before = *JointLoadStates.Find(record);
            Assert.That(before.BreakingForce, Is.GreaterThan(0.0f), "the smoothed force has started to rise");

            BackendHost.Detach();
            try
            {
                Assert.That(JointLoadStates.Find(record) == null, Is.True, "no step state holds the loads");
                Assert.That(record->force.Equals(before.Force) && record->torque.Equals(before.Torque), Is.True, "the record has the loads");
                Assert.That(record->breakingForce, Is.EqualTo(before.BreakingForce), "the record has the smoothed force");
                Assert.That(record->breakingTorque, Is.EqualTo(before.BreakingTorque), "the record has the smoothed torque");
                Assert.That(record->fullForce.Equals(before.FullForce) && record->fullTorque.Equals(before.FullTorque), Is.True,
                    "the record has the full loads");
                Assert.That(record->smoothedFullForce, Is.EqualTo(before.SmoothedFullForce));
                Assert.That(record->smoothedFullTorque, Is.EqualTo(before.SmoothedFullTorque));
            }
            finally
            {
                BackendHost.Attach();
            }

            Assert.That((float3)joint.currentForce, Is.EqualTo(before.Force), "GetForces reads the record before any step");

            // A query makes the world's step state again, which reads the loads from the records.
            UnityEngine.Physics.Raycast(new Vector3(0.0f, 10.0f, 0.0f), Vector3.down, 20.0f);
            JointLoadState* after = JointLoadStates.Find(record);
            Assert.That(after != null, Is.True, "the new step state holds the loads");
            AssertSame(before, *after, "after attaching");
            Simulate(100);
            AssertVector(new Vector3(0.0f, 3.0f * 9.81f, 0.0f), joint.currentForce, 0.5f, "the body's weight");
        }
    }
}
