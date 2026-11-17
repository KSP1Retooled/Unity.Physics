using NUnit.Framework;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;
using MotionData = Unity.Physics.MotionData;
using MotionVelocity = Unity.Physics.MotionVelocity;
using PhysicsRigidBody = Unity.Physics.RigidBody;

namespace KSP.Retooled.Physics.Tests
{
    /// <summary>
    /// The copies that a body's row in the store keeps of its owner's serial number and of its collider's expansion
    /// factor (<see cref="NativeStore.BodyData.serial"/> and <see cref="NativeStore.BodyData.colliderExpansion"/>). The
    /// body build reads these copies instead of the owner's record and the collider's memory. After each change that
    /// can touch them, each test checks that every row's copies equal the values that they copy, and that the build
    /// gives the same bodies and motions, to the last bit, as a build that reads the owners and the colliders.
    /// </summary>
    public unsafe class BuildRowTests : PhysicsScene
    {
        [SetUp]
        public void RequireRetooled()
        {
            if (!BackendInfo.IsRetooled)
            {
                Assert.Ignore("The store's rows are part of the Unity Physics backend.");
            }
        }

        private static NativeStore.WorldData* Store => StepKernel.Current->Store;

        /// <summary>Checks the copies in every row of the store against the owner's serial number and the collider.</summary>
        private static void AssertRowsHoldTheirSources(string when)
        {
            NativeStore.WorldData* store = Store;
            for (int i = 0; i < (int)store->bodyCount; i++)
            {
                NativeStore.BodyData* row = store->bodies + i;
                Assert.That(row->owner != null, Is.True, $"{when}: row {i} has an owner");
                Assert.That(row->serial, Is.EqualTo(row->owner->serial), $"{when}: the serial number in row {i}");
                Assert.That(row->serial, Is.Not.EqualTo(0u), $"{when}: row {i} has a serial number");
                float expected = row->collider->Value.IsCreated ? row->collider->Value.Value.MassProperties.AngularExpansionFactor : 0.0f;
                Assert.That(math.asuint(row->colliderExpansion), Is.EqualTo(math.asuint(expected)), $"{when}: the collider's expansion factor in row {i}");
            }
        }

        /// <summary>
        /// Builds the store's bodies twice, once from the rows' copies and once from the owners and the colliders, and
        /// checks that the two builds are the same to the last bit. Returns the number of dynamic bodies.
        /// </summary>
        private static int AssertBuildsAgree(string when, float deltaTime = Step)
        {
            NativeStore.WorldData* store = Store;
            using var dynamicIndices = new NativeList<int>(Allocator.TempJob);
            using var staticIndices = new NativeList<int>(Allocator.TempJob);
            using var builtIndices = new NativeList<int>(Allocator.TempJob);
            using var matches = new NativeReference<int>(Allocator.TempJob);
            new PartitionBodiesJob
            {
                World = store,
                DynamicIndices = dynamicIndices,
                StaticIndices = staticIndices,
                BuiltIndices = builtIndices,
                MotionsMatchStore = matches,
            }.Run();

            int count = dynamicIndices.Length + staticIndices.Length + 1;
            int dynamicCount = dynamicIndices.Length;
            using var bodies = new NativeArray<PhysicsRigidBody>(count, Allocator.TempJob);
            using var motions = new NativeArray<MotionData>(dynamicCount, Allocator.TempJob);
            using var velocities = new NativeArray<MotionVelocity>(dynamicCount, Allocator.TempJob);
            using var sourceBodies = new NativeArray<PhysicsRigidBody>(count, Allocator.TempJob);
            using var sourceMotions = new NativeArray<MotionData>(dynamicCount, Allocator.TempJob);
            using var sourceVelocities = new NativeArray<MotionVelocity>(dynamicCount, Allocator.TempJob);
            for (int pass = 0; pass < 2; pass++)
            {
                bool readSources = pass == 1;
                var job = new BuildBodiesJob
                {
                    Store = store->bodies,
                    DynamicIndices = dynamicIndices.AsArray(),
                    StaticIndices = staticIndices.AsArray(),
                    Bodies = readSources ? sourceBodies : bodies,
                    MotionDatas = readSources ? sourceMotions : motions,
                    MotionVelocities = readSources ? sourceVelocities : velocities,
                    DeltaTime = deltaTime,
                    ReadSources = readSources,
                };

                job.Schedule(job.Count, 16).Complete();
            }

            for (int i = 0; i < count; i++)
            {
                PhysicsRigidBody a = bodies[i], b = sourceBodies[i];
                bool same = a.Collider.GetUnsafePtr() == b.Collider.GetUnsafePtr() && a.Entity == b.Entity && a.CustomTags == b.CustomTags
                    && math.asuint(a.Scale) == math.asuint(b.Scale)
                    && math.all(math.asuint(a.WorldFromBody.pos) == math.asuint(b.WorldFromBody.pos))
                    && math.all(math.asuint(a.WorldFromBody.rot.value) == math.asuint(b.WorldFromBody.rot.value));
                Assert.That(same, Is.True, $"{when}: built body {i}, entity {a.Entity} instead of {b.Entity}");
            }

            for (int i = 0; i < dynamicCount; i++)
            {
                MotionData a = motions[i], b = sourceMotions[i];
                Assert.That(Unity.Collections.LowLevel.Unsafe.UnsafeUtility.MemCmp(&a, &b, sizeof(MotionData)), Is.EqualTo(0), $"{when}: motion {i}");
                MotionVelocity va = velocities[i], vb = sourceVelocities[i];
                Assert.That(Unity.Collections.LowLevel.Unsafe.UnsafeUtility.MemCmp(&va, &vb, sizeof(MotionVelocity)), Is.EqualTo(0),
                    $"{when}: motion velocity {i}, expansion {va.AngularExpansionFactor} instead of {vb.AngularExpansionFactor}");
            }

            return dynamicCount;
        }

        private static void AssertAgree(string when)
        {
            AssertRowsHoldTheirSources(when);
            AssertBuildsAgree(when);
            AssertBuildsAgree($"{when}, for queries", 0.0f);
        }

        private Rigidbody OffCentreBox(Vector3 position, float mass)
        {
            Rigidbody body = DynamicBox(position, mass);
            body.centerOfMass = new Vector3(0.2f, -0.1f, 0.05f);
            return body;
        }

        [Test]
        public void TheRowsFollowBodiesThatAreMadeAndDestroyed()
        {
            // The first step makes the step's state, which these checks read.
            Ground();
            Simulate(1);
            Rigidbody first = OffCentreBox(new Vector3(0.0f, 0.5f, 0.0f), 2.0f);
            OffCentreBox(new Vector3(3.0f, 0.5f, 0.0f), 3.0f);
            Rigidbody last = OffCentreBox(new Vector3(6.0f, 0.5f, 0.0f), 4.0f);
            AssertAgree("before the first step, with no colliders built");
            Simulate(2);
            AssertAgree("after two steps");

            // The store moves the last body into the row of the destroyed body.
            NativeStore.WorldData* store = Store;
            uint lastSerial = store->bodies[store->bodyCount - 1].serial;
            uint count = store->bodyCount;
            Object.DestroyImmediate(first.gameObject);
            Assert.That(store->bodyCount, Is.EqualTo(count - 1), "the body is gone");
            AssertAgree("after the first body was destroyed");
            Assert.That(FindSerial(store, lastSerial), Is.True, "the moved row keeps its serial number");
            Assert.That(last != null, Is.True, "the last body lives");
            Simulate(1);
            AssertAgree("after a step with the moved row");

            // A new body gets a new serial number, and gets its collider's expansion factor when the step builds its
            // collider.
            OffCentreBox(new Vector3(-3.0f, 0.5f, 0.0f), 1.0f);
            AssertAgree("after a body was made");
            Simulate(1);
            AssertAgree("after the new body's first step");
        }

        private static bool FindSerial(NativeStore.WorldData* store, uint serial)
        {
            for (int i = 0; i < (int)store->bodyCount; i++)
            {
                if (store->bodies[i].serial == serial)
                {
                    return true;
                }
            }

            return false;
        }

        [Test]
        public void TheRowsFollowNewCollidersAndMasses()
        {
            Ground();
            Rigidbody body = OffCentreBox(new Vector3(0.0f, 0.5f, 0.0f), 2.0f);
            OffCentreBox(new Vector3(3.0f, 0.5f, 0.0f), 3.0f);
            Simulate(2);
            float before = Store->bodies[0].colliderExpansion + Store->bodies[1].colliderExpansion;
            Assert.That(before, Is.GreaterThan(0.0f), "the boxes' colliders have an expansion factor");

            // A larger box gives the body a new collider with a larger expansion factor.
            body.GetComponent<BoxCollider>().size = new Vector3(2.0f, 1.0f, 3.0f);
            AssertAgree("after the collider changed, before the step builds it");
            Simulate(1);
            AssertAgree("after the step built the new collider");
            float after = Store->bodies[0].colliderExpansion + Store->bodies[1].colliderExpansion;
            Assert.That(after, Is.GreaterThan(before), "the larger box's expansion factor");

            // A second collider makes the body's collider a compound.
            BoxCollider second = body.gameObject.AddComponent<BoxCollider>();
            second.center = new Vector3(0.0f, 1.0f, 0.0f);
            Simulate(1);
            AssertAgree("after the body got a compound collider");

            body.mass = 7.0f;
            body.centerOfMass = new Vector3(-0.3f, 0.2f, 0.1f);
            body.inertiaTensor = new Vector3(1.0f, 2.0f, 3.0f);
            AssertAgree("after the mass changed");
            Simulate(1);
            AssertAgree("after a step with the new mass");

            // A body with no collider left has no expansion factor from a collider.
            Object.DestroyImmediate(second);
            Object.DestroyImmediate(body.GetComponent<BoxCollider>());
            Simulate(1);
            AssertAgree("after the body lost its colliders");
        }

        [Test]
        public void TheRowsFollowKinematicAndSleepingBodies()
        {
            Ground();
            Rigidbody moved = OffCentreBox(new Vector3(0.0f, 0.5f, 0.0f), 2.0f);
            Rigidbody sleeper = OffCentreBox(new Vector3(3.0f, 0.5f, 0.0f), 3.0f);
            Simulate(2);

            moved.isKinematic = true;
            AssertAgree("after the body was made kinematic");
            Simulate(1);
            moved.MovePosition(new Vector3(0.0f, 1.0f, 0.5f));
            moved.MoveRotation(Quaternion.Euler(10.0f, 20.0f, 30.0f));
            AssertAgree("with a kinematic target");
            Simulate(1);
            AssertAgree("after the kinematic body moved");

            sleeper.Sleep();
            AssertAgree("after the body was put to sleep");
            Simulate(1);
            AssertAgree("after a step with a sleeping body");

            moved.isKinematic = false;
            sleeper.WakeUp();
            Simulate(1);
            AssertAgree("after both bodies move again");
        }
    }
}
