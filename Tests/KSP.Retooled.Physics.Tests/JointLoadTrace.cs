using System;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using System.Text;
using NUnit.Framework;
using UnityEngine;
using Object = UnityEngine.Object;

namespace KSP.Retooled.Physics.Tests
{
    /// <summary>
    /// Writes, for every step of a few scenes, a hash of every joint's force and torque, of every body's pose, of every
    /// body's velocity, of the bodies and motions of the world that the step built, and of the collision messages, and
    /// the number of joints that are left. Two commits that must give the same results to the last bit give the same
    /// file. TRACE_TAG names the output, Logs/JointLoadTrace-&lt;tag&gt;.txt. TRACE_BREAK sets the break force of the
    /// Tu-160's joints, in newtons (default 2e4). The scenes are the Tu-160 dropped 2 m onto the ground, and a tower of
    /// stack joints that a heavy box falls on, beside a chain that hangs from the world and a pair of boxes that falls
    /// asleep.
    /// </summary>
    [Explicit("A trace for comparing two commits")]
    public class JointLoadTrace : PhysicsScene
    {
        private const int Steps = 200;

        private static ulong Mix(ulong hash, float value)
        {
            hash ^= (uint)BitConverter.SingleToInt32Bits(value);
            return hash * 1099511628211UL;
        }

        private static ulong Mix(ulong hash, Vector3 value)
        {
            return Mix(Mix(Mix(hash, value.x), value.y), value.z);
        }

        /// <summary>
        /// Mixes every collision message that its GameObject receives into one hash: the kind of message, the other
        /// collider's name, and each contact's point, normal, separation and impulse.
        /// </summary>
        [ExecuteAlways]
        internal class ContactHash : MonoBehaviour
        {
            public static ulong Hash = 14695981039346656037UL;
            public static int Messages;

            private void Add(float kind, Collision collision)
            {
                Messages++;
                Hash = Mix(Hash, kind);
                Hash = Mix(Hash, collision.collider != null ? collision.collider.name.Length : -1);
                Hash = Mix(Hash, collision.impulse);
                for (int i = 0; i < collision.contactCount; i++)
                {
                    ContactPoint contact = collision.GetContact(i);
                    Hash = Mix(Mix(Mix(Hash, contact.point), contact.normal), contact.separation);
                    Hash = Mix(Hash, contact.impulse);
                }
            }

            private void OnCollisionEnter(Collision collision)
            {
                Add(1.0f, collision);
            }

            private void OnCollisionStay(Collision collision)
            {
                Add(2.0f, collision);
            }

            private void OnCollisionExit(Collision collision)
            {
                Add(3.0f, collision);
            }
        }

        private static ulong Mix(ulong hash, int value)
        {
            hash ^= (uint)value;
            return hash * 1099511628211UL;
        }

        /// <summary>
        /// A hash of the bodies and motions of the world that the step built: every field except the colliders'
        /// addresses and the serial numbers, which differ between runs. It also mixes in whether each body has a collider
        /// and a serial number.
        /// </summary>
        private static unsafe ulong WorldHash()
        {
            ulong hash = 14695981039346656037UL;
            Unity.Physics.PhysicsWorld world = StepKernel.Current->World;
            foreach (Unity.Physics.RigidBody body in world.Bodies)
            {
                hash = Mix(hash, body.Collider.IsCreated ? 1 : 0);
                hash = Mix(hash, body.Entity.Index);
                hash = Mix(hash, body.Entity.Version > 0 ? 1 : body.Entity.Version);
                hash = Mix(hash, body.Scale);
                hash = Mix(hash, body.CustomTags);
                hash = Mix(Mix(hash, (Vector3)body.WorldFromBody.pos), (Vector4)body.WorldFromBody.rot.value);
            }

            foreach (Unity.Physics.MotionData motion in world.MotionDatas)
            {
                hash = Mix(Mix(hash, (Vector3)motion.WorldFromMotion.pos), (Vector4)motion.WorldFromMotion.rot.value);
                hash = Mix(Mix(hash, (Vector3)motion.BodyFromMotion.pos), (Vector4)motion.BodyFromMotion.rot.value);
                hash = Mix(Mix(hash, motion.LinearDamping), motion.AngularDamping);
            }

            foreach (Unity.Physics.MotionVelocity velocity in world.MotionVelocities)
            {
                hash = Mix(Mix(Mix(hash, (Vector3)velocity.LinearVelocity), (Vector3)velocity.AngularVelocity), (Vector3)velocity.InverseInertia);
                hash = Mix(Mix(Mix(hash, velocity.InverseMass), velocity.AngularExpansionFactor), velocity.GravityFactor);
            }

            return hash;
        }

        private static ulong Mix(ulong hash, Vector4 value)
        {
            return Mix(Mix(Mix(Mix(hash, value.x), value.y), value.z), value.w);
        }

        private static void Record(StringBuilder trace, int step, List<ConfigurableJoint> joints, List<Rigidbody> bodies)
        {
            ulong velocities = 14695981039346656037UL;
            foreach (Rigidbody body in bodies)
            {
                if (body != null)
                {
                    velocities = Mix(Mix(velocities, body.linearVelocity), body.angularVelocity);
                }
            }

            ulong world = WorldHash();
            ulong contacts = ContactHash.Hash;
            int messages = ContactHash.Messages;
            ContactHash.Hash = 14695981039346656037UL;
            ContactHash.Messages = 0;
            RecordLoadsAndPoses(trace, step, joints, bodies);
            trace.Append($"{step} velocities {velocities:x16} world {world:x16} contacts {contacts:x16} messages {messages}\n");
        }

        private static void RecordLoadsAndPoses(StringBuilder trace, int step, List<ConfigurableJoint> joints, List<Rigidbody> bodies)
        {
            ulong loads = 14695981039346656037UL, poses = 14695981039346656037UL;
            int left = 0;
            foreach (ConfigurableJoint joint in joints)
            {
                if (joint == null)
                {
                    loads = Mix(loads, -1.0f);
                    continue;
                }

                left++;
                loads = Mix(Mix(loads, joint.currentForce), joint.currentTorque);
            }

            foreach (Rigidbody body in bodies)
            {
                poses = Mix(Mix(poses, body.position), body.rotation.eulerAngles);
            }

            trace.Append($"{step} loads {loads:x16} poses {poses:x16} joints {left}\n");
        }

        private static ConfigurableJoint StackJoint(Rigidbody host, Rigidbody target, Vector3 node, float breakForce)
        {
            ConfigurableJoint first = null;
            for (int i = 0; i < 3; i++)
            {
                Vector3 anchor = node + Quaternion.AngleAxis(120.0f * i, Vector3.up) * new Vector3(0.3f, 0.0f, 0.0f);
                ConfigurableJoint joint = host.gameObject.AddComponent<ConfigurableJoint>();
                joint.connectedBody = target;
                joint.autoConfigureConnectedAnchor = false;
                joint.anchor = host.transform.InverseTransformPoint(anchor);
                joint.connectedAnchor = target.transform.InverseTransformPoint(anchor);
                joint.xMotion = joint.yMotion = joint.zMotion = ConfigurableJointMotion.Locked;
                joint.angularXMotion = ConfigurableJointMotion.Locked;
                joint.angularYMotion = joint.angularZMotion = ConfigurableJointMotion.Limited;
                joint.angularYLimit = new SoftJointLimit { limit = 177.0f };
                joint.angularZLimit = new SoftJointLimit { limit = 177.0f };
                joint.rotationDriveMode = RotationDriveMode.XYAndZ;
                joint.angularYZDrive = new JointDrive { positionSpring = 20000.0f, positionDamper = 200.0f, maximumForce = 1e20f };
                joint.breakForce = breakForce;
                first = first ?? joint;
            }

            return first;
        }

        [Test]
        public void Write()
        {
            if (!BackendInfo.IsRetooled)
            {
                Assert.Ignore("The trace is for the Unity Physics backend.");
            }

            string tag = Environment.GetEnvironmentVariable("TRACE_TAG") ?? "run";
            float breakForce = float.TryParse(Environment.GetEnvironmentVariable("TRACE_BREAK"), out float value) ? value : 2e4f;
            var trace = new StringBuilder();
            ContactHash.Hash = 14695981039346656037UL;
            ContactHash.Messages = 0;

            // The Tu-160, built as the bench builds it, 2 m above the ground.
            var bench = new VesselForestBench();
            MethodInfo build = typeof(VesselForestBench).GetMethod("Build", BindingFlags.NonPublic | BindingFlags.Instance);
            bench.SetUpScene();

            // As in the bench, the craft's parts don't collide with each other.
            bool collided = !UnityEngine.Physics.GetIgnoreLayerCollision(30, 30);
            UnityEngine.Physics.IgnoreLayerCollision(30, 30, true);
            try
            {
                var bodies = (List<Rigidbody>)build.Invoke(bench, new object[] { VesselForestBench.Load("tu160"), true, true });
                var joints = new List<ConfigurableJoint>();
                foreach (Rigidbody body in bodies)
                {
                    body.position += new Vector3(0.0f, 2.0f, 0.0f);
                    body.transform.position = body.position;
                    body.gameObject.AddComponent<ContactHash>();
                    foreach (ConfigurableJoint joint in body.GetComponents<ConfigurableJoint>())
                    {
                        joint.breakForce = breakForce;
                        joints.Add(joint);
                    }
                }

                trace.Append($"tu160: {bodies.Count} bodies, {joints.Count} joints, break force {breakForce}\n");
                for (int step = 0; step < Steps; step++)
                {
                    Simulate(1);
                    Record(trace, step, joints, bodies);
                }
            }
            finally
            {
                bench.TearDownScene();
                UnityEngine.Physics.IgnoreLayerCollision(30, 30, !collided);
            }

            // A tower of stack joints on the ground that a heavy box falls on, a chain that hangs from the world, and a
            // pair of boxes that falls asleep on the ground.
            {
                Ground();
                var bodies = new List<Rigidbody>();
                var joints = new List<ConfigurableJoint>();
                for (int i = 0; i < 10; i++)
                {
                    Rigidbody body = DynamicBox(new Vector3(0.0f, 0.5f + i, 0.0f), 2.0f);
                    bodies.Add(body);
                    if (i > 0)
                    {
                        StackJoint(body, bodies[i - 1], new Vector3(0.0f, i, 0.0f), 3000.0f);
                        joints.AddRange(body.GetComponents<ConfigurableJoint>());
                    }
                }

                bodies.Add(DynamicBox(new Vector3(0.0f, 14.0f, 0.0f), 400.0f));
                Rigidbody above = null;
                for (int i = 0; i < 6; i++)
                {
                    Rigidbody link = DynamicBox(new Vector3(5.0f, 15.0f - 1.5f * i, 0.0f), 1.0f + i);
                    ConfigurableJoint joint = link.gameObject.AddComponent<ConfigurableJoint>();
                    joint.connectedBody = above;
                    joint.autoConfigureConnectedAnchor = false;
                    joint.anchor = new Vector3(0.0f, 0.75f, 0.0f);
                    joint.connectedAnchor = above != null ? new Vector3(0.0f, -0.75f, 0.0f) : new Vector3(5.0f, 15.75f, 0.0f);
                    joint.xMotion = joint.yMotion = joint.zMotion = ConfigurableJointMotion.Locked;
                    joint.breakForce = 150.0f;
                    joint.breakTorque = 400.0f;
                    joints.Add(joint);
                    bodies.Add(link);
                    above = link;
                }

                Rigidbody left = DynamicBox(new Vector3(-5.0f, 0.5f, 0.0f), 1.0f);
                Rigidbody right = DynamicBox(new Vector3(-6.0f, 0.5f, 0.0f), 1.0f);
                StackJoint(right, left, new Vector3(-5.5f, 0.5f, 0.0f), 1e6f);
                joints.AddRange(right.GetComponents<ConfigurableJoint>());
                bodies.Add(left);
                bodies.Add(right);

                foreach (Rigidbody body in bodies)
                {
                    body.gameObject.AddComponent<ContactHash>();
                }

                trace.Append($"tower: {bodies.Count} bodies, {joints.Count} joints\n");
                for (int step = 0; step < Steps; step++)
                {
                    Simulate(1);
                    Record(trace, step, joints, bodies);
                    if (step == 100)
                    {
                        // Halve the limits of the chain between steps.
                        foreach (ConfigurableJoint joint in joints)
                        {
                            if (joint != null && joint.breakForce == 150.0f)
                            {
                                joint.breakForce = 75.0f;
                            }
                        }
                    }
                }
            }

            File.WriteAllText(Path.Combine(Application.dataPath, "..", "Logs", $"JointLoadTrace-{tag}.txt"), trace.ToString());
        }
    }
}
