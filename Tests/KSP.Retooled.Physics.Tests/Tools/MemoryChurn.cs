using System.Collections.Generic;
using System;
using System.IO;
using System.Text;
using KSP.Retooled.Physics.Commands;
using KSP.Retooled.Physics.Stepping;
using NUnit.Framework;
using Unity.Mathematics;
using Unity.Physics.Configuration;
using Unity.Physics.Store.Bodies;
using Unity.Physics.Store.World;
using UnityEngine;
using PhysicsScene = KSP.Retooled.Physics.Tests.Scenes.PhysicsScene;
using Object = UnityEngine.Object;

namespace KSP.Retooled.Physics.Tests.Tools
{
    [Explicit("Run once with each cache configuration and compare the traces.")]
    public unsafe class MemoryChurn : PhysicsScene
    {
        private const int Seeds = 50;
        private const int Steps = 100;

        [Test]
        public void Write()
        {
            Assume.That(BackendInfo.IsRetooled);
            Assert.That(StepCacheSwitches.MemoryChecks, Is.True, "set RETOOLED_MEMORY_CHECKS=1");
            string tag = StepCacheSwitches.ClearSolverSlots ? "off" : "on";
            string path = Path.Combine(Application.dataPath, "..", "Logs", $"MemoryChurn-{tag}.txt");
            using var output = new StreamWriter(path);
            for (int seed = 0; seed < Seeds; seed++)
            {
                Run(seed, output);
                DestroyAll();
                output.Flush();
            }
        }

        private void Run(int seed, StreamWriter output)
        {
            var random = new System.Random(seed);
            var bodies = new List<Rigidbody>();
            var joints = new List<ConfigurableJoint>();
            var statics = new List<GameObject>();
            var debris = new List<Rigidbody>();
            var origin = Vector3.zero;
            int nextName = 0;
            JointLoadTrace.ContactHash.Hash = 14695981039346656037UL;
            JointLoadTrace.ContactHash.Messages = 0;

            Rigidbody AddBody(Vector3 position, bool gravity)
            {
                Rigidbody body = DynamicBox(position + origin);
                body.name = $"body-{nextName++}";
                body.useGravity = gravity;
                body.gameObject.AddComponent<JointLoadTrace.ContactHash>();
                bodies.Add(body);
                return body;
            }

            ConfigurableJoint Connect(Rigidbody body, Rigidbody other)
            {
                var joint = body.gameObject.AddComponent<ConfigurableJoint>();
                joint.connectedBody = other;
                joint.autoConfigureConnectedAnchor = false;
                joint.anchor = Vector3.zero;
                joint.connectedAnchor = other == null ? body.position : other.transform.InverseTransformPoint(body.position);
                joint.xMotion = joint.yMotion = joint.zMotion = ConfigurableJointMotion.Locked;
                joint.angularXMotion = joint.angularYMotion = joint.angularZMotion = ConfigurableJointMotion.Locked;
                joints.Add(joint);
                return joint;
            }

            // Create debris before the clamp so removing it can swap a surviving row and move the anchor index.
            debris.Add(AddBody(new Vector3(18, 3, 0), true));
            Rigidbody held = AddBody(new Vector3(-20, 5, 0), false);
            ConfigurableJoint clamp = Connect(held, null);
            Rigidbody heldPart = AddBody(new Vector3(-20, 7, 0), false);
            Connect(heldPart, held);
            Rigidbody heldTip = AddBody(new Vector3(-20, 9, 0), false);
            Connect(heldTip, heldPart);

            var otherParts = new Rigidbody[4];
            for (int i = 0; i < otherParts.Length; i++)
            {
                otherParts[i] = AddBody(new Vector3(0, 3 + 2 * i, 0), true);
                if (i > 0)
                {
                    Connect(otherParts[i], otherParts[i - 1]);
                }
            }

            Rigidbody wheelBody = AddBody(new Vector3(8, 1.2f, 0), true);
            GameObject wheelHolder = Track(new GameObject("wheel"));
            wheelHolder.transform.SetParent(wheelBody.transform, false);
            wheelHolder.AddComponent<WheelCollider>().radius = 0.5f;
            statics.Add(Box(new Vector3(0, -0.5f, 0), new Vector3(64, 1, 30)));
            Rigidbody switched = AddBody(new Vector3(0, 20, 20), false);
            bool simulated = true;

            output.WriteLine($"seed {seed}");
            for (int step = 0; step < Steps; step++)
            {
                if (step > 0 && step % 4 == 0)
                {
                    switch (step <= 32 ? step / 4 - 1 : random.Next(8))
                    {
                        case 0:
                            if (debris.Count != 0)
                            {
                                int removed = random.Next(debris.Count);
                                Object.DestroyImmediate(debris[removed].gameObject);
                                debris.RemoveAt(removed);
                            }
                            break;
                        case 1:
                            debris.Add(AddBody(new Vector3(16 + random.Next(12), 4 + random.Next(8), random.Next(-8, 9)), true));
                            break;
                        case 2:
                            if (joints.Count > 3)
                            {
                                ConfigurableJoint removed = joints[joints.Count - 1];
                                joints.RemoveAt(joints.Count - 1);
                                Object.DestroyImmediate(removed);
                            }
                            break;
                        case 3:
                            Connect(otherParts[3], otherParts[2]);
                            break;
                        case 4:
                            if (statics.Count > 1)
                            {
                                Object.DestroyImmediate(statics[statics.Count - 1]);
                                statics.RemoveAt(statics.Count - 1);
                            }
                            else
                            {
                                statics.Add(Box(origin + new Vector3(20, 1, 0), new Vector3(3, 2, 4)));
                            }
                            break;
                        case 5:
                            otherParts[0].isKinematic = !otherParts[0].isKinematic;
                            break;
                        case 6:
                            simulated = !simulated;
                            SetSimulated(switched, simulated);
                            break;
                        case 7:
                            Vector3 shift = new Vector3(random.Next(2) == 0 ? -512 : 512, 0, 0);
                            foreach (Rigidbody body in bodies)
                            {
                                if (body != null)
                                {
                                    body.transform.position += shift;
                                }
                            }
                            foreach (GameObject item in statics)
                            {
                                item.transform.position += shift;
                            }
                            clamp.connectedAnchor += shift;
                            origin += shift;
                            UnityEngine.Physics.SyncTransforms();
                            break;
                    }
                }

                if (step == 64)
                {
                    for (int i = 0; i < 48; i++)
                    {
                        debris.Add(AddBody(new Vector3(40 + 2 * i, 4, 0), true));
                    }
                }

                Simulate();
                bodies.RemoveAll(body => body == null);
                var trace = new StringBuilder();
                JointLoadTrace.Record(trace, step, joints, bodies);
                output.Write(trace);
                foreach (Rigidbody body in bodies)
                {
                    Vector3 position = body.position;
                    Vector3 speed = body.linearVelocity;
                    Quaternion rotation = body.rotation;
                    Vector3 spin = body.angularVelocity;
                    Assert.That(math.all(math.isfinite((float3)position)) && math.all(math.isfinite((float3)speed))
                        && math.all(math.isfinite((float3)spin)) && math.all(math.isfinite(((quaternion)rotation).value)), Is.True,
                        $"seed {seed}, step {step}, {body.name} has non-finite motion");
                    output.WriteLine(FormattableString.Invariant($"body {body.name} {position.x:R} {position.y:R} {position.z:R} {rotation.x:R} {rotation.y:R} {rotation.z:R} {rotation.w:R} {speed.x:R} {speed.y:R} {speed.z:R} {spin.x:R} {spin.y:R} {spin.z:R}"));
                }
                Assert.That(held.linearVelocity.magnitude, Is.LessThan(0.005f), $"seed {seed}, step {step}, clamp root");
                Assert.That(heldPart.linearVelocity.magnitude, Is.LessThan(0.005f), $"seed {seed}, step {step}, clamp part");
                Assert.That(heldTip.linearVelocity.magnitude, Is.LessThan(0.005f), $"seed {seed}, step {step}, clamp tip");
                AssertVector(new Vector3(-20, 5, 0) + origin, held.position, 0.002f, $"seed {seed}, step {step}, clamp root");
            }
        }

        private static void SetSimulated(Rigidbody body, bool enabled)
        {
            var stepper = StepEntryPoints.Current;
            WorldData* store = stepper->Store;
            for (int row = 0; row < store->bodyCount; row++)
            {
                BodyData* candidate = store->bodies + row;
                if (!math.all(candidate->pose->Position == (float3)body.position))
                {
                    continue;
                }

                CommandArgs args = default;
                args.flags = (uint)BodyFlags.Simulated;
                args.enabled = enabled ? 1u : 0u;
                BodyCommands.Run((uint)BodyCommand.SetFlags, candidate->owner, &args, stepper->Globals);
                return;
            }
            Assert.Fail($"Could not find the isolated body at {body.position} in the store.");
        }
    }
}
