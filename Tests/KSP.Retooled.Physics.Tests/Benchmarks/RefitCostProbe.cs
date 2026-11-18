using Unity.Physics.Configuration;
using PhysicsScene = KSP.Retooled.Physics.Tests.Scenes.PhysicsScene;
using KSP.Retooled.Physics.Stepping;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Physics;
using UnityEngine;
using Debug = UnityEngine.Debug;

namespace KSP.Retooled.Physics.Tests.Benchmarks
{
    using Graph = VesselForestBench.Graph;

    /// <summary>
    /// Measure dynamic refit stages separately in Burst, excluding scheduling costs, and count next-build cache
    /// reuse.
    /// PROBE_CRAFT selects the craft (default tu160x4). Write Logs/RefitProbe.txt.
    /// </summary>
    [Explicit("profiling")]
    [Category("Slow")]
    public unsafe class RefitCostProbe : PhysicsScene
    {
        private const int PartLayer = 30;
        private const int Repeats = 31;

        private static ConfigurableJoint Joint(Rigidbody host, Rigidbody target, Vector3 anchor, float spring, bool locked)
        {
            ConfigurableJoint joint = host.gameObject.AddComponent<ConfigurableJoint>();
            joint.connectedBody = target;
            joint.autoConfigureConnectedAnchor = false;
            joint.anchor = host.transform.InverseTransformPoint(anchor);
            joint.connectedAnchor = target.transform.InverseTransformPoint(anchor);
            if (locked)
            {
                joint.xMotion = joint.yMotion = joint.zMotion = ConfigurableJointMotion.Locked;
                joint.angularXMotion = joint.angularYMotion = joint.angularZMotion = ConfigurableJointMotion.Locked;
                return joint;
            }

            joint.xMotion = joint.yMotion = joint.zMotion = ConfigurableJointMotion.Limited;
            joint.angularXMotion = ConfigurableJointMotion.Locked;
            joint.angularYMotion = joint.angularZMotion = ConfigurableJointMotion.Limited;
            joint.linearLimit = new SoftJointLimit { limit = 1.0f };
            joint.angularYLimit = new SoftJointLimit { limit = 177.0f };
            joint.angularZLimit = new SoftJointLimit { limit = 177.0f };
            joint.xDrive = joint.yDrive = joint.zDrive = new JointDrive { positionSpring = 1e20f, maximumForce = 1e20f };
            joint.rotationDriveMode = RotationDriveMode.XYAndZ;
            joint.angularYZDrive = new JointDrive { positionSpring = spring, positionDamper = 0.02f * spring, maximumForce = 1e20f };
            return joint;
        }

        private void Build(Graph graph)
        {
            var bodies = new List<Rigidbody>(graph.Parent.Length);
            for (int i = 0; i < graph.Parent.Length; i++)
            {
                float mass = Mathf.Max(graph.Mass[i], 1e-3f);
                float inertia = Mathf.Max((graph.Inertia[i].x + graph.Inertia[i].y + graph.Inertia[i].z) / 3.0f, 1e-6f);
                float side = Mathf.Clamp(Mathf.Sqrt(6.0f * inertia / mass), 0.05f, 5.0f);
                GameObject part = Box(graph.Position[i], Vector3.one * side);
                part.layer = PartLayer;
                Rigidbody body = part.AddComponent<Rigidbody>();
                body.mass = mass;
                body.sleepThreshold = 0.0f;
                bodies.Add(body);
            }

            for (int i = 0; i < graph.Parent.Length; i++)
            {
                if (graph.Parent[i] >= 0)
                {
                    Joint(bodies[i], bodies[graph.Parent[i]], graph.Joint[i], graph.Swing[i] > 0.0f ? graph.Swing[i] : 1e5f, false);
                }
            }

            for (int c = 0; c < graph.ClosureA.Length; c++)
            {
                int a = graph.ClosureA[c], b = graph.ClosureB[c];
                Joint(bodies[b], bodies[a], 0.5f * (graph.Position[a] + graph.Position[b]), 0.0f, true);
            }
        }

        // Which part of the work a probe job does.
        private const int PoseOnly = 0;
        private const int BoundsOnly = 1;
        private const int ExpandOnly = 2;
        private const int WholeBody = 3;
        private const int WholeBodyWithCache = 4;
        private const int NormalizedPose = 5;
        private const int TouchCollider = 6;
        private const int LocalBounds = 7;
        private const int BoxBoundsInline = 8;
        private const int WholeBodyWithCacheNoShapes = 9;

        [BurstCompile]
        private struct BodyProbeJob : IJob
        {
            [NativeDisableUnsafePtrRestriction] public MotionData* Motions;
            [NativeDisableUnsafePtrRestriction] public MotionVelocity* Velocities;
            [NativeDisableUnsafePtrRestriction] public RigidBody* Bodies;
            [NativeDisableUnsafePtrRestriction] public Aabb* Aabbs;
            [NativeDisableUnsafePtrRestriction] public RigidTransform* Poses;
            [NativeDisableUnsafePtrRestriction] public Aabb* Locals;
            public DynamicBoundsCache.View Cache;
            public int Count;
            public int Mode;
            public float Margin;
            public float TimeStep;

            public void Execute()
            {
                switch (Mode)
                {
                    case PoseOnly:
                        for (int i = 0; i < Count; i++)
                        {
                            Poses[i] = math.mul(Motions[i].WorldFromMotion, math.inverse(Motions[i].BodyFromMotion));
                        }

                        break;
                    case NormalizedPose:
                        for (int i = 0; i < Count; i++)
                        {
                            RigidTransform pose = math.mul(Motions[i].WorldFromMotion, math.inverse(Motions[i].BodyFromMotion));
                            Poses[i] = new RigidTransform(math.normalize(pose.rot), pose.pos);
                        }

                        break;
                    case BoundsOnly:
                        for (int i = 0; i < Count; i++)
                        {
                            Aabbs[i] = Bodies[i].CalculateAabb();
                        }

                        break;
                    case TouchCollider:
                        int sum = 0;
                        for (int i = 0; i < Count; i++)
                        {
                            sum += (int)((Unity.Physics.Collider*)Bodies[i].Collider.GetUnsafePtr())->Type;
                        }

                        Aabbs[0].Min.x = sum;
                        break;
                    case LocalBounds:
                        for (int i = 0; i < Count; i++)
                        {
                            Aabbs[i] = Unity.Physics.Math.TransformAabb(Locals[i], Bodies[i].WorldFromBody);
                        }

                        break;
                    case BoxBoundsInline:
                        for (int i = 0; i < Count; i++)
                        {
                            Aabbs[i] = ((Unity.Physics.BoxCollider*)Bodies[i].Collider.GetUnsafePtr())->CalculateAabb(Bodies[i].WorldFromBody);
                        }

                        break;
                    case ExpandOnly:
                        for (int i = 0; i < Count; i++)
                        {
                            Aabbs[i] = Broadphase.PrepareDynamicBodyDataJob.ExpandAabb(Aabbs[i], Margin, float3.zero, TimeStep, Velocities[i]);
                        }

                        break;
                    default:
                        DynamicBoundsCache.View cache = Mode == WholeBodyWithCache || Mode == WholeBodyWithCacheNoShapes ? Cache : default;
                        if (Mode == WholeBodyWithCacheNoShapes)
                        {
                            cache.Shapes = null;
                        }

                        for (int i = 0; i < Count; i++)
                        {
                            CollisionWorld.UpdateDynamicBoundsJob.UpdateBody(i, Motions, Velocities, Bodies, Aabbs, Margin, float3.zero, TimeStep, cache);
                        }

                        break;
                }
            }
        }

        [BurstCompile]
        private struct NodeProbeJob : IJob
        {
            public Broadphase.Tree Tree;
            [NativeDisableUnsafePtrRestriction] public Aabb* Aabbs;

            public void Execute()
            {
                BoundingVolumeHierarchy bvh = Tree.BoundingVolumeHierarchy;
                for (int slot = 0; slot < BoundingVolumeHierarchy.RefitSubtreeCount; slot++)
                {
                    int root = bvh.FindRefitSubtree(slot);
                    if (root != 0)
                    {
                        bvh.RefitSubtree(root, Aabbs);
                    }
                }

                bvh.RefitAboveCut();
            }
        }

        private static double Median(List<double> values)
        {
            values.Sort();
            return values[values.Count / 2];
        }

        [Test]
        [Timeout(1800000)]
        public void Measure()
        {
            string craft = Environment.GetEnvironmentVariable("PROBE_CRAFT") ?? "tu160x4";
            bool ignored = UnityEngine.Physics.GetIgnoreLayerCollision(PartLayer, PartLayer);
            UnityEngine.Physics.IgnoreLayerCollision(PartLayer, PartLayer, true);
            try
            {
                MeasureCraft(craft);
            }
            finally
            {
                UnityEngine.Physics.IgnoreLayerCollision(PartLayer, PartLayer, ignored);
            }
        }

        private void MeasureCraft(string craft)
        {
            Assert.IsTrue(BackendHost.WaitForCompiledKernels(300.0), "the kernels still run in Mono");
            Build(VesselForestBench.Load(craft));
            Simulate(30);
            StepEntryPoints.Current->FinishPendingRefit();

            var report = new StringBuilder();
            PhysicsWorld world = StepEntryPoints.Current->World;
            int count = world.NumDynamicBodies;
            report.Append($"{craft}: {count} dynamic bodies, {world.CollisionWorld.Broadphase.DynamicTree.BoundingVolumeHierarchy.Domain} \n");

            // The kinds of the bodies' colliders.
            var kinds = new Dictionary<string, int>();
            for (int i = 0; i < count; i++)
            {
                RigidBody body = world.DynamicBodies[i];
                string kind = body.Collider.IsCreated ? body.Collider.Value.Type.ToString() : "none";
                kinds[kind] = kinds.TryGetValue(kind, out int n) ? n + 1 : 1;
            }

            report.Append("collider kinds: " + string.Join(", ", kinds.Select(k => $"{k.Key} {k.Value}")) + "\n");

            // Work on copies, so that the world stays as the step left it.
            var motions = new NativeArray<MotionData>(world.MotionDatas, Allocator.Persistent);
            var velocities = new NativeArray<MotionVelocity>(world.MotionVelocities, Allocator.Persistent);
            var bodies = new NativeArray<RigidBody>(world.DynamicBodies, Allocator.Persistent);
            var aabbs = new NativeArray<Aabb>(count, Allocator.Persistent);
            var poses = new NativeArray<RigidTransform>(count, Allocator.Persistent);
            var locals = new NativeArray<Aabb>(count, Allocator.Persistent);
            var cacheVersion = (uint*)UnsafeUtility.Malloc(4, 4, Allocator.Persistent);
            *cacheVersion = 1;
            var cache = new DynamicBoundsCache(cacheVersion, Allocator.Persistent);
            try
            {
                DynamicBoundsCache.View view = cache.Resize(count);
                var job = new BodyProbeJob
                {
                    Motions = (MotionData*)motions.GetUnsafePtr(),
                    Velocities = (MotionVelocity*)velocities.GetUnsafePtr(),
                    Bodies = (RigidBody*)bodies.GetUnsafePtr(),
                    Aabbs = (Aabb*)aabbs.GetUnsafePtr(),
                    Poses = (RigidTransform*)poses.GetUnsafePtr(),
                    Cache = view,
                    Count = count,
                    Margin = world.CollisionWorld.CollisionTolerance * 0.5f,
                    TimeStep = Step,
                };

                report.Append($"Burst safety checks {Unity.Burst.BurstCompiler.Options.EnableBurstSafetyChecks}, collections checks " +
#if ENABLE_UNITY_COLLECTIONS_CHECKS
                    "on\n");
#else
                    "off\n");
#endif
                bool boxes = true;
                for (int i = 0; i < count; i++)
                {
                    RigidBody body = bodies[i];
                    boxes &= body.Collider.IsCreated && body.Collider.Value.Type == ColliderType.Box;
                    locals[i] = body.Collider.IsCreated ? body.Collider.Value.CalculateAabb() : default;
                }

                job.Locals = (Aabb*)locals.GetUnsafePtr();
                string[] names = { "pose", "collider bounds", "widen by speed", "whole body", "whole body with cache", "pose normalized",
                    "collider type only", "local bounds moved", "box bounds, no switch", "whole body, cache without shapes" };
                var modes = new List<int> { PoseOnly, NormalizedPose, BoundsOnly, ExpandOnly, WholeBody, WholeBodyWithCache, TouchCollider, LocalBounds, BoundsOnly, WholeBodyWithCacheNoShapes };
                if (boxes)
                {
                    modes.Add(BoxBoundsInline);
                }

                foreach (int mode in modes)
                {
                    job.Mode = mode;
                    var times = new List<double>();
                    for (int r = 0; r < Repeats; r++)
                    {
                        long start = Stopwatch.GetTimestamp();
                        job.Run();
                        times.Add((Stopwatch.GetTimestamp() - start) * 1e6 / Stopwatch.Frequency);
                    }

                    double median = Median(times);
                    report.Append($"  {names[mode],-24} {median,8:F1} us, {median * 1000.0 / count,6:F1} ns per body\n");
                }

                // The node refit on the world's own tree, with the bounds that the world's last refit would give. The
                // last whole-body run above left those bounds in the array.
                var nodes = new NodeProbeJob { Tree = world.CollisionWorld.Broadphase.DynamicTree, Aabbs = (Aabb*)aabbs.GetUnsafePtr() };
                var nodeTimes = new List<double>();
                for (int r = 0; r < Repeats; r++)
                {
                    long start = Stopwatch.GetTimestamp();
                    nodes.Run();
                    nodeTimes.Add((Stopwatch.GetTimestamp() - start) * 1e6 / Stopwatch.Frequency);
                }

                int nodeCount = world.CollisionWorld.Broadphase.DynamicTree.BoundingVolumeHierarchy.NodeCount;
                report.Append($"  {"node refit",-24} {Median(nodeTimes),8:F1} us for {nodeCount} nodes, {Median(nodeTimes) * 1000.0 / count,6:F1} ns per body\n");

                // How many bodies keep the last bits of their rotation when the write-back normalizes it.
                int changed = 0;
                for (int i = 0; i < count; i++)
                {
                    RigidTransform pose = math.mul(motions[i].WorldFromMotion, math.inverse(motions[i].BodyFromMotion));
                    if (math.any(math.asuint(math.normalize(pose.rot).value) != math.asuint(pose.rot.value)))
                    {
                        changed++;
                    }
                }

                report.Append($"bodies whose rotation the write-back's normalize changes: {changed} of {count}\n");

                // The bounds of convex hulls, with one hull for all bodies, so that this measures the work and not the
                // memory. Each body keeps its pose.
                foreach (int points in new[] { 8, 16, 32, 64, 128, 255 })
                {
                    var input = new NativeArray<float3>(points, Allocator.Temp);
                    var random = new Unity.Mathematics.Random(1);
                    for (int i = 0; i < points; i++)
                    {
                        input[i] = random.NextFloat3Direction() * new float3(1.0f, 2.0f, 0.5f);
                    }

                    ConvexHullGenerationParameters parameters = ConvexHullGenerationParameters.Default;
                    parameters.BevelRadius = 0.0f;
                    parameters.SimplificationTolerance = 0.0f;
                    Unity.Entities.BlobAssetReference<Unity.Physics.Collider> hull = ConvexCollider.Create(input, parameters);
                    try
                    {
                        int vertices = ((ConvexCollider*)hull.GetUnsafePtr())->ConvexHull.Vertices.Length;
                        for (int i = 0; i < count; i++)
                        {
                            RigidBody body = bodies[i];
                            body.Collider = hull;
                            bodies[i] = body;
                        }

                        job.Mode = BoundsOnly;
                        var times = new List<double>();
                        for (int r = 0; r < Repeats; r++)
                        {
                            long start = Stopwatch.GetTimestamp();
                            job.Run();
                            times.Add((Stopwatch.GetTimestamp() - start) * 1e6 / Stopwatch.Frequency);
                        }

                        report.Append($"  hull of {vertices,3} vertices: {Median(times) * 1000.0 / count,6:F1} ns per body\n");
                    }
                    finally
                    {
                        hull.Dispose();
                        input.Dispose();
                    }
                }
            }
            finally
            {
                motions.Dispose();
                velocities.Dispose();
                bodies.Dispose();
                aabbs.Dispose();
                poses.Dispose();
                locals.Dispose();
                cache.Dispose();
                UnsafeUtility.Free(cacheVersion, Allocator.Persistent);
            }

            // The bodies that take their bounds from the cache in the build of each step.
            bool wasChecking = BroadphaseSwitches.BoundsCheckEnabled;
            BroadphaseSwitches.BoundsCheckEnabled = true;
            long hits0 = DynamicBoundsCache.Counts->Hits;
            long mismatches0 = DynamicBoundsCache.Counts->Mismatches;
            const int checkedSteps = 20;
            Simulate(checkedSteps);
            StepEntryPoints.Current->FinishPendingRefit();
            BroadphaseSwitches.BoundsCheckEnabled = wasChecking;
            long hits = DynamicBoundsCache.Counts->Hits - hits0;
            report.Append($"cache: {hits / (double)checkedSteps:F0} of {count} bodies per step took their bounds from the cache, " +
                $"{DynamicBoundsCache.Counts->Mismatches - mismatches0} mismatches\n");

            string text = report.ToString();
            Debug.Log("[RefitCostProbe]\n" + text);
            File.AppendAllText(Path.Combine(Application.dataPath, "..", "Logs", "RefitProbe.txt"), text);
        }
    }
}
