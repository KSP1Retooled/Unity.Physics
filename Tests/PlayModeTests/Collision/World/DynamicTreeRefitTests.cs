using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Entities;
using Unity.Jobs;
using Unity.Jobs.LowLevel.Unsafe;
using Unity.Mathematics;
using UnityEngine;
using static Unity.Physics.BoundingVolumeHierarchy;

namespace Unity.Physics.Tests.Collision.World
{
    /// <summary>
    /// The step fits the dynamic tree to the solved poses in parallel, one subtree at a time, instead of in one job
    /// that walks the whole tree. These tests check that the parallel refit leaves every node exactly as the serial
    /// refit (<see cref="BoundingVolumeHierarchy.RefitFromRoot"/>) leaves it, for trees from the serial and the parallel
    /// build, small trees whose leaves sit above the cut, and a deep tree that overflows the walk's inline stack.
    /// </summary>
    class DynamicTreeRefitTests
    {
        const float TimeStep = 1.0f / 50.0f;
        static readonly float3 Gravity = new float3(0.0f, -9.81f, 0.0f);

        /// <summary>
        /// A world of dynamic boxes of different sizes. Each body has its own collider, so that the bodies' colliders
        /// are spread through memory as in a real scene.
        /// </summary>
        sealed class BoxWorld : IDisposable
        {
            public PhysicsWorld World;
            readonly List<BlobAssetReference<Collider>> colliders = new List<BlobAssetReference<Collider>>();

            public BoxWorld(IReadOnlyList<float3> positions, IReadOnlyList<float> sizes, uint seed)
            {
                int count = positions.Count;
                World = new PhysicsWorld(0, count, 0);
                var random = new Unity.Mathematics.Random(seed);
                NativeArray<Physics.RigidBody> bodies = World.DynamicBodies;
                for (int i = 0; i < count; i++)
                {
                    BlobAssetReference<Collider> collider = BoxCollider.Create(new BoxGeometry
                    {
                        Center = float3.zero,
                        Orientation = quaternion.identity,
                        Size = new float3(sizes[i]),
                        BevelRadius = math.min(0.01f, sizes[i] * 0.25f),
                    });
                    colliders.Add(collider);
                    var pose = new RigidTransform(random.NextQuaternionRotation(), positions[i]);
                    bodies[i] = new Physics.RigidBody
                    {
                        Collider = collider,
                        Entity = new Entity { Index = i + 1 },
                        Scale = 1.0f,
                        WorldFromBody = pose,
                    };
                    World.DynamicsWorld.m_MotionDatas[i] = new MotionData
                    {
                        WorldFromMotion = pose,
                        BodyFromMotion = RigidTransform.identity,
                    };
                    World.DynamicsWorld.m_MotionVelocities[i] = new MotionVelocity
                    {
                        LinearVelocity = random.NextFloat3(-5.0f, 5.0f),
                        AngularVelocity = random.NextFloat3(-1.0f, 1.0f),
                        InverseInertia = new float3(1.0f),
                        InverseMass = 1.0f,
                        AngularExpansionFactor = 0.5f * sizes[i],
                        GravityFactor = 1.0f,
                    };
                }
            }

            /// <summary>Moves every body far from where the tree was built, so that the tree's groups end up spread out.</summary>
            public void Move(uint seed, float distance)
            {
                var random = new Unity.Mathematics.Random(seed);
                for (int i = 0; i < World.NumDynamicBodies; i++)
                {
                    MotionData motion = World.DynamicsWorld.m_MotionDatas[i];
                    motion.WorldFromMotion = new RigidTransform(math.mul(random.NextQuaternionRotation(), motion.WorldFromMotion.rot),
                        motion.WorldFromMotion.pos + random.NextFloat3(-distance, distance));
                    World.DynamicsWorld.m_MotionDatas[i] = motion;
                }
            }

            /// <summary>Gives body i's collider the filter. Each body has a collider of its own, so no other body changes.</summary>
            public void SetFilter(int i, CollisionFilter filter)
            {
                colliders[i].Value.SetCollisionFilter(filter);
            }

            public void Dispose()
            {
                World.Dispose();
                foreach (BlobAssetReference<Collider> collider in colliders)
                {
                    collider.Dispose();
                }
            }
        }

        static (float3[] positions, float[] sizes) RandomScene(int count, uint seed)
        {
            var random = new Unity.Mathematics.Random(seed);
            var positions = new float3[count];
            var sizes = new float[count];
            for (int i = 0; i < count; i++)
            {
                // Clusters of bodies, as in several craft, with some bodies in the same place.
                float3 cluster = new float3((i % 7) * 40.0f, 0.0f, (i % 3) * 25.0f);
                positions[i] = i % 11 == 0 ? cluster : cluster + random.NextFloat3(-15.0f, 15.0f);
                sizes[i] = random.NextFloat(0.05f, 3.0f);
            }

            return (positions, sizes);
        }

        /// <summary>The nodes that the root reaches, found by walking the tree.</summary>
        static unsafe List<int> ReachedNodes(Node* nodes)
        {
            var reached = new List<int>();
            var pending = new Stack<int>();
            pending.Push(1);
            while (pending.Count > 0)
            {
                int index = pending.Pop();
                Node* node = nodes + index;
                reached.Add(index);
                if (node->IsInternal)
                {
                    for (int j = 0; j < 4; j++)
                    {
                        if (node->IsInternalValid(j))
                        {
                            pending.Push(node->Data[j]);
                        }
                    }
                }
            }

            return reached;
        }

        /// <summary>
        /// Builds the dynamic tree, moves the bodies, refits the tree as the step does, and refits a copy of the tree
        /// from before the step's refit with the serial refit. Every byte of every node must match, including the
        /// nodes that the root doesn't reach, which neither refit may touch. Returns the number of nodes in the tree's
        /// list that the root doesn't reach.
        /// </summary>
        static unsafe int CheckRefit(int count, int threads, uint seed)
        {
            (float3[] positions, float[] sizes) = RandomScene(count, seed);
            using var scene = new BoxWorld(positions, sizes, seed);
            ref PhysicsWorld world = ref scene.World;
            Broadphase.Tree tree = world.CollisionWorld.Broadphase.DynamicTree;
            world.CollisionWorld.Broadphase.ScheduleDynamicTreeBuildJobs(ref world, TimeStep, Gravity, threads, default).Complete();

            scene.Move(seed + 1, 60.0f);
            var before = new NativeArray<Node>(tree.Nodes.AsArray(), Allocator.TempJob);
            world.CollisionWorld.ScheduleRefitDynamicTree(ref world, TimeStep, Gravity, default).Complete();

            // The step's refit has moved the bodies to their motions' poses, so the bounds can be computed from them.
            // A Burst job computes them, as the step does, because code that Burst compiles can round differently
            // from the same code in C#.
            var aabbs = new NativeArray<Aabb>(count, Allocator.TempJob);
            new BoundsJob
            {
                MotionDatas = world.MotionDatas,
                MotionVelocities = world.MotionVelocities,
                RigidBodies = world.DynamicBodies,
                Aabbs = aabbs,
                Margin = world.CollisionWorld.CollisionTolerance * 0.5f,
                Gravity = Gravity,
            }.Schedule(count, 32, default).Complete();
            new SerialRefitJob { Nodes = (Node*)before.GetUnsafePtr(), NodeCount = before.Length, Aabbs = aabbs }.Run();

            NativeArray<Node> after = tree.Nodes.AsArray();
            Assert.That(after.Length, Is.EqualTo(before.Length));
            for (int n = 0; n < after.Length; n++)
            {
                if (UnsafeUtility.MemCmp((Node*)after.GetUnsafeReadOnlyPtr() + n, (Node*)before.GetUnsafePtr() + n, sizeof(Node)) != 0)
                {
                    Node a = after[n], b = before[n];
                    Assert.Fail($"node {n} of {after.Length} differs from the serial refit, with {count} bodies and {threads} threads: "
                        + $"leaf {a.IsLeaf}, data {a.Data}, compound {a.Bounds.GetCompoundAabb().Min} {a.Bounds.GetCompoundAabb().Max} "
                        + $"against {b.Bounds.GetCompoundAabb().Min} {b.Bounds.GetCompoundAabb().Max}");
                }
            }

            // Every leaf that the root reaches must hold the bounds of its bodies, and every body must be in one of
            // them, so that a mistake that both refits make can't hide.
            var seen = new bool[count];
            foreach (int n in ReachedNodes((Node*)after.GetUnsafeReadOnlyPtr()))
            {
                Node node = after[n];
                for (int j = 0; j < 4 && node.IsLeaf; j++)
                {
                    if (node.IsLeafValid(j))
                    {
                        Aabb leaf = node.Bounds.GetAabb(j);
                        Assert.That(leaf.Min, Is.EqualTo(aabbs[node.Data[j]].Min));
                        Assert.That(leaf.Max, Is.EqualTo(aabbs[node.Data[j]].Max));
                        seen[node.Data[j]] = true;
                    }
                }
            }

            Assert.That(seen, Is.All.True, "every body is in a leaf");
            aabbs.Dispose();
            before.Dispose();
            return after.Length - 1 - ReachedNodes((Node*)after.GetUnsafeReadOnlyPtr()).Count;
        }

        [Test]
        public void RefitMatchesTheSerialRefitForSmallTrees([Values(1, 2, 4, 5, 9, 17, 64, 65, 300)] int count)
        {
            CheckRefit(count, 1, (uint)count);
            CheckRefit(count, JobsUtility.JobWorkerCount + 1, (uint)count + 100);
        }

        [Test]
        public void RefitMatchesTheSerialRefitForASeriallyBuiltTree()
        {
            CheckRefit(4000, 1, 7);
        }

        /// <summary>
        /// The parallel build gives each branch a range of nodes that is large enough for any branch, so the tree's
        /// list holds unused nodes between the branches. Neither refit may touch them.
        /// </summary>
        [Test]
        public void RefitMatchesTheSerialRefitForATreeWithUnusedNodes()
        {
            int unused = CheckRefit(4000, 8, 11);
            Assert.That(unused, Is.GreaterThan(0), "the parallel build should leave unused nodes between its branches");
        }

        /// <summary>
        /// A chain of internal nodes, each with one leaf, is far deeper than any tree from the builder. Walking it
        /// moves the walk's stack from the stack frame to the heap.
        /// </summary>
        [Test]
        public unsafe void RefitMatchesTheSerialRefitForADeepTree()
        {
            const int depth = 400;
            int nodeCount = 2 + 2 * depth;
            var nodes = new NativeArray<Node>(nodeCount, Allocator.Temp);
            var aabbs = new NativeArray<Aabb>(depth + 1, Allocator.Temp);
            var random = new Unity.Mathematics.Random(3);
            for (int b = 0; b <= depth; b++)
            {
                float3 center = random.NextFloat3(-100.0f, 100.0f);
                aabbs[b] = new Aabb { Min = center - random.NextFloat(0.1f, 2.0f), Max = center + random.NextFloat(0.1f, 2.0f) };
            }

            // Node 1 + 2k is the chain's k-th internal node, and node 2 + 2k is its leaf. The chain's child is the
            // first child, so that the walk visits it last and keeps an entry for every level on its stack.
            nodes[0] = Node.Empty;
            for (int k = 0; k < depth; k++)
            {
                Node chain = Node.Empty;
                chain.Data = new int4(k + 1 < depth ? 3 + 2 * k : 2 * depth + 1, 2 + 2 * k, 0, 0);
                nodes[1 + 2 * k] = chain;
                Node leaf = Node.EmptyLeaf;
                leaf.Data = new int4(k, -1, -1, -1);
                nodes[2 + 2 * k] = leaf;
            }

            Node last = Node.EmptyLeaf;
            last.Data = new int4(depth, -1, -1, -1);
            nodes[2 * depth + 1] = last;

            var serialNodes = new NativeArray<Node>(nodes, Allocator.Temp);
            new BoundingVolumeHierarchy((Node*)serialNodes.GetUnsafePtr(), nodeCount, null).RefitFromRoot(aabbs);

            var parallel = new BoundingVolumeHierarchy((Node*)nodes.GetUnsafePtr(), nodeCount, null);
            for (int slot = RefitSubtreeCount - 1; slot >= 0; slot--)
            {
                int root = parallel.FindRefitSubtree(slot);
                if (root != 0)
                {
                    parallel.RefitSubtree(root, aabbs);
                }
            }

            parallel.RefitAboveCut();
            Assert.That(UnsafeUtility.MemCmp(nodes.GetUnsafePtr(), serialNodes.GetUnsafePtr(), (long)nodeCount * sizeof(Node)), Is.Zero);

            Aabb all = aabbs[0];
            for (int b = 1; b <= depth; b++)
            {
                all.Include(aabbs[b]);
            }

            Assert.That(nodes[1].Bounds.GetCompoundAabb().Min, Is.EqualTo(all.Min));
            Assert.That(nodes[1].Bounds.GetCompoundAabb().Max, Is.EqualTo(all.Max));
        }

        /// <summary>Every node that the root reaches is the root of exactly one slot's subtree or lies above the cut.</summary>
        [Test]
        public unsafe void EveryNodeBelowTheCutIsFittedOnce([Values(1, 5, 17, 300, 4000)] int count)
        {
            (float3[] positions, float[] sizes) = RandomScene(count, 5);
            using var scene = new BoxWorld(positions, sizes, 5);
            ref PhysicsWorld world = ref scene.World;
            Broadphase.Tree tree = world.CollisionWorld.Broadphase.DynamicTree;
            world.CollisionWorld.Broadphase.ScheduleDynamicTreeBuildJobs(ref world, TimeStep, Gravity, 8, default).Complete();

            BoundingVolumeHierarchy bvh = tree.BoundingVolumeHierarchy;
            var owner = new Dictionary<int, int>();
            for (int slot = 0; slot < RefitSubtreeCount; slot++)
            {
                int root = bvh.FindRefitSubtree(slot);
                if (root != 0)
                {
                    Assert.That(owner.ContainsKey(root), Is.False, $"node {root} is the subtree of slots {owner.GetValueOrDefault(root)} and {slot}");
                    owner[root] = slot;
                }
            }

            // Walk the tree to the cut. Each node at the cut, and each leaf above it, must be a subtree's root.
            var nodes = (Node*)tree.Nodes.GetUnsafeReadOnlyPtr();
            var pending = new Stack<(int index, int depth)>();
            pending.Push((1, 0));
            int roots = 0;
            while (pending.Count > 0)
            {
                (int index, int depth) = pending.Pop();
                Node* node = nodes + index;
                if (node->IsLeaf || depth == RefitCutDepth)
                {
                    Assert.That(owner.ContainsKey(index), Is.True, $"node {index} at depth {depth} has no slot");
                    roots++;
                    continue;
                }

                for (int j = 0; j < 4; j++)
                {
                    if (node->IsInternalValid(j))
                    {
                        pending.Push((node->Data[j], depth + 1));
                    }
                }
            }

            Assert.That(owner.Count, Is.EqualTo(roots));
        }

        /// <summary>The bounds of each body in the leaves of the dynamic tree, by the body's index.</summary>
        static unsafe Aabb[] LeafBounds(Broadphase.Tree tree, int count)
        {
            var bounds = new Aabb[count];
            var nodes = (Node*)tree.Nodes.GetUnsafeReadOnlyPtr();
            foreach (int n in ReachedNodes(nodes))
            {
                for (int j = 0; j < 4 && nodes[n].IsLeaf; j++)
                {
                    if (nodes[n].IsLeafValid(j))
                    {
                        bounds[nodes[n].Data[j]] = nodes[n].Bounds.GetAabb(j);
                    }
                }
            }

            return bounds;
        }

        /// <summary>
        /// Does to the bodies what the step's write-back and the next build do to them: each body gets its motion's
        /// pose, with the rotation normalized if <paramref name="normalize"/> is true, and a new velocity, as gravity
        /// and forces give it.
        /// </summary>
        static void WriteBack(ref PhysicsWorld world, uint seed, bool normalize = true)
        {
            // A Burst job normalizes the rotations, as the step's write-back does, because C# can round differently.
            if (normalize)
            {
                new NormalizeRotationsJob { RigidBodies = world.DynamicBodies }.Run(world.NumDynamicBodies);
            }

            var random = new Unity.Mathematics.Random(seed);
            for (int i = 0; i < world.NumDynamicBodies; i++)
            {
                MotionVelocity velocity = world.DynamicsWorld.m_MotionVelocities[i];
                velocity.LinearVelocity += random.NextFloat3(-1.0f, 1.0f);
                world.DynamicsWorld.m_MotionVelocities[i] = velocity;
            }
        }

        [BurstCompile(CompileSynchronously = true)]
        struct NormalizeRotationsJob : IJobParallelFor
        {
            public NativeArray<Physics.RigidBody> RigidBodies;

            public void Execute(int i)
            {
                Physics.RigidBody body = RigidBodies[i];
                body.WorldFromBody = new RigidTransform(math.normalize(body.WorldFromBody.rot), body.WorldFromBody.pos);
                RigidBodies[i] = body;
            }
        }

        /// <summary>
        /// Does what the backend's body build does for the cache: marks each body that has exactly the collider and pose
        /// that the refit left at its index.
        /// </summary>
        static unsafe void MarkSame(NativeArray<Physics.RigidBody> bodies, Physics.RigidBody[] refitted, DynamicBoundsCache cache)
        {
            DynamicBoundsCache.View view = cache.CurrentView;
            for (int i = 0; i < bodies.Length; i++)
            {
                view.Same[i] = DynamicBoundsCache.View.Matches(refitted[i], bodies[i]) ? (byte)1 : (byte)0;
            }
        }

        /// <summary>
        /// A new body at the index of a destroyed body can get a collider at the address of the destroyed body's
        /// collider, at the same pose. It isn't the same body, so it must not take the old body's bounds from the cache.
        /// </summary>
        [Test]
        public void ANewBodyWithAColliderAtTheOldAddressDoesNotMatch()
        {
            BlobAssetReference<Collider> collider = BoxCollider.Create(new BoxGeometry { Orientation = quaternion.identity, Size = new float3(1.0f) });
            try
            {
                var before = new Physics.RigidBody
                {
                    Collider = collider,
                    WorldFromBody = RigidTransform.identity,
                    Scale = 1.0f,
                    Entity = new Entity { Index = 3, Version = 1 },
                };
                Physics.RigidBody after = before;
                Assert.That(DynamicBoundsCache.View.Matches(before, after), Is.True, "the same body at the same pose");

                after.Entity = new Entity { Index = 3, Version = 2 };
                Assert.That(DynamicBoundsCache.View.Matches(before, after), Is.False, "a new body with a collider at the old address");
            }
            finally
            {
                collider.Dispose();
            }
        }

        /// <summary>
        /// An entry that held a destroyed body's collider takes the filter again when it is written for a new body
        /// whose collider has the same address. The test changes the filter of one collider between the two writes,
        /// in place of a new collider that the heap put at the old address.
        /// </summary>
        [Test]
        public unsafe void AnEntryWrittenForANewBodyAtTheOldColliderAddressTakesTheNewFilter()
        {
            BlobAssetReference<Collider> collider = BoxCollider.Create(new BoxGeometry { Orientation = quaternion.identity, Size = new float3(1.0f) });
            uint colliderVersion = 1;
            var cache = new DynamicBoundsCache(&colliderVersion, Allocator.Temp);
            try
            {
                DynamicBoundsCache.View view = cache.Resize(1);
                var body = new Physics.RigidBody
                {
                    Collider = collider,
                    WorldFromBody = RigidTransform.identity,
                    Scale = 1.0f,
                    Entity = new Entity { Index = 3, Version = 1 },
                };
                view.Write(0, body, body.CalculateAabb());

                var other = new CollisionFilter { BelongsTo = 2, CollidesWith = 4 };
                collider.Value.SetCollisionFilter(other);
                body.Entity = new Entity { Index = 3, Version = 2 };
                view.Write(0, body, body.CalculateAabb());

                Assert.That(view.Colliders[0].Filter.Equals(other), Is.True, "the entry holds the new body's filter");
            }
            finally
            {
                cache.Dispose();
                collider.Dispose();
            }
        }

        /// <summary>
        /// Computes the bounds of one body at many poses, through the cache's shape entry and from the collider, and
        /// counts the poses where the two differ in any bit. The first call makes the entry from the collider.
        /// </summary>
        [BurstCompile(CompileSynchronously = true)]
        unsafe struct ShapeBoundsJob : IJob
        {
            public Physics.RigidBody Body;
            public DynamicBoundsCache.View View;
            public NativeArray<RigidTransform> Poses;
            public NativeReference<int> Mismatches;
            public NativeReference<int> FromShapes;

            public void Execute()
            {
                Physics.RigidBody body = Body;
                for (int i = 0; i < Poses.Length; i++)
                {
                    body.WorldFromBody = Poses[i];
                    Aabb fromShape = View.ColliderBounds(0, &body);
                    Aabb fromCollider = body.CalculateAabb();
                    if (math.any(math.asuint(fromShape.Min) != math.asuint(fromCollider.Min)) || math.any(math.asuint(fromShape.Max) != math.asuint(fromCollider.Max)))
                    {
                        Mismatches.Value++;
                    }

                    if (View.Shapes[0].Kind != DynamicBoundsCache.ShapeBounds.ReadCollider)
                    {
                        FromShapes.Value++;
                    }
                }
            }
        }

        static BlobAssetReference<Collider> ShapeCollider(string kind)
        {
            var orientation = quaternion.Euler(0.3f, 1.1f, -0.7f);
            var center = new float3(0.4f, -1.3f, 2.2f);
            switch (kind)
            {
                case "box":
                    return BoxCollider.Create(new BoxGeometry { Orientation = orientation, Center = center, Size = new float3(1.0f, 2.5f, 0.3f), BevelRadius = 0.05f });
                case "sphere":
                    return SphereCollider.Create(new SphereGeometry { Center = center, Radius = 0.7f });
                case "capsule":
                    return CapsuleCollider.Create(new CapsuleGeometry { Vertex0 = center, Vertex1 = -center, Radius = 0.3f });
                case "cylinder":
                    return CylinderCollider.Create(new CylinderGeometry { Orientation = orientation, Center = center, Height = 2.0f, Radius = 0.5f, SideCount = 12 });
                case "convex":
                {
                    var points = new NativeArray<float3>(40, Allocator.Temp);
                    var random = new Unity.Mathematics.Random(7);
                    for (int i = 0; i < points.Length; i++)
                    {
                        points[i] = center + random.NextFloat3Direction() * new float3(1.0f, 2.0f, 0.5f);
                    }

                    BlobAssetReference<Collider> hull = ConvexCollider.Create(points, ConvexHullGenerationParameters.Default);
                    points.Dispose();
                    return hull;
                }
                case "mesh":
                {
                    var vertices = new NativeArray<float3>(4, Allocator.Temp);
                    vertices[0] = center;
                    vertices[1] = center + new float3(3.0f, 0.0f, 0.0f);
                    vertices[2] = center + new float3(0.0f, 0.0f, 3.0f);
                    vertices[3] = center + new float3(3.0f, 1.0f, 3.0f);
                    var triangles = new NativeArray<int3>(2, Allocator.Temp);
                    triangles[0] = new int3(0, 1, 2);
                    triangles[1] = new int3(1, 3, 2);
                    BlobAssetReference<Collider> mesh = MeshCollider.Create(vertices, triangles);
                    vertices.Dispose();
                    triangles.Dispose();
                    return mesh;
                }
                default:
                {
                    BlobAssetReference<Collider> box = ShapeCollider("box");
                    BlobAssetReference<Collider> sphere = ShapeCollider("sphere");
                    var children = new NativeArray<CompoundCollider.ColliderBlobInstance>(2, Allocator.Temp);
                    children[0] = new CompoundCollider.ColliderBlobInstance { Collider = box, CompoundFromChild = new RigidTransform(orientation, new float3(1.0f, 0.0f, 0.0f)) };
                    children[1] = new CompoundCollider.ColliderBlobInstance { Collider = sphere, CompoundFromChild = new RigidTransform(quaternion.identity, new float3(-2.0f, 0.5f, 0.0f)) };
                    BlobAssetReference<Collider> compound = CompoundCollider.Create(children);
                    children.Dispose();
                    box.Dispose();
                    sphere.Dispose();
                    return compound;
                }
            }
        }

        /// <summary>
        /// The bounds that the cache computes from what it keeps of a body's collider are the bounds that the collider
        /// computes, to the last bit, at poses near and far from the origin. A convex hull and a cylinder keep reading
        /// the collider.
        /// </summary>
        [TestCase("box", true)]
        [TestCase("sphere", true)]
        [TestCase("capsule", true)]
        [TestCase("compound", true)]
        [TestCase("mesh", true)]
        [TestCase("convex", false)]
        [TestCase("cylinder", false)]
        public unsafe void ShapeBoundsAreTheCollidersBoundsToTheLastBit(string kind, bool fromShapes)
        {
            BlobAssetReference<Collider> collider = ShapeCollider(kind);
            uint colliderVersion = 1;
            var cache = new DynamicBoundsCache(&colliderVersion, Allocator.TempJob);
            var poses = new NativeArray<RigidTransform>(4000, Allocator.TempJob);
            var mismatches = new NativeReference<int>(0, Allocator.TempJob);
            var shaped = new NativeReference<int>(0, Allocator.TempJob);
            bool wasEnabled = DynamicBoundsCache.ShapesEnabled;
            try
            {
                DynamicBoundsCache.ShapesEnabled = true;
                var random = new Unity.Mathematics.Random(11);
                for (int i = 0; i < poses.Length; i++)
                {
                    float reach = i % 4 == 0 ? 1.0f : i % 4 == 1 ? 100.0f : i % 4 == 2 ? 1e4f : 1e6f;
                    poses[i] = new RigidTransform(random.NextQuaternionRotation(), random.NextFloat3(-reach, reach));
                }

                new ShapeBoundsJob
                {
                    Body = new Physics.RigidBody { Collider = collider, Scale = 1.0f, Entity = new Entity { Index = 3, Version = 1 } },
                    View = cache.Resize(1),
                    Poses = poses,
                    Mismatches = mismatches,
                    FromShapes = shaped,
                }.Run();

                Assert.That(mismatches.Value, Is.Zero, $"{kind}: poses where the bounds differ");
                Assert.That(shaped.Value, Is.EqualTo(fromShapes ? poses.Length : 0), $"{kind}: poses where the cache kept the shape");
            }
            finally
            {
                DynamicBoundsCache.ShapesEnabled = wasEnabled;
                cache.Dispose();
                poses.Dispose();
                mismatches.Dispose();
                shaped.Dispose();
                collider.Dispose();
            }
        }

        /// <summary>
        /// The cache makes a body's shape entry again from the collider when the index gets another collider or
        /// another body, or when any body's collider changed since the entry was made, and only then.
        /// </summary>
        [Test]
        public unsafe void AShapeEntryIsMadeAgainForAnotherColliderBodyOrColliderVersion()
        {
            BlobAssetReference<Collider> small = BoxCollider.Create(new BoxGeometry { Orientation = quaternion.identity, Size = new float3(1.0f) });
            BlobAssetReference<Collider> large = BoxCollider.Create(new BoxGeometry { Orientation = quaternion.identity, Size = new float3(4.0f) });
            uint colliderVersion = 1;
            var cache = new DynamicBoundsCache(&colliderVersion, Allocator.Temp);
            bool wasEnabled = DynamicBoundsCache.ShapesEnabled;
            try
            {
                DynamicBoundsCache.ShapesEnabled = true;
                DynamicBoundsCache.View view = cache.Resize(1);

                // The test changes an entry by hand below, which the check would count as a mismatch.
                view.Check = null;
                var body = new Physics.RigidBody
                {
                    Collider = small,
                    WorldFromBody = new RigidTransform(quaternion.Euler(0.2f, 0.4f, 0.6f), new float3(5.0f, 1.0f, -2.0f)),
                    Scale = 1.0f,
                    Entity = new Entity { Index = 3, Version = 1 },
                };

                AssertSameBounds(body.CalculateAabb(), view.ColliderBounds(0, &body), "first call");
                Assert.That(view.Shapes[0].Collider, Is.EqualTo((long)small.GetUnsafePtr()));
                Assert.That(view.Shapes[0].Local.Max.x, Is.EqualTo(0.5f));

                // Another collider at the index.
                body.Collider = large;
                AssertSameBounds(body.CalculateAabb(), view.ColliderBounds(0, &body), "another collider");
                Assert.That(view.Shapes[0].Local.Max.x, Is.EqualTo(2.0f), "the entry holds the new collider's shape");

                // Another body with the same collider: the entry is made again, for the new body.
                body.Entity = new Entity { Index = 3, Version = 2 };
                view.ColliderBounds(0, &body);
                Assert.That(view.Shapes[0].Body, Is.EqualTo(body.Entity), "the entry belongs to the new body");

                // A collider change anywhere: the entry is made again. The test changes the entry's shape by hand, as
                // a collider at the same address with another shape would, and the new version replaces it.
                view.Shapes[0].Local.Max.x = 100.0f;
                AssertSameBounds(view.Shapes[0].At(body.WorldFromBody), view.ColliderBounds(0, &body), "the entry holds while the version holds");
                colliderVersion++;
                AssertSameBounds(body.CalculateAabb(), view.ColliderBounds(0, &body), "after a collider change");
                Assert.That(view.Shapes[0].Local.Max.x, Is.EqualTo(2.0f), "the entry was made again");
                Assert.That(view.Shapes[0].Version, Is.EqualTo(colliderVersion));

                // A scaled body never uses the entry.
                body.Scale = 2.0f;
                AssertSameBounds(body.CalculateAabb(), view.ColliderBounds(0, &body), "scaled");
            }
            finally
            {
                DynamicBoundsCache.ShapesEnabled = wasEnabled;
                cache.Dispose();
                small.Dispose();
                large.Dispose();
            }
        }

        static void AssertSameBounds(Aabb expected, Aabb actual, string what)
        {
            Assert.That(math.asuint(actual.Min), Is.EqualTo(math.asuint(expected.Min)), what);
            Assert.That(math.asuint(actual.Max), Is.EqualTo(math.asuint(expected.Max)), what);
        }

        /// <summary>
        /// The pairs of bodies that the step's parallel search finds in the trees, each with the lower index first,
        /// with the pair's solver type, sorted.
        /// </summary>
        static List<(int, int, int)> OverlapPairs(ref PhysicsWorld world)
        {
            SimulationJobHandles handles = world.CollisionWorld.Broadphase.ScheduleFindOverlapsJobs(
                out NativeStream dynamicPairs, out NativeStream staticPairs, default, multiThreaded: true);
            handles.FinalExecutionHandle.Complete();
            handles.FinalDisposeHandle.Complete();
            var pairs = new List<(int, int, int)>();
            foreach (NativeStream stream in new[] { dynamicPairs, staticPairs })
            {
                NativeStream.Reader reader = stream.AsReader();
                for (int i = 0; i < reader.ForEachCount; i++)
                {
                    int count = reader.BeginForEachIndex(i);
                    for (int j = 0; j < count; j++)
                    {
                        Broadphase.OverlapResult pair = reader.Read<Broadphase.OverlapResult>();
                        int a = pair.BodyPair.BodyIndexA, b = pair.BodyPair.BodyIndexB;
                        pairs.Add((math.min(a, b), math.max(a, b), (int)pair.SolverType));
                    }

                    reader.EndForEachIndex();
                }

                stream.Dispose();
            }

            pairs.Sort();
            return pairs;
        }

        /// <summary>
        /// A build that refits the tree of the last build (<see cref="Broadphase.ScheduleDynamicTreeRefitBuildJobs"/>),
        /// after the bodies have moved and some bodies' filters have changed, gives every body the bounds that a build
        /// from scratch gives it, to the last bit, and the search finds exactly the pairs that it finds in the tree of
        /// a build from scratch, though the two trees have different shapes. Some bodies first collide with nothing,
        /// so that the nodes above them exclude everything, and then collide with everything, so that a pair is lost
        /// unless the refit builds the nodes' filters again. Other bodies go the other way, or join a group that never
        /// collides with itself.
        /// </summary>
        [Test]
        public unsafe void ARefitBuildFindsThePairsOfABuildFromScratch([Values(1, 8)] int threads, [Values(false, true)] bool changeFilters)
        {
            const int count = 2000;
            (float3[] positions, float[] sizes) = RandomScene(count, 21);
            using var scene = new BoxWorld(positions, sizes, 21);
            ref PhysicsWorld world = ref scene.World;
            Broadphase.Tree tree = world.CollisionWorld.Broadphase.DynamicTree;
            var none = new CollisionFilter { BelongsTo = 0u, CollidesWith = 0u };
            var grouped = new CollisionFilter { BelongsTo = ~0u, CollidesWith = ~0u, GroupIndex = -3 };
            for (int i = 0; i < count; i += 5)
            {
                scene.SetFilter(i, none);
            }

            var results = (DynamicTreeRefitResults*)UnsafeUtility.Malloc(sizeof(DynamicTreeRefitResults), 16, Allocator.Persistent);
            UnsafeUtility.MemClear(results, sizeof(DynamicTreeRefitResults));
            try
            {
                world.CollisionWorld.Broadphase.ScheduleDynamicTreeBuildJobs(ref world, TimeStep, Gravity, threads, default).Complete();
                Node[] built = tree.Nodes.AsArray().ToArray();

                // As at the end of a step and the start of the next.
                scene.Move(22, 8.0f);
                world.CollisionWorld.ScheduleRefitDynamicTree(ref world, TimeStep, Gravity, default).Complete();
                WriteBack(ref world, 23);
                if (changeFilters)
                {
                    for (int i = 0; i < count; i += 5)
                    {
                        scene.SetFilter(i, CollisionFilter.Default);
                        scene.SetFilter(i + 1, i % 2 == 0 ? none : grouped);
                    }
                }

                world.CollisionWorld.Broadphase.ScheduleDynamicTreeRefitBuildJobs(ref world, TimeStep, Gravity, default, default, results).Complete();
                Assert.That(results->FiltersChanged, Is.EqualTo(changeFilters ? 1 : 0), "the refit saw whether a filter changed");
                Assert.That(results->Quality, Is.GreaterThan(1.0f), "the nodes hold more area than the bodies");
                NativeArray<Node> refitted = tree.Nodes.AsArray();
                for (int n = 0; n < refitted.Length; n++)
                {
                    Assert.That(refitted[n].Data, Is.EqualTo(built[n].Data), $"node {n} keeps the shape of the build");
                }

                Aabb[] refittedBounds = LeafBounds(tree, count);
                CollisionFilter[] refittedFilters = tree.BodyFilters.AsArray().ToArray();
                List<(int, int, int)> refittedPairs = OverlapPairs(ref world);

                world.CollisionWorld.Broadphase.ScheduleDynamicTreeBuildJobs(ref world, TimeStep, Gravity, threads, default).Complete();
                Aabb[] builtBounds = LeafBounds(tree, count);
                List<(int, int, int)> builtPairs = OverlapPairs(ref world);
                int moved = 0;
                for (int n = 0; n < math.min(built.Length, tree.Nodes.Length); n++)
                {
                    moved += math.any(tree.Nodes[n].Data != built[n].Data) ? 1 : 0;
                }

                UnityEngine.Debug.Log($"[DynamicTreeRefitTests] {builtPairs.Count} pairs, {moved} nodes of the new build differ from the old one's, quality {results->Quality}");
                Assert.That(moved, Is.GreaterThan(0), "the build from scratch made a tree of another shape");
                Assert.That(builtPairs.Count, Is.GreaterThan(100), "the scene has pairs");
                for (int i = 0; i < count; i++)
                {
                    AssertSameBounds(builtBounds[i], refittedBounds[i], $"body {i}'s bounds");
                    Assert.That(refittedFilters[i], Is.EqualTo(tree.BodyFilters[i]), $"body {i}'s filter");
                }

                Assert.That(refittedPairs, Is.EqualTo(builtPairs));
            }
            finally
            {
                UnsafeUtility.Free(results, Allocator.Persistent);
            }
        }

        /// <summary>
        /// With the bounds cache, a refit build reads a body's filter from the cache only for the body that the cache
        /// holds. A body at the same index whose collider keeps its address, but which is another body (another
        /// entity) with another filter, as when the heap gives a new body's collider the address of a destroyed body's
        /// collider, gets its own filter, and the refit builds the nodes' filters again. The search then finds exactly
        /// the pairs of a build from scratch.
        /// </summary>
        [Test]
        public unsafe void ARefitBuildWithTheCacheTellsANewBodyAtAnOldColliderAddress()
        {
            const int count = 1000;
            (float3[] positions, float[] sizes) = RandomScene(count, 31);
            using var scene = new BoxWorld(positions, sizes, 31);
            ref PhysicsWorld world = ref scene.World;
            var none = new CollisionFilter { BelongsTo = 0u, CollidesWith = 0u };
            for (int i = 0; i < count; i += 4)
            {
                scene.SetFilter(i, none);
            }

            uint version = 1;
            var cache = new DynamicBoundsCache(&version, Allocator.Persistent);
            var results = (DynamicTreeRefitResults*)UnsafeUtility.Malloc(sizeof(DynamicTreeRefitResults), 16, Allocator.Persistent);
            UnsafeUtility.MemClear(results, sizeof(DynamicTreeRefitResults));
            try
            {
                world.CollisionWorld.Broadphase.ScheduleDynamicTreeBuildJobs(ref world, TimeStep, Gravity, 8, default, cache).Complete();
                cache.MarkCurrent();
                world.CollisionWorld.ScheduleRefitDynamicTree(ref world, TimeStep, Gravity, default, cache).Complete();
                Physics.RigidBody[] refitted = world.DynamicBodies.ToArray();

                // Every fourth body becomes another body with the same collider memory and a filter that collides.
                NativeArray<Physics.RigidBody> bodies = world.DynamicBodies;
                for (int i = 0; i < count; i += 4)
                {
                    scene.SetFilter(i, CollisionFilter.Default);
                    Physics.RigidBody body = bodies[i];
                    body.Entity = new Entity { Index = body.Entity.Index, Version = 7 };
                    bodies[i] = body;
                }

                MarkSame(bodies, refitted, cache);
                world.CollisionWorld.Broadphase.ScheduleDynamicTreeRefitBuildJobs(ref world, TimeStep, Gravity, default, cache, results).Complete();
                Assert.That(results->FiltersChanged, Is.EqualTo(1), "the refit saw the new bodies' filters");
                Broadphase.Tree tree = world.CollisionWorld.Broadphase.DynamicTree;
                for (int i = 0; i < count; i += 4)
                {
                    Assert.That(tree.BodyFilters[i], Is.EqualTo(CollisionFilter.Default), $"body {i}'s filter");
                }

                List<(int, int, int)> refittedPairs = OverlapPairs(ref world);
                world.CollisionWorld.Broadphase.ScheduleDynamicTreeBuildJobs(ref world, TimeStep, Gravity, 8, default).Complete();
                List<(int, int, int)> builtPairs = OverlapPairs(ref world);
                Assert.That(builtPairs.Count, Is.GreaterThan(100), "the scene has pairs");
                Assert.That(refittedPairs, Is.EqualTo(builtPairs));
            }
            finally
            {
                UnsafeUtility.Free(results, Allocator.Persistent);
                cache.Dispose();
            }
        }

        /// <summary>
        /// The refit leaves each body's collider bounds in the cache at the body's new pose, and the next build takes
        /// them from there for each body that is still at that pose. That build gives every body the same bounds, to the
        /// last bit, and the same filter, as a build without the cache.
        /// </summary>
        [Test]
        public unsafe void ABuildWithTheBoundsFromTheRefitMatchesABuildWithout([Values(1, 8)] int threads)
        {
            const int count = 3000;
            (float3[] positions, float[] sizes) = RandomScene(count, 11);
            using var scene = new BoxWorld(positions, sizes, 11);
            ref PhysicsWorld world = ref scene.World;
            Broadphase.Tree tree = world.CollisionWorld.Broadphase.DynamicTree;
            uint version = 1;
            var cache = new DynamicBoundsCache(&version, Allocator.Persistent);
            try
            {
                world.CollisionWorld.Broadphase.ScheduleDynamicTreeBuildJobs(ref world, TimeStep, Gravity, threads, default, cache).Complete();
                cache.MarkCurrent();
                scene.Move(12, 20.0f);
                world.CollisionWorld.ScheduleRefitDynamicTree(ref world, TimeStep, Gravity, default, cache).Complete();
                Physics.RigidBody[] refitted = world.DynamicBodies.ToArray();
                WriteBack(ref world, 13);
                MarkSame(world.DynamicBodies, refitted, cache);

                // The refit leaves the bounds at the motions' poses. The write-back's normalizing changes the last bits
                // of some rotations, and those bodies compute their bounds again.
                DynamicBoundsCache.View view = cache.CurrentView;
                NativeArray<Physics.RigidBody> bodies = world.DynamicBodies;
                int held = 0;
                for (int i = 0; i < count; i++)
                {
                    held += view.Holds(i, bodies[i]) ? 1 : 0;
                }

                UnityEngine.Debug.Log($"[DynamicTreeRefitTests] the cache holds {held} of {count} bodies where the write-back put them");
                Assert.That(held, Is.GreaterThan(count / 2), "the cache holds most bodies where the write-back put them");

                world.CollisionWorld.Broadphase.ScheduleDynamicTreeBuildJobs(ref world, TimeStep, Gravity, threads, default, cache).Complete();
                Aabb[] cached = LeafBounds(tree, count);
                CollisionFilter[] cachedFilters = tree.BodyFilters.AsArray().ToArray();
                bool[] cachedResponds = tree.RespondsToCollision.AsArray().ToArray();

                world.CollisionWorld.Broadphase.ScheduleDynamicTreeBuildJobs(ref world, TimeStep, Gravity, threads, default).Complete();
                Aabb[] computed = LeafBounds(tree, count);
                for (int i = 0; i < count; i++)
                {
                    AssertSameBounds(computed[i], cached[i], $"body {i}'s bounds");
                    Assert.That(cachedFilters[i], Is.EqualTo(tree.BodyFilters[i]), $"body {i}'s filter");
                    Assert.That(cachedResponds[i], Is.EqualTo(tree.RespondsToCollision[i]), $"body {i}'s response");
                }
            }
            finally
            {
                cache.Dispose();
            }
        }

        /// <summary>
        /// A body takes its collider bounds from the cache only when the cache holds its collider at exactly its pose,
        /// and only while no collider has changed since the cache recorded the colliders' version. The test fills the
        /// cache with bounds that no collider has, so that the bounds in the tree show where they came from.
        /// </summary>
        [Test]
        public unsafe void ABodyTakesItsBoundsFromTheCacheOnlyForItsColliderAndPose()
        {
            const int count = 64;
            (float3[] positions, float[] sizes) = RandomScene(count, 21);
            using var scene = new BoxWorld(positions, sizes, 21);
            ref PhysicsWorld world = ref scene.World;
            Broadphase.Tree tree = world.CollisionWorld.Broadphase.DynamicTree;
            uint version = 1;
            var cache = new DynamicBoundsCache(&version, Allocator.Persistent);
            BlobAssetReference<Collider> other = BoxCollider.Create(new BoxGeometry
            {
                Center = float3.zero,
                Orientation = quaternion.identity,
                Size = new float3(2.0f),
            });
            try
            {
                world.CollisionWorld.Broadphase.ScheduleDynamicTreeBuildJobs(ref world, TimeStep, Gravity, 1, default, cache).Complete();
                cache.MarkCurrent();
                scene.Move(22, 20.0f);
                world.CollisionWorld.ScheduleRefitDynamicTree(ref world, TimeStep, Gravity, default, cache).Complete();
                Physics.RigidBody[] refitted = world.DynamicBodies.ToArray();
                WriteBack(ref world, 23, false);

                var planted = new Aabb { Min = new float3(-500.0f), Max = new float3(500.0f) };
                DynamicBoundsCache.View view = cache.CurrentView;
                for (int i = 0; i < count; i++)
                {
                    view.Bounds[i] = planted;
                }

                // Body 0 moves by one bit of its position, and body 1 gets another collider.
                NativeArray<Physics.RigidBody> bodies = world.DynamicBodies;
                Physics.RigidBody moved = bodies[0];
                moved.WorldFromBody.pos.x = math.asfloat(math.asuint(moved.WorldFromBody.pos.x) + 1u);
                bodies[0] = moved;
                Physics.RigidBody changed = bodies[1];
                changed.Collider = other;
                bodies[1] = changed;
                MarkSame(bodies, refitted, cache);

                world.CollisionWorld.Broadphase.ScheduleDynamicTreeBuildJobs(ref world, TimeStep, Gravity, 1, default, cache).Complete();
                Aabb[] fromCache = LeafBounds(tree, count);
                world.CollisionWorld.Broadphase.ScheduleDynamicTreeBuildJobs(ref world, TimeStep, Gravity, 1, default).Complete();
                Aabb[] computed = LeafBounds(tree, count);
                AssertSameBounds(computed[0], fromCache[0], "the body that moved computes its bounds");
                AssertSameBounds(computed[1], fromCache[1], "the body with another collider computes its bounds");
                for (int i = 2; i < count; i++)
                {
                    Assert.That(fromCache[i].Contains(planted), Is.True, $"body {i} takes its bounds from the cache");
                }

                // The cache now holds the right bounds for bodies 0 and 1. Plant the false bounds again, and change
                // the colliders' version: then no body takes its bounds from the cache.
                for (int i = 0; i < count; i++)
                {
                    view.Bounds[i] = planted;
                }

                version++;
                world.CollisionWorld.Broadphase.ScheduleDynamicTreeBuildJobs(ref world, TimeStep, Gravity, 1, default, cache).Complete();
                Aabb[] afterChange = LeafBounds(tree, count);
                for (int i = 0; i < count; i++)
                {
                    AssertSameBounds(computed[i], afterChange[i], $"body {i} computes its bounds after a collider changed");
                }
            }
            finally
            {
                cache.Dispose();
                other.Dispose();
            }
        }

        // The benchmark below times the jobs in this file, which are compiled before their first run, and not the
        // step's jobs, which Burst may still be compiling when a test starts.

        [BurstCompile(CompileSynchronously = true)]
        struct BoundsJob : IJobParallelFor
        {
            [ReadOnly] public NativeArray<MotionData> MotionDatas;
            [ReadOnly] public NativeArray<MotionVelocity> MotionVelocities;
            public NativeArray<Physics.RigidBody> RigidBodies;
            [WriteOnly] public NativeArray<Aabb> Aabbs;
            public float Margin;
            public float3 Gravity;

            public void Execute(int i)
            {
                Physics.RigidBody body = RigidBodies[i];
                body.WorldFromBody = math.mul(MotionDatas[i].WorldFromMotion, math.inverse(MotionDatas[i].BodyFromMotion));
                RigidBodies[i] = body;
                Aabbs[i] = Broadphase.PrepareDynamicBodyDataJob.CalculateAabb(ref body, Margin, Gravity, TimeStep, MotionVelocities[i]);
            }
        }

        [BurstCompile(CompileSynchronously = true)]
        unsafe struct SerialRefitJob : IJob
        {
            [NativeDisableUnsafePtrRestriction] public Node* Nodes;
            public int NodeCount;
            [ReadOnly] public NativeArray<Aabb> Aabbs;

            public void Execute()
            {
                new BoundingVolumeHierarchy(Nodes, NodeCount, null).RefitFromRoot(Aabbs);
            }
        }

        /// <summary>Only the listing of the nodes that <see cref="BoundingVolumeHierarchy.RefitFromRoot"/> makes first.</summary>
        [BurstCompile(CompileSynchronously = true)]
        unsafe struct ListingJob : IJob
        {
            [NativeDisableUnsafePtrRestriction] public Node* Nodes;
            public int NodeCount;
            public NativeArray<int> Listed;

            public void Execute()
            {
                int* order = (int*)UnsafeUtility.Malloc((long)NodeCount * sizeof(int), 4, Allocator.Temp);
                int* pending = (int*)UnsafeUtility.Malloc((long)NodeCount * sizeof(int), 4, Allocator.Temp);
                int listed = 0;
                int waiting = 0;
                pending[waiting++] = 1;
                while (waiting > 0)
                {
                    int index = pending[--waiting];
                    order[listed++] = index;
                    Node* node = Nodes + index;
                    if (node->IsInternal)
                    {
                        for (int j = 0; j < 4; j++)
                        {
                            if (node->IsInternalValid(j))
                            {
                                pending[waiting++] = node->Data[j];
                            }
                        }
                    }
                }

                Listed[0] = listed;
                UnsafeUtility.Free(order, Allocator.Temp);
                UnsafeUtility.Free(pending, Allocator.Temp);
            }
        }

        [BurstCompile(CompileSynchronously = true)]
        unsafe struct SubtreesJob : IJobParallelFor
        {
            [NativeDisableUnsafePtrRestriction] public Node* Nodes;
            public int NodeCount;
            [ReadOnly] public NativeArray<Aabb> Aabbs;

            public void Execute(int slot)
            {
                var bvh = new BoundingVolumeHierarchy(Nodes, NodeCount, null);
                int root = bvh.FindRefitSubtree(slot);
                if (root != 0)
                {
                    bvh.RefitSubtree(root, Aabbs);
                }
            }
        }

        [BurstCompile(CompileSynchronously = true)]
        unsafe struct TopJob : IJob
        {
            [NativeDisableUnsafePtrRestriction] public Node* Nodes;
            public int NodeCount;

            public void Execute()
            {
                new BoundingVolumeHierarchy(Nodes, NodeCount, null).RefitAboveCut();
            }
        }

        /// <summary>Writes a buffer much larger than the caches, so that the next job finds nothing of its data there.</summary>
        [BurstCompile(CompileSynchronously = true)]
        struct FlushJob : IJobParallelFor
        {
            public NativeArray<int4> Buffer;
            public int Pass;

            public void Execute(int i)
            {
                Buffer[i] = Buffer[i] + Pass;
            }
        }

        /// <summary>
        /// Times the parts of the dynamic tree's refit on a fleet of four Tu-160s (10,868 boxes), with the caches
        /// flushed before each run (as at the end of a step, after the solve has used the caches) and without.
        /// Results go to the log and to Logs/DynamicTreeRefitBench.txt.
        /// </summary>
        [Test]
        [Explicit("A benchmark")]
        public unsafe void DynamicTreeRefitBench()
        {
            string craft = Environment.GetEnvironmentVariable("RETOOLED_BENCH_CRAFT") ?? "tu160";
            int copies = int.TryParse(Environment.GetEnvironmentVariable("RETOOLED_BENCH_COPIES"), out int c) ? c : 4;
            KSP.Retooled.Physics.Tests.VesselGraph graph = KSP.Retooled.Physics.Tests.VesselGraph.Load(craft);
            double left = graph.Position.Min(p => p.x), right = graph.Position.Max(p => p.x);
            var positions = new List<float3>();
            var sizes = new List<float>();
            for (int copy = 0; copy < copies; copy++)
            {
                for (int i = 0; i < graph.Count; i++)
                {
                    double inertia = math.max(math.csum(graph.Inertia[i]) / 3.0, 1e-6);
                    double mass = math.max(graph.Mass[i], 1e-3);
                    sizes.Add((float)math.clamp(math.sqrt(6.0 * inertia / mass), 0.05, 5.0));
                    positions.Add((float3)(graph.Position[i] + new double3(copy * (right - left + 10.0), 0.0, 0.0)));
                }
            }

            using var scene = new BoxWorld(positions, sizes, 1);
            ref PhysicsWorld world = ref scene.World;
            int count = world.NumDynamicBodies;
            int threads = JobsUtility.JobWorkerCount + 1;
            Broadphase.Tree tree = world.CollisionWorld.Broadphase.DynamicTree;
            world.CollisionWorld.Broadphase.ScheduleDynamicTreeBuildJobs(ref world, TimeStep, float3.zero, threads, default).Complete();
            scene.Move(2, 0.01f);

            var nodes = (Node*)tree.Nodes.GetUnsafePtr();
            int nodeCount = tree.Nodes.Length;
            int reached = ReachedNodes(nodes).Count;
            int leaves = 0, slots = 0, depth = 0;
            var pending = new Stack<(int, int)>();
            pending.Push((1, 0));
            while (pending.Count > 0)
            {
                (int index, int d) = pending.Pop();
                depth = math.max(depth, d);
                Node* node = nodes + index;
                if (node->IsLeaf)
                {
                    leaves++;
                    slots += math.csum(math.select(int4.zero, new int4(1), node->AreLeavesValid));
                    continue;
                }

                for (int j = 0; j < 4; j++)
                {
                    if (node->IsInternalValid(j))
                    {
                        pending.Push((node->Data[j], d + 1));
                    }
                }
            }

            var aabbs = new NativeArray<Aabb>(count, Allocator.TempJob);
            var listed = new NativeArray<int>(1, Allocator.TempJob);
            var flush = new NativeArray<int4>(4 << 20, Allocator.TempJob); // 64 MB
            float margin = world.CollisionWorld.CollisionTolerance * 0.5f;
            var times = new Dictionary<string, List<double>>();
            void Time(string name, Func<JobHandle> schedule)
            {
                var watch = Stopwatch.StartNew();
                schedule().Complete();
                if (!times.TryGetValue(name, out List<double> list))
                {
                    times[name] = list = new List<double>();
                }

                list.Add(watch.Elapsed.TotalMilliseconds);
            }

            NativeArray<MotionData> motionDatas = world.MotionDatas;
            NativeArray<MotionVelocity> motionVelocities = world.MotionVelocities;
            NativeArray<Physics.RigidBody> dynamicBodies = world.DynamicBodies;
            JobHandle Bounds() => new BoundsJob
            {
                MotionDatas = motionDatas,
                MotionVelocities = motionVelocities,
                RigidBodies = dynamicBodies,
                Aabbs = aabbs,
                Margin = margin,
            }.Schedule(count, 32, default);

            JobHandle Serial() => new SerialRefitJob { Nodes = nodes, NodeCount = nodeCount, Aabbs = aabbs }.Schedule();
            JobHandle Listing() => new ListingJob { Nodes = nodes, NodeCount = nodeCount, Listed = listed }.Schedule();
            JobHandle Parallel()
            {
                JobHandle fitted = new SubtreesJob { Nodes = nodes, NodeCount = nodeCount, Aabbs = aabbs }.Schedule(RefitSubtreeCount, 1, default);
                return new TopJob { Nodes = nodes, NodeCount = nodeCount }.Schedule(fitted);
            }

            void Flush(int pass) => new FlushJob { Buffer = flush, Pass = pass }.Schedule(flush.Length, 4096, default).Complete();

            for (int round = 0; round < 60; round++)
            {
                bool measured = round >= 10;
                foreach (bool cold in new[] { true, false })
                {
                    string state = cold ? "cold" : "warm";
                    if (cold) Flush(round);
                    Time($"bounds (parallel), {state}", Bounds);
                    if (cold) Flush(round);
                    Time($"serial refit, {state}", Serial);
                    if (cold) Flush(round);
                    Time($"listing only, {state}", Listing);
                    if (cold) Flush(round);
                    Time($"parallel refit, {state}", Parallel);
                    if (cold) Flush(round);
                    Bounds().Complete();
                    Time($"serial refit after a parallel bounds job, {state}", Serial);
                    if (cold) Flush(round);
                    Bounds().Complete();
                    Time($"parallel refit after a parallel bounds job, {state}", Parallel);
                }

                if (!measured)
                {
                    times.Clear();
                }
            }

            aabbs.Dispose();
            listed.Dispose();
            flush.Dispose();

            var text = new System.Text.StringBuilder();
            text.AppendLine($"{craft} x{copies}: {count} bodies, {nodeCount} nodes in the list, {reached} reached, {leaves} leaves "
                + $"holding {slots} bodies ({(double)slots / leaves:F2} per leaf), depth {depth}, node size {sizeof(Node)} B, "
                + $"{threads} threads, {RefitSubtreeCount} subtrees");
            foreach (KeyValuePair<string, List<double>> entry in times)
            {
                List<double> sorted = entry.Value.OrderBy(t => t).ToList();
                text.AppendLine($"  {entry.Key,-52} min {sorted[0]:F4} ms  median {sorted[sorted.Count / 2]:F4} ms");
            }

            UnityEngine.Debug.Log(text.ToString());
            File.AppendAllText(Path.Combine(Application.dataPath, "..", "Logs", "DynamicTreeRefitBench.txt"), text + "\n");
        }
    }
}
