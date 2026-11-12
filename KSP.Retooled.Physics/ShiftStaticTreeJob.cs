using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Physics;

namespace KSP.Retooled.Physics
{
    /// <summary>
    /// Decides whether the static tree has to be built again, or can be moved instead. KSP's floating origin moves every static
    /// by the same offset, often every frame, and building the tree again costs a bounding box of every static. When the statics
    /// only moved, nearly all of them by one offset, this moves the tree's bounds and writes 0 to <see cref="Rebuild"/>;
    /// otherwise it writes 1.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Unity works each static's world pose out in floats, under parents that may be a planet's width away, so the statics
    /// don't all move by exactly the offset: a terrain collider under a planet's centre 600 km off misses it by centimetres.
    /// Each node's bounds grow by the furthest any static below it missed, plus the rounding of the move, which keeps every
    /// static inside its bounds. Each move starts from the bounds as built, so the rounding of the moves doesn't add up.
    /// </para>
    /// <para>
    /// Some statics move their own way, turning or moving with parents of their own. Each of those, and each static that drifts too far
    /// from the offset, gets a bounding box at its new pose, and the nodes above it are fitted around their children again.
    /// Once too many statics need that, building the tree again is the better deal.
    /// </para>
    /// </remarks>
    [BurstCompile]
    internal unsafe struct ShiftStaticTreeJob : IJob
    {
        /// <summary>
        /// The most statics a move fits one by one. Fitting one costs a bounding box and a walk up the tree, where building
        /// the tree again costs a box for every static and sorting them all, so a quarter of them is still the better deal.
        /// In flight about a hundred of KSP's statics move their own way every frame.
        /// </summary>
        private readonly int MaxStrays => math.max(64, StaticBodies.Length / 4);

        /// <summary>How far a static may miss the offset, in metres, before it is fitted on its own.</summary>
        private const float MaxMiss = 1.0f;

        /// <summary>The built statics this build, the anchor last.</summary>
        [ReadOnly]
        public NativeArray<RigidBody> StaticBodies;

        // The static tree's lists. Taken without their safety handles, because the tree's build jobs, scheduled after this
        // one, ask for the same lists as arrays on the main thread.
        [NativeDisableUnsafePtrRestriction]
        public UnsafeList<BoundingVolumeHierarchy.Node>* Nodes;

        [NativeDisableUnsafePtrRestriction]
        public UnsafeList<FourTransposedAabbs>* MeshSubAabbs;

        public ShiftState State;

        [NativeDisableUnsafePtrRestriction]
        public NativeStore.BodyData* Store;

        /// <summary>
        /// The bodies whose colliders this build's refresh built: new bodies, and bodies whose shapes changed. A static among
        /// them may have changed in ways its pose and collider pointer don't show, such as its filter, and a collider freed
        /// since the tree was built can come back at the same address. A build without a refresh passes the last refresh's,
        /// which at worst builds the tree again for nothing.
        /// </summary>
        [ReadOnly]
        [NativeDisableContainerSafetyRestriction]
        public NativeList<ColliderBuilder.Work> Refreshed;

        /// <summary>False when the tree may not have been built from the held statics, such as after a build that failed.</summary>
        public bool TreeMatchesHeld;

        /// <summary>What the tree's build adds around each static's bounding box.</summary>
        public float Margin;

        /// <summary>What the static tree's build jobs read. Only this job writes it, and they wait for it.</summary>
        [NativeDisableContainerSafetyRestriction]
        public NativeReference<int> Rebuild;

        /// <summary>A static as the tree was built from it.</summary>
        internal struct TreeStatic
        {
            public RigidTransform WorldFromBody;
            public long Collider;
        }

        /// <summary>
        /// What the job keeps between builds. The world's step state owns it. Only this job touches it, and each build waits
        /// for the last, so it does without safety handles, like the tree's lists.
        /// </summary>
        internal struct ShiftState
        {
            /// <summary>The statics the tree was last built from, in the order of the built statics.</summary>
            [NativeDisableContainerSafetyRestriction]
            public NativeList<TreeStatic> Held;

            /// <summary>The tree's node bounds and mesh bounds as built, once a move has copied them.</summary>
            [NativeDisableContainerSafetyRestriction]
            public NativeList<FourTransposedAabbs> BuiltNodes;

            [NativeDisableContainerSafetyRestriction]
            public NativeList<FourTransposedAabbs> BuiltMeshes;

            /// <summary>The nodes reachable from the root, each after its parent.</summary>
            [NativeDisableContainerSafetyRestriction]
            public NativeList<int> Reached;

            /// <summary>For each node, the slot in its parent that holds it, as node * 4 + slot; -1 for the root.</summary>
            [NativeDisableContainerSafetyRestriction]
            public NativeList<int> SlotOfNode;

            /// <summary>For each static, the leaf slot that holds it, as node * 4 + slot.</summary>
            [NativeDisableContainerSafetyRestriction]
            public NativeList<int> SlotOfStatic;

            /// <summary>1 once the tree as built is copied, 0 while the copies are stale.</summary>
            [NativeDisableContainerSafetyRestriction]
            public NativeReference<int> Copied;

            internal static ShiftState Create()
            {
                return new ShiftState
                {
                    Held = new NativeList<TreeStatic>(Allocator.Persistent),
                    BuiltNodes = new NativeList<FourTransposedAabbs>(Allocator.Persistent),
                    BuiltMeshes = new NativeList<FourTransposedAabbs>(Allocator.Persistent),
                    Reached = new NativeList<int>(Allocator.Persistent),
                    SlotOfNode = new NativeList<int>(Allocator.Persistent),
                    SlotOfStatic = new NativeList<int>(Allocator.Persistent),
                    Copied = new NativeReference<int>(0, Allocator.Persistent),
                };
            }

            internal void Dispose()
            {
                Held.Dispose();
                BuiltNodes.Dispose();
                BuiltMeshes.Dispose();
                Reached.Dispose();
                SlotOfNode.Dispose();
                SlotOfStatic.Dispose();
                Copied.Dispose();
            }
        }

        public void Execute()
        {
            var strays = new NativeList<int>(MaxStrays, Allocator.Temp);
            var misses = new NativeArray<float>(StaticBodies.Length, Allocator.Temp);
            float3 offset = float3.zero;
            bool shift = TreeMatchesHeld
                && State.Held.Length == StaticBodies.Length
                && !RefreshedAStatic()
                && FindOffset(strays, misses, out offset);

            if (shift && State.Copied.Value == 0)
            {
                shift = CopyBuiltTree();
            }

            if (shift)
            {
                MoveBounds(offset, misses);
                FitStrays(strays);
                Rebuild.Value = 0;
            }
            else
            {
                Hold();
                State.Copied.Value = 0;
                Rebuild.Value = 1;
            }
        }

        private readonly bool RefreshedAStatic()
        {
            for (int i = 0; i < Refreshed.Length; i++)
            {
                if ((Store[Refreshed[i].Body].flags & NativeStore.BodyFlags.Static) != 0)
                {
                    return true;
                }
            }

            return false;
        }

        /// <summary>
        /// The offset that moved most statics since the tree was built, how far each of those missed it, and the statics that
        /// moved some other way. False if a collider changed, or too many statics stray. Statics without a collider never
        /// touch anything, so they don't count.
        /// </summary>
        private readonly bool FindOffset(NativeList<int> strays, NativeArray<float> misses, out float3 offset)
        {
            offset = float3.zero;
            TreeStatic* held = State.Held.GetUnsafeReadOnlyPtr();
            RigidBody* bodies = (RigidBody*)StaticBodies.GetUnsafeReadOnlyPtr();
            for (int i = 0; i < StaticBodies.Length; i++)
            {
                if ((long)bodies[i].Collider.GetUnsafePtr() != held[i].Collider)
                {
                    return false;
                }
            }

            // Try the moves of a few statics spread through the list, in case the first one tried strays itself.
            const int Candidates = 4;
            for (int candidate = 0; candidate < Candidates; candidate++)
            {
                int index = FirstWithCollider(candidate * StaticBodies.Length / Candidates);
                if (index < 0)
                {
                    return candidate == 0;
                }

                double3 moved = (double3)bodies[index].WorldFromBody.pos - held[index].WorldFromBody.pos;
                offset = (float3)moved;
                if (Sort(bodies, held, moved, (double3)offset, strays, misses))
                {
                    return true;
                }
            }

            return false;
        }

        /// <summary>
        /// Sorts the statics into those that moved by the offset, with how far each missed it, and strays. False as soon as
        /// there are too many strays.
        /// </summary>
        private readonly bool Sort(RigidBody* bodies, TreeStatic* held, double3 moved, double3 offset, NativeList<int> strays, NativeArray<float> misses)
        {
            strays.Clear();

            // The offset itself rounds to a float, which every static misses by as well.
            double rounding = math.cmax(math.abs(offset - moved));
            for (int i = 0; i < StaticBodies.Length; i++)
            {
                if (!bodies[i].Collider.IsCreated)
                {
                    continue;
                }

                // Differences of floats are exact in doubles.
                double missed = math.cmax(math.abs((double3)bodies[i].WorldFromBody.pos - held[i].WorldFromBody.pos - moved));
                if (missed > MaxMiss || math.any(bodies[i].WorldFromBody.rot.value != held[i].WorldFromBody.rot.value))
                {
                    if (strays.Length == MaxStrays)
                    {
                        return false;
                    }

                    strays.Add(i);
                    misses[i] = 0.0f;
                }
                else
                {
                    misses[i] = (float)(missed + rounding);
                }
            }

            return true;
        }

        private readonly int FirstWithCollider(int from)
        {
            for (int i = from; i < StaticBodies.Length; i++)
            {
                if (StaticBodies[i].Collider.IsCreated)
                {
                    return i;
                }
            }

            return -1;
        }

        /// <summary>
        /// Copies the tree as built, which every move starts from until the next build, and maps where each node and static
        /// sits in it. A tree built on several threads leaves gaps between its branches, and the nodes in them hold anything
        /// at all, so only the nodes reachable from the root count. False if a static is missing from the leaves.
        /// </summary>
        private bool CopyBuiltTree()
        {
            State.BuiltNodes.ResizeUninitialized(Nodes->Length);
            State.SlotOfNode.ResizeUninitialized(Nodes->Length);
            for (int i = 0; i < Nodes->Length; i++)
            {
                State.BuiltNodes[i] = (*Nodes)[i].Bounds;
            }

            State.BuiltMeshes.ResizeUninitialized(MeshSubAabbs->Length);
            for (int i = 0; i < MeshSubAabbs->Length; i++)
            {
                State.BuiltMeshes[i] = (*MeshSubAabbs)[i];
            }

            State.SlotOfStatic.ResizeUninitialized(StaticBodies.Length);
            for (int i = 0; i < StaticBodies.Length; i++)
            {
                State.SlotOfStatic[i] = -1;
            }

            State.Reached.Clear();
            if (Nodes->Length > 1)
            {
                State.Reached.Add(1);
                State.SlotOfNode[1] = -1;
            }

            for (int next = 0; next < State.Reached.Length; next++)
            {
                int index = State.Reached[next];
                BoundingVolumeHierarchy.Node node = (*Nodes)[index];
                for (int slot = 0; slot < 4; slot++)
                {
                    if (!node.IsChildValid(slot))
                    {
                        continue;
                    }

                    int child = node.Data[slot];
                    if (node.IsLeaf)
                    {
                        if (child < State.SlotOfStatic.Length)
                        {
                            State.SlotOfStatic[child] = index * 4 + slot;
                        }
                    }
                    else
                    {
                        State.SlotOfNode[child] = index * 4 + slot;
                        State.Reached.Add(child);
                    }
                }
            }

            for (int i = 0; i < StaticBodies.Length; i++)
            {
                if (StaticBodies[i].Collider.IsCreated && State.SlotOfStatic[i] < 0)
                {
                    return false;
                }
            }

            State.Copied.Value = 1;
            return true;
        }

        /// <summary>
        /// Moves the reachable nodes and every body's mesh bounds from where they were built. Each node grows by the most any
        /// static below it missed the offset, plus its own rounding. A node's bounds hold its children's, so both of those are
        /// at least its children's, and it still holds them once all moved.
        /// </summary>
        private readonly void MoveBounds(float3 offset, NativeArray<float> misses)
        {
            float distance = math.cmax(math.abs(offset)) + MaxMiss;

            // Children come after their parents, so going backwards finds each node's children done.
            var growths = new NativeArray<float>(Nodes->Length, Allocator.Temp);
            for (int r = State.Reached.Length - 1; r >= 0; r--)
            {
                int index = State.Reached[r];
                BoundingVolumeHierarchy.Node node = (*Nodes)[index];
                FourTransposedAabbs built = State.BuiltNodes[index];
                float largest = 0.0f;
                float missed = 0.0f;
                for (int slot = 0; slot < 4; slot++)
                {
                    if (!node.IsChildValid(slot))
                    {
                        continue;
                    }

                    largest = math.max(largest, Magnitude(built.GetAabb(slot)));
                    int child = node.Data[slot];
                    missed = math.max(missed, node.IsLeaf ? (child < misses.Length ? misses[child] : 0.0f) : growths[child]);
                }

                growths[index] = missed + Rounding(largest, distance);
                Nodes->ElementAt(index).Bounds = Moved(built, offset, growths[index]);
            }

            for (int i = 0; i < MeshSubAabbs->Length; i++)
            {
                FourTransposedAabbs built = State.BuiltMeshes[i];
                float largest = 0.0f;
                for (int slot = 0; slot < 4; slot++)
                {
                    Aabb aabb = built.GetAabb(slot);
                    if (aabb.IsValid)
                    {
                        largest = math.max(largest, Magnitude(aabb));
                    }
                }

                float missed = i < misses.Length ? misses[i] : 0.0f;
                MeshSubAabbs->ElementAt(i) = Moved(built, offset, missed + Rounding(largest, distance));
            }
        }

        private static float Magnitude(Aabb aabb)
        {
            return math.cmax(math.max(math.abs(aabb.Min), math.abs(aabb.Max)));
        }

        /// <summary>
        /// Twice the most that moving a bound of this size rounds it by: once adding the growth to the offset, once adding
        /// that to the bound. Far from the origin a float's step is large, and so is this.
        /// </summary>
        private static float Rounding(float largest, float distance)
        {
            return (largest + 2.0f * distance) * (1.0f / (1 << 22));
        }

        private static FourTransposedAabbs Moved(FourTransposedAabbs bounds, float3 offset, float growth)
        {
            return new FourTransposedAabbs
            {
                Lx = bounds.Lx + (offset.x - growth),
                Hx = bounds.Hx + (offset.x + growth),
                Ly = bounds.Ly + (offset.y - growth),
                Hy = bounds.Hy + (offset.y + growth),
                Lz = bounds.Lz + (offset.z - growth),
                Hz = bounds.Hz + (offset.z + growth),
            };
        }

        /// <summary>
        /// Gives each stray its bounding box at its new pose, as the tree's build would, then fits each node above it around
        /// its children. Fitting reads the children as they are, so strays that share a node see each other's boxes.
        /// </summary>
        private readonly void FitStrays(NativeList<int> strays)
        {
            foreach (int index in strays)
            {
                RigidBody body = StaticBodies[index];
                int leaf = State.SlotOfStatic[index];
                Nodes->ElementAt(leaf >> 2).Bounds.SetAabb(leaf & 3, Broadphase.PrepareStaticBodyDataJob.CalculateAabb(ref body, Margin));
                MeshSubAabbs->ElementAt(index) = Broadphase.PrepareStaticBodyDataJob.ComputeMeshSubAabbs(ref body);

                for (int node = leaf >> 2; State.SlotOfNode[node] >= 0; node = State.SlotOfNode[node] >> 2)
                {
                    int slot = State.SlotOfNode[node];
                    Aabb fitted = (*Nodes)[node].Bounds.GetCompoundAabb();
                    Nodes->ElementAt(slot >> 2).Bounds.SetAabb(slot & 3, fitted);
                }
            }
        }

        /// <summary>Holds the statics the tree is about to be built from.</summary>
        private void Hold()
        {
            State.Held.ResizeUninitialized(StaticBodies.Length);
            for (int i = 0; i < StaticBodies.Length; i++)
            {
                RigidBody body = StaticBodies[i];
                State.Held[i] = new TreeStatic
                {
                    WorldFromBody = body.WorldFromBody,
                    Collider = (long)body.Collider.GetUnsafePtr(),
                };
            }
        }
    }
}
