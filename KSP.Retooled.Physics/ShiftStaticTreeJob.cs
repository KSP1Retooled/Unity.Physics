using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Physics;

namespace KSP.Retooled.Physics
{
    /// <summary>
    /// Determines whether to rebuild the static tree or move its bounds. KSP's floating origin moves every static body
    /// by the same offset, often every frame. Rebuilding the tree requires calculating a bounding box for every static
    /// body. If the bodies have only moved and nearly all share one offset, this job moves the tree's bounds and writes
    /// 0 to <see cref="Rebuild"/>. Otherwise it writes 1.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Unity calculates each static body's world pose using floats. A parent transform may be a planet's width away, so
    /// floating-point rounding prevents all bodies from moving by exactly the same offset. For example, a terrain
    /// collider parented to a planet's centre 600 km away can deviate by centimetres. This job expands each node's
    /// bounds to cover the largest deviation among the bodies below it, plus rounding during movement. This keeps every
    /// static body within its bounds. Each move uses the original bounds from the last build, so rounding errors do not
    /// accumulate across moves.
    /// </para>
    /// <para>
    /// Some static bodies move independently because they rotate or follow their own parent transforms. For each such
    /// body, and each body that deviates too far from the common offset, this job recalculates the bounding box at the
    /// new pose. It then fits the bounds of ancestor nodes around their children again. If too many bodies require this
    /// work, rebuilding the tree is cheaper.
    /// </para>
    /// </remarks>
    [BurstCompile]
    internal unsafe struct ShiftStaticTreeJob : IJob
    {
        /// <summary>
        /// Limits how many static bodies can have their bounds fitted individually during a move. Fitting one body
        /// requires calculating its bounding box and walking up the tree. Rebuilding requires calculating boxes for all
        /// static bodies and sorting them, so fitting up to a quarter of them is still cheaper. The limit is at least
        /// 64 bodies. During flight, about a hundred of KSP's static bodies move independently every frame.
        /// </summary>
        private readonly int MaxStrays => math.max(64, StaticBodies.Length / 4);

        /// <summary>
        /// Specifies how far a static body can deviate from the common offset, in metres, before its bounds must be
        /// fitted individually.
        /// </summary>
        private const float MaxMiss = 1.0f;

        /// <summary>Contains the static bodies for the current build, with the anchor body last.</summary>
        [ReadOnly]
        public NativeArray<RigidBody> StaticBodies;

        // Access the lists for the static tree without safety handles. The jobs that build the tree are scheduled after
        // this job, and scheduling them requires accessing the same lists as arrays on the main thread.
        [NativeDisableUnsafePtrRestriction]
        public UnsafeList<BoundingVolumeHierarchy.Node>* Nodes;

        [NativeDisableUnsafePtrRestriction]
        public UnsafeList<FourTransposedAabbs>* MeshSubAabbs;

        public ShiftState State;

        [NativeDisableUnsafePtrRestriction]
        public NativeStore.BodyData* Store;

        /// <summary>
        /// Lists the bodies whose colliders were rebuilt during the refresh for this build: new bodies and bodies whose
        /// shapes changed. A static body's collider may have changed in ways that its pose and pointer do not reveal,
        /// such as a change to its filter. A collider freed since the tree was built may also be replaced by another at
        /// the same address. A build that does not refresh colliders uses the list from the last refresh. This may
        /// cause an unnecessary rebuild of the tree, but is otherwise safe.
        /// </summary>
        [ReadOnly]
        [NativeDisableContainerSafetyRestriction]
        public NativeList<ColliderBuilder.Work> Refreshed;

        /// <summary>
        /// Indicates whether the tree was built from the saved static bodies. This is false after a failed build, for
        /// example.
        /// </summary>
        public bool TreeMatchesHeld;

        /// <summary>Specifies the margin added around each static body's bounding box when building the tree.</summary>
        public float Margin;

        /// <summary>
        /// Tells the jobs that build the static tree whether a rebuild is needed. Only this job writes the value, and
        /// those jobs wait for this job to finish.
        /// </summary>
        [NativeDisableContainerSafetyRestriction]
        public NativeReference<int> Rebuild;

        /// <summary>Records a static body as it was when the tree was built.</summary>
        internal struct TreeStatic
        {
            public RigidTransform WorldFromBody;
            public long Collider;
        }

        /// <summary>
        /// Stores data that this job retains between builds. The world's step state owns the data, but only this job
        /// accesses it. Each build waits for the previous build, so these containers do not need safety handles, just
        /// as the lists for the tree do not.
        /// </summary>
        internal struct ShiftState
        {
            /// <summary>
            /// Stores the static bodies used for the last tree build, in the same order as the built static bodies.
            /// </summary>
            [NativeDisableContainerSafetyRestriction]
            public NativeList<TreeStatic> Held;

            /// <summary>
            /// Stores the original bounds of tree nodes and meshes after they have been copied for a move.
            /// </summary>
            [NativeDisableContainerSafetyRestriction]
            public NativeList<FourTransposedAabbs> BuiltNodes;

            [NativeDisableContainerSafetyRestriction]
            public NativeList<FourTransposedAabbs> BuiltMeshes;

            /// <summary>Lists the nodes reachable from the root, with each node after its parent.</summary>
            [NativeDisableContainerSafetyRestriction]
            public NativeList<int> Reached;

            /// <summary>
            /// Stores the slot for each node in its parent as parent node * 4 + slot. The root has the value -1.
            /// </summary>
            [NativeDisableContainerSafetyRestriction]
            public NativeList<int> SlotOfNode;

            /// <summary>Stores the leaf slot for each static body as node * 4 + slot.</summary>
            [NativeDisableContainerSafetyRestriction]
            public NativeList<int> SlotOfStatic;

            /// <summary>
            /// Contains 1 once the original tree bounds have been copied, or 0 while those copies are stale.
            /// </summary>
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
        /// Finds the common offset of most static bodies since the tree was built. Records how far each body deviated
        /// from that offset and lists bodies that moved differently. Returns false if a collider pointer changed or too
        /// many bodies moved differently. Bodies without colliders cannot collide with anything, so they are excluded
        /// from the movement checks.
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

            // Try offsets from a few static bodies spread through the list. The first body tried may itself have moved
            // independently of the others.
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
        /// Separates static bodies into those that followed the common offset and those that need their bounds fitted
        /// individually. Records how far each body in the first group deviated from the offset. Returns false as soon
        /// as too many bodies need individual fitting.
        /// </summary>
        private readonly bool Sort(RigidBody* bodies, TreeStatic* held, double3 moved, double3 offset, NativeList<int> strays, NativeArray<float> misses)
        {
            strays.Clear();

            // Converting the offset to a float adds rounding error. Include that error in the deviation for every
            // static body.
            double rounding = math.cmax(math.abs(offset - moved));
            for (int i = 0; i < StaticBodies.Length; i++)
            {
                if (!bodies[i].Collider.IsCreated)
                {
                    continue;
                }

                // Subtract float positions in double precision to avoid rounding the difference to float precision.
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
        /// Copies the original tree bounds and records where each node and static body is stored. Every move uses these
        /// original bounds until the next rebuild. Building a tree on several threads leaves gaps between branches.
        /// Nodes in those gaps may contain arbitrary data, so only nodes reachable from the root are used. Returns
        /// false if a static body with a collider is missing from the leaves.
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
        /// Moves the bounds of reachable nodes and the mesh bounds for every body from their original positions when
        /// the tree was built. Each node's bounds are expanded to cover deviations from the common offset and rounding
        /// during movement. Expansion propagates from children to parents. A parent's original bounds enclose its
        /// children's bounds, so its rounding allowance is at least as large as theirs. The expanded bounds therefore
        /// still enclose the children after movement.
        /// </summary>
        private readonly void MoveBounds(float3 offset, NativeArray<float> misses)
        {
            float distance = math.cmax(math.abs(offset)) + MaxMiss;

            // Children appear after their parents in the list. Process the list backwards so that each child is
            // processed before its parent.
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
        /// Returns twice the maximum rounding error when moving a bound of this size. Rounding occurs when adding the
        /// expansion to the offset and again when adding that result to the bound. Floating-point values are spaced
        /// further apart far from the origin, so the allowance grows with distance.
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
        /// Recalculates the bounding box at the current pose for each body that moved differently, using the same
        /// calculation as a tree build. Then fits the bounds of its ancestors around their children. Each fit reads the
        /// current child bounds, so updates for bodies with a shared ancestor include the boxes that were already
        /// recalculated.
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

        /// <summary>Saves the static bodies that will be used to build the tree.</summary>
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
