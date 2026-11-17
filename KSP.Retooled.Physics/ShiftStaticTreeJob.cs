using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Physics;
using Unity.Physics.Store.Bodies;
using Unity.Physics.Collision.Colliders.Building;

namespace KSP.Retooled.Physics
{
    /// <summary>
    /// Shift static tree bounds for a common body offset; write Rebuild = 1 when a rebuild is required.
    /// Always shift from the last built bounds to avoid accumulated rounding. Expand bounds for pose rounding
    /// and refit independently moving bodies when their count is small.
    /// </summary>
    [BurstCompile]
    internal unsafe struct ShiftStaticTreeJob : IJob
    {
        /// <summary>
        /// Maximum number of independently moving bodies to refit before rebuilding the static tree.
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
        public BodyData* Store;

        /// <summary>
        /// Bodies whose colliders were rebuilt. Treat these as changed even when pose and pointer match,
        /// because memory may have been reused or collider settings changed.
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
        /// Persistent tree shift data owned by StepState. Only serialized build jobs access these containers.
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
                if ((Store[Refreshed[i].Body].flags & BodyFlags.Static) != 0)
                {
                    return true;
                }
            }

            return false;
        }

        /// <summary>
        /// Find the common static body offset and record deviations and independently moving bodies.
        /// Return false if colliders changed or too many bodies need individual refitting.
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
        /// Separate bodies following the common offset from those needing individual refits.
        /// Return false when the refit count exceeds the limit.
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
        /// Save built bounds and tree locations for later shifts. Visit only reachable nodes because parallel
        /// builds leave gaps.
        /// Return false if a collidable static body is missing from the leaves.
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
        /// Shift original node bounds and expand them for body deviations and floating-point rounding.
        /// Propagate expansion from children to parents to preserve containment.
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
        /// Twice the maximum rounding error when shifting a bound, accounting for both additions.
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
        /// Recompute bounds for independently moving bodies and refit their ancestors using current child bounds.
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
