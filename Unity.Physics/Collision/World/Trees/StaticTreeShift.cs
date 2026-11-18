using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Physics.Store.Bodies;
using Unity.Physics.Collision.Colliders.Building;

namespace Unity.Physics.Collision.World.Trees
{
    internal static partial class StaticTreeShift
    {
        /// <summary>Records a static body as it was when the tree was built.</summary>
        internal struct TreeStatic
        {
            public RigidTransform WorldFromBody;
            public long Collider;
        }

        /// <summary>
        /// Persistent tree shift data owned by StoreStepper. Only serialized build jobs access these containers.
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
    }
}
