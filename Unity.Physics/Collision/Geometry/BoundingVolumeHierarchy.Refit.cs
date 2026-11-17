using System.Runtime.CompilerServices;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;

namespace Unity.Physics
{
    internal partial struct BoundingVolumeHierarchy
    {
        /// <summary>
        /// Refit reachable nodes without changing tree structure, processing children before parents.
        /// Skip gaps left by parallel builds. Used as the reference for parallel refit tests.
        /// </summary>
        internal unsafe void RefitFromRoot(NativeArray<Aabb> aabbs)
        {
            Node* baseNode = m_Nodes;
            Aabb* bounds = (Aabb*)aabbs.GetUnsafeReadOnlyPtr();

            // Every node in the tree is listed once, so the list never holds more than the node count. The pending
            // stack holds at most three entries for each level of the tree, plus one.
            int capacity = m_NodesList != null ? m_NodesList->Length : m_MaxNodeCount;
            int* order = (int*)UnsafeUtility.Malloc((long)capacity * sizeof(int), UnsafeUtility.AlignOf<int>(), Allocator.Temp);
            int* pending = (int*)UnsafeUtility.Malloc((long)capacity * sizeof(int), UnsafeUtility.AlignOf<int>(), Allocator.Temp);
            int listed = 0;
            int waiting = 0;
            pending[waiting++] = 1;
            while (waiting > 0)
            {
                int index = pending[--waiting];
                SafetyChecks.CheckIndexAndThrow(index, m_MaxNodeCount);
                order[listed++] = index;
                Node* node = baseNode + index;
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

            for (int i = listed - 1; i >= 0; i--)
            {
                Node* node = baseNode + order[i];
                if (node->IsLeaf)
                {
                    FitLeafNode(bounds, node);
                }
                else
                {
                    FitInternalNode(baseNode, node);
                }
            }

            UnsafeUtility.Free(order, Allocator.Temp);
            UnsafeUtility.Free(pending, Allocator.Temp);
        }

        /// <summary>
        /// Depth at which parallel refitting splits the tree into subtrees.
        /// </summary>
        internal const int RefitCutDepth = 3;

        /// <summary>The number of subtrees that a parallel refit can fit at the same time.</summary>
        internal const int RefitSubtreeCount = 1 << (2 * RefitCutDepth);

        /// <summary>
        /// Return the root assigned to a refit slot, or zero if absent.
        /// Use base-four slot digits to select children; assign leaves above the cut to one slot only.
        /// </summary>
        internal unsafe int FindRefitSubtree(int slot)
        {
            Node* baseNode = m_Nodes;
            int index = 1;
            for (int level = 0; level < RefitCutDepth; level++)
            {
                int shift = 2 * (RefitCutDepth - 1 - level);
                Node* node = baseNode + index;
                if (node->IsLeaf)
                {
                    int remainingDigits = slot & ((1 << (shift + 2)) - 1);
                    return remainingDigits == 0 ? index : 0;
                }

                int child = (slot >> shift) & 3;
                if (!node->IsInternalValid(child))
                {
                    return 0;
                }

                index = node->Data[child];
            }

            return index;
        }

        /// <summary>
        /// Refit a subtree, including its root, after its children. Match RefitFromRoot bounds.
        /// </summary>
        internal unsafe void RefitSubtree(int root, NativeArray<Aabb> aabbs)
        {
            RefitSubtree(root, (Aabb*)aabbs.GetUnsafeReadOnlyPtr());
        }

        /// <summary>As <see cref="RefitSubtree(int, NativeArray{Aabb})"/>, with the bounds of the leaves' elements.</summary>
        internal unsafe void RefitSubtree(int root, Aabb* bounds)
        {
            var fitter = new BoundsFitter { Bounds = bounds };
            RefitSubtree(root, ref fitter);
        }

        /// <summary>
        /// Per-node refit operations. Fit leaves directly and internal nodes after their children.
        /// </summary>
        internal unsafe interface INodeFitter
        {
            void FitLeaf(Node* baseNode, int index);
            void FitInternal(Node* baseNode, int index);
        }

        /// <summary>Fits the bounds of each node to the bounds of the leaves' elements, and does nothing else.</summary>
        internal unsafe struct BoundsFitter : INodeFitter
        {
            public Aabb* Bounds;

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public void FitLeaf(Node* baseNode, int index) => FitLeafNode(Bounds, baseNode + index);

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public void FitInternal(Node* baseNode, int index) => FitInternalNode(baseNode, baseNode + index);
        }

        /// <summary>
        /// Refit bounds and measure node and leaf surface areas for tree quality.
        /// Optionally rebuild node filters as unions of child filters.
        /// </summary>
        internal unsafe struct MeasuringFitter : INodeFitter
        {
            public Aabb* Bounds;

            /// <summary>The filters of the elements, or null to leave the nodes' filters as they are.</summary>
            public CollisionFilter* LeafFilters;

            /// <summary>The filters of the nodes, which the fitter writes when <see cref="LeafFilters"/> is set.</summary>
            public CollisionFilter* NodeFilters;

            public float NodeArea;
            public float LeafArea;

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public void FitLeaf(Node* baseNode, int index)
            {
                Node* node = baseNode + index;
                FitLeafNode(Bounds, node);
                CollisionFilter filter = CollisionFilter.Zero;
                bool first = true;
                for (int j = 0; j < 4; j++)
                {
                    if (node->IsLeafValid(j))
                    {
                        LeafArea += Bounds[node->Data[j]].SurfaceArea;
                        if (LeafFilters != null)
                        {
                            CollisionFilter leafFilter = LeafFilters[node->Data[j]];
                            filter = first ? leafFilter : CollisionFilter.CreateUnion(filter, leafFilter);
                            first = false;
                        }
                    }
                }

                if (LeafFilters != null)
                {
                    NodeFilters[index] = filter;
                }
            }

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public void FitInternal(Node* baseNode, int index)
            {
                Node* node = baseNode + index;
                FitInternalNode(baseNode, node);
                CollisionFilter filter = CollisionFilter.Zero;
                bool first = true;
                for (int j = 0; j < 4; j++)
                {
                    if (node->IsInternalValid(j))
                    {
                        NodeArea += node->Bounds.GetAabb(j).SurfaceArea;
                        if (LeafFilters != null)
                        {
                            CollisionFilter childFilter = NodeFilters[node->Data[j]];
                            filter = first ? childFilter : CollisionFilter.CreateUnion(filter, childFilter);
                            first = false;
                        }
                    }
                }

                if (LeafFilters != null)
                {
                    NodeFilters[index] = filter;
                }
            }
        }

        /// <summary>
        /// Refit root and its descendants depth-first, processing children before their parents without changing
        /// structure.
        /// </summary>
        internal unsafe void RefitSubtree<T>(int root, ref T fitter) where T : struct, INodeFitter
        {
            Node* baseNode = m_Nodes;

            // Negative complemented indices mark internal nodes waiting for their children.
            // Use the heap if a deep tree exceeds the inline stack.
            const int inlineCapacity = 256;
            int* inlineStack = stackalloc int[inlineCapacity];
            int* stack = inlineStack;
            int capacity = inlineCapacity;
            int count = 0;
            stack[count++] = root;
            while (count > 0)
            {
                int entry = stack[--count];
                if (entry < 0)
                {
                    fitter.FitInternal(baseNode, ~entry);
                    continue;
                }

                SafetyChecks.CheckIndexAndThrow(entry, m_MaxNodeCount);
                Node* node = baseNode + entry;
                if (node->IsLeaf)
                {
                    fitter.FitLeaf(baseNode, entry);
                    continue;
                }

                if (count + 5 > capacity)
                {
                    int* grown = (int*)UnsafeUtility.Malloc((long)capacity * 2 * sizeof(int), UnsafeUtility.AlignOf<int>(), Allocator.Temp);
                    UnsafeUtility.MemCpy(grown, stack, (long)count * sizeof(int));
                    if (stack != inlineStack)
                    {
                        UnsafeUtility.Free(stack, Allocator.Temp);
                    }

                    stack = grown;
                    capacity *= 2;
                }

                stack[count++] = ~entry;
                for (int j = 0; j < 4; j++)
                {
                    if (node->IsInternalValid(j))
                    {
                        stack[count++] = node->Data[j];
                    }
                }
            }

            if (stack != inlineStack)
            {
                UnsafeUtility.Free(stack, Allocator.Temp);
            }
        }

        /// <summary>
        /// Fit internal nodes above the parallel cut after all assigned subtrees finish.
        /// </summary>
        internal unsafe void RefitAboveCut()
        {
            var fitter = new BoundsFitter();
            FitAboveCut(m_Nodes, 1, 0, ref fitter);
        }

        /// <summary>As <see cref="RefitAboveCut()"/>, with <paramref name="fitter"/> to fit each node.</summary>
        internal unsafe void RefitAboveCut<T>(ref T fitter) where T : struct, INodeFitter
        {
            FitAboveCut(m_Nodes, 1, 0, ref fitter);
        }

        static unsafe void FitAboveCut<T>(Node* baseNode, int index, int depth, ref T fitter) where T : struct, INodeFitter
        {
            Node* node = baseNode + index;
            if (node->IsLeaf || depth >= RefitCutDepth)
            {
                return;
            }

            for (int j = 0; j < 4; j++)
            {
                if (node->IsInternalValid(j))
                {
                    FitAboveCut(baseNode, node->Data[j], depth + 1, ref fitter);
                }
            }

            fitter.FitInternal(baseNode, index);
        }

        // Only the bounds change. The counts of elements and free slots stay as the build left them.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static unsafe void FitLeafNode(Aabb* bounds, Node* node)
        {
            for (int j = 0; j < 4; ++j)
            {
                node->Bounds.SetAabb(j, node->IsLeafValid(j) ? bounds[node->Data[j]] : Aabb.Empty);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static unsafe void FitInternalNode(Node* baseNode, Node* node)
        {
            for (int j = 0; j < 4; j++)
            {
                node->Bounds.SetAabb(j, node->IsInternalValid(j) ? baseNode[node->Data[j]].Bounds.GetCompoundAabb() : Aabb.Empty);
            }
        }
    }
}
