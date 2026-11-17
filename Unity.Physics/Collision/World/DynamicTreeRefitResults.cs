namespace Unity.Physics
{
    /// <summary>
    /// Refit results used to decide whether to reuse the dynamic tree again.
    /// Keep at a fixed address and read on the main thread after jobs complete.
    /// </summary>
    internal unsafe struct DynamicTreeRefitResults
    {
        /// <summary>1 when a body's filter differed from the filter that the tree held for its index, and 0 otherwise.</summary>
        public int FiltersChanged;

        /// <summary>The sum of the surface areas of the bounds of every node below the root.</summary>
        public float NodeArea;

        /// <summary>The sum of the surface areas of the bounds of every body in the tree.</summary>
        public float LeafArea;

        /// <summary>The two sums for each subtree below the cut, which the subtree's iteration writes.</summary>
        public fixed float SlotNodeAreas[BoundingVolumeHierarchy.RefitSubtreeCount];
        public fixed float SlotLeafAreas[BoundingVolumeHierarchy.RefitSubtreeCount];

        /// <summary>
        /// Node surface area divided by leaf surface area, or zero for an empty tree.
        /// Higher values indicate worse spatial grouping, normalized for changing body bounds.
        /// </summary>
        public float Quality => LeafArea > 0.0f ? NodeArea / LeafArea : 0.0f;

        /// <summary>The fitter for the refit's jobs, which builds the nodes' filters again when a body's filter changed.</summary>
        internal BoundingVolumeHierarchy.MeasuringFitter Fitter(Broadphase.Tree tree, Aabb* bounds)
        {
            return new BoundingVolumeHierarchy.MeasuringFitter
            {
                Bounds = bounds,
                LeafFilters = FiltersChanged != 0 ? tree.BodyFilters.GetUnsafeList()->Ptr : null,
                NodeFilters = tree.NodeFilters.GetUnsafeList()->Ptr,
            };
        }
    }
}
