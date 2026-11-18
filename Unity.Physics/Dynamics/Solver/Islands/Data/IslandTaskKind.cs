namespace Unity.Physics.Dynamics.Solver.Islands.Data
{
    /// <summary>Labels the stages that solve teams publish and island helpers dispatch.</summary>
    internal static class IslandTaskKind
    {
        /// <summary>Identifies the direct solver stages that run around the sparse solver stages.</summary>
        internal const int PredictDriftKind = -1;
        internal const int ApplyImpulsesKind = -2;
        internal const int SetupBodiesKind = -3;
        internal const int SummarizeJacobiansKind = -4;
        internal const int FillBlocksKind = -5;

        /// <summary>
        /// Identifies one phase of an iteration for pairs solved iteratively. The context points to that phase's job.
        /// </summary>
        internal const int IterativePhaseKind = -6;

        /// <summary>
        /// Identifies the stages for factorization, forward substitution and backward substitution, each processed in
        /// chunks of a vessel forest.
        /// </summary>
        internal const int ForestFactorKind = -7;
        internal const int ForestForwardKind = -8;
        internal const int ForestSubstituteKind = -9;

        /// <summary>
        /// Identifies a stage that assigns rows to forest sections or computes body impulses and loads on locked
        /// pairs for those sections.
        /// </summary>
        internal const int ForestSectionsKind = -10;

        /// <summary>
        /// Identifies the stage that classifies a direct island's rows for the forest, processing a range of blocks per
        /// task.
        /// </summary>
        internal const int ForestClassifyKind = -11;

        /// <summary>
        /// Identifies a stage that processes a range of sections, links or parts per task while solving a vessel
        /// island.
        /// </summary>
        internal const int ForestVesselKind = -12;

        /// <summary>Identifies the stage that fills the boundary block of a forest solve, by ranges of its rows.</summary>
        internal const int ForestBoundaryKind = -13;

        /// <summary>Identifies each strut solve stage that a team shares across lanes.</summary>
        internal const int StrutLoadsKind = -14;
        internal const int StrutForwardKind = -15;
        internal const int StrutBackKind = -16;
        internal const int StrutLinksKind = -17;
    }
}
