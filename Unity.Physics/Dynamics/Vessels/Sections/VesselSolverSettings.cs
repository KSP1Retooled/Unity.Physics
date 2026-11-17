namespace Unity.Physics.Dynamics.Vessels.Sections
{
    /// <summary>Settings for solving vessel sections.</summary>
    internal struct VesselSolverSettings
    {
        public double TimeStep;

        /// <summary>
        /// Hard-row spring period. Compute stiffness from reduced section mass or inertia so that light connectors
        /// do not soften heavy subtrees.
        /// </summary>
        public double HardPeriod;

        /// <summary>The damping for a hard row, as a fraction of critical damping for the same reduced mass.</summary>
        public double HardDampingRatio;

        /// <summary>Sections with at least this many active closures are treated as hubs and eliminated last.</summary>
        public int HubClosures;

        /// <summary>
        /// The approximate number of independent pieces of factorization work to distribute among job threads.
        /// </summary>
        public int ChunkTarget;

        /// <summary>
        /// Use minimum-degree ordering to reduce fill from struts. Otherwise order sections leaves-first with hubs
        /// last.
        /// </summary>
        public bool MinimumDegree;

        /// <summary>
        /// Include centripetal force in locked-edge loads. Disable to balance loads exactly against active-row
        /// impulses;
        /// the solver itself has no centripetal or gyroscopic terms.
        /// </summary>
        public bool CentripetalLoads;

        public static VesselSolverSettings Default => new VesselSolverSettings
        {
            TimeStep = 0.02,
            HardPeriod = 0.001,
            HardDampingRatio = 1.0,
            HubClosures = 16,
            ChunkTarget = 96,
            MinimumDegree = true,
            CentripetalLoads = true,
        };
    }
}
