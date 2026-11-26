using Unity.Mathematics;

namespace Unity.Physics.Dynamics.Solver.Forest
{
    /// <summary>
    /// Boundary size limits for vessel islands. Zero fields use defaults; negative fields disable limits,
    /// except SelfContactManifolds, which never exceeds MaxSelfContactManifolds.
    /// </summary>
    internal struct BoundaryCaps
    {
        /// <summary>
        /// The most contact points against the world or against bodies outside the vessel, over the whole island. The
        /// solve keeps the deepest points.
        /// </summary>
        public int OutsideContactPoints;

        /// <summary>The most points that the solve keeps from one contact manifold. It keeps the deepest.</summary>
        public int PointsPerManifold;

        /// <summary>
        /// Maximum retained self-contact manifolds, selected deepest first and limited by MaxSelfContactManifolds.
        /// Dropped manifolds retain their previous impulses.
        /// </summary>
        public int SelfContactManifolds;

        /// <summary>The most rounds of block pivoting in one solve of the boundary.</summary>
        public int PivotRounds;

        /// <summary>
        /// Start boundary solves from builder guesses rather than remembered states. Benchmark option.
        /// </summary>
        public bool ColdStart;

        /// <summary>
        /// Refactor the free rows every pivot round instead of updating their previous factor. Benchmark option.
        /// </summary>
        public bool RefactorEveryRound;

        /// <summary>
        /// Keep the factor prefix before the first removed row and reinsert the remainder, when RefactorEveryRound
        /// is false.
        /// </summary>
        public bool KeepFactorPrefix;

        /// <summary>
        /// Use boundary row order for the first factor instead of prioritizing rows with stable states. Benchmark
        /// option.
        /// </summary>
        public bool UnorderedFactor;

        /// <summary>The way that the boundary's rounds change their factor, from <see cref="RefactorEveryRound"/> and <see cref="KeepFactorPrefix"/>.</summary>
        public Unity.Numerics.Linear.Solvers.Lowlevel.FactorUpdate FactorUpdate => RefactorEveryRound
            ? Unity.Numerics.Linear.Solvers.Lowlevel.FactorUpdate.Refactor
            : KeepFactorPrefix ? Unity.Numerics.Linear.Solvers.Lowlevel.FactorUpdate.KeepPrefix : Unity.Numerics.Linear.Solvers.Lowlevel.BlockPrincipalPivoting.DefaultUpdate;

        /// <summary>
        /// Identify external boundary bodies by world index instead of stable entity identity. Benchmark option.
        /// </summary>
        public bool KeyByWorldIndex;

        // Keep the version 13 recording layout until the format changes.
        private byte recordingPadding;

        /// <summary>
        /// Disable held-row corrections, preserving their warm impulses and omitting push-out. Benchmark option.
        /// </summary>
        public bool IdleHeldRows;

        /// <summary>
        /// How far, in metres or radians, a one-sided joint row must stay from its stop over the step for the solve to
        /// leave it out of the boundary.
        /// </summary>
        public const float DormantMargin = 0.005f;

        public const int DefaultOutsideContactPoints = 256;
        public const int DefaultPointsPerManifold = 4;
        public const int DefaultSelfContactManifolds = 8;
        public const int MaxSelfContactManifolds = 25;
        public const int DefaultPivotRounds = 8;

        /// <summary>The limit on outside contact points that applies, after defaults.</summary>
        public int OutsidePointLimit => Limit(OutsideContactPoints, DefaultOutsideContactPoints);

        /// <summary>The limit on points per manifold that applies, after defaults.</summary>
        public int PointLimit => Limit(PointsPerManifold, DefaultPointsPerManifold);

        /// <summary>
        /// Effective self-contact manifold cap, at most 25 even when a negative setting disables other caps.
        /// </summary>
        public int SelfManifoldLimit =>
            Unity.Mathematics.math.min(Limit(SelfContactManifolds, DefaultSelfContactManifolds), MaxSelfContactManifolds);

        /// <summary>The limit on pivoting rounds that applies, after defaults.</summary>
        public int PivotRoundLimit => Limit(PivotRounds, DefaultPivotRounds);

        static int Limit(int value, int fallback) => value == 0 ? fallback : value < 0 ? int.MaxValue : value;
    }
}
