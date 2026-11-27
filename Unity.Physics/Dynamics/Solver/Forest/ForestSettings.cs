using Unity.Mathematics;
using Unity.Physics.Dynamics.Solver.Sparse;

namespace Unity.Physics.Dynamics.Solver.Forest
{
    /// <summary>
    /// Solve unbounded forest joint rows structurally and all other rows through a dense boundary.
    /// Fall back to SparseConstraintSolver without forest rows or above BoundaryRowCap.
    /// Optional locking merges low-deflection tree pairs into rigid sections until they reach YieldAngle.
    /// </summary>
    internal struct ForestSettings
    {
        public int BoundaryRowCap;
        public bool Locking;

        /// <summary>Radians: an edge whose springs would deflect less than this under its last load may lock.</summary>
        public float LockAngle;

        /// <summary>Radians: a locked edge whose load would deflect its springs more than this yields.</summary>
        public float YieldAngle;

        /// <summary>
        /// Minimum steps before changing a lock state again.
        /// Yield immediately above twice YieldAngle or when the owner revokes permission.
        /// </summary>
        public int MinimumHoldSteps;

        /// <summary>
        /// Exponential smoothing duration for lock/yield deflection tests, in steps. Zero or one uses the last
        /// step's mean.
        /// The twice-yield test always uses the unsmoothed mean.
        /// </summary>
        public int DeflectionSmoothingSteps;

        /// <summary>The limits on a vessel island's boundary block, see <see cref="BoundaryCaps"/>.</summary>
        public BoundaryCaps Caps;

        public const float DefaultLockAngle = 0.5e-3f;
        public const float DefaultYieldAngle = 2e-3f;
        public const int DefaultMinimumHoldSteps = 10;
        public const int DefaultDeflectionSmoothingSteps = 20;

        public static ForestSettings Default => new ForestSettings
        {
            BoundaryRowCap = 512,
            Locking = false,
            LockAngle = DefaultLockAngle,
            YieldAngle = DefaultYieldAngle,
            MinimumHoldSteps = DefaultMinimumHoldSteps,
            DeflectionSmoothingSteps = DefaultDeflectionSmoothingSteps,
        };
    }
}
