using Unity.Physics.Dynamics.Solver.Forest;

namespace Unity.Physics.Configuration
{
    /// <summary>Settings that the step reads outside the store.</summary>
    internal struct RecordedPhysicsSettings
    {
        internal bool Vessels;
        internal bool Fused;
        internal bool FusedContacts;
        internal bool ServoEdges;
        internal int FusedParts;
        internal int MinimumParts;
        internal bool StrutLoads;
        internal int StrutExperiment;
        internal SolverSwitches.SubstepFolding Folding;
        internal int SerialIslandPairLimit;
        internal bool Forest;
        internal int BoundaryRowCap;
        internal BoundaryCaps Caps;
        internal bool Locking;
        internal float LockAngle;
        internal float YieldAngle;
        internal float LockFraction;
        internal float YieldFraction;
        internal int HoldSteps;
        internal int SmoothingSteps;
        internal bool ClearSolverSlots;
        internal bool RefillVesselEdges;
        internal bool StaticReuse;
        internal bool PairSortReuse;
        internal bool TreeReuse;
        internal bool AlwaysRefit;
        internal bool BoundsCache;
        internal bool ShapeBounds;
        internal double RowLinearCap;
        internal double RowAngularCap;

        internal static RecordedPhysicsSettings Read() => new RecordedPhysicsSettings
        {
            Vessels = VesselSwitches.Enabled,
            Fused = SolverSwitches.VesselFused,
            FusedContacts = SolverSwitches.FusedContacts,
            ServoEdges = VesselSwitches.ServoEdges,
            FusedParts = SolverSwitches.FusedVesselParts,
            MinimumParts = VesselSwitches.MinimumParts,
            StrutLoads = VesselSwitches.StrutLoadsEnabled,
            StrutExperiment = VesselSwitches.StrutExperiment,
            Folding = SolverSwitches.UnfusedFolding,
            SerialIslandPairLimit = SolverSwitches.SerialIslandPairLimit,
            Forest = ForestSwitches.Enabled,
            BoundaryRowCap = ForestSwitches.BoundaryRowCap,
            Caps = ForestSwitches.Caps,
            Locking = ForestSwitches.Locking,
            LockAngle = ForestSwitches.LockAngle,
            YieldAngle = ForestSwitches.YieldAngle,
            LockFraction = ForestSwitches.LockFraction,
            YieldFraction = ForestSwitches.YieldFraction,
            HoldSteps = ForestSwitches.MinimumHoldSteps,
            SmoothingSteps = ForestSwitches.DeflectionSmoothingSteps,
            ClearSolverSlots = StepCacheSwitches.ClearSolverSlots,
            RefillVesselEdges = StepCacheSwitches.RefillVesselEdges,
            StaticReuse = StepCacheSwitches.StaticReuseEnabled,
            PairSortReuse = StepCacheSwitches.PairSortReuseEnabled,
            TreeReuse = BroadphaseSwitches.TreeReuseEnabled,
            AlwaysRefit = BroadphaseSwitches.TreeAlwaysRefitted,
            BoundsCache = BroadphaseSwitches.BoundsCacheEnabled,
            ShapeBounds = BroadphaseSwitches.ShapeBoundsEnabled,
            RowLinearCap = VesselRowSwitches.LinearCap,
            RowAngularCap = VesselRowSwitches.AngularCap,
        };

        internal readonly void Apply()
        {
            VesselSwitches.Enabled = Vessels;
            VesselSwitches.ServoEdges = ServoEdges;
            SolverSwitches.VesselFused = Fused;
            SolverSwitches.FusedContacts = FusedContacts;
            SolverSwitches.FusedVesselParts = FusedParts;
            VesselSwitches.MinimumParts = MinimumParts;
            VesselSwitches.StrutLoadsEnabled = StrutLoads;
            VesselSwitches.StrutExperiment = StrutExperiment;
            SolverSwitches.UnfusedFolding = Folding;
            SolverSwitches.SerialIslandPairLimit = SerialIslandPairLimit;
            ForestSwitches.Enabled = Forest;
            ForestSwitches.BoundaryRowCap = BoundaryRowCap;
            ForestSwitches.Caps = Caps;
            ForestSwitches.Locking = Locking;
            ForestSwitches.LockAngle = LockAngle;
            ForestSwitches.YieldAngle = YieldAngle;
            ForestSwitches.LockFraction = LockFraction;
            ForestSwitches.YieldFraction = YieldFraction;
            ForestSwitches.MinimumHoldSteps = HoldSteps;
            ForestSwitches.DeflectionSmoothingSteps = SmoothingSteps;
            StepCacheSwitches.ClearSolverSlots = ClearSolverSlots;
            StepCacheSwitches.RefillVesselEdges = RefillVesselEdges;
            StepCacheSwitches.StaticReuseEnabled = StaticReuse;
            StepCacheSwitches.PairSortReuseEnabled = PairSortReuse;
            if (AlwaysRefit)
            {
                BroadphaseSwitches.TreeAlwaysRefitted = true;
            }
            else
            {
                BroadphaseSwitches.TreeReuseEnabled = TreeReuse;
            }
            BroadphaseSwitches.BoundsCacheEnabled = BoundsCache;
            BroadphaseSwitches.ShapeBoundsEnabled = ShapeBounds;

            // A cap read as 0 was off. A negative value keeps it off, where 0 would restore the default.
            VesselRowSwitches.LinearCap = RowLinearCap > 0.0 ? RowLinearCap : -1.0;
            VesselRowSwitches.AngularCap = RowAngularCap > 0.0 ? RowAngularCap : -1.0;
        }
    }
}
