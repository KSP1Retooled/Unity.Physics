namespace KSP.Retooled.Physics.Tests.Conformance.Vessels
{
    /// <summary>
    /// Runs VesselStepTests at the configured minimum size for fused solving.
    /// </summary>
    public class FusedVesselStepTests : VesselStepTests
    {
        protected override int FusedParts => 0;
    }
}
