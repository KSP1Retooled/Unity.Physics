namespace KSP.Retooled.Physics.Tests.Conformance.Vessels
{
    /// <summary>
    /// Runs VesselModeTests at the configured minimum size for fused solving.
    /// </summary>
    public class FusedVesselModeTests : VesselModeTests
    {
        protected override int FusedParts => 0;
    }
}
