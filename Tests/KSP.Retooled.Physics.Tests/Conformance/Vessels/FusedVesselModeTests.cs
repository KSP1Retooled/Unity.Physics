namespace KSP.Retooled.Physics.Tests.Conformance.Vessels
{
    /// <summary>
    /// Run VesselModeTests with fused solving enabled for every eligible island.
    /// </summary>
    public class FusedVesselModeTests : VesselModeTests
    {
        protected override int FusedParts => 0;
    }
}
