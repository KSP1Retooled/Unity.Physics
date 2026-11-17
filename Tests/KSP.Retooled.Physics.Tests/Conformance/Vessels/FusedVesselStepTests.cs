namespace KSP.Retooled.Physics.Tests.Conformance.Vessels
{
    /// <summary>
    /// Run VesselStepTests with fused solving enabled for every eligible island.
    /// </summary>
    public class FusedVesselStepTests : VesselStepTests
    {
        protected override int FusedParts => 0;
    }
}
