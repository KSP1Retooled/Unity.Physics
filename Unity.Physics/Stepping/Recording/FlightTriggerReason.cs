namespace Unity.Physics.Stepping.Recording
{
    internal enum FlightTriggerReason
    {
        None,
        RestoredMotion,
        NonFiniteMotion,
        LinearVelocity,
        AngularVelocity,
        ReorderedBodies,
    }
}
