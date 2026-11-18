using System;

namespace Unity.Physics.Stepping.Recording
{
    [Flags]
    internal enum RecordedSolverPath
    {
        Independent = 0,
        PlainIsland = 1,
        UnfusedVessel = 2,
        FusedVessel = 4,
        Forest = 8,
        BoundaryBlock = 16,
        Sparse = 32,
        PushOut = 64,
        IterativeContacts = 128,
    }
}
