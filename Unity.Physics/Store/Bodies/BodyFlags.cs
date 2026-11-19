using System;

namespace Unity.Physics.Store.Bodies
{
    [Flags]
    internal enum BodyFlags : uint
    {
        Gravity = 1 << 0,
        Kinematic = 1 << 1,
        Simulated = 1 << 2,
        Static = 1 << 8,
        Sleeping = 1 << 9,
        HasKinematicTarget = 1 << 10,
        ColliderDirty = 1 << 11,

        // Only the shapes' poses have changed since the collider was built, so the children of a compound collider
        // can be reused.
        CollidersMoved = 1 << 12,
        ArticulationLink = 1 << 13,
        PublicMask = Gravity | Kinematic | Simulated,
    }
}
