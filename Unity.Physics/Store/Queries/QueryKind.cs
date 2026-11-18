namespace Unity.Physics.Store.Queries
{
    /// <summary>Matches QueryKind in Store.h.</summary>
    internal enum QueryKind : uint
    {
        Ray = 0,
        RayAll = 1,
        RayAgainstShape = 2,
        OverlapAny = 3,
        OverlapAll = 4,
        OverlapBroadphase = 5,
        Cast = 6,
        CastAll = 7,
        BodyCast = 8,
        BodyCastAll = 9,
        ClosestPoint = 10,
        Penetration = 11,
        CastAgainstShape = 12,
    }
}
