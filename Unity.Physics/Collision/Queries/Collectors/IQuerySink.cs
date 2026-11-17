using Unity.Burst;
using Unity.Collections;
using Unity.Entities;
using Unity.Mathematics;
using Unity.Physics.Collision.Colliders.Building;
using Unity.Physics.Store.Bodies;
using Unity.Physics.Store.Queries;
using Unity.Physics.Store.World;
using Unity.Physics.Collision.Queries.Snapshot;

namespace Unity.Physics.Collision.Queries.Collectors
{
    /// <summary>Receives the hits that a <see cref="QueryCollector{T,S}"/> accepts.</summary>
    internal unsafe interface IQuerySink<T>
        where T : unmanaged, IQueryResult
    {
        bool EarlyOut { get; }

        float MaxFraction { get; }

        int NumHits { get; }

        /// <summary>
        /// Return false to stop sending hits. serial is the body's Entity.Version, used for distance tie-breaking.
        /// </summary>
        bool Add(in T hit, QueryShape* shape, uint serial, bool backface, bool overlappedAtStart);
    }
}
