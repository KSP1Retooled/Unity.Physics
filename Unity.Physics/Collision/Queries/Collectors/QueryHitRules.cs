using Unity.Burst;
using Unity.Collections;
using Unity.Entities;
using Unity.Mathematics;
using Unity.Physics.Collision.Colliders.Building;
using Unity.Physics.Store.Bodies;
using Unity.Physics.Store.Queries;
using Unity.Physics.Store.World;

namespace Unity.Physics.Collision.Queries.Collectors
{
    /// <summary>Specifies how to handle hits for each kind of query.</summary>
    internal enum QueryHitRules
    {
        /// <summary>A ray that starts inside a shape doesn't hit it.</summary>
        Ray,

        /// <summary>
        /// Treats hits at the start of a sweep as overlaps. If any triangle in a mesh overlaps, the whole mesh counts
        /// as overlapped.
        /// </summary>
        Sweep,

        /// <summary>Treats every hit as an overlap and accepts either side of a triangle.</summary>
        Overlap,
    }
}
