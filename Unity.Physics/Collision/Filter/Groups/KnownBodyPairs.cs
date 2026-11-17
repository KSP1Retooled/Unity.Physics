using System;
using System.Collections.Generic;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Entities;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Physics.Dynamics.Bodies;
using Unity.Physics.Store.Bodies;
using Unity.Physics.Store.Memory;
using Unity.Physics.Store.Reports;
using StoreContactPoint = Unity.Physics.Store.Reports.ContactPoint;
using TriggerEvent = Unity.Physics.Store.Reports.TriggerEvent;
using Unity.Physics.Store.Shapes;
using Unity.Physics.Store.World;
using Unity.Physics.Stepping;
using Unity.Physics.Collision.Colliders.Building;
using Unity.Physics.Collision.Filter.Groups;
using Unity.Physics.Collision.Queries.Collectors;
using Unity.Physics.Dynamics.Joint.Grouping;
using Unity.Physics.Dynamics.Vessels.Detection;

namespace Unity.Physics.Collision.Filter.Groups
{
    internal static class KnownBodyPairs
    {
        /// <summary>
        /// Cached collision filtering result for a body pair, with filter/build versions and body serials.
        /// Serials distinguish bodies that reused the same store rows.
        /// </summary>
        internal struct KnownBodyPair
        {
            public uint IgnoreVersion;
            public uint WholeBuilds;
            public uint SerialLow;
            public uint SerialHigh;
            public bool Ignored;
        }
    }
}
