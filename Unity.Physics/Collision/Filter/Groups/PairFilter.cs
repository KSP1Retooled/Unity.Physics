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
    internal static partial class PairFilter
    {
        /// <summary>
        /// Store newly computed body-pair filtering results. Clear the cache if it exceeds 32 entries per body plus
        /// 1024.
        /// </summary>
        [BurstCompile]
        internal struct RememberBodyPairsJob : IJob
        {
            private const int KeptPerBody = 32;

            public NativeHashMap<ulong, KnownBodyPairs.KnownBodyPair> Known;

            public NativeQueue<DropIgnoredPairsJob.KeyValue> Learned;

            public int Bodies;

            public void Execute()
            {
                if (Known.Count + Learned.Count > KeptPerBody * Bodies + 1024)
                {
                    Known.Clear();
                }

                while (Learned.TryDequeue(out DropIgnoredPairsJob.KeyValue learned))
                {
                    Known[learned.Key] = learned.Value;
                }
            }
        }
    }
}
