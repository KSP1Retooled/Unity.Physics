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
using static Unity.Physics.Dynamics.Contacts.ManifoldRecords;

namespace Unity.Physics.Dynamics.Contacts
{
    internal static class ContactBuckets
    {
        /// <summary>
        /// Stores the records and points retained by <see cref="ContactMaterials.ReadContactsJob"/> from one bucket of the narrow phase
        /// stream. A record's first point is indexed relative to the start of that bucket's points.
        /// </summary>
        internal struct ContactBucket
        {
            public UnsafeList<ManifoldRecord> Records;
            public UnsafeList<ManifoldPoint> Points;

            /// <summary>
            /// Indicates whether a manifold contains a moving body touching a sleeping body and should wake it during
            /// contact processing.
            /// </summary>
            public bool Wakes;

            internal void Dispose()
            {
                Records.Dispose();
                Points.Dispose();
            }
        }

        /// <summary>
        /// Prepare reusable lists for the narrow phase stream's buckets and record the count for later jobs.
        /// </summary>
        [BurstCompile]
        internal struct PrepareContactBucketsJob : IJob
        {
            [ReadOnly]
            public NativeStream.Reader Contacts;

            public NativeList<ContactBucket> Buckets;

            public NativeReference<int> Filled;

            public void Execute()
            {
                Filled.Value = Contacts.ForEachCount;
                while (Buckets.Length < Contacts.ForEachCount)
                {
                    Buckets.Add(new ContactBucket
                    {
                        Records = new UnsafeList<ManifoldRecord>(16, Allocator.Persistent),
                        Points = new UnsafeList<ManifoldPoint>(64, Allocator.Persistent),
                    });
                }
            }
        }

        /// <summary>
        /// Merges the records retained by <see cref="ContactMaterials.ReadContactsJob"/> into this step's manifolds, in bucket order.
        /// </summary>
        [BurstCompile]
        internal unsafe struct MergeContactBucketsJob : IJob
        {
            [ReadOnly]
            public NativeReference<int> Filled;

            [ReadOnly]
            public NativeList<ContactBucket> Buckets;

            [NativeDisableContainerSafetyRestriction]
            public NativeList<ManifoldRecord> Records;

            [NativeDisableContainerSafetyRestriction]
            public NativeList<ManifoldPoint> Points;

            public void Execute()
            {
                for (int bucket = 0; bucket < Filled.Value; bucket++)
                {
                    ContactBucket kept = Buckets[bucket];
                    int firstRecord = Records.Length;
                    int pointOffset = Points.Length;
                    Records.AddRange(kept.Records.Ptr, kept.Records.Length);
                    Points.AddRange(kept.Points.Ptr, kept.Points.Length);
                    for (int i = firstRecord; i < Records.Length; i++)
                    {
                        Records.ElementAt(i).FirstPoint += pointOffset;
                    }
                }
            }
        }
    }
}
