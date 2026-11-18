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

        internal static JobHandle ScheduleMerge(NativeReference<int> filled, NativeList<ContactBucket> buckets,
            NativeList<int2> offsets, NativeList<ManifoldRecord> records, NativeList<ManifoldPoint> points, JobHandle dependency)
        {
            JobHandle sized = new SizeMergedContactsJob
            {
                Filled = filled,
                Buckets = buckets,
                Offsets = offsets,
                Records = records,
                Points = points,
            }.Schedule(dependency);
            return new CopyContactBucketsJob
            {
                Buckets = buckets,
                Offsets = offsets.AsDeferredJobArray(),
                Records = records.AsDeferredJobArray(),
                Points = points.AsDeferredJobArray(),
            }.ScheduleUnsafe(filled, 1, sized);
        }

        /// <summary>Finds where each bucket goes and sizes the output lists in bucket order.</summary>
        [BurstCompile]
        private struct SizeMergedContactsJob : IJob
        {
            [ReadOnly]
            public NativeReference<int> Filled;

            [ReadOnly]
            public NativeList<ContactBucket> Buckets;

            public NativeList<int2> Offsets;
            public NativeList<ManifoldRecord> Records;
            public NativeList<ManifoldPoint> Points;

            public void Execute()
            {
                int count = Filled.Value;
                Offsets.ResizeUninitialized(count);
                int2 next = int2.zero;
                for (int bucket = 0; bucket < count; bucket++)
                {
                    Offsets[bucket] = next;
                    next += new int2(Buckets[bucket].Records.Length, Buckets[bucket].Points.Length);
                }

                Records.ResizeUninitialized(next.x);
                Points.ResizeUninitialized(next.y);
            }
        }

        /// <summary>Copies each bucket into its own range and adjusts the first point of each record.</summary>
        [BurstCompile]
        private unsafe struct CopyContactBucketsJob : IJobParallelForDefer
        {
            [ReadOnly]
            public NativeList<ContactBucket> Buckets;

            [ReadOnly]
            public NativeArray<int2> Offsets;

            [NativeDisableParallelForRestriction]
            public NativeArray<ManifoldRecord> Records;

            [NativeDisableParallelForRestriction]
            public NativeArray<ManifoldPoint> Points;

            public void Execute(int bucket)
            {
                ContactBucket source = Buckets[bucket];
                int2 offset = Offsets[bucket];
                var records = (ManifoldRecord*)Records.GetUnsafePtr() + offset.x;
                UnsafeUtility.MemCpy(records, source.Records.Ptr, (long)source.Records.Length * sizeof(ManifoldRecord));
                UnsafeUtility.MemCpy((ManifoldPoint*)Points.GetUnsafePtr() + offset.y, source.Points.Ptr,
                    (long)source.Points.Length * sizeof(ManifoldPoint));

                for (int i = 0; i < source.Records.Length; i++)
                {
                    records[i].FirstPoint += offset.y;
                }
            }
        }
    }
}
