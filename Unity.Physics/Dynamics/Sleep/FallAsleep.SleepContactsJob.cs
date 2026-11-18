using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Physics.Store.Bodies;
using Unity.Physics.Store.Joints;
using Unity.Physics.Store.Memory;
using Unity.Physics.Store.World;
using Unity.Physics.Stepping;
using Unity.Physics.Dynamics.Joint.Translation;
using Unity.Physics.Dynamics.Bodies;
using Unity.Physics.Dynamics.Contacts;
using static Unity.Physics.Dynamics.Contacts.ManifoldRecords;

namespace Unity.Physics.Dynamics.Sleep
{
    internal static partial class FallAsleep
    {
        /// <summary>
        /// Processes contacts for <see cref="SleepJob"/>. It connects solved bodies that touch and counts each body's
        /// contact points. It also records whether each body touches a body that is not solved, including a static or
        /// kinematic body.
        /// </summary>
        [BurstCompile]
        internal unsafe struct SleepContactsJob : IJob
        {
            [NativeDisableUnsafePtrRestriction]
            public WorldData* World;

            [ReadOnly]
            public NativeList<ManifoldRecord> Records;

            public NativeList<int> Parents;

            public NativeList<int> Contacts;

            public NativeList<bool> TouchesStill;

            public void Execute()
            {
                int count = (int)World->bodyCount;
                Parents.ResizeUninitialized(count);
                Contacts.ResizeUninitialized(count);
                TouchesStill.ResizeUninitialized(count);
                UnsafeUtility.MemClear(Contacts.GetUnsafePtr(), (long)count * sizeof(int));
                UnsafeUtility.MemClear(TouchesStill.GetUnsafePtr(), count);
                NativeArray<int> parents = Parents.AsArray();
                for (int i = 0; i < count; i++)
                {
                    parents[i] = i;
                }

                for (int i = 0; i < Records.Length; i++)
                {
                    ManifoldRecord record = Records[i];
                    if (!record.Touching || record.Trigger)
                    {
                        continue;
                    }

                    bool solvedA = SleepJob.Solved(World->bodies + record.RowA);
                    bool solvedB = SleepJob.Solved(World->bodies + record.RowB);
                    Contacts[record.RowA] += record.PointCount;
                    Contacts[record.RowB] += record.PointCount;
                    TouchesStill[record.RowA] |= !solvedB;
                    TouchesStill[record.RowB] |= !solvedA;
                    if (solvedA && solvedB)
                    {
                        SleepJob.Union(parents, record.RowA, record.RowB);
                    }
                }
            }
        }
    }
}
