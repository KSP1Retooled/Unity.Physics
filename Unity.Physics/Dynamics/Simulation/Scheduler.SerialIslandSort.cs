using Unity.Burst;
using Unity.Collections;
using Unity.Physics.Configuration;
using Unity.Jobs;

namespace Unity.Physics
{
    partial struct DispatchPairSequencer
    {
        internal static void SortPairsForIslands(in PhysicsWorld world, NativeStream.Reader dynamicPairs,
            NativeStream.Reader staticPairs, NativeList<DispatchPair> unsortedPairs, NativeList<DispatchPair> sortedPairs)
        {
            using var pairCount = new NativeReference<int>(Allocator.Temp);
            new SortPairsForIslandsJob
            {
                Joints = world.Joints,
                DynamicVsDynamicPairs = dynamicPairs,
                StaticVsDynamicPairs = staticPairs,
                NumBodies = world.NumBodies,
                UnsortedPairs = unsortedPairs,
                SortedPairs = sortedPairs,
                PairCount = pairCount,
            }.Execute();
        }

        /// <summary>
        /// Sort island dispatch pairs and disable contacts forbidden by joints, matching the parallel chain's
        /// order.
        /// Record the pair count for next step's scheduling choice.
        /// </summary>
        [BurstCompile]
        struct SortPairsForIslandsJob : IJob
        {
            [ReadOnly] public NativeArray<Joint> Joints;
            public NativeStream.Reader DynamicVsDynamicPairs;
            public NativeStream.Reader StaticVsDynamicPairs;
            public int NumBodies;
            public NativeList<DispatchPair> UnsortedPairs;
            public NativeList<DispatchPair> SortedPairs;
            public NativeReference<int> PairCount;
            public IslandPairSortCache Cache;

            public void Execute()
            {
                int count = Count(DynamicVsDynamicPairs) + Count(StaticVsDynamicPairs);
                for (int i = 0; i < Joints.Length; i++)
                {
                    count += Joints[i].BodyPair.IsValid ? 1 : 0;
                }

                PairCount.Value = count;
                UnsortedPairs.ResizeUninitialized(count);
                SortedPairs.ResizeUninitialized(count);
                if (count == 0)
                {
                    Cache.Clear();
                    return;
                }

                NativeArray<DispatchPair> pairs = UnsortedPairs.AsArray();
                int next = 0;
                Append(DynamicVsDynamicPairs, pairs, ref next);
                Append(StaticVsDynamicPairs, pairs, ref next);
                for (int i = 0; i < Joints.Length; i++)
                {
                    Joint joint = Joints[i];
                    if (joint.BodyPair.IsValid)
                    {
                        pairs[next++] = DispatchPair.CreateJoint(joint.BodyPair, i, joint.EnableCollision, joint.SolverType);
                    }
                }

                bool keepPairs = Cache.IsCreated && StepCacheSwitches.PairSortReuseEnabled;
                if (keepPairs && Cache.TryCopySorted(UnsortedPairs, SortedPairs))
                {
                    IslandPairSortCounters.RecordReuse();
                    return;
                }

                // After the radix sort, entry b of the histogram is the index just past body b's pairs.
                var histogram = new NativeArray<int>(NumBodies + 1, Allocator.Temp, NativeArrayOptions.ClearMemory);
                RadixSortPerBodyAJob.RadixSortPerBodyA(pairs, SortedPairs.AsArray(), histogram, NumBodies - 1);
                var subArrays = new SortSubArraysJob { InOutArray = SortedPairs.AsArray(), NextElementIndex = histogram };
                for (int b = 0; b < histogram.Length; b++)
                {
                    subArrays.Execute(b);
                }

                histogram.Dispose();
                SetDisableCollisionFlags(SortedPairs);
                if (keepPairs)
                {
                    Cache.Remember(UnsortedPairs, SortedPairs);
                }
            }

            static int Count(NativeStream.Reader reader)
            {
                int count = 0;
                for (int i = 0; i < reader.ForEachCount; i++)
                {
                    count += reader.BeginForEachIndex(i);
                }

                return count;
            }

            static void Append(NativeStream.Reader reader, NativeArray<DispatchPair> pairs, ref int next)
            {
                for (int i = 0; i < reader.ForEachCount; i++)
                {
                    int count = reader.BeginForEachIndex(i);
                    for (int j = 0; j < count; j++)
                    {
                        pairs[next++] = DispatchPair.CreateCollisionPair(reader.Read<Broadphase.OverlapResult>());
                    }

                    reader.EndForEachIndex();
                }
            }
        }

        /// <summary>Records the number of pairs that the parallel chain sorted. See <see cref="SortPairsForIslandsJob"/>.</summary>
        [BurstCompile]
        struct CountIslandPairsJob : IJob
        {
            [ReadOnly] public NativeList<DispatchPair> Pairs;
            public NativeReference<int> PairCount;

            public void Execute()
            {
                PairCount.Value = Pairs.Length;
            }
        }
    }
}
