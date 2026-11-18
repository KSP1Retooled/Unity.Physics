using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Physics.Dynamics.Solver.Forest;
using Unity.Physics.Dynamics.Solver.Islands.Data;
using static Unity.Physics.Solver;

namespace Unity.Physics.Dynamics.Solver.Islands.Scheduling
{
    internal static partial class IslandSolveScheduler
    {
        /// <summary>
        /// List direct-island Jacobians in iterator order, joints before contacts, and count first-substep rows.
        /// </summary>
        [BurstCompile]
        unsafe struct IndexDirectIslandsJob : IJob
        {
            [ReadOnly]
            public DispatchPairSequencer.DirectSolverSchedulerInfo Info;

            public NativeStream.Reader JacobiansReader;

            public DirectIslandWork Work;

            public IslandSolverCache Cache;

            /// <summary>The world's bodies, which tell the cache who each island's first body is.</summary>
            [ReadOnly]
            public NativeArray<RigidBody> Bodies;

            /// <summary>The number of moving bodies in the world, which the islands' shared index of bodies covers.</summary>
            public int MotionCount;

            public void Execute()
            {
                int islandCount = Info.NumIslands;

                // The job that found the islands also listed each island's vessels. Without that list, the islands have
                // no vessels.
                bool vesselsKnown = Info.IslandVesselInfos.Length == islandCount;
                Work.IslandVessels.Clear();
                // The cache gives its slots to the islands after MarkFusedVesselIslandsJob has found the fused islands,
                // which keep their state in slots of their own.
                if (Cache.IsCreated)
                {
                    UnsafeList<int>* firstBodies = Work.Buffers.FirstBodies;
                    UnsafeList<ulong>* firstIdentities = Work.Buffers.FirstIdentities;
                    firstBodies->Resize(islandCount);
                    firstIdentities->Resize(islandCount);
                    var bodies = (RigidBody*)Bodies.GetUnsafeReadOnlyPtr();
                    for (int island = 0; island < islandCount; island++)
                    {
                        int first = Info.IslandBodies[Info.FirstIslandBodyIndices[island]];
                        firstBodies->Ptr[island] = first;
                        firstIdentities->Ptr[island] = ForestIsland.BodyIdentity(bodies, Bodies.Length, first);
                    }

                    Cache.ClearCompactIndex(MotionCount);
                }

                Work.Islands.ResizeUninitialized(islandCount);
                Work.Buffers.Bounded->Resize(islandCount);

                // Count rows while listing headers, then convert counts to offsets to avoid rereading cold
                // headers.
                UnsafeList<int>* headerRows = Work.Buffers.HeaderRows;
                headerRows->Clear();
                for (int island = 0; island < islandCount; island++)
                {
                    int numContacts = Info.IslandContactCounts[island];
                    int firstHeader = Work.Headers.Length;
                    bool touchesSomething = false;

                    // The iterator reads joint pairs before contact pairs. Identify joint Jacobians by excluding the
                    // Contact and Trigger types, which are produced only by contact pairs.
                    int jointHeaders = 0;
                    var iterator = new IslandJacobianIterator(JacobiansReader, Info, island, numContacts);
                    while (iterator.HasJacobiansLeft())
                    {
                        ref JacobianHeader header = ref iterator.ReadJacobianHeader();
                        touchesSomething |= header.Type == JacobianType.Contact && header.Enabled;
                        jointHeaders += header.Type != JacobianType.Contact && header.Type != JacobianType.Trigger ? 1 : 0;
                        Work.Headers.Add(new IslandHeader
                        {
                            Header = (JacobianHeader*)UnsafeUtility.AddressOf(ref header),
                            Island = island,
                        });
                        headerRows->Add(GetConstraintRows(ref header));
                    }

                    int firstVessel = Work.IslandVessels.Length, vesselParts = 0, otherBodies = 0;
                    if (vesselsKnown)
                    {
                        DispatchPairSequencer.IslandVesselInfo vessels = Info.IslandVesselInfos[island];
                        Work.IslandVessels.AddRange((int*)Info.IslandVessels.GetUnsafeReadOnlyPtr() + vessels.FirstVessel, vessels.VesselCount);
                        vesselParts = vessels.VesselParts;
                        otherBodies = vessels.OtherBodies;
                    }

                    Work.Islands[island] = new DirectIsland
                    {
                        FirstHeader = firstHeader,
                        JointEnd = firstHeader + jointHeaders,
                        SolveEnd = touchesSomething ? Work.Headers.Length : firstHeader + jointHeaders,
                        HeaderEnd = Work.Headers.Length,
                        ContactPairs = touchesSomething ? numContacts : 0,
                        FirstVessel = firstVessel,
                        VesselCount = Work.IslandVessels.Length - firstVessel,
                        VesselParts = vesselParts,
                        OtherBodies = otherBodies,
                    };
                }

                CountIslandRowsJob.CountRows(Work, true);
            }
        }
    }
}
