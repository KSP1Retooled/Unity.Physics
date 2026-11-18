using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using Unity.Physics.Base.Threading;
using static Unity.Physics.Dynamics.Solver.Forest.ForestSolver;
using Unity.Physics.Dynamics.Solver.Forest;
using Unity.Physics.Dynamics.Solver.Islands.Data;
using Unity.Physics.Dynamics.Solver.Islands.Scheduling;
using Unity.Physics.Dynamics.Vessels.Sections;
using Unity.Profiling;

namespace Unity.Physics.Dynamics.Solver.Vessels
{
    internal static partial class VesselIslandSolver
    {
        /// <summary>
        /// Shares a section's strut solve across the team. The leader solves the top between the two substitution stages.
        /// The system fixes the chunks and ranges independently of the lane count.
        /// </summary>
        static unsafe void RunStrutLoads(ref ForestRun run, int s, VesselEquilibrium.ApplyMode mode)
        {
            ForestIsland* forest = run.Forest;
            ref VesselContext kernel = ref forest->Context;
            ref VesselEquilibrium equilibrium = ref forest->Equilibrium;
            if (s >= equilibrium.SectionEntry.Length)
            {
                return;
            }

            using (StrutLoads.BookkeepingMarker.Auto())
            {
                equilibrium.ClearClosures(ref kernel, s);
            }

            int entryIndex = equilibrium.SectionEntry[s];
            if (entryIndex < 0 || VesselEquilibrium.SkipSolve(mode))
            {
                return;
            }

            using ProfilerMarker.AutoScope span = StrutLoads.SolveMarker.Auto();
            long started = Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.Timestamp;
            SectionEquilibrium* entry = equilibrium.Entries.Ptr + entryIndex;
            var tasks = new StrutTasks
            {
                Context = (VesselContext*)UnsafeUtility.AddressOf(ref kernel),
                Equilibrium = (VesselEquilibrium*)UnsafeUtility.AddressOf(ref equilibrium),
                Entry = entry,
                Section = s,
                Add = mode == VesselEquilibrium.ApplyMode.Add,
                FromSum = mode == VesselEquilibrium.ApplyMode.Finish,
            };
            if (mode != VesselEquilibrium.ApplyMode.Finish)
            {
                tasks.Count = entry->Count;
                WorkTeam.RunAll<IslandTaskDispatch>(run.Team, &tasks, IslandTaskKind.StrutLoadsKind, (entry->Count + VesselItemsPerTask - 1) / VesselItemsPerTask, null);
            }

            long loaded = Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.Timestamp;
            if (mode == VesselEquilibrium.ApplyMode.Add)
            {
                entry->Summed++;
                System.Threading.Interlocked.Add(ref StrutLoads.Counts.Data.LoadTicks, loaded - started);
                System.Threading.Interlocked.Add(ref StrutLoads.Counts.Data.AccumulateTicks, loaded - started);
                return;
            }

            if (mode == VesselEquilibrium.ApplyMode.Finish && entry->Summed == 0)
            {
                return;
            }

            WorkTeam.RunAll<IslandTaskDispatch>(run.Team, &tasks, IslandTaskKind.StrutForwardKind, entry->Chunks.ChunkCount, null);
            long topStarted = Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.Timestamp;
            SectionEquilibriumSolve.Top(entry, tasks.FromSum);
            System.Threading.Interlocked.Add(ref StrutLoads.Counts.Data.TopTicks, Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.Timestamp - topStarted);
            WorkTeam.RunAll<IslandTaskDispatch>(run.Team, &tasks, IslandTaskKind.StrutBackKind, entry->Chunks.ChunkCount, null);
            long substituted = Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.Timestamp;
            tasks.Count = entry->Links.Length;
            WorkTeam.RunAll<IslandTaskDispatch>(run.Team, &tasks, IslandTaskKind.StrutLinksKind, (entry->Links.Length + VesselItemsPerTask - 1) / VesselItemsPerTask, null);
            long finished = Unity.Profiling.LowLevel.Unsafe.ProfilerUnsafeUtility.Timestamp;
            System.Threading.Interlocked.Increment(ref StrutLoads.Counts.Data.Solves);
            System.Threading.Interlocked.Add(ref StrutLoads.Counts.Data.SolveTicks, finished - started);
            System.Threading.Interlocked.Add(ref StrutLoads.Counts.Data.LoadTicks, loaded - started);
            System.Threading.Interlocked.Add(ref StrutLoads.Counts.Data.SubstituteTicks, substituted - loaded);
            System.Threading.Interlocked.Add(ref StrutLoads.Counts.Data.LinkTicks, finished - substituted);
        }

        unsafe struct StrutTasks
        {
            public VesselContext* Context;
            public VesselEquilibrium* Equilibrium;
            public SectionEquilibrium* Entry;
            public int Section;
            public int Count;
            public bool Add;
            public bool FromSum;
        }

        internal static unsafe void StrutLoadTask(void* context, int task)
        {
            var tasks = (StrutTasks*)context;
            int from = task * VesselItemsPerTask;
            int to = math.min(from + VesselItemsPerTask, tasks->Count);
            tasks->Equilibrium->Loads(ref *tasks->Context, tasks->Section, from, to, tasks->Add);
        }

        internal static unsafe void StrutForwardTask(void* context, int task)
        {
            var tasks = (StrutTasks*)context;
            SectionEquilibriumSolve.ForwardChunk(tasks->Entry, task, tasks->FromSum);
        }

        internal static unsafe void StrutBackTask(void* context, int task)
        {
            var tasks = (StrutTasks*)context;
            SectionEquilibriumSolve.BackChunk(tasks->Entry, task);
        }

        internal static unsafe void StrutLinkTask(void* context, int task)
        {
            var tasks = (StrutTasks*)context;
            int from = task * VesselItemsPerTask;
            int to = math.min(from + VesselItemsPerTask, tasks->Count);
            tasks->Equilibrium->LinkLoads(ref *tasks->Context, tasks->Section, from, to);
        }
    }
}
