using Unity.Physics.Dynamics.Solver.Islands.Scheduling;
using Unity.Physics.Dynamics.Solver.Vessels;
using Unity.Physics.Dynamics.Solver.Forest;
using System.Threading;
using Unity.Burst.Intrinsics;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Physics.Dynamics.Solver.Sparse;
using Unity.Physics.Dynamics.Vessels.Sections;
using Unity.Profiling;
using Unity.Profiling.LowLevel.Unsafe;

namespace Unity.Physics.Base.Threading
{
    /// <summary>
    /// Coordinate a running leader and helper lanes within one solver job.
    /// The leader publishes independent tasks and also executes them. Late or absent helpers must not prevent
    /// completion.
    /// </summary>
    internal unsafe struct WorkTeam
    {
        /// <summary>
        /// Idle helper spin duration before address waiting, in microseconds.
        /// </summary>
        const long SpinMicroseconds = 50;

        /// <summary>
        /// Idle helper timeout in microseconds before returning to the job system. Checked after waking as well as
        /// while spinning.
        /// </summary>
        const long PatienceMicroseconds = 2000;

        /// <summary>The number of spin pauses between a helper's clock reads.</summary>
        const int SpinsPerClockRead = 64;

        static readonly ProfilerMarker ParkMarker = new ProfilerMarker(ProfilerCategory.Physics, "Physics.Team.Park");

        // The leader's part of a stage that it publishes to the helpers, and a stage of one task, which the leader runs
        // alone while the helpers stay parked.
        static readonly ProfilerMarker LeadStageMarker = new ProfilerMarker(ProfilerCategory.Physics, "Physics.Team.LeadStage");
        static readonly ProfilerMarker LeadAloneMarker = new ProfilerMarker(ProfilerCategory.Physics, "Physics.Team.LeadAlone");

        /// <summary>Identifies the direct solver stages that run around the sparse solver stages.</summary>
        public const int PredictDriftKind = -1;
        public const int ApplyImpulsesKind = -2;
        public const int SetupBodiesKind = -3;
        public const int SummarizeJacobiansKind = -4;
        public const int FillBlocksKind = -5;

        /// <summary>
        /// Identifies one phase of an iteration for pairs solved iteratively. The context points to that phase's job.
        /// </summary>
        public const int IterativePhaseKind = -6;

        /// <summary>
        /// Identifies the stages for factorization, forward substitution and backward substitution, each processed in
        /// chunks of a vessel forest.
        /// </summary>
        public const int ForestFactorKind = -7;
        public const int ForestForwardKind = -8;
        public const int ForestSubstituteKind = -9;

        /// <summary>
        /// Identifies a stage that assigns rows to forest sections or computes body impulses and loads on locked
        /// pairs for those sections.
        /// </summary>
        public const int ForestSectionsKind = -10;

        /// <summary>
        /// Identifies the stage that classifies a direct island's rows for the forest, processing a range of blocks per
        /// task.
        /// </summary>
        public const int ForestClassifyKind = -11;

        /// <summary>
        /// Identifies a stage that processes a range of sections, links or parts per task while solving a vessel
        /// island.
        /// </summary>
        public const int ForestVesselKind = -12;

        /// <summary>Identifies the stage that fills the boundary block of a forest solve, by ranges of its rows.</summary>
        public const int ForestBoundaryKind = -13;

        struct Stage
        {
            public void* Context;
            public double* Vector;
            public int Kind;
            public int TaskCount;
        }

        int leader;
        int finished;

        // Durations in timestamp ticks. The main thread sets them because only it can read the clock's conversion rate.
        long spinTicks;
        long patienceTicks;

        // Separate solve flags from frequently modified task counters by at least one cache line.
        fixed byte separation[64];

        // The upper 32 bits store the current stage's generation; the lower 32 bits store the next task index. Check
        // and update both together so a lane cannot claim a task from a stage that has ended.
        long next;
        int done;

        // Blocked helper count and wake signal. A woken helper passes the wake to another when work is
        // available.
        int parked;
        int signal;

        // When the leader published the current stage, and the last generation in which a lane other than the leader
        // claimed a task. WorkTeamCounters uses them to measure how long helpers take to arrive.
        long publishedAt;
        long helpedGeneration;

        // The leader writes a stage before publishing its generation. Successive generations alternate between these
        // two slots. A slot therefore remains unchanged while its generation is current.
        Stage even;
        Stage odd;

        /// <summary>Attempts to become the leader. Returns true for the first lane that calls it.</summary>
        public bool TryLead()
        {
            return Interlocked.CompareExchange(ref leader, 1, 0) == 0;
        }

        /// <summary>Sets how long helpers spin and wait. The main thread calls it before the team's job runs.</summary>
        public void Configure()
        {
            spinTicks = AddressWait.TicksFromMicroseconds(SpinMicroseconds);
            patienceTicks = AddressWait.TicksFromMicroseconds(PatienceMicroseconds);
        }

        /// <summary>Whether the leader has called <see cref="Finish"/>, so that the team will publish no more stages.</summary>
        public bool Finished => Volatile.Read(ref finished) != 0;

        /// <summary>Releases the helpers. The leader calls this when no more stages remain to be published.</summary>
        public void Finish()
        {
            Interlocked.Exchange(ref finished, 1);
            if (Volatile.Read(ref parked) > 0)
            {
                Interlocked.Increment(ref signal);
                AddressWait.WakeAll((int*)UnsafeUtility.AddressOf(ref signal));
            }
        }

        /// <summary>
        /// Run a stage's tasks with available helpers and wait for completion.
        /// context points to the solver or the direct stage's working data.
        /// </summary>
        public void Lead(void* context, int kind, int taskCount, double* vector)
        {
            if (taskCount == 0)
            {
                return;
            }

            // The leader runs a stage that has one task by itself and does not publish it. If the leader published it,
            // the parked helpers would wake up, find nothing to do, and spin again before they park.
            if (taskCount == 1)
            {
                using (LeadAloneMarker.Auto())
                {
                    Run(new Stage { Context = context, Vector = vector, Kind = kind, TaskCount = 1 }, 0);
                }

                return;
            }

            using ProfilerMarker.AutoScope span = LeadStageMarker.Auto();

            long generation = (Interlocked.Read(ref next) >> 32) + 1;
            var stage = new Stage { Context = context, Vector = vector, Kind = kind, TaskCount = taskCount };
            if ((generation & 1) == 0)
            {
                even = stage;
            }
            else
            {
                odd = stage;
            }

            publishedAt = ProfilerUnsafeUtility.Timestamp;
            Interlocked.Increment(ref WorkTeamCounters.Shared.Data.Stages);
            Interlocked.Exchange(ref done, 0);
            Interlocked.Exchange(ref next, generation << 32);

            // The exchange above is a full memory fence, as is the helper's increment of parked before blocking. Either
            // the helper sees the new generation and stays active, or the leader sees the blocked helper and wakes it.
            if (Volatile.Read(ref parked) > 0)
            {
                Interlocked.Increment(ref signal);
                AddressWait.WakeOne((int*)UnsafeUtility.AddressOf(ref signal));
            }

            Work(generation, stage, true);

            // Poll using reads that do not lock the cache line. A locked read would take ownership of that line from
            // lanes that are still finishing tasks.
            while (Volatile.Read(ref done) < taskCount)
            {
                Common.Pause();
            }
        }

        /// <summary>
        /// Help published stages until the leader finishes or the idle timeout expires.
        /// Spin briefly, then use address waiting when available.
        /// </summary>
        public void Help()
        {
            long seen = 0;
            int spins = 0;
            long idleSince = ProfilerUnsafeUtility.Timestamp;
            while (Volatile.Read(ref finished) == 0)
            {
                long generation = Interlocked.Read(ref next) >> 32;
                if (generation != seen)
                {
                    seen = generation;
                    spins = 0;
                    Work(generation, (generation & 1) == 0 ? even : odd, false);
                    idleSince = ProfilerUnsafeUtility.Timestamp;
                    continue;
                }

                Common.Pause();
                if (++spins % SpinsPerClockRead != 0)
                {
                    continue;
                }

                long idle = ProfilerUnsafeUtility.Timestamp - idleSince;
                if (idle > patienceTicks)
                {
                    return;
                }

                if (AddressWait.Available && idle > spinTicks)
                {
                    Park(seen);
                }
            }
        }

        /// <summary>
        /// Claim available tasks from the current stage without waiting. Return whether an open stage was found.
        /// </summary>
        public bool TryWork()
        {
            long current = Interlocked.Read(ref next);
            long generation = current >> 32;
            if (Volatile.Read(ref finished) != 0 || generation == 0)
            {
                return false;
            }

            // Publish a generation after writing its alternating slot. Check the generation before each task
            // claim.
            Stage stage = (generation & 1) == 0 ? even : odd;
            if ((current & uint.MaxValue) >= stage.TaskCount)
            {
                return false;
            }

            Work(generation, stage, false);
            return true;
        }

        /// <summary>The time that an idle helper spins before it blocks, in timestamp ticks.</summary>
        public long SpinTicks => spinTicks;

        /// <summary>
        /// Waits for a signal from the leader unless a new stage is available or the leader has finished.
        /// </summary>
        void Park(long seen)
        {
            int* word = (int*)UnsafeUtility.AddressOf(ref signal);
            Interlocked.Increment(ref parked);
            int expected = Volatile.Read(ref signal);
            if (Volatile.Read(ref finished) == 0 && Interlocked.Read(ref next) >> 32 == seen)
            {
                using (ParkMarker.Auto())
                {
                    AddressWait.Wait(word, expected);
                }
            }

            Interlocked.Decrement(ref parked);

            // Wake another helper so that the leader needs only one wake call, regardless of the helper count. This
            // lane returns to Help to find its own work.
            if (Interlocked.Read(ref next) >> 32 != seen && Volatile.Read(ref parked) > 0)
            {
                AddressWait.WakeOne(word);
            }
        }

        static void Run(in Stage stage, int task)
        {
            if (stage.Kind == PredictDriftKind)
            {
                Solver.PredictDriftTask(stage.Context, task);
            }
            else if (stage.Kind == ApplyImpulsesKind)
            {
                Solver.ApplyImpulsesTask(stage.Context, task);
            }
            else if (stage.Kind == SetupBodiesKind)
            {
                Solver.SetupBodiesTask(stage.Context, task);
            }
            else if (stage.Kind == SummarizeJacobiansKind)
            {
                Solver.SummarizeJacobiansTask(stage.Context, task);
            }
            else if (stage.Kind == FillBlocksKind)
            {
                Solver.FillBlocksTask(stage.Context, task);
            }
            else if (stage.Kind == IterativePhaseKind)
            {
                IslandSolveScheduler.IterativePhaseTask(stage.Context, task);
            }
            else if (stage.Kind == ForestFactorKind)
            {
                ForestSolver.ForestFactorTask(stage.Context, task);
            }
            else if (stage.Kind == ForestForwardKind)
            {
                ((VesselContext*)stage.Context)->ForwardChunk(task);
            }
            else if (stage.Kind == ForestSubstituteKind)
            {
                ((VesselContext*)stage.Context)->SubstituteChunk(task);
            }
            else if (stage.Kind == ForestSectionsKind)
            {
                ForestSolver.ForestSectionTask(stage.Context, task);
            }
            else if (stage.Kind == ForestClassifyKind)
            {
                ForestSolver.ForestClassifyTask(stage.Context, task);
            }
            else if (stage.Kind == ForestVesselKind)
            {
                VesselIslandSolver.ForestVesselTask(stage.Context, task);
            }
            else if (stage.Kind == ForestBoundaryKind)
            {
                ForestSolver.ForestBoundaryTask(stage.Context, task);
            }
            else
            {
                ((SparseConstraintSolver*)stage.Context)->RunTask(stage.Kind, task, stage.Vector);
            }
        }

        /// <summary>Runs a stage's tasks on the team, or sequentially on this thread if no team is supplied.</summary>
        public static void RunAll(WorkTeam* team, void* context, int kind, int taskCount, double* vector)
        {
            if (team != null)
            {
                team->Lead(context, kind, taskCount, vector);
                return;
            }

            var stage = new Stage { Context = context, Vector = vector, Kind = kind, TaskCount = taskCount };
            for (int task = 0; task < taskCount; task++)
            {
                Run(stage, task);
            }
        }

        void Work(long generation, Stage stage, bool leading)
        {
            while (true)
            {
                long current = Interlocked.Read(ref next);
                long task = current & uint.MaxValue;
                if (current >> 32 != generation || task >= stage.TaskCount)
                {
                    return;
                }

                if (Interlocked.CompareExchange(ref next, current + 1, current) != current)
                {
                    continue;
                }

                // The first task that a lane other than the leader claims in a stage measures how long the helpers took
                // to arrive after the stage was published.
                long helped = Volatile.Read(ref helpedGeneration);
                if (!leading && helped != generation && Interlocked.CompareExchange(ref helpedGeneration, generation, helped) == helped)
                {
                    ref WorkTeamCounters counters = ref WorkTeamCounters.Shared.Data;
                    Interlocked.Increment(ref counters.HelpedStages);
                    Interlocked.Add(ref counters.ArrivalTicks, ProfilerUnsafeUtility.Timestamp - publishedAt);
                }

                Run(stage, (int)task);
                Interlocked.Increment(ref done);
            }
        }
    }
}
