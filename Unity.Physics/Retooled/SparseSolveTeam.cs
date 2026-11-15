using System.Threading;
using Unity.Burst.Intrinsics;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Profiling;
using Unity.Profiling.LowLevel.Unsafe;

namespace Unity.Physics
{
    /// <summary>
    /// Coordinates lanes within one job to solve large direct islands with the sparse solver. The first lane to run
    /// becomes the leader. It performs the solve and publishes stages that can be split into independent tasks, while
    /// also taking tasks itself. Other lanes help with those stages until the leader finishes. Helpers wait only for a
    /// leader that is already running. A lane that starts late or never starts therefore reduces parallelism without
    /// preventing progress.
    /// </summary>
    internal unsafe struct SparseSolveTeam
    {
        /// <summary>
        /// How long an idle helper spins before blocking. If a stage starts during this interval, the helper can work
        /// immediately. A blocked helper takes a few microseconds to wake, which is short compared with this interval.
        /// </summary>
        const long SpinMicroseconds = 50;

        /// <summary>
        /// How long an idle helper waits before leaving the leader to finish alone. This returns helper threads to the
        /// job system if the leader spends a long time on serial work.
        /// </summary>
        const long PatienceMicroseconds = 2000;

        /// <summary>The number of spin pauses between a helper's clock reads.</summary>
        const int SpinsPerClockRead = 64;

        static readonly ProfilerMarker ParkMarker = new ProfilerMarker(ProfilerCategory.Physics, "Retooled.Physics.Island.Park");

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
        /// Identifies factorization and forward substitution by chunks of a vessel forest, and its backward
        /// substitution.
        /// </summary>
        public const int ForestFactorKind = -7;
        public const int ForestForwardKind = -8;
        public const int ForestSubstituteKind = -9;

        /// <summary>
        /// Identifies a stage that assigns rows to forest sections or resolves the bodies and locked loads for those
        /// sections.
        /// </summary>
        public const int ForestSectionsKind = -10;

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

        // Separate flags written once per solve from counters updated for every task by at least one cache line,
        // regardless of the team's starting address. Reading a flag then avoids loading a cache line that working lanes
        // are modifying.
        fixed byte separation[64];

        // The high half stores the current stage's generation, and the low half stores its next task index. Checking
        // both prevents a lane from taking a task from a stage that has ended.
        long next;
        int done;

        // The number of blocked helpers and the signal word on which they wait. The leader changes the word before
        // waking a helper. If that helper finds a new stage, it wakes another helper. The leader therefore needs only
        // one wake call, regardless of the number of blocked helpers.
        int parked;
        int signal;

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
        /// Runs all tasks in a stage using this lane and any helpers, then returns when every task is complete. For
        /// sparse solver stages, the context points to the sparse solver. For the direct solver's stages, it points to
        /// the data that the stage reads and writes.
        /// </summary>
        public void Lead(void* context, int kind, int taskCount, double* vector)
        {
            if (taskCount == 0)
            {
                return;
            }

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

            Interlocked.Exchange(ref done, 0);
            Interlocked.Exchange(ref next, generation << 32);

            // The exchange above is a full memory fence, as is the helper's increment of parked before blocking. Either
            // the helper sees the new generation and stays active, or the leader sees the blocked helper and wakes it.
            if (Volatile.Read(ref parked) > 0)
            {
                Interlocked.Increment(ref signal);
                AddressWait.WakeOne((int*)UnsafeUtility.AddressOf(ref signal));
            }

            Work(generation, stage);

            // Poll with reads that do not lock the cache line. A locked read would take ownership from lanes still
            // finishing tasks.
            while (Volatile.Read(ref done) < taskCount)
            {
                Common.Pause();
            }
        }

        /// <summary>
        /// Takes tasks from each stage that the leader publishes until the leader finishes or this lane exceeds its
        /// idle timeout. Between stages, the lane spins briefly and then blocks while waiting for a signal from the
        /// leader.
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
                    Work(generation, (generation & 1) == 0 ? even : odd);
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
                Solver.IterativePhaseTask(stage.Context, task);
            }
            else if (stage.Kind == ForestFactorKind)
            {
                Solver.ForestFactorTask(stage.Context, task);
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
                Solver.ForestSectionTask(stage.Context, task);
            }
            else
            {
                ((SparseConstraintSolver*)stage.Context)->RunTask(stage.Kind, task, stage.Vector);
            }
        }

        /// <summary>Runs a stage's tasks on the team, or sequentially on this thread if no team is supplied.</summary>
        public static void RunAll(SparseSolveTeam* team, void* context, int kind, int taskCount, double* vector)
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

        void Work(long generation, Stage stage)
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

                Run(stage, (int)task);
                Interlocked.Increment(ref done);
            }
        }
    }
}
