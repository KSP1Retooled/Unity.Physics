using System.Threading;
using Unity.Burst.Intrinsics;

namespace Unity.Physics
{
    /// <summary>
    /// The lanes of one job, sharing the sparse solves of the big direct islands. The first lane to run leads: it solves, and
    /// hands out each stage that splits into independent tasks, taking tasks itself too. The other lanes help with those stages
    /// until it finishes. A helper only waits on a leader that is already running, so a lane that starts late or never starts
    /// costs parallelism and never progress.
    /// </summary>
    internal unsafe struct SparseSolveTeam
    {
        /// <summary>
        /// How many times a helper spins with no stage to work on before it leaves the leader to finish alone, so the lanes
        /// give their threads back if the leader stays serial for long.
        /// </summary>
        const int PatienceSpins = 1 << 20;

        /// <summary>Stages of the direct solver's own, around a sparse solver's.</summary>
        public const int PredictDriftKind = -1;
        public const int ApplyImpulsesKind = -2;
        public const int SetupBodiesKind = -3;

        struct Stage
        {
            public void* Context;
            public double* Vector;
            public int Kind;
            public int TaskCount;
        }

        int leader;
        int finished;

        // Keeps the flags written once per solve a cache line apart from the counters every task updates, wherever the team
        // starts, so reading a flag doesn't pull in a line that the working lanes are writing.
        fixed byte separation[64];

        // The current stage's generation in the high half and its next task in the low half, so that a lane can't take a
        // task from a stage that has ended.
        long next;
        int done;

        // The leader writes a stage before it publishes the stage's generation, and generations alternate between these two.
        // The one a lane reads therefore stays the same for as long as its generation is current.
        Stage even;
        Stage odd;

        /// <summary>Whether this lane is the first to ask, and so leads.</summary>
        public bool TryLead()
        {
            return Interlocked.CompareExchange(ref leader, 1, 0) == 0;
        }

        /// <summary>Lets the helpers go. The leader calls it once it has nothing more to hand out.</summary>
        public void Finish()
        {
            Interlocked.Exchange(ref finished, 1);
        }

        /// <summary>
        /// Runs every task of a stage, on this lane and any helping, and returns once all are done. The context is the sparse
        /// solver whose stage it is, or for the direct solver's own kinds what that stage reads and writes.
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
            Work(generation, stage);

            // Waiting lanes poll with plain reads: a locked read takes the line from the lanes still finishing tasks.
            while (Volatile.Read(ref done) < taskCount)
            {
                Common.Pause();
            }
        }

        /// <summary>Takes tasks from each stage the leader hands out, until it finishes or the lane runs out of patience.</summary>
        public void Help()
        {
            long seen = 0;
            int spins = 0;
            while (Volatile.Read(ref finished) == 0)
            {
                long generation = Interlocked.Read(ref next) >> 32;
                if (generation != seen)
                {
                    seen = generation;
                    spins = 0;
                    Work(generation, (generation & 1) == 0 ? even : odd);
                    continue;
                }

                if (++spins > PatienceSpins)
                {
                    return;
                }

                Common.Pause();
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
            else
            {
                ((SparseConstraintSolver*)stage.Context)->RunTask(stage.Kind, task, stage.Vector);
            }
        }

        /// <summary>Runs a stage's tasks on a team, or one after another on this thread if there is none.</summary>
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
