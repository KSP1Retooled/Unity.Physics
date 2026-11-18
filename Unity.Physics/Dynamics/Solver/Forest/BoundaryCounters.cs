using System.Threading;
using Unity.Burst;
using Unity.Numerics.Linear.Solvers.Lowlevel;

namespace Unity.Physics.Dynamics.Solver.Forest
{
    /// <summary>
    /// Accumulate vessel boundary solve counters since Reset. Read between steps.
    /// </summary>
    internal struct BoundaryCounters
    {
        /// <summary>Velocity solves of a vessel island's boundary, and the rows and pivoting rounds that they took.</summary>
        public long Solves;
        public long Rows;
        public long Rounds;

        /// <summary>The most rounds that one solve took.</summary>
        public long MostRounds;

        /// <summary>Solves that used every round that the cap allows.</summary>
        public long CappedSolves;

        /// <summary>
        /// Solves reaching the round cap without a feasible answer, finished with projected Gauss-Seidel.
        /// </summary>
        public long UnsettledSolves;

        /// <summary>Rows that started from the state that the island remembered for them.</summary>
        public long WarmRows;

        /// <summary>Contact points held out of the boundary, and self-contact manifolds held out of it.</summary>
        public long HeldPoints;
        public long HeldSelfManifolds;

        /// <summary>Contact manifolds against bodies outside the vessel that the cap on outside points left with no point.</summary>
        public long EmptiedOutsideManifolds;

        // Held normal rows returning warm impulses for separation, and rows receiving deep-contact push-out.
        public long HeldReleases;
        public long HeldPushes;

        /// <summary>Held normal rows that pushed in a velocity solve because their contact was closing.</summary>
        public long HeldSupports;

        /// <summary>The most outside contact points and the most self-contact manifolds that one boundary kept.</summary>
        public long MostOutsidePoints;
        public long MostSelfManifolds;

        /// <summary>Joint rows that the solves left out of the boundary because they were dormant.</summary>
        public long DormantRows;
        public long RowKindSamples;
        public long ContactRows;
        public long AngularStopRows;
        public long OtherRows;
        public long StopRetries;

        /// <summary>
        /// What the solves did with the factor of their free rows, see <see cref="BlockPrincipalPivotingStats"/>. Ticks are in
        /// profiler timestamp units.
        /// </summary>
        public BlockPrincipalPivotingStats Factor;

        /// <summary>The time spent in the dense solve of the boundary, in profiler timestamp units.</summary>
        public long SolveTicks;

        // Warm rows whose final state differs from memory, counted separately for first and later substeps.
        public long FirstSubstepChanges;
        public long LaterSubstepChanges;
        public long FirstSubstepSolves;

        // Push-out boundary solve counts, pivot rounds and elapsed time.
        public long PushSolves;
        public long PushRounds;
        public long PushTicks;

        // Time choosing boundary rows, preparing factor paths and filling W.
        public long ChooseTicks;
        public long PathTicks;
        public long FillTicks;

        // Time applying vessel row impulses and processing deep-contact push-out.
        public long ApplyTicks;
        public long PushOutTicks;

        static readonly SharedStatic<BoundaryCounters> Shared = SharedStatic<BoundaryCounters>.GetOrCreate<BoundaryCounters>();

        /// <summary>The counts so far. Read them on the main thread between steps.</summary>
        public static BoundaryCounters Current => Shared.Data;

        public static void Reset()
        {
            Shared.Data = default;
        }

        internal static void AddRowKinds(int contacts, int stops, int others)
        {
            ref BoundaryCounters data = ref Shared.Data;
            Interlocked.Increment(ref data.RowKindSamples);
            Interlocked.Add(ref data.ContactRows, contacts);
            Interlocked.Add(ref data.AngularStopRows, stops);
            Interlocked.Add(ref data.OtherRows, others);
        }

        internal static void AddStopRetry()
        {
            Interlocked.Increment(ref Shared.Data.StopRetries);
        }

        internal static void AddSolve(int rows, int rounds, bool capped, bool unsettled, int warmRows)
        {
            ref BoundaryCounters data = ref Shared.Data;
            Interlocked.Increment(ref data.Solves);
            Interlocked.Add(ref data.Rows, rows);
            Interlocked.Add(ref data.Rounds, rounds);
            Interlocked.Add(ref data.WarmRows, warmRows);
            if (capped)
            {
                Interlocked.Increment(ref data.CappedSolves);
            }

            if (unsettled)
            {
                Interlocked.Increment(ref data.UnsettledSolves);
            }

            Most(ref data.MostRounds, rounds);
        }

        internal static void AddCaps(int heldPoints, int heldManifolds, int outsidePoints, int selfManifolds, int emptiedManifolds)
        {
            ref BoundaryCounters data = ref Shared.Data;
            Interlocked.Add(ref data.HeldPoints, heldPoints);
            Interlocked.Add(ref data.HeldSelfManifolds, heldManifolds);
            Interlocked.Add(ref data.EmptiedOutsideManifolds, emptiedManifolds);
            Most(ref data.MostOutsidePoints, outsidePoints);
            Most(ref data.MostSelfManifolds, selfManifolds);
        }

        internal static void AddFactor(in BlockPrincipalPivotingStats stats, long solveTicks, int changes, bool firstSubstep)
        {
            ref BoundaryCounters data = ref Shared.Data;
            Interlocked.Add(ref data.Factor.Rounds, stats.Rounds);
            Interlocked.Add(ref data.Factor.CouplingRounds, stats.CouplingRounds);
            Interlocked.Add(ref data.Factor.Factorizations, stats.Factorizations);
            Interlocked.Add(ref data.Factor.Updates, stats.Updates);
            Interlocked.Add(ref data.Factor.Reuses, stats.Reuses);
            Interlocked.Add(ref data.Factor.AddedRows, stats.AddedRows);
            Interlocked.Add(ref data.Factor.RemovedRows, stats.RemovedRows);
            Interlocked.Add(ref data.Factor.FactorTicks, stats.FactorTicks);
            Interlocked.Add(ref data.Factor.SolveTicks, stats.SolveTicks);
            Interlocked.Add(ref data.Factor.PivotTicks, stats.PivotTicks);
            Interlocked.Add(ref data.Factor.FullFactorTicks, stats.FullFactorTicks);
            Interlocked.Add(ref data.Factor.FreeRows, stats.FreeRows);
            Most(ref data.Factor.MostFreeRows, stats.MostFreeRows);
            Fewest(ref data.Factor.FewestFreeRows, stats.FewestFreeRows);
            Interlocked.Add(ref data.SolveTicks, solveTicks);
            if (firstSubstep)
            {
                Interlocked.Increment(ref data.FirstSubstepSolves);
                Interlocked.Add(ref data.FirstSubstepChanges, changes);
            }
            else
            {
                Interlocked.Add(ref data.LaterSubstepChanges, changes);
            }
        }

        internal static void AddPush(int rounds, long ticks)
        {
            ref BoundaryCounters data = ref Shared.Data;
            Interlocked.Increment(ref data.PushSolves);
            Interlocked.Add(ref data.PushRounds, rounds);
            Interlocked.Add(ref data.PushTicks, ticks);
        }

        internal static void AddFinish(long applyTicks, long pushOutTicks)
        {
            ref BoundaryCounters data = ref Shared.Data;
            Interlocked.Add(ref data.ApplyTicks, applyTicks);
            Interlocked.Add(ref data.PushOutTicks, pushOutTicks);
        }

        internal static void AddBuild(long chooseTicks, long pathTicks, long fillTicks)
        {
            ref BoundaryCounters data = ref Shared.Data;
            Interlocked.Add(ref data.ChooseTicks, chooseTicks);
            Interlocked.Add(ref data.PathTicks, pathTicks);
            Interlocked.Add(ref data.FillTicks, fillTicks);
        }

        internal static void AddDormant(int dormantRows)
        {
            Interlocked.Add(ref Shared.Data.DormantRows, dormantRows);
        }

        internal static void AddHeld(int releases, int pushes, int supports)
        {
            Interlocked.Add(ref Shared.Data.HeldReleases, releases);
            Interlocked.Add(ref Shared.Data.HeldPushes, pushes);
            Interlocked.Add(ref Shared.Data.HeldSupports, supports);
        }

        /// <summary>Lowers <paramref name="fewest"/> to <paramref name="value"/>. Zero counts as no value in both.</summary>
        static void Fewest(ref long fewest, long value)
        {
            long seen = Interlocked.Read(ref fewest);
            while (value > 0 && (seen == 0 || value < seen))
            {
                long previous = Interlocked.CompareExchange(ref fewest, value, seen);
                if (previous == seen)
                {
                    break;
                }

                seen = previous;
            }
        }

        static void Most(ref long most, long value)
        {
            long seen = Interlocked.Read(ref most);
            while (value > seen)
            {
                long previous = Interlocked.CompareExchange(ref most, value, seen);
                if (previous == seen)
                {
                    break;
                }

                seen = previous;
            }
        }
    }
}
