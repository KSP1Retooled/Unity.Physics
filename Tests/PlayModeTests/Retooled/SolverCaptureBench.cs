using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Physics;
using Unity.Physics.Numerics.Retooled;
using Unity.Profiling.LowLevel.Unsafe;
using UnityEngine;
using Debug = UnityEngine.Debug;

namespace KSP.Retooled.Physics.Tests
{
    /// <summary>
    /// Solves all problems saved by <see cref="SolverCapture"/> again, using one thread in Burst. Writes the time for
    /// each stage of each problem to Library/SolverCapture/bench.csv. Also reports the difference between the new
    /// impulses and those recorded during the flight, to check whether a faster solver produces the same answer. Run it
    /// with `unity exec --method`.
    /// </summary>
    public static unsafe class SolverCaptureBench
    {
        private const int Repeats = 5;
        private const double BurstWaitSeconds = 120.0;

        // Burst safety checks are off in player builds and on by default in the editor. The solver accesses lists
        // often, so disable the checks here to measure performance as it would be in a game.
        [BurstCompile(DisableSafetyChecks = true)]
        private struct SolveJob : IJob
        {
            [NativeDisableUnsafePtrRestriction]
            public SparseConstraintSolver.Problem* Problem;

            [NativeDisableUnsafePtrRestriction]
            public float* Impulses;

            [NativeDisableUnsafePtrRestriction]
            public SparseConstraintSolver.Statistics* Statistics;

            [NativeDisableUnsafePtrRestriction]
            public bool* RanManaged;

            [NativeDisableUnsafePtrRestriction]
            public SparseConstraintSolver* Solver;

            public bool Bounded;

            public void Execute()
            {
                MarkManaged();
                Solver->Solve(*Problem, Impulses, Bounded);
                *Statistics = Solver->Stats;
            }

            [BurstDiscard]
            private void MarkManaged()
            {
                *RanManaged = true;
            }
        }

        private readonly struct Sample
        {
            public readonly double Total;
            public readonly SparseConstraintSolver.Statistics Statistics;

            public Sample(double total, SparseConstraintSolver.Statistics statistics)
            {
                Total = total;
                Statistics = statistics;
            }
        }

        public static void Run()
        {
            string root = Path.GetFullPath(Path.Combine(Application.dataPath, "..", SolverCapture.CaptureRoot));
            string[] files = Directory.GetFiles(root, "*.bin", SearchOption.AllDirectories);
            Array.Sort(files, StringComparer.Ordinal);
            if (files.Length == 0)
            {
                throw new InvalidOperationException($"No captures under {root}");
            }

            StringBuilder csv = new StringBuilder("capture,rows,bodies,blocks,nodes,factor_scalars,pivots,factorizations,");
            csv.AppendLine("refinements,residual,ms,analyze_ms,assemble_ms,factorize_ms,substitute_ms,pivot_ms,impulse_error");

            WaitForBurst(SolverCapture.Read(files[0]));
            foreach (IGrouping<string, string> run in files.GroupBy(RunOf))
            {
                List<string> names = run.Select(file => Path.GetRelativePath(root, file).Replace('\\', '/')).ToList();
                List<SolverCapture.Saved> problems = run.Select(SolverCapture.Read).ToList();
                foreach (string line in Measure(names, problems))
                {
                    csv.AppendLine(line);
                    Debug.Log($"[SolverCaptureBench] {line}");
                }
            }

            File.WriteAllText(Path.Combine(root, "bench.csv"), csv.ToString());
        }

        /// <summary>
        /// Repeatedly solves one run's captures in order on this thread, so a sampling profiler can observe steady
        /// work. Logs the time per solve. The argument specifies the run under Library/SolverCapture and the duration
        /// in seconds, separated by a colon, as in "pegasus/004:30". Each pass starts with a fresh solver, so the first
        /// solve analyzes and factorizes the problem as it would during a flight.
        /// </summary>
        public static void Loop(string argument)
        {
            string[] parts = argument.Split(':');
            string folder = Path.GetFullPath(Path.Combine(Application.dataPath, "..", SolverCapture.CaptureRoot,
                Path.GetDirectoryName(parts[0])));
            string[] files = Directory.GetFiles(folder, Path.GetFileName(parts[0]) + "-*.bin");
            Array.Sort(files, StringComparer.Ordinal);
            double seconds = double.Parse(parts[1], CultureInfo.InvariantCulture);
            List<SolverCapture.Saved> run = files.Select(SolverCapture.Read).ToList();
            WaitForBurst(run[0]);

            float[] impulses = new float[run.Max(saved => saved.RowCount)];
            var times = new List<double>();
            long substituteTime = 0;
            Stopwatch total = Stopwatch.StartNew();
            while (total.Elapsed.TotalSeconds < seconds)
            {
                var solver = new SparseConstraintSolver(Allocator.Persistent);
                foreach (SolverCapture.Saved saved in run)
                {
                    Stopwatch watch = Stopwatch.StartNew();
                    Solve(ref solver, saved, impulses, out SparseConstraintSolver.Statistics statistics);
                    times.Add(watch.Elapsed.TotalMilliseconds * 1000.0);
                    substituteTime += statistics.SubstituteTime;
                }

                solver.Dispose();
            }

            times.Sort();
            ProfilerUnsafeUtility.TimestampConversionRatio ratio = ProfilerUnsafeUtility.TimestampToNanosecondsConversionRatio;
            double substituteMicroseconds = substituteTime * (double)ratio.Numerator / ratio.Denominator / 1e3 / times.Count;
            Debug.Log($"[SolverCaptureBench] {parts[0]}: {times.Count} solves, median {times[times.Count / 2]:F1} us, "
                + $"fastest {times[0]:F1} us, substitution {substituteMicroseconds:F1} us per solve");
        }

        /// <summary>Files named run-solve.bin are consecutive solves of one run.</summary>
        private static string RunOf(string file)
        {
            string name = Path.GetFileNameWithoutExtension(file);
            int dash = name.IndexOf('-');
            return Path.Combine(Path.GetDirectoryName(file), dash < 0 ? name : name.Substring(0, dash));
        }

        /// <summary>
        /// Waits for Burst compilation to finish by checking whether the job still runs as managed code.
        /// </summary>
        private static void WaitForBurst(SolverCapture.Saved saved)
        {
            float[] impulses = new float[saved.RowCount];
            Stopwatch wait = Stopwatch.StartNew();
            var solver = new SparseConstraintSolver(Allocator.Persistent);
            while (Solve(ref solver, saved, impulses, out _))
            {
                if (wait.Elapsed.TotalSeconds > BurstWaitSeconds)
                {
                    solver.Dispose();
                    throw new TimeoutException("Burst never compiled the solver");
                }

                System.Threading.Thread.Sleep(250);
            }

            solver.Dispose();
        }

        /// <summary>
        /// Solves a run in order with one solver, as during a flight. Each repetition starts with a fresh solver.
        /// </summary>
        private static IEnumerable<string> Measure(List<string> names, List<SolverCapture.Saved> problems)
        {
            List<Sample>[] samples = names.Select(_ => new List<Sample>()).ToArray();
            double[] errors = new double[names.Count];
            for (int repeat = 0; repeat < Repeats; repeat++)
            {
                var solver = new SparseConstraintSolver(Allocator.Persistent);
                for (int i = 0; i < problems.Count; i++)
                {
                    SolverCapture.Saved saved = problems[i];
                    float[] impulses = new float[saved.RowCount];
                    Stopwatch watch = Stopwatch.StartNew();
                    bool managed = Solve(ref solver, saved, impulses, out SparseConstraintSolver.Statistics statistics);
                    watch.Stop();
                    if (managed)
                    {
                        solver.Dispose();
                        throw new InvalidOperationException("The solve ran without Burst");
                    }

                    samples[i].Add(new Sample(watch.Elapsed.TotalMilliseconds, statistics));
                    for (int row = 0; row < saved.RowCount; row++)
                    {
                        errors[i] = math.max(errors[i], math.abs(impulses[row] - saved.Impulses[row]));
                    }
                }

                solver.Dispose();
            }

            for (int i = 0; i < problems.Count; i++)
            {
                SolverCapture.Saved saved = problems[i];
                samples[i].Sort((a, b) => a.Total.CompareTo(b.Total));
                Sample median = samples[i][samples[i].Count / 2];
                SparseConstraintSolver.Statistics stats = median.Statistics;
                yield return string.Join(",", names[i], Count(saved.RowCount), Count(saved.Bodies.Length),
                    Count(saved.Blocks.Length), Count(stats.Nodes), Count(stats.FactorScalars), Count(stats.PivotingIterations),
                    Count(stats.Factorizations), Count(stats.Refinements), Number(stats.Residual), Time(median.Total),
                    Milliseconds(stats.AnalyzeTime), Milliseconds(stats.AssembleTime), Milliseconds(stats.FactorizeTime),
                    Milliseconds(stats.SubstituteTime), Milliseconds(stats.PivotTime), Number(errors[i]));
            }
        }

        /// <summary>Solves on this thread and returns whether the job ran as managed code.</summary>
        private static bool Solve(ref SparseConstraintSolver solver, SolverCapture.Saved saved, float[] impulses,
            out SparseConstraintSolver.Statistics statistics)
        {
            SparseConstraintSolver.Statistics result = default;
            bool ranManaged = false;
            fixed (SparseConstraintSolver.Body* bodies = saved.Bodies)
            fixed (SparseConstraintSolver.ConstraintBlock* blocks = saved.Blocks)
            fixed (float* jacobian = saved.Jacobian)
            fixed (float* epsilon = saved.Epsilon)
            fixed (float* rightHandSide = saved.RightHandSide)
            fixed (float* lower = saved.Lower)
            fixed (float* upper = saved.Upper)
            fixed (byte* initialStates = saved.InitialStates)
            fixed (BoundCoupling* couplings = saved.Couplings)
            fixed (float* output = impulses)
            fixed (SparseConstraintSolver* solverPointer = &solver)
            {
                var problem = new SparseConstraintSolver.Problem
                {
                    BodyCount = saved.Bodies.Length,
                    Bodies = bodies,
                    BlockCount = saved.Blocks.Length,
                    Blocks = blocks,
                    RowCount = saved.RowCount,
                    Jacobian = jacobian,
                    Epsilon = epsilon,
                    RightHandSide = rightHandSide,
                    Lower = lower,
                    Upper = upper,
                    InitialStates = initialStates,
                    CouplingCount = saved.Couplings.Length,
                    Couplings = couplings,
                };

                new SolveJob
                {
                    Problem = &problem,
                    Impulses = output,
                    Statistics = &result,
                    RanManaged = &ranManaged,
                    Solver = solverPointer,
                    Bounded = saved.Bounded,
                }.Run();
            }

            statistics = result;
            return ranManaged;
        }

        private static string Count(long value)
        {
            return value.ToString(CultureInfo.InvariantCulture);
        }

        private static string Number(double value)
        {
            return value.ToString("G4", CultureInfo.InvariantCulture);
        }

        private static string Time(double milliseconds)
        {
            return milliseconds.ToString("F3", CultureInfo.InvariantCulture);
        }

        private static string Milliseconds(long timestamp)
        {
            ProfilerUnsafeUtility.TimestampConversionRatio ratio = ProfilerUnsafeUtility.TimestampToNanosecondsConversionRatio;
            return Time(timestamp * (double)ratio.Numerator / ratio.Denominator / 1e6);
        }
    }
}
