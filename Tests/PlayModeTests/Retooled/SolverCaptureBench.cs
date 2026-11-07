using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Text;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Physics;
using Unity.Profiling.LowLevel.Unsafe;
using UnityEngine;
using Debug = UnityEngine.Debug;

namespace KSP.Retooled.PhysicsBackend.Tests
{
    /// <summary>
    /// Replays everything <see cref="SolverCapture"/> saved through the solver, on one thread in Burst, and writes each
    /// problem's time by stage to Library/SolverCapture/bench.csv. It also reports how far the impulses are from the ones the
    /// flight got, so a faster solver can be shown to give the same answer. Run it with `unity exec --method`.
    /// </summary>
    public static unsafe class SolverCaptureBench
    {
        private const int Repeats = 5;
        private const double BurstWaitSeconds = 120.0;

        [BurstCompile]
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

            public bool Bounded;

            public void Execute()
            {
                MarkManaged();
                var solver = new SparseConstraintSolver(Allocator.Temp);
                solver.Solve(*Problem, Impulses, Bounded);
                *Statistics = solver.Stats;
                solver.Dispose();
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

            StringBuilder csv = new StringBuilder("capture,rows,bodies,blocks,nodes,factor_scalars,pivots,ms,analyze_ms,");
            csv.AppendLine("assemble_ms,factorize_ms,substitute_ms,pivot_ms,impulse_error");

            WaitForBurst(SolverCapture.Read(files[0]));
            foreach (string file in files)
            {
                SolverCapture.Saved saved = SolverCapture.Read(file);
                string name = Path.GetRelativePath(root, file).Replace('\\', '/');
                string line = Measure(name, saved);
                csv.AppendLine(line);
                Debug.Log($"[SolverCaptureBench] {line}");
            }

            File.WriteAllText(Path.Combine(root, "bench.csv"), csv.ToString());
        }

        /// <summary>Burst compiles in the background; a job that ran managed says so.</summary>
        private static void WaitForBurst(SolverCapture.Saved saved)
        {
            float[] impulses = new float[saved.RowCount];
            Stopwatch wait = Stopwatch.StartNew();
            while (Solve(saved, impulses, out _))
            {
                if (wait.Elapsed.TotalSeconds > BurstWaitSeconds)
                {
                    throw new TimeoutException("Burst never compiled the solver");
                }

                System.Threading.Thread.Sleep(250);
            }
        }

        private static string Measure(string name, SolverCapture.Saved saved)
        {
            float[] impulses = new float[saved.RowCount];
            List<Sample> samples = new List<Sample>();
            for (int repeat = 0; repeat < Repeats; repeat++)
            {
                Stopwatch watch = Stopwatch.StartNew();
                bool managed = Solve(saved, impulses, out SparseConstraintSolver.Statistics statistics);
                watch.Stop();
                if (managed)
                {
                    throw new InvalidOperationException("The solve ran without Burst");
                }

                samples.Add(new Sample(watch.Elapsed.TotalMilliseconds, statistics));
            }

            samples.Sort((a, b) => a.Total.CompareTo(b.Total));
            Sample median = samples[samples.Count / 2];
            SparseConstraintSolver.Statistics stats = median.Statistics;

            double error = 0.0;
            for (int row = 0; row < saved.RowCount; row++)
            {
                error = math.max(error, math.abs(impulses[row] - saved.Impulses[row]));
            }

            return string.Join(",", name, Count(saved.RowCount), Count(saved.Bodies.Length), Count(saved.Blocks.Length),
                Count(stats.Nodes), Count(stats.FactorScalars), Count(stats.PivotingIterations), Time(median.Total),
                Milliseconds(stats.AnalyzeTime), Milliseconds(stats.AssembleTime), Milliseconds(stats.FactorizeTime),
                Milliseconds(stats.SubstituteTime), Milliseconds(stats.PivotTime),
                error.ToString("G4", CultureInfo.InvariantCulture));
        }

        /// <summary>Solves on this thread and returns whether that ran managed.</summary>
        private static bool Solve(SolverCapture.Saved saved, float[] impulses, out SparseConstraintSolver.Statistics statistics)
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
            fixed (float* output = impulses)
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
                };

                new SolveJob
                {
                    Problem = &problem,
                    Impulses = output,
                    Statistics = &result,
                    RanManaged = &ranManaged,
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
