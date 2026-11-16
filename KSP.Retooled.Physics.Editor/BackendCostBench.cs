using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Text;
using UnityEngine;
using Debug = UnityEngine.Debug;

namespace KSP.Retooled.Physics.Editor
{
    /// <summary>
    /// Measures the elapsed time per command on the selected backend, in nanoseconds. Run it on each backend with
    /// `unity exec --backend`. The benchmark writes Library/RetooledPhysics/cost-(backend).csv. It repeats each case
    /// and reports the median, so an occasional pause on the machine does not determine the reported time.
    /// </summary>
    public static class BackendCostBench
    {
        private const int Bodies = 1000;
        private const int Repeats = 7;

        // Repeat enough calls that the overhead of invoking the delegate and running the loop is negligible.
        private const int FastCalls = 2_000_000;
        private const int SlowCalls = 100_000;
        private const float DeltaTime = 0.02f;
        private const double BurstWaitSeconds = 120.0;

        private static readonly StringBuilder Report = new StringBuilder("case,calls,ns_per_call,best_ns\n");

        private static float sink;

        private static void Measure(string name, int calls, Action run)
        {
            List<double> samples = new List<double>();
            for (int repeat = 0; repeat < Repeats; repeat++)
            {
                Stopwatch watch = Stopwatch.StartNew();
                run();
                watch.Stop();
                samples.Add(watch.Elapsed.TotalMilliseconds * 1e6 / calls);
            }

            samples.Sort();
            double median = samples[samples.Count / 2];
            string line = string.Format(CultureInfo.InvariantCulture, "{0},{1},{2:F1},{3:F1}", name, calls, median, samples[0]);
            Report.AppendLine(line);
            Debug.Log("[CostBench] " + line);
        }

        public static void Run()
        {
            // The command cases below measure the kernels, which run in Mono until their compile finishes.
            if (BackendInfo.IsRetooled && !BackendHost.WaitForCompiledKernels(BurstWaitSeconds))
            {
                Debug.LogWarning($"[CostBench] the kernels still run in Mono after {BurstWaitSeconds} s");
            }

            UnityEngine.Physics.simulationMode = SimulationMode.Script;
            UnityEngine.Physics.gravity = Vector3.zero;
            UnityEngine.Physics.autoSyncTransforms = false;

            Rigidbody[] bodies = new Rigidbody[Bodies];
            for (int i = 0; i < Bodies; i++)
            {
                GameObject box = new GameObject("box");
                box.transform.position = new Vector3((i % 32) * 3.0f, (i / 32) * 3.0f, 0.0f);
                box.AddComponent<BoxCollider>();
                bodies[i] = box.AddComponent<Rigidbody>();
                bodies[i].useGravity = false;
            }

            UnityEngine.Physics.SyncTransforms();
            UnityEngine.Physics.Simulate(DeltaTime);

            Rigidbody body = bodies[0];
            Measure("baseline: empty loop", FastCalls, () =>
            {
                float sum = 0.0f;
                for (int i = 0; i < FastCalls; i++)
                {
                    sum += i;
                }

                sink += sum;
            });
            Measure("rb.linearVelocity get", FastCalls, () =>
            {
                float sum = 0.0f;
                for (int i = 0; i < FastCalls; i++)
                {
                    sum += body.linearVelocity.x;
                }

                sink += sum;
            });
            Measure("rb.linearVelocity set", FastCalls, () =>
            {
                for (int i = 0; i < FastCalls; i++)
                {
                    body.linearVelocity = Vector3.zero;
                }
            });
            Measure("rb.position get", FastCalls, () =>
            {
                float sum = 0.0f;
                for (int i = 0; i < FastCalls; i++)
                {
                    sum += body.position.x;
                }

                sink += sum;
            });
            Measure("rb.mass get", FastCalls, () =>
            {
                float sum = 0.0f;
                for (int i = 0; i < FastCalls; i++)
                {
                    sum += body.mass;
                }

                sink += sum;
            });
            Measure("rb.AddForce", FastCalls, () =>
            {
                Vector3 force = new Vector3(0.0f, 1.0f, 0.0f);
                for (int i = 0; i < FastCalls; i++)
                {
                    body.AddForce(force);
                }
            });
            UnityEngine.Physics.Simulate(DeltaTime);

            Measure("rb.AddForceAtPosition", FastCalls, () =>
            {
                Vector3 force = new Vector3(0.0f, 1.0f, 0.0f);
                for (int i = 0; i < FastCalls; i++)
                {
                    body.AddForceAtPosition(force, Vector3.one);
                }
            });
            UnityEngine.Physics.Simulate(DeltaTime);

            WaitForBurstQueries(body);
            Measure("Physics.Raycast hit", SlowCalls, () =>
            {
                Vector3 origin = body.position + new Vector3(0.0f, 0.0f, -5.0f);
                for (int i = 0; i < SlowCalls; i++)
                {
                    UnityEngine.Physics.Raycast(origin, Vector3.forward, 10.0f);
                }
            });

            MeasureTransformSync(bodies);
            MeasureStep(bodies);

            string folder = Path.Combine("Library", "RetooledPhysics");
            Directory.CreateDirectory(folder);
            string file = Path.Combine(folder, $"cost-{(BackendInfo.IsRetooled ? "retooled" : "physx")}.csv");
            File.WriteAllText(file, Report.ToString());
            Debug.Log($"[CostBench] wrote {file} ({sink})");
        }

        /// <summary>
        /// Waits for queries to run in Burst instead of Mono, or reports that the wait timed out. This identifies which
        /// version of the code was used for the raycast measurement.
        /// </summary>
        private static void WaitForBurstQueries(Rigidbody body)
        {
            if (!BackendInfo.IsRetooled)
            {
                return;
            }

            Stopwatch watch = Stopwatch.StartNew();
            while (watch.Elapsed.TotalSeconds < BurstWaitSeconds)
            {
                long before = BackendHost.ManagedQueries;
                UnityEngine.Physics.Raycast(body.position + new Vector3(0.0f, 0.0f, -5.0f), Vector3.forward, 10.0f);
                if (BackendHost.ManagedQueries == before)
                {
                    Debug.Log($"[CostBench] queries run in Burst after {watch.Elapsed.TotalSeconds:F1} s");
                    return;
                }

                System.Threading.Thread.Sleep(50);
            }

            Debug.LogWarning($"[CostBench] queries still run in Mono after {BurstWaitSeconds} s");
        }

        /// <summary>
        /// Measures a floating origin shift by moving every body's transform and synchronizing the changes to the
        /// backend.
        /// </summary>
        private static void MeasureTransformSync(Rigidbody[] bodies)
        {
            Transform[] transforms = new Transform[bodies.Length];
            for (int i = 0; i < bodies.Length; i++)
            {
                transforms[i] = bodies[i].transform;
            }

            const int Passes = 20;
            float offset = 0.0f;
            Measure("transform.position write and SyncTransforms, per body", Passes * bodies.Length, () =>
            {
                for (int pass = 0; pass < Passes; pass++)
                {
                    offset = offset > 0.0f ? -0.5f : 0.5f;
                    Vector3 shift = new Vector3(offset, 0.0f, 0.0f);
                    for (int i = 0; i < transforms.Length; i++)
                    {
                        transforms[i].position += shift;
                    }

                    UnityEngine.Physics.SyncTransforms();
                }
            });
            Measure("SyncTransforms alone with nothing moved, per body", Passes * bodies.Length, () =>
            {
                for (int pass = 0; pass < Passes; pass++)
                {
                    UnityEngine.Physics.SyncTransforms();
                }
            });
            Measure("rb.position set, per body", Passes * bodies.Length, () =>
            {
                for (int pass = 0; pass < Passes; pass++)
                {
                    for (int i = 0; i < bodies.Length; i++)
                    {
                        bodies[i].position = transforms[i].position;
                    }
                }
            });
        }

        /// <summary>The per-step cost of a thousand awake bodies with nothing to solve, most of which is write-back.</summary>
        private static void MeasureStep(Rigidbody[] bodies)
        {
            for (int i = 0; i < bodies.Length; i++)
            {
                bodies[i].linearVelocity = new Vector3(0.0f, 0.0f, 0.01f);
            }

            const int Steps = 50;
            Measure("Physics.Simulate with interpolation off, per body", Steps * bodies.Length, () =>
            {
                for (int step = 0; step < Steps; step++)
                {
                    UnityEngine.Physics.Simulate(DeltaTime);
                }
            });

            for (int i = 0; i < bodies.Length; i++)
            {
                bodies[i].interpolation = RigidbodyInterpolation.Interpolate;
            }

            Measure("Physics.Simulate with interpolation on, per body", Steps * bodies.Length, () =>
            {
                for (int step = 0; step < Steps; step++)
                {
                    UnityEngine.Physics.Simulate(DeltaTime);
                }
            });
        }
    }
}
