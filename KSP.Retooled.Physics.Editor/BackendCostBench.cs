using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Text;
using UnityEngine;
using Debug = UnityEngine.Debug;

namespace KSP.Retooled.PhysicsBackend.Editor
{
    /// <summary>
    /// What one command costs through the selected backend, in wall-clock nanoseconds per call. Run it on each backend with
    /// `unity exec --backend`; it writes Library/RetooledPhysics/cost-(backend).csv. Every case repeats and reports its
    /// median, so a stray pause on the machine does not set a number.
    /// </summary>
    public static class BackendCostBench
    {
        private const int Bodies = 1000;
        private const int Repeats = 7;

        // Large enough that the delegate call and loop are a rounding error.
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
            Physics.simulationMode = SimulationMode.Script;
            Physics.gravity = Vector3.zero;
            Physics.autoSyncTransforms = false;

            Rigidbody[] bodies = new Rigidbody[Bodies];
            for (int i = 0; i < Bodies; i++)
            {
                GameObject box = new GameObject("box");
                box.transform.position = new Vector3((i % 32) * 3.0f, (i / 32) * 3.0f, 0.0f);
                box.AddComponent<BoxCollider>();
                bodies[i] = box.AddComponent<Rigidbody>();
                bodies[i].useGravity = false;
            }

            Physics.SyncTransforms();
            Physics.Simulate(DeltaTime);

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
            Physics.Simulate(DeltaTime);

            Measure("rb.AddForceAtPosition", FastCalls, () =>
            {
                Vector3 force = new Vector3(0.0f, 1.0f, 0.0f);
                for (int i = 0; i < FastCalls; i++)
                {
                    body.AddForceAtPosition(force, Vector3.one);
                }
            });
            Physics.Simulate(DeltaTime);

            WaitForBurstQueries(body);
            Measure("Physics.Raycast hit", SlowCalls, () =>
            {
                Vector3 origin = body.position + new Vector3(0.0f, 0.0f, -5.0f);
                for (int i = 0; i < SlowCalls; i++)
                {
                    Physics.Raycast(origin, Vector3.forward, 10.0f);
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
        /// Holds the bench until queries stop running in Mono, or reports that they did not, so the raycast number says
        /// which code it measured.
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
                Physics.Raycast(body.position + new Vector3(0.0f, 0.0f, -5.0f), Vector3.forward, 10.0f);
                if (BackendHost.ManagedQueries == before)
                {
                    Debug.Log($"[CostBench] queries run in Burst after {watch.Elapsed.TotalSeconds:F1} s");
                    return;
                }

                System.Threading.Thread.Sleep(50);
            }

            Debug.LogWarning($"[CostBench] queries still run in Mono after {BurstWaitSeconds} s");
        }

        /// <summary>A floating origin shift: every body's transform moves, and a sync hands them all to the backend.</summary>
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

                    Physics.SyncTransforms();
                }
            });
            Measure("SyncTransforms alone with nothing moved, per body", Passes * bodies.Length, () =>
            {
                for (int pass = 0; pass < Passes; pass++)
                {
                    Physics.SyncTransforms();
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
                    Physics.Simulate(DeltaTime);
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
                    Physics.Simulate(DeltaTime);
                }
            });
        }
    }
}
