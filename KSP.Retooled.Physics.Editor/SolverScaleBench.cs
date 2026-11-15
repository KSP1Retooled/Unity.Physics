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
    /// Measures the time per physics step as the craft size increases, in milliseconds. Each craft has a central stack
    /// and pairs of radial parts braced by struts. Parts are joined as in stock KSP, and thrust is applied to the
    /// bottom part. Run it on each backend with `unity exec --backend`. The benchmark writes
    /// Library/RetooledPhysics/solver-scale-(backend).csv and stops increasing the craft size when steps take too long.
    /// </summary>
    public static class SolverScaleBench
    {
        private const int Steps = 12;
        private const int SettleSteps = 3;
        private const double BurstWaitSeconds = 90.0;
        private const double GiveUpMilliseconds = 1500.0;
        private const float DeltaTime = 0.02f;

        private static readonly int[] StackHeights = { 15, 30, 60, 120, 240, 480 };

        private sealed class Craft
        {
            public readonly List<Rigidbody> Parts = new List<Rigidbody>();
            public int Joints;
        }

        public static void Run()
        {
            SimulationMode simulationMode = UnityEngine.Physics.simulationMode;
            UnityEngine.Physics.simulationMode = SimulationMode.Script;

            string backend = BackendInfo.IsRetooled ? "retooled" : "physx";
            StringBuilder csv = new StringBuilder("parts,joints,ms_per_step,best_ms\n");
            WaitForBurst();

            foreach (int height in StackHeights)
            {
                Craft craft = Build(height);
                List<double> steps = Measure(craft);
                Destroy(craft);

                steps.Sort();
                double median = steps[steps.Count / 2];
                string line = string.Format(CultureInfo.InvariantCulture, "{0},{1},{2:F3},{3:F3}", craft.Parts.Count, craft.Joints, median, steps[0]);
                csv.AppendLine(line);
                Debug.Log($"[SolverScale] {backend} {line}");
                if (median > GiveUpMilliseconds)
                {
                    break;
                }
            }

            string folder = Path.GetFullPath(Path.Combine(Application.dataPath, "..", "Library", "RetooledPhysics"));
            Directory.CreateDirectory(folder);
            File.WriteAllText(Path.Combine(folder, $"solver-scale-{backend}.csv"), csv.ToString());
            UnityEngine.Physics.simulationMode = simulationMode;
        }

        /// <summary>
        /// Steps a small craft while Burst compiles in the background. Stops when the median time improves by less than
        /// ten percent after at least ten seconds, or when the wait times out.
        /// </summary>
        private static void WaitForBurst()
        {
            Craft craft = Build(StackHeights[0]);
            Stopwatch wait = Stopwatch.StartNew();
            double last = double.MaxValue;
            while (wait.Elapsed.TotalSeconds < BurstWaitSeconds)
            {
                List<double> steps = Measure(craft);
                steps.Sort();
                double median = steps[steps.Count / 2];
                if (wait.Elapsed.TotalSeconds > 10.0 && median > last * 0.9)
                {
                    break;
                }

                last = median;
                System.Threading.Thread.Sleep(500);
            }

            Destroy(craft);
        }

        private static List<double> Measure(Craft craft)
        {
            List<double> steps = new List<double>();
            for (int step = 0; step < Steps + SettleSteps; step++)
            {
                craft.Parts[0].AddForce(Vector3.up * craft.Parts.Count * 15.0f);
                Stopwatch watch = Stopwatch.StartNew();
                UnityEngine.Physics.Simulate(DeltaTime);
                watch.Stop();
                if (step >= SettleSteps)
                {
                    steps.Add(watch.Elapsed.TotalMilliseconds);
                }

                if (watch.Elapsed.TotalMilliseconds > GiveUpMilliseconds * 3.0 && steps.Count > 0)
                {
                    break;
                }
            }

            return steps;
        }

        private static Craft Build(int height)
        {
            Craft craft = new Craft();
            Rigidbody below = null;
            Rigidbody lastRadial = null;
            for (int level = 0; level < height; level++)
            {
                Rigidbody part = Part(new Vector3(0.0f, level, 0.0f), craft);
                if (below != null)
                {
                    Join(part, below, new Vector3(0.0f, level - 0.5f, 0.0f), craft);
                }

                if (level % 3 == 1)
                {
                    Rigidbody left = Radial(part, level, -1.0f, craft);
                    Rigidbody right = Radial(part, level, 1.0f, craft);
                    if (lastRadial != null)
                    {
                        Strut(left, lastRadial, craft);
                    }

                    lastRadial = right;
                }

                below = part;
            }

            return craft;
        }

        private static Rigidbody Radial(Rigidbody stack, int level, float side, Craft craft)
        {
            Rigidbody radial = Part(new Vector3(side * 1.05f, level, 0.0f), craft);
            Join(radial, stack, new Vector3(side * 0.55f, level, 0.0f), craft);
            return radial;
        }

        private static Rigidbody Part(Vector3 position, Craft craft)
        {
            GameObject box = GameObject.CreatePrimitive(PrimitiveType.Cube);
            box.transform.position = position;
            Object.DestroyImmediate(box.GetComponent<BoxCollider>());
            Rigidbody body = box.AddComponent<Rigidbody>();
            body.mass = 1.0f;
            body.useGravity = false;
            body.inertiaTensor = Vector3.one * 0.2f;
            body.centerOfMass = Vector3.zero;
            craft.Parts.Add(body);
            return body;
        }

        /// <summary>
        /// Creates a joint configured as in stock KSP: large linear and angular limits, drives that hold the parts
        /// together, and one locked angular axis.
        /// </summary>
        private static void Join(Rigidbody host, Rigidbody target, Vector3 worldAnchor, Craft craft)
        {
            ConfigurableJoint joint = host.gameObject.AddComponent<ConfigurableJoint>();
            joint.connectedBody = target;
            joint.autoConfigureConnectedAnchor = false;
            joint.anchor = host.transform.InverseTransformPoint(worldAnchor);
            joint.connectedAnchor = target.transform.InverseTransformPoint(worldAnchor);
            joint.xMotion = joint.yMotion = joint.zMotion = ConfigurableJointMotion.Limited;
            joint.angularXMotion = joint.angularYMotion = ConfigurableJointMotion.Limited;
            joint.angularZMotion = ConfigurableJointMotion.Locked;
            joint.linearLimit = new SoftJointLimit { limit = 1.0f };
            joint.highAngularXLimit = new SoftJointLimit { limit = 177.0f };
            joint.lowAngularXLimit = new SoftJointLimit { limit = -177.0f };
            joint.angularYLimit = new SoftJointLimit { limit = 177.0f };
            JointDrive linear = new JointDrive { positionSpring = 1e20f, maximumForce = 1e20f };
            joint.xDrive = joint.yDrive = joint.zDrive = linear;
            JointDrive angular = new JointDrive { positionSpring = 60000.0f, maximumForce = 1e20f };
            joint.rotationDriveMode = RotationDriveMode.XYAndZ;
            joint.angularXDrive = joint.angularYZDrive = angular;
            craft.Joints++;
        }

        private static void Strut(Rigidbody host, Rigidbody target, Craft craft)
        {
            ConfigurableJoint joint = host.gameObject.AddComponent<ConfigurableJoint>();
            joint.connectedBody = target;
            joint.xMotion = joint.yMotion = joint.zMotion = ConfigurableJointMotion.Locked;
            joint.angularXMotion = joint.angularYMotion = joint.angularZMotion = ConfigurableJointMotion.Locked;
            craft.Joints++;
        }

        private static void Destroy(Craft craft)
        {
            foreach (Rigidbody part in craft.Parts)
            {
                Object.DestroyImmediate(part.gameObject);
            }
        }
    }
}
