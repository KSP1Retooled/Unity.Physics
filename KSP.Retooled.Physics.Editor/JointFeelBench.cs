using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using UnityEngine;

namespace KSP.Retooled.Physics
{
    /// <summary>
    /// The feel rigs of the joint plan, built from Rigidbody and ConfigurableJoint as stock parts are: stacks and a cantilever
    /// with real part masses and spacing, in tonnes, kilonewtons and metres. Run with
    /// <c>RetooledCli unity exec --method KSP.Retooled.Physics.JointFeelBench.Run</c> on either backend. Writes
    /// Library/RetooledPhysics/joint-feel-&lt;backend&gt;.csv.
    /// </summary>
    public static class JointFeelBench
    {
        private const float StackRadius = 0.625f;

        private readonly struct PartSpec
        {
            public readonly float Mass;
            public readonly float Height;
            public readonly float Radius;

            public PartSpec(float mass, float height, float radius)
            {
                Mass = mass;
                Height = height;
                Radius = radius;
            }
        }

        private sealed class Rig
        {
            public string Name;
            public string Parts;
            public float Acceleration;
            public float TipForce;
            public float StepTime = 0.02f;
            public float Duration = 10.0f;
            public bool FixedBase;
            public bool Soft;
        }

        private readonly struct Result
        {
            public readonly float MaxBend;
            public readonly float EquilibriumMean;
            public readonly float LateAmplitude;
            public readonly float MaxGap;

            public Result(float maxBend, float equilibriumMean, float lateAmplitude, float maxGap)
            {
                MaxBend = maxBend;
                EquilibriumMean = equilibriumMean;
                LateAmplitude = lateAmplitude;
                MaxGap = maxGap;
            }
        }

        public static void Run()
        {
            Rig[] rigs =
            {
                new Rig { Name = "stack12_kick", Parts = "TTTTTTTTTTTP", Acceleration = 15.0f },
                new Rig { Name = "heavy_on_decoupler", Parts = "TLLLLLLDRP", Acceleration = 8.0f },
                new Rig { Name = "stack12_warp4", Parts = "TTTTTTTTTTTP", Acceleration = 15.0f, StepTime = 0.08f },
                new Rig { Name = "cantilever_20kN", Parts = "TTTTTTTTTT", TipForce = 20.0f, FixedBase = true, Duration = 15.0f },
                new Rig { Name = "stack12_soft", Parts = "TTTTTTTTTTTP", Acceleration = 15.0f, Soft = true },
            };

            SimulationMode simulationMode = UnityEngine.Physics.simulationMode;
            Vector3 gravity = UnityEngine.Physics.gravity;
            UnityEngine.Physics.simulationMode = SimulationMode.Script;
            UnityEngine.Physics.gravity = Vector3.zero;
            UnityEngine.Physics.defaultSolverIterations = 50;
            UnityEngine.Physics.defaultSolverVelocityIterations = 1;

            string backend = BackendInfo.IsRetooled ? "retooled" : "physx";
            StringBuilder csv = new StringBuilder("rig,maxBendDeg,eqMeanDeg,lateAmpDeg,maxAnchorGap_m\n");
            foreach (Rig rig in rigs)
            {
                Result result = RunRig(rig);
                string line = string.Format(CultureInfo.InvariantCulture, "{0},{1:F3},{2:F3},{3:F3},{4:F5}", rig.Name, result.MaxBend, result.EquilibriumMean, result.LateAmplitude, result.MaxGap);
                csv.AppendLine(line);
                Debug.Log($"[JointFeel] {backend} {line}");
            }

            string folder = Path.GetFullPath(Path.Combine(Application.dataPath, "..", "Library", "RetooledPhysics"));
            Directory.CreateDirectory(folder);
            File.WriteAllText(Path.Combine(folder, $"joint-feel-{backend}.csv"), csv.ToString());

            UnityEngine.Physics.simulationMode = simulationMode;
            UnityEngine.Physics.gravity = gravity;
        }

        private static PartSpec Part(char code)
        {
            switch (code)
            {
                case 'T':
                    return new PartSpec(2.25f, 1.894f, StackRadius);
                case 'L':
                    return new PartSpec(4.5f, 3.7625f, StackRadius);
                case 'P':
                    return new PartSpec(0.84f, 1.047f, StackRadius);
                case 'D':
                    return new PartSpec(0.04f, 0.1f, StackRadius);
                default:
                    return new PartSpec(9.0f, 1.84f, 1.25f);
            }
        }

        private static Vector3 CylinderInertia(PartSpec part)
        {
            float across = part.Mass * (3.0f * part.Radius * part.Radius + part.Height * part.Height) / 12.0f;
            return new Vector3(across, part.Mass * part.Radius * part.Radius / 2.0f, across);
        }

        private static Result RunRig(Rig rig)
        {
            int count = rig.Parts.Length;
            PartSpec[] parts = new PartSpec[count];
            Rigidbody[] bodies = new Rigidbody[count];
            ConfigurableJoint[] joints = new ConfigurableJoint[count];
            GameObject root = new GameObject(rig.Name);

            float height = 0.0f;
            float totalMass = 0.0f;
            for (int i = 0; i < count; i++)
            {
                parts[i] = Part(rig.Parts[i]);
                totalMass += parts[i].Mass;

                GameObject part = new GameObject($"part{i}");
                part.transform.SetParent(root.transform, false);
                part.transform.position = new Vector3(0.0f, height + parts[i].Height / 2.0f, 0.0f);
                height += parts[i].Height;

                Rigidbody body = part.AddComponent<Rigidbody>();
                body.useGravity = false;
                body.mass = parts[i].Mass;
                body.centerOfMass = Vector3.zero;
                body.inertiaTensor = CylinderInertia(parts[i]);
                body.inertiaTensorRotation = Quaternion.identity;
                body.linearDamping = 0.0f;
                body.angularDamping = 0.0f;
                body.maxAngularVelocity = 50.0f;
                body.solverIterations = 50;
                body.solverVelocityIterations = 1;
                body.isKinematic = rig.FixedBase && i == 0;
                bodies[i] = body;

                if (i > 0)
                {
                    joints[i] = Attach(part, bodies[i - 1], parts[i], parts[i - 1], rig.Soft);
                }
            }

            UnityEngine.Physics.SyncTransforms();

            List<float> times = new List<float>();
            List<float> bends = new List<float>();
            float maxGap = 0.0f;
            float thrust = rig.Acceleration * totalMass;
            int steps = Mathf.RoundToInt(rig.Duration / rig.StepTime);
            for (int step = 0; step < steps; step++)
            {
                float time = step * rig.StepTime;
                if (thrust > 0.0f)
                {
                    bodies[0].AddForce(bodies[0].rotation * new Vector3(0.0f, thrust, 0.0f));
                    if (time >= 1.0f && time < 1.3f)
                    {
                        bodies[0].AddForce(new Vector3(0.05f * thrust, 0.0f, 0.0f));
                    }
                }

                if (rig.TipForce > 0.0f)
                {
                    bodies[count - 1].AddForce(new Vector3(rig.TipForce, 0.0f, 0.0f));
                }

                UnityEngine.Physics.Simulate(rig.StepTime);

                times.Add(time + rig.StepTime);
                bends.Add(Bend(bodies[0].rotation, bodies[count - 1].rotation));
                for (int i = 1; i < count; i++)
                {
                    Vector3 own = joints[i].transform.TransformPoint(joints[i].anchor);
                    Vector3 other = bodies[i - 1].transform.TransformPoint(joints[i].connectedAnchor);
                    maxGap = Mathf.Max(maxGap, (own - other).magnitude);
                }
            }

            Object.DestroyImmediate(root);
            return Measure(times, bends, rig.Duration, maxGap);
        }

        private static ConfigurableJoint Attach(GameObject part, Rigidbody parent, PartSpec own, PartSpec below, bool soft)
        {
            ConfigurableJoint joint = part.AddComponent<ConfigurableJoint>();
            joint.connectedBody = parent;
            joint.autoConfigureConnectedAnchor = false;
            joint.anchor = new Vector3(0.0f, -own.Height / 2.0f, 0.0f);
            joint.connectedAnchor = new Vector3(0.0f, below.Height / 2.0f, 0.0f);
            joint.axis = Vector3.up;
            joint.secondaryAxis = Vector3.right;

            joint.xMotion = joint.yMotion = joint.zMotion = ConfigurableJointMotion.Locked;
            ConfigurableJointMotion angular = soft ? ConfigurableJointMotion.Limited : ConfigurableJointMotion.Locked;
            joint.angularXMotion = joint.angularYMotion = joint.angularZMotion = angular;
            if (soft)
            {
                SoftJointLimit limit = new SoftJointLimit { limit = 180.0f };
                joint.highAngularXLimit = joint.lowAngularXLimit = joint.angularYLimit = joint.angularZLimit = limit;
                JointDrive drive = new JointDrive { positionSpring = 6000.0f, positionDamper = 0.0f, maximumForce = 1e20f };
                joint.rotationDriveMode = RotationDriveMode.XYAndZ;
                joint.angularXDrive = joint.angularYZDrive = drive;
            }

            joint.breakForce = float.PositiveInfinity;
            joint.breakTorque = float.PositiveInfinity;
            return joint;
        }

        /// <summary>The signed angle about z between the first and last part's up vectors, in degrees.</summary>
        private static float Bend(Quaternion bottom, Quaternion top)
        {
            Vector3 up = bottom * Vector3.up;
            Vector3 tip = top * Vector3.up;
            return Mathf.Rad2Deg * Mathf.Atan2(Vector3.Cross(up, tip).z, Vector3.Dot(up, tip));
        }

        private static Result Measure(List<float> times, List<float> bends, float duration, float maxGap)
        {
            float max = 0.0f;
            float low = float.MaxValue;
            float high = float.MinValue;
            float sum = 0.0f;
            int counted = 0;
            for (int i = 0; i < bends.Count; i++)
            {
                max = Mathf.Max(max, Mathf.Abs(bends[i]));
                if (times[i] >= duration - 2.0f)
                {
                    low = Mathf.Min(low, bends[i]);
                    high = Mathf.Max(high, bends[i]);
                }

                if (times[i] >= 1.0f)
                {
                    sum += bends[i];
                    counted++;
                }
            }

            return new Result(max, counted > 0 ? sum / counted : 0.0f, (high - low) / 2.0f, maxGap);
        }
    }
}
