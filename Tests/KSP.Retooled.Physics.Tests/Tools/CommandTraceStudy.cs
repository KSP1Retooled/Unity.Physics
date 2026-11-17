using PhysicsScene = KSP.Retooled.Physics.Tests.Scenes.PhysicsScene;
using KSP.Retooled.Physics.Commands;
using KSP.Retooled.Physics.Interop;
using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using NUnit.Framework;
using UnityEngine;

namespace KSP.Retooled.Physics.Tests.Tools
{
    /// <summary>
    /// Trace command categories and IDs by thread across physics stages. Requires RETOOLED_COMMAND_TRACE=ON.
    /// Write results to the log and Logs/CommandTrace.txt.
    /// </summary>
    [Explicit("A study that needs a plugin built with RETOOLED_COMMAND_TRACE")]
    [Category("Slow")]
    public unsafe class CommandTraceStudy : PhysicsScene
    {
        private const int Awake = 100;
        private const int Asleep = 10;
        private const int Kinematic = 10;
        private const int Moved = 10;
        private const int Statics = 10;

        [DllImport(NativeLibrary.Library)]
        private static extern void RetooledPhysics_ReadCommandTrace(uint* counts);

        private static readonly string[] Categories = { "?", "Sdk", "World", "Body", "Shape", "Query", "Joint", "Articulation", "Extension" };

        private static string Name(int category, int id)
        {
            string name = category switch
            {
                2 => Enum.GetName(typeof(WorldCommand), (uint)id),
                3 => Enum.GetName(typeof(BodyCommand), (uint)id),
                4 => Enum.GetName(typeof(ShapeCommand), (uint)id),
                _ => null,
            };
            // The plugin counts a GetPose that it forwards to the kernel as category 15.
            string categoryName = category == 15 ? "ForwardedToKernel" : category < Categories.Length ? Categories[category] : category.ToString();
            name = category == 15 ? Enum.GetName(typeof(BodyCommand), (uint)id) : name;
            return $"{categoryName}.{name ?? id.ToString()}";
        }

        private static string Read(string stage)
        {
            uint[] counts = new uint[2 * 16 * 256];
            fixed (uint* pointer = counts)
            {
                RetooledPhysics_ReadCommandTrace(pointer);
            }

            var text = new StringBuilder($"  {stage}:");
            for (int thread = 0; thread < 2; thread++)
            {
                for (int category = 0; category < 16; category++)
                {
                    for (int id = 0; id < 256; id++)
                    {
                        uint count = counts[(thread * 16 + category) * 256 + id];
                        if (count != 0)
                        {
                            text.Append($" {(thread == 0 ? "main" : "worker")} {Name(category, id)} x{count};");
                        }
                    }
                }
            }

            return text.ToString();
        }

        [Test]
        public void Trace()
        {
            UnityEngine.Physics.autoSyncTransforms = false;
            var bodies = new List<Rigidbody>();
            for (int i = 0; i < Awake + Asleep + Kinematic + Moved; i++)
            {
                Rigidbody body = WeightlessBox(new Vector3(3.0f * (i % 20), 10.0f, 3.0f * (i / 20)));
                body.sleepThreshold = 0.0f;
                body.angularVelocity = new Vector3(0.3f, 0.2f, 0.1f);
                Transform model = new GameObject("model").transform;
                model.SetParent(body.transform, false);
                for (int c = 1; c < 4; c++)
                {
                    new GameObject("mesh").transform.SetParent(model, false);
                }

                bodies.Add(body);
            }

            for (int i = 0; i < Statics; i++)
            {
                Box(new Vector3(3.0f * i, -5.0f, -10.0f), Vector3.one);
            }

            for (int i = 0; i < Kinematic; i++)
            {
                bodies[Awake + Asleep + i].isKinematic = true;
            }

            UnityEngine.Physics.Simulate(Step);
            UnityEngine.Physics.Simulate(Step);
            for (int i = 0; i < Asleep; i++)
            {
                Rigidbody body = bodies[Awake + i];
                body.sleepThreshold = 1.0f;
                body.linearVelocity = Vector3.zero;
                body.angularVelocity = Vector3.zero;
                body.Sleep();
            }

            UnityEngine.PhysicsScene scene = UnityEngine.Physics.defaultPhysicsScene;
            var report = new StringBuilder($"[CommandTraceStudy] {Awake} awake bodies with 4 child transforms, {Asleep} asleep, {Kinematic} kinematic with targets, {Moved} moved by script, {Statics} static boxes\n");
            report.AppendLine(Read("before the steps"));
            for (int step = 0; step < 3; step++)
            {
                for (int i = 0; i < Kinematic; i++)
                {
                    Rigidbody body = bodies[Awake + Asleep + i];
                    body.MovePosition(body.position + new Vector3(0.01f, 0.0f, 0.0f));
                }

                for (int i = 0; i < Moved; i++)
                {
                    Transform transform = bodies[Awake + Asleep + Kinematic + i].transform;
                    transform.position += new Vector3(0.0f, 0.01f, 0.0f);
                }

                report.AppendLine($" step {step}");
                report.AppendLine(Read("script"));
                scene.RunSimulationStages(Step, SimulationStage.PrepareSimulation, SimulationOption.All);
                report.AppendLine(Read("Prepare"));
                scene.RunSimulationStages(Step, SimulationStage.RunSimulation, SimulationOption.All);
                report.AppendLine(Read("Run"));
                scene.RunSimulationStages(Step, SimulationStage.PublishSimulationResults, SimulationOption.All);
                report.AppendLine(Read("Publish"));
            }

            UnityEngine.Debug.Log(report.ToString());
            File.WriteAllText(Path.Combine(Application.dataPath, "..", "Logs", "CommandTrace.txt"), report.ToString());
        }
    }
}
