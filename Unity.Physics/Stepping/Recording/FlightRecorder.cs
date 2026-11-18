using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Unity.Mathematics;
using Unity.Physics.Configuration;
using Unity.Physics.Dynamics.Joint.Loads;
using Unity.Physics.Store.Bodies;
using Unity.Physics.Store.World;
using UnityEngine;

namespace Unity.Physics.Stepping.Recording
{
    internal sealed unsafe class FlightRecorder
    {
        private static FlightRecorder latest;
        private static readonly HashSet<FlightRecorder> active = new HashSet<FlightRecorder>();
        private static long generation;
        private static int queued;
        private static string lastSave;
        private static Task writerTask = Task.CompletedTask;

        private readonly Queue<RecordedStep> window = new Queue<RecordedStep>();
        private RecordedGeometry geometry = new RecordedGeometry();
        private readonly long currentGeneration = generation;
        private long stepNumber;
        private long triggerNumber;
        private int postSteps;
        private RecordingWriter frame = new RecordingWriter();
        private RecordedStep reusable;
        private int lastInputLength = 4096;
        private RecordedBody[] before = Array.Empty<RecordedBody>();
        private RecordedBody[] lastBodies = Array.Empty<RecordedBody>();
        private int lastBodyCount;
        private readonly Dictionary<uint, int> lastRows = new Dictionary<uint, int>();

        internal static FlightRecorder Start(FlightRecorder previous)
        {
            if (previous != null && previous.currentGeneration == generation)
            {
                return previous;
            }

            var recorder = new FlightRecorder();
            active.Add(recorder);
            return recorder;
        }

        internal static void Clear()
        {
            generation++;
            latest = null;
            foreach (FlightRecorder recorder in active)
            {
                recorder.window.Clear();
                recorder.geometry = null;
                recorder.frame = null;
                recorder.reusable = null;
                recorder.before = Array.Empty<RecordedBody>();
                recorder.lastBodies = Array.Empty<RecordedBody>();
                recorder.lastRows.Clear();
            }

            active.Clear();
        }

        internal static void Forget(FlightRecorder recorder)
        {
            if (recorder != null && recorder.triggerNumber != 0)
            {
                recorder.Save($"automatic trigger at step {recorder.triggerNumber}, world closed");
            }

            if (latest == recorder)
            {
                latest = null;
            }

            active.Remove(recorder);
        }

        internal static string SaveLatest(string reason) => latest?.Save(reason);

        internal static string Status(bool enabled)
        {
            FlightRecorder recorder = latest;
            long bytes = 0;
            var geometry = new HashSet<byte[]>();
            if (recorder != null)
            {
                foreach (RecordedStep step in recorder.window)
                {
                    bytes += step.Input.Length + (long)step.Bodies.Length * sizeof(RecordedBody);
                    geometry.UnionWith(step.Geometry);
                }

                foreach (byte[] payload in geometry)
                {
                    bytes += payload.Length;
                }
            }

            return $"Enabled={enabled}, Steps={recorder?.window.Count ?? 0}, Bytes={bytes}, "
                + $"PostTriggerStepsRemaining={recorder?.postSteps ?? 0}, SavesQueued={queued}, LastSave={Volatile.Read(ref lastSave)}";
        }

        internal RecordedStep Begin(StoreStepper* stepper, float deltaTime)
        {
            WorldData* world = stepper->Store;
            RecordedStep step = reusable ?? new RecordedStep();
            reusable = null;
            step.Number = ++stepNumber;
            step.DeltaTime = deltaTime;
            step.ContactStep = stepper->RecorderContactStep;
            step.Geometry.Clear();
            step.Trigger = null;
            frame.BeginFrame(step.Input, lastInputLength);
            if (before.Length < world->bodyCount)
            {
                Array.Resize(ref before, checked((int)world->bodyCount));
            }

            fixed (RecordedBody* input = before)
            {
                RecordedBody.Capture(world, stepper->Globals->engineObjectIdOffset, null, null, input);
            }

            RecordStore.Write(stepper, step, geometry, frame);
            lastInputLength = step.InputLength;
            geometry.Finish(step.Number);
            return step;
        }

        internal void End(StoreStepper* stepper, RecordedStep step)
        {
            WorldData* world = stepper->Store;
            if (currentGeneration != generation || step.ContactStep == stepper->RecorderContactStep)
            {
                return;
            }

            int bodyCount = checked((int)world->bodyCount);
            RecordedBody[] bodies = step.Bodies;
            if (bodies == null || bodies.Length < bodyCount)
            {
                bodies = new RecordedBody[bodyCount];
            }
            fixed (RecordedBody* output = bodies)
            {
                RecordedBody.Capture(world, stepper->Globals->engineObjectIdOffset, stepper->RecorderIslands,
                    stepper->RecorderRestoredMotion, output);
            }

            var manifolds = stepper->Manifolds;
            for (int i = 0; i < manifolds.Length; i++)
            {
                var contact = manifolds[i];
                if (!contact.Trigger)
                {
                    NoteIterative(bodies, bodyCount, contact.RowA);
                    NoteIterative(bodies, bodyCount, contact.RowB);
                }
            }

            int triggerRow = -1;
            string reason = null;
            RecordedBody triggerBefore = default;
            bool canTrigger = triggerNumber == 0 && HasFileSlot;
            lastRows.Clear();
            for (int i = 0; canTrigger && i < bodyCount; i++)
            {
                RecordedBody input = before[i];
                reason = (bodies[i].SolverPath & (int)RecordedSolverPath.RestoredMotion) != 0
                    ? "write-back restored non-finite motion"
                    : Check(input, bodies[i]);
                if (reason == null && input.Serial != 0 && TryPrevious(i, input.Serial, out RecordedBody previous))
                {
                    reason = Check(previous, input);
                    if (reason != null)
                    {
                        input = previous;
                    }
                }

                if (reason != null)
                {
                    triggerRow = i;
                    triggerBefore = input;
                    break;
                }
            }

            step.Bodies = bodies;
            step.BodyCount = bodyCount;
            window.Enqueue(step);
            lastBodies = bodies;
            lastBodyCount = bodyCount;

            latest = this;
            int windowSize = math.clamp(FlightRecorderSwitches.WindowSteps, 1, 10_000);
            if (triggerNumber == 0)
            {
                while (window.Count > windowSize)
                {
                    RecordedStep removed = window.Dequeue();
                    if (!removed.QueuedForSave)
                    {
                        reusable = removed;
                    }
                }

                if (triggerRow >= 0)
                {
                    triggerNumber = step.Number;
                    postSteps = math.clamp(FlightRecorderSwitches.PostTriggerSteps, 0, 1_000);
                    step.Trigger = Summary(stepper, step, triggerRow, triggerBefore, before[triggerRow], reason);
                }
            }
            else
            {
                postSteps--;
            }

            if (triggerNumber != 0 && postSteps == 0)
            {
                Save($"automatic trigger at step {triggerNumber}");
                triggerNumber = 0;
            }
        }

        private bool TryPrevious(int row, uint serial, out RecordedBody previous)
        {
            if (row < lastBodyCount && lastBodies[row].Serial == serial)
            {
                previous = lastBodies[row];
                return true;
            }

            if (lastRows.Count == 0)
            {
                for (int i = 0; i < lastBodyCount; i++)
                {
                    lastRows[lastBodies[i].Serial] = i;
                }
            }

            if (lastRows.TryGetValue(serial, out row))
            {
                previous = lastBodies[row];
                return true;
            }

            previous = default;
            return false;
        }

        private static void NoteIterative(RecordedBody[] bodies, int count, int row)
        {
            if ((uint)row < (uint)count && bodies[row].Island == -1 && bodies[row].Dynamic
                && (bodies[row].Flags & BodyFlags.Sleeping) == 0)
            {
                bodies[row].SolverPath |= (int)RecordedSolverPath.IterativeContacts;
            }
        }

        private static string Check(RecordedBody first, RecordedBody second)
        {
            if (!first.Finite || !second.Finite)
            {
                return "non-finite pose or velocity";
            }

            if (second.Dynamic)
            {
                float linear = FlightRecorderSwitches.LinearVelocityThreshold;
                if (linear >= 0 && math.lengthsq(second.Linear - first.Linear) > linear * linear)
                {
                    return "linear velocity change";
                }

                float angular = FlightRecorderSwitches.AngularVelocityThreshold;
                if (angular >= 0 && math.lengthsq(second.WorldAngular - first.WorldAngular) > angular * angular)
                {
                    return "angular velocity change";
                }
            }

            return null;
        }

        private static string Summary(StoreStepper* stepper, RecordedStep step, int row, RecordedBody previous,
            RecordedBody input, string reason)
        {
            RecordedBody body = step.Bodies[row];
            var text = new StringBuilder();
            text.AppendLine($"Step {step.Number}, {reason}, body {body.Serial}, engine ID {body.EngineId}, row {row}");
            text.AppendLine($"Position {body.Position}, island {body.Island}, size {body.IslandSize}, path {(RecordedSolverPath)body.SolverPath}, flags {body.Flags}");
            text.AppendLine($"Linear velocity {previous.Linear} (step input {input.Linear}) → {body.Linear}");
            text.AppendLine($"Angular velocity {previous.WorldAngular} (step input {input.WorldAngular}) → {body.WorldAngular}");
            WorldData* world = stepper->Store;
            var loads = JointLoadStates.Of(world);
            for (uint i = 0; i < world->jointCount; i++)
            {
                var joint = world->joints[i];
                if ((joint->body0 != null && joint->body0->index == row) || (joint->body1 != null && joint->body1->index == row))
                {
                    var state = loads != null ? loads->Ptr[i] : JointLoadState.Of(joint);
                    text.AppendLine($"Joint {joint->serial}, engine ID {EngineId(joint->userData, stepper->Globals->engineObjectIdOffset)}, "
                        + $"ends {BodyLabel(joint->body0)} and {BodyLabel(joint->body1)}, "
                        + $"force {state.Force}, torque {state.Torque}, break load {state.BreakingForce}/{state.BreakingTorque}");
                }
            }

            var manifolds = stepper->Manifolds;
            var points = stepper->ManifoldPoints;
            for (int i = 0; i < manifolds.Length; i++)
            {
                var contact = manifolds[i];
                if (contact.RowA != row && contact.RowB != row)
                {
                    continue;
                }

                int other = contact.RowA == row ? contact.RowB : contact.RowA;
                string label = other >= 0 && other < world->bodyCount ? BodyLabel(world->bodies[other].owner) : "world";
                var otherShape = contact.RowA == row ? contact.ShapeB : contact.ShapeA;
                ulong collider = otherShape == null ? 0 : EngineId(otherShape->userData, stepper->Globals->engineObjectIdOffset);
                text.AppendLine($"Contact with {label}, collider engine ID {collider}, normal {contact.Normal} from row {contact.RowB} toward row {contact.RowA}, "
                    + $"normal impulse {contact.Impulse:R}, warm impulse {contact.WarmImpulse:R}");
                for (int p = 0; p < contact.PointCount; p++)
                {
                    var point = points[contact.FirstPoint + p];
                    text.AppendLine($"  Point {point.Position}, separation {point.Distance:R}, reported normal impulse {contact.Impulse / contact.PointCount:R}");
                }
            }

            return text.ToString();
        }

        private static string BodyLabel(Body* body) => body == null ? "world" : $"body {body->serial} (row {body->index})";

        internal static ulong EngineId(void* userData, uint offset) => userData == null ? 0 : *(ulong*)((byte*)userData + offset);

        private static bool HasFileSlot => queued < math.max(0, FlightRecorderSwitches.MaxFiles);

        private string Save(string reason)
        {
            if (window.Count == 0 || !HasFileSlot)
            {
                return null;
            }

            try
            {
                string folder = FlightRecorderSwitches.OutputDirectory;
                if (string.IsNullOrWhiteSpace(folder))
                {
                    folder = Path.Combine(Application.persistentDataPath, "PhysicsRecordings");
                }

                string path = Path.GetFullPath(Path.Combine(folder, $"physics-{DateTime.UtcNow:yyyyMMdd-HHmmss}-{Guid.NewGuid():N}.rfr"));
                string unity = Application.unityVersion;
                RecordedStep[] steps = window.ToArray();
                // The writer owns these buffers even after their steps leave the window.
                foreach (RecordedStep step in steps)
                {
                    step.QueuedForSave = true;
                }
                queued++;
                Volatile.Write(ref lastSave, $"queued {path}");
                writerTask = writerTask.ContinueWith(_ => WriteFile(path, reason, unity, steps), TaskScheduler.Default);
                return path;
            }
            catch (Exception error)
            {
                Volatile.Write(ref lastSave, error.Message);
                Debug.LogWarning($"[FlightRecorder] Could not queue a save: {error.Message}");
                return null;
            }
        }

        private static void WriteFile(string path, string reason, string unity, RecordedStep[] steps)
        {
            string temporary = path + ".tmp";
            try
            {
                Directory.CreateDirectory(Path.GetDirectoryName(path));
                using (var file = new FileStream(temporary, FileMode.CreateNew, FileAccess.Write))
                using (var gzip = new GZipStream(file, System.IO.Compression.CompressionLevel.Fastest))
                using (var writer = new BinaryWriter(gzip))
                {
                    writer.Write(RecordingBinary.Magic);
                    writer.Write(RecordingBinary.Version);
                    writer.Write(IntPtr.Size);
                    writer.Write(BitConverter.IsLittleEndian);
                    writer.Write(unity);
                    writer.Write(reason);
                    var summary = new StringBuilder();
                    var geometry = new HashSet<byte[]>();
                    foreach (RecordedStep step in steps)
                    {
                        geometry.UnionWith(step.Geometry);
                        summary.Append(step.Trigger);
                    }

                    writer.Write(summary.ToString());
                    writer.Write(geometry.Count);
                    foreach (byte[] payload in geometry)
                    {
                        writer.Write(payload.Length);
                        writer.Write(payload);
                    }

                    writer.Write(steps.Length);
                    foreach (RecordedStep step in steps)
                    {
                        writer.Write(step.Number);
                        writer.Write(step.DeltaTime);
                        writer.Write(step.InputLength);
                        writer.Write(step.Input, 0, step.InputLength);
                        writer.Write(step.BodyCount);
                        fixed (RecordedBody* bodies = step.Bodies)
                        {
                            writer.Write(new ReadOnlySpan<byte>(bodies, step.BodyCount * sizeof(RecordedBody)));
                        }
                    }
                }

                File.Move(temporary, path);
                Volatile.Write(ref lastSave, path);
                Debug.Log($"[FlightRecorder] Saved {path} ({reason})");
            }
            catch (Exception error)
            {
                Volatile.Write(ref lastSave, error.Message);
                Debug.LogWarning($"[FlightRecorder] Could not save {path}: {error.Message}");
                try
                {
                    File.Delete(temporary);
                }
                catch (IOException)
                {
                }
                catch (UnauthorizedAccessException)
                {
                }
            }
        }
    }
}
