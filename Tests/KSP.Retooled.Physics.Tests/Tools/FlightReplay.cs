using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using KSP.Retooled.Physics.Stepping;
using Unity.Collections;
using Unity.Jobs;
using Unity.Physics;
using Unity.Physics.Configuration;
using Unity.Physics.Dynamics.Vessels.Detection;
using Unity.Physics.Stepping;
using Unity.Physics.Stepping.Recording;

namespace KSP.Retooled.Physics.Tests.Tools
{
    internal static unsafe class FlightReplay
    {
        internal delegate void Inspect(long step, StoreStepper* world);

        internal static int Run(string path, long stopAt, TextWriter summary, Inspect before = null, Inspect after = null,
            Action<long, string> mismatch = null, long startAt = long.MinValue)
        {
            RecordedPhysicsSettings settings = RecordedPhysicsSettings.Read();
            StoreStepper* original = StepEntryPoints.Current;
            long vesselId = VesselCompiler.RecordingId;
            try
            {
                using var file = File.OpenRead(path);
                using var gzip = new GZipStream(file, CompressionMode.Decompress);
                using var reader = new BinaryReader(gzip);
                if (reader.ReadString() != RecordingBinary.Magic)
                {
                    throw new InvalidDataException("Unsupported flight recorder format.");
                }

                int version = reader.ReadInt32();
                if (version != RecordingBinary.Version && version != RecordingBinary.VersionBeforeServoEdges)
                {
                    throw new InvalidDataException("Unsupported flight recorder format.");
                }

                if (reader.ReadInt32() != IntPtr.Size || reader.ReadBoolean() != BitConverter.IsLittleEndian)
                {
                    throw new InvalidDataException("The recording used a different native layout.");
                }

                summary.WriteLine($"Recorded in Unity {reader.ReadString()}.");
                summary.WriteLine(reader.ReadString());
                summary.Write(reader.ReadString());
                var payloads = ReadGeometry(reader);
                int count = reader.ReadCount(11_000);
                int replayed = 0;
                for (int i = 0; i < count; i++)
                {
                    long number = reader.ReadInt64();
                    if (number > stopAt)
                    {
                        break;
                    }

                    float dt = reader.ReadSingle();
                    int length = reader.ReadCount(1 << 29);
                    byte[] bytes = reader.ReadBytes(length);
                    if (bytes.Length != length)
                    {
                        throw new EndOfStreamException();
                    }

                    if (number < startAt)
                    {
                        // Every step carries its own input, so the steps before the first wanted one are only read past.
                        int skipped = reader.ReadCount(1 << 24);
                        reader.ReadBytes(checked(skipped * sizeof(RecordedBody)));
                        continue;
                    }

                    using var input = new BinaryReader(new MemoryStream(bytes, writable: false));
                    using var store = new ReplayStore();
                    store.Read(input, payloads, version);
                    if (input.BaseStream.Position != input.BaseStream.Length)
                    {
                        throw new InvalidDataException($"Extra input bytes at step {number}.");
                    }

                    StepEntryPoints.Current = store.Stepper;
                    before?.Invoke(number, store.Stepper);
                    Step(store.Stepper, dt);
                    Compare(number, reader, store.Stepper, mismatch);
                    after?.Invoke(number, store.Stepper);
                    replayed++;
                }

                summary.WriteLine(mismatch == null
                    ? $"Replayed {replayed} steps. Every pose and velocity matched bit for bit."
                    : $"Replayed {replayed} steps. Differences from the recording went to the caller.");
                return replayed;
            }
            finally
            {
                StepEntryPoints.Current = original;
                VesselCompiler.RecordingId = vesselId;
                settings.Apply();
            }
        }

        private static Dictionary<int, byte[]> ReadGeometry(BinaryReader reader)
        {
            var payloads = new Dictionary<int, byte[]>();
            int count = reader.ReadCount(1 << 24);
            for (int i = 0; i < count; i++)
            {
                int length = reader.ReadCount(1 << 29);
                byte[] data = reader.ReadBytes(length);
                if (length < sizeof(int) || data.Length != length)
                {
                    throw new InvalidDataException("Invalid geometry payload.");
                }

                payloads.Add(BitConverter.ToInt32(data, 0), data);
            }

            return payloads;
        }

        private static void Step(StoreStepper* state, float dt)
        {
            Simulation simulation = Simulation.Create();
            var singleton = new SimulationSingleton();
            singleton.InitializeFromSimulation(ref simulation);
            NativeReference<int> staticsChanged = default;
            JobHandle dependency = default;
            try
            {
                StepEntryPoints.ApplyForces(dt);
                StepEntryPoints.BuildWorld(dt, &dependency);
                staticsChanged = new NativeReference<int>(state->StaticsChanged, Allocator.TempJob);
                PhysicsStep settings = state->Settings;
                bool multiThreaded = settings.MultiThreaded > 0;
                var input = new SimulationStepInput
                {
                    World = state->World,
                    TimeStep = dt,
                    Gravity = settings.Gravity,
                    EnableGyroscopicTorque = settings.EnableGyroscopicTorque,
                    SynchronizeCollisionWorld = settings.SynchronizeCollisionWorld > 0,
                    NumSolverIterations = settings.SolverIterationCount,
                    DirectSolverSettings = settings.DirectSolverSettings,
                    MaxDynamicDepenetrationVelocity = settings.MaxDynamicDepenetrationVelocity,
                    MaxStaticDepenetrationVelocity = settings.MaxStaticDepenetrationVelocity,
                    SolverStabilizationHeuristicSettings = settings.SolverStabilizationHeuristicSettings,
                    HaveStaticBodiesChanged = staticsChanged.AsReadOnly(),
                };

                dependency = simulation.ScheduleBroadphaseJobs(input, dependency, multiThreaded).FinalExecutionHandle;
                StepEntryPoints.SortIslands(&singleton, &input, &dependency);
                dependency = simulation.ScheduleNarrowphaseJobs(input, dependency, multiThreaded).FinalExecutionHandle;
                StepEntryPoints.ReadContacts(&singleton, dt, &dependency);
                dependency = simulation.ScheduleCreateJacobiansJobs(input, dependency, multiThreaded).FinalExecutionHandle;
                StepEntryPoints.PrepareSolve(&singleton, &input, &dependency);
                dependency = simulation.ScheduleSolveAndIntegrateJobs(input, dependency, multiThreaded).FinalExecutionHandle;
                StepEntryPoints.FinishPreparedStep(dt, &dependency);
            }
            finally
            {
                dependency.Complete();
                simulation.FinalJobHandle.Complete();
                simulation.Dispose();
                if (staticsChanged.IsCreated)
                {
                    staticsChanged.Dispose();
                }
            }
        }

        // With a mismatch callback, a step that differs from the recording reports its first difference and the count
        // of differing bodies, and the replay goes on: every step starts from its own recorded input.
        private static void Compare(long number, BinaryReader reader, StoreStepper* stepper, Action<long, string> mismatch)
        {
            var store = stepper->Store;
            string first = null;
            int different = 0;
            float worst = 0f;
            uint worstBody = 0;
            int count = reader.ReadCount(1 << 24);
            if (count != store->bodyCount)
            {
                throw new InvalidDataException($"Step {number} has {store->bodyCount} replayed bodies, expected {count}.");
            }

            for (int i = 0; i < count; i++)
            {
                RecordedBody expected = reader.ReadValue<RecordedBody>();
                RecordedBody actual = RecordedBody.Read(store->bodies + i, stepper->Globals->engineObjectIdOffset);
                try
                {
                    if (expected.Serial != actual.Serial)
                    {
                        throw new InvalidDataException($"Step {number}, row {i}: expected body {expected.Serial}, got {actual.Serial}.");
                    }

                    Check(number, expected, nameof(expected.Position), expected.Position, actual.Position);
                    Check(number, expected, nameof(expected.Rotation), expected.Rotation.value, actual.Rotation.value);
                    Check(number, expected, nameof(expected.Linear), expected.Linear, actual.Linear);
                    Check(number, expected, nameof(expected.Angular), expected.Angular, actual.Angular);
                }
                catch (InvalidDataException error) when (mismatch != null)
                {
                    first ??= error.Message;
                    different++;
                    float change = Unity.Mathematics.math.length(expected.Linear - actual.Linear);
                    if (change > worst)
                    {
                        worst = change;
                        worstBody = expected.Serial;
                    }
                }
            }

            if (first != null)
            {
                mismatch(number, $"{different} of {count} bodies differ from the recording; the largest linear velocity difference "
                    + $"is {worst:R} m/s on body {worstBody}. First: {first}");
            }
        }

        private static void Check<T>(long step, RecordedBody body, string field, T expected, T actual) where T : unmanaged
        {
            if (!RecordingBinary.Same(expected, actual))
            {
                throw new InvalidDataException($"Step {step}, body {body.Serial}, engine ID {body.EngineId}, {field}: "
                    + $"expected {expected}, got {actual}. First differing byte {FirstDifferentByte(expected, actual)}.");
            }
        }

        private static int FirstDifferentByte<T>(T expected, T actual) where T : unmanaged
        {
            byte* a = (byte*)&expected;
            byte* b = (byte*)&actual;
            for (int i = 0; i < sizeof(T); i++)
            {
                if (a[i] != b[i])
                {
                    return i;
                }
            }

            return -1;
        }
    }
}
