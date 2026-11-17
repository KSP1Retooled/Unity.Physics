using System;
using System.IO;
using System.Threading;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Physics;
using Unity.Physics.Dynamics.Solver.Sparse;
using Unity.Numerics.Linear.Solvers.Lowlevel;
using UnityEditor;
using UnityEngine;

namespace KSP.Retooled.Physics.Tests
{
    /// <summary>
    /// Capture direct solver problems to Library/SolverCapture/(RETOOLED_SOLVER_CAPTURE).
    /// Sample RunLength consecutive qualifying problems every SolveInterval, up to MaxRuns groups.
    /// </summary>
    [InitializeOnLoad]
    public static unsafe class SolverCapture
    {
        public const string CaptureRoot = "Library/SolverCapture";

        private const string NameVariable = "RETOOLED_SOLVER_CAPTURE";
        private const int SolveInterval = 1000;
        private const int RunLength = 12;
        private const int MaxRuns = 8;
        private const int SmallestRows = 64;
        private const int Magic = 0x33505352;

        private static string folder;
        private static int solves;

        static SolverCapture()
        {
            string name = Environment.GetEnvironmentVariable(NameVariable);
            if (string.IsNullOrEmpty(name))
            {
                return;
            }

            folder = Path.GetFullPath(Path.Combine(Application.dataPath, "..", CaptureRoot, name));
            Directory.CreateDirectory(folder);
            AssemblyReloadEvents.beforeAssemblyReload += () => SparseProblemObserver.Set(null);
            SparseProblemObserver.Set(Observe);
        }

        /// <summary>A saved problem, whose arrays last as long as it does.</summary>
        internal sealed class Saved
        {
            public SparseConstraintSolver.Body[] Bodies;
            public SparseConstraintSolver.ConstraintBlock[] Blocks;
            public float[] Jacobian;
            public float[] Epsilon;
            public float[] RightHandSide;
            public float[] Lower;
            public float[] Upper;
            public byte[] InitialStates;
            public BoundCoupling[] Couplings;
            public float[] Impulses;
            public bool Bounded;

            public int RowCount => Epsilon.Length;
        }

        internal static Saved Read(string path)
        {
            using BinaryReader reader = new BinaryReader(File.OpenRead(path));
            if (reader.ReadInt32() != Magic)
            {
                throw new InvalidDataException($"{path} isn't a solver capture");
            }

            int bodyCount = reader.ReadInt32();
            int blockCount = reader.ReadInt32();
            int rowCount = reader.ReadInt32();
            int couplingCount = reader.ReadInt32();
            Saved saved = new Saved
            {
                Bounded = reader.ReadBoolean(),
                Bodies = new SparseConstraintSolver.Body[bodyCount],
                Blocks = new SparseConstraintSolver.ConstraintBlock[blockCount],
                Jacobian = new float[rowCount * SparseConstraintSolver.ColumnsPerRow],
                Epsilon = new float[rowCount],
                RightHandSide = new float[rowCount],
                Lower = new float[rowCount],
                Upper = new float[rowCount],
                InitialStates = new byte[rowCount],
                Couplings = new BoundCoupling[couplingCount],
                Impulses = new float[rowCount],
            };

            ReadArray(reader, saved.Bodies);
            ReadArray(reader, saved.Blocks);
            ReadArray(reader, saved.Jacobian);
            ReadArray(reader, saved.Epsilon);
            ReadArray(reader, saved.RightHandSide);
            ReadArray(reader, saved.Lower);
            ReadArray(reader, saved.Upper);
            ReadArray(reader, saved.InitialStates);
            ReadArray(reader, saved.Couplings);
            ReadArray(reader, saved.Impulses);
            return saved;
        }

        private static void Observe(SparseConstraintSolver.Problem* problem, float* impulses, byte bounded)
        {
            // Catch exceptions here so that they cannot propagate back into the Burst code that called this method.
            try
            {
                if (problem->RowCount < SmallestRows)
                {
                    return;
                }

                int solve = Interlocked.Increment(ref solves);
                int run = solve / SolveInterval;
                int inRun = solve % SolveInterval;
                if (run >= 1 && run <= MaxRuns && inRun < RunLength)
                {
                    Write(Path.Combine(folder, $"{run:D3}-{inRun:D2}.bin"), problem, impulses, bounded != 0);
                }
            }
            catch (Exception exception)
            {
                Debug.LogException(exception);
            }
        }

        private static void Write(string path, SparseConstraintSolver.Problem* problem, float* impulses, bool bounded)
        {
            using BinaryWriter writer = new BinaryWriter(File.Create(path));
            writer.Write(Magic);
            writer.Write(problem->BodyCount);
            writer.Write(problem->BlockCount);
            writer.Write(problem->RowCount);
            writer.Write(problem->CouplingCount);
            writer.Write(bounded);

            int rows = problem->RowCount;
            WriteBytes(writer, problem->Bodies, problem->BodyCount * sizeof(SparseConstraintSolver.Body));
            WriteBytes(writer, problem->Blocks, problem->BlockCount * sizeof(SparseConstraintSolver.ConstraintBlock));
            WriteBytes(writer, problem->Jacobian, rows * SparseConstraintSolver.ColumnsPerRow * sizeof(float));
            WriteBytes(writer, problem->Epsilon, rows * sizeof(float));
            WriteBytes(writer, problem->RightHandSide, rows * sizeof(float));
            WriteBytes(writer, problem->Lower, rows * sizeof(float));
            WriteBytes(writer, problem->Upper, rows * sizeof(float));
            WriteBytes(writer, problem->InitialStates, rows);
            WriteBytes(writer, problem->Couplings, problem->CouplingCount * sizeof(BoundCoupling));
            WriteBytes(writer, impulses, rows * sizeof(float));
        }

        private static void WriteBytes(BinaryWriter writer, void* data, int bytes)
        {
            writer.Write(new ReadOnlySpan<byte>(data, bytes));
        }

        private static void ReadArray<T>(BinaryReader reader, T[] array) where T : unmanaged
        {
            fixed (T* pointer = array)
            {
                Span<byte> bytes = new Span<byte>(pointer, array.Length * UnsafeUtility.SizeOf<T>());
                while (bytes.Length > 0)
                {
                    int read = reader.BaseStream.Read(bytes);
                    if (read == 0)
                    {
                        throw new EndOfStreamException();
                    }

                    bytes = bytes.Slice(read);
                }
            }
        }
    }
}
