using System.IO;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Physics.Stepping.Recording;

namespace Unity.Physics.Dynamics.Solver.Sparse
{
    internal unsafe partial struct SparseConstraintSolver
    {
        internal readonly void WriteRecording(RecordingWriter writer)
        {
            writer.Write(analyzedBodies);
            if (analyzedBodies < 0)
            {
                return;
            }

            writer.Write(analyzedBounded);
            writer.WriteList(analyzedBlocks);
            writer.WriteList(rowNode);
            writer.WriteList(lower);
            writer.WriteList(upper);

            // The next solve can refine an earlier factor and answer even when matrix values changed.
            writer.Write(factored);
            if (factored)
            {
                writer.WriteList(factor);
                writer.WriteList(inverseDiagonal);
                writer.WriteList(state);
                writer.Write(answered);
                if (answered)
                {
                    writer.WriteList(solution);
                }
            }
        }

        internal void ReadRecording(BinaryReader reader)
        {
            int bodies = reader.ReadInt32();
            if (bodies < 0)
            {
                return;
            }

            bool bounded = reader.ReadBoolean();
            var blocks = new UnsafeList<ConstraintBlock>(0, Allocator.Temp);
            var recordedNodes = new UnsafeList<int>(0, Allocator.Temp);
            try
            {
                reader.ReadList(ref blocks);
                reader.ReadList(ref recordedNodes);
                RestoreAnalysis(bodies, bounded, blocks, recordedNodes);
            }
            finally
            {
                recordedNodes.Dispose();
                blocks.Dispose();
            }

            reader.ReadList(ref lower);
            reader.ReadList(ref upper);
            factored = reader.ReadBoolean();
            if (factored)
            {
                reader.ReadList(ref factor);
                reader.ReadList(ref inverseDiagonal);
                reader.ReadList(ref state);
                answered = reader.ReadBoolean();
                if (answered)
                {
                    reader.ReadList(ref solution);
                }
            }
        }

        private void RestoreAnalysis(int bodies, bool bounded, UnsafeList<ConstraintBlock> blocks, UnsafeList<int> recordedNodes)
        {
            // Symbolic analysis only uses the body pairs, the row ranges and which rows were fixed.
            // Give those rows equal bounds so Analyze builds the same node and factor layout.
            int rows = recordedNodes.Length;
            using var fixedLower = new NativeArray<float>(rows, Allocator.Temp);
            using var fixedUpper = new NativeArray<float>(rows, Allocator.Temp);
            var upperBounds = (float*)fixedUpper.GetUnsafePtr();
            coupled.Clear();
            coupled.Resize(rows, NativeArrayOptions.ClearMemory);
            for (int row = 0; row < rows; row++)
            {
                upperBounds[row] = recordedNodes[row] == NoNode ? 0f : 1f;
            }

            var problem = new Problem
            {
                BodyCount = bodies,
                RowCount = rows,
                BlockCount = blocks.Length,
                Blocks = blocks.Ptr,
                Lower = (float*)fixedLower.GetUnsafeReadOnlyPtr(),
                Upper = upperBounds,
            };
            Analyze(problem, bounded);
        }
    }
}
