using System.IO;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Physics.Stepping.Recording;

namespace Unity.Physics.Dynamics.Solver.Sparse
{
    internal unsafe partial struct SparseConstraintSolver
    {
        /// <summary>The cached factor, symbolic ordering, bounds, and previous solution can affect the next answer.</summary>
        internal readonly void WriteRecording(RecordingWriter writer)
        {
            writer.Write(analyzedBodies);
            writer.Write(analyzedBounded);
            writer.Write(factored);
            writer.Write(answered);
            writer.WriteList(nodes);
            writer.WriteList(order);
            writer.WriteList(neighbours);
            writer.WriteList(taskStarts);
            writer.WriteList(taskPositions);
            writer.WriteList(topPositions);
            writer.WriteList(boundaryPositions);
            writer.WriteList(topUpdateStarts);
            writer.WriteList(topUpdates);
            writer.WriteList(topRowStarts);
            writer.WriteList(topRows);
            writer.WriteList(nodeReady);
            writer.WriteList(bodyRowStarts);
            writer.WriteList(bodyRows);
            writer.WriteList(stageWorst);
            writer.WriteList(rowNode);
            writer.WriteList(rowInNode);
            writer.WriteList(rowUnknown);
            writer.WriteList(factor);
            writer.WriteList(right);
            writer.WriteList(solution);
            writer.WriteList(correction);
            writer.WriteList(inverseDiagonal);
            writer.WriteList(bodySpeedScale);
            writer.WriteList(state);
            writer.WriteList(lower);
            writer.WriteList(upper);
            writer.WriteList(coupled);
            writer.WriteList(analyzedBlocks);
        }

        internal void ReadRecording(BinaryReader reader)
        {
            analyzedBodies = reader.ReadInt32();
            analyzedBounded = reader.ReadBoolean();
            factored = reader.ReadBoolean();
            answered = reader.ReadBoolean();
            reader.ReadList(ref nodes);
            reader.ReadList(ref order);
            reader.ReadList(ref neighbours);
            reader.ReadList(ref taskStarts);
            reader.ReadList(ref taskPositions);
            reader.ReadList(ref topPositions);
            reader.ReadList(ref boundaryPositions);
            reader.ReadList(ref topUpdateStarts);
            reader.ReadList(ref topUpdates);
            reader.ReadList(ref topRowStarts);
            reader.ReadList(ref topRows);
            reader.ReadList(ref nodeReady);
            reader.ReadList(ref bodyRowStarts);
            reader.ReadList(ref bodyRows);
            reader.ReadList(ref stageWorst);
            reader.ReadList(ref rowNode);
            reader.ReadList(ref rowInNode);
            reader.ReadList(ref rowUnknown);
            reader.ReadList(ref factor);
            reader.ReadList(ref right);
            reader.ReadList(ref solution);
            reader.ReadList(ref correction);
            reader.ReadList(ref inverseDiagonal);
            reader.ReadList(ref bodySpeedScale);
            reader.ReadList(ref state);
            reader.ReadList(ref lower);
            reader.ReadList(ref upper);
            reader.ReadList(ref coupled);
            reader.ReadList(ref analyzedBlocks);

            readyStamp = 0;
            UnsafeUtility.MemClear(nodeReady.Ptr, (long)nodeReady.Length * sizeof(int));
        }
    }
}
