using System.IO;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Physics.Stepping.Recording;

namespace Unity.Physics.Dynamics.Solver.Forest
{
    internal unsafe partial struct ForestIsland
    {
        internal bool RecordedLayoutPending;

        /// <summary>Keep locks, the forest pairs and their inverses, and boundary history.</summary>
        internal readonly void WriteRecording(RecordingWriter writer)
        {
            writer.Write(VesselMode);
            writer.Write(BoundaryMemory.CapCount);
            writer.Write(TakesStates);
            writer.WriteList(PairWorldKey);
            writer.WriteList(PairLocked);
            writer.WriteList(PairDeflection);
            writer.WriteList(PairHeld);
            writer.WriteList(VesselSignature);
            writer.WriteList(VesselPartStart);
            writer.WriteList(PartLocked);
            writer.WriteList(PartHeld);
            writer.WriteList(PartRatio);
            writer.WriteList(TakenIds);
            writer.WriteList(TakenVersions);
            writer.WriteList(TakenStarts);
            writer.WriteList(TakenLocked);
            writer.WriteList(TakenHeld);
            writer.WriteList(TakenRatio);
            writer.WriteList(BoundaryMemory.States);
            bool hasPairs = Signature.Length > 0 && Signature[0] != ulong.MaxValue;
            writer.Write(hasPairs);
            if (hasPairs)
            {
                writer.Write(DeflectionCount);
                writer.WriteList(Signature);
                writer.WriteList(BlockPair);
                writer.WriteList(PairBodies);
                writer.WriteList(PairTree);
                writer.WriteList(PairEdge);
                writer.WriteList(PairDeflectionSum);
                writer.WriteList(PairDeflectionStep);
                writer.WriteList(PairContact);
                writer.WriteList(PairBlockStart);
                writer.WriteList(PairBlocks);
                writer.WriteList(PairInverse);
                writer.WriteList(PairInverseKey);
                writer.WriteList(PairInverseRows);
                writer.WriteList(BodyPart);
                writer.WriteList(PartBody);
                writer.WriteList(PartPair);
                writer.WriteList(Parts);
                writer.WriteList(Edges);
            }
        }

        internal void ReadRecording(BinaryReader reader)
        {
            if (!BoundaryMemory.States.IsCreated)
            {
                BoundaryMemory.States = new UnsafeList<ulong>(0, Allocator.Persistent);
            }

            VesselMode = reader.ReadBoolean();
            BoundaryMemory.CapCount = reader.ReadUInt32();
            TakesStates = reader.ReadBoolean();
            reader.ReadList(ref PairWorldKey);
            reader.ReadList(ref PairLocked);
            reader.ReadList(ref PairDeflection);
            reader.ReadList(ref PairHeld);
            reader.ReadList(ref VesselSignature);
            reader.ReadList(ref VesselPartStart);
            reader.ReadList(ref PartLocked);
            reader.ReadList(ref PartHeld);
            reader.ReadList(ref PartRatio);
            reader.ReadList(ref TakenIds);
            reader.ReadList(ref TakenVersions);
            reader.ReadList(ref TakenStarts);
            reader.ReadList(ref TakenLocked);
            reader.ReadList(ref TakenHeld);
            reader.ReadList(ref TakenRatio);
            reader.ReadList(ref BoundaryMemory.States);
            if (reader.ReadBoolean())
            {
                DeflectionCount = reader.ReadInt32();
                reader.ReadList(ref Signature);
                reader.ReadList(ref BlockPair);
                reader.ReadList(ref PairBodies);
                reader.ReadList(ref PairTree);
                reader.ReadList(ref PairEdge);
                reader.ReadList(ref PairDeflectionSum);
                reader.ReadList(ref PairDeflectionStep);
                reader.ReadList(ref PairContact);
                reader.ReadList(ref PairBlockStart);
                reader.ReadList(ref PairBlocks);
                reader.ReadList(ref PairInverse);
                reader.ReadList(ref PairInverseKey);
                reader.ReadList(ref PairInverseRows);
                reader.ReadList(ref BodyPart);
                reader.ReadList(ref PartBody);
                reader.ReadList(ref PartPair);
                reader.ReadList(ref Parts);
                reader.ReadList(ref Edges);
            }
            else
            {
                Signature.Add(ulong.MaxValue);
            }

            // The next solve derives the section arrays from the restored pairs or vessel edges.
            RecordedLayoutPending = true;
        }
    }
}
