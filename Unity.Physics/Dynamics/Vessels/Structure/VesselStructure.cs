using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;

namespace Unity.Physics.Dynamics.Vessels.Structure
{
    /// <summary>
    /// Rigid sections, elimination order, symbolic factor and parallel chunks for one vessel.
    /// Rebuild when edge states change. Number sections by elimination order and store 6 by 6 blocks column-wise.
    /// Chunks own subtrees and private top-update buffers; the top job combines them without concurrent writes.
    /// </summary>
    internal struct VesselStructure : System.IDisposable
    {
        public const int SectionCount = 0;
        public const int OffDiagonalCount = 1;
        public const int BufferBlockCount = 2;
        public const int BufferRhsCount = 3;
        public const int ChunkCount = 4;
        public const int TopCount = 5;
        public const int MaxStructure = 6;
        public const int LinkCount = 7;
        public const int HubCount = 8;
        public const int FillCount = 9;
        public const int HeaderLength = 10;

        public UnsafeList<int> Header;

        /// <summary>Each part's section.</summary>
        public UnsafeList<int> PartSection;

        /// <summary>Each section's parts, parents before children, from SectionPartStart[s] to SectionPartStart[s + 1].</summary>
        public UnsafeList<int> SectionPartStart;

        public UnsafeList<int> SectionParts;

        /// <summary>Each column's structure (later sections it couples to, ascending), from ColumnStart[s].</summary>
        public UnsafeList<int> ColumnStart;

        public UnsafeList<int> Rows;

        /// <summary>The elimination tree: each section's first structure entry, or -1.</summary>
        public UnsafeList<int> EliminationParent;

        /// <summary>
        /// For column s, from UpdateStart[s]: for each structure entry p in turn, the block its diagonal update goes to,
        /// then the block for each pair (p, q &gt; p).
        /// </summary>
        public UnsafeList<int> UpdateStart;

        public UnsafeList<int> UpdateTarget;

        /// <summary>For each structure entry, where the forward substitution's update to that row goes.</summary>
        public UnsafeList<int> RhsTarget;

        /// <summary>The links owned by each section, from LinkStart[s].</summary>
        public UnsafeList<int> LinkStart;

        public UnsafeList<VesselLink> Links;

        /// <summary>Every link touching each section, from IncidentStart[s].</summary>
        public UnsafeList<int> IncidentStart;

        public UnsafeList<int> Incident;

        /// <summary>The link of each part's live tree edge, or -1, and of each closure, or -1 when it is inert.</summary>
        public UnsafeList<int> PartLink;

        public UnsafeList<int> ClosureLink;

        /// <summary>Each chunk's columns, ascending, from ChunkStart[c]; then the top's columns, ascending.</summary>
        public UnsafeList<int> ChunkStart;

        public UnsafeList<int> ChunkColumns;
        public UnsafeList<int> TopColumns;

        /// <summary>Each chunk's buffer blocks and right-hand sides, and where the top job adds each one.</summary>
        public UnsafeList<int> ChunkBufferStart;

        public UnsafeList<int> BufferTarget;
        public UnsafeList<int> ChunkRhsStart;
        public UnsafeList<int> RhsBufferTarget;

        public VesselStructure(Allocator allocator)
        {
            Header = new UnsafeList<int>(HeaderLength, allocator);
            Header.Resize(HeaderLength, NativeArrayOptions.ClearMemory);
            PartSection = new UnsafeList<int>(0, allocator);
            SectionPartStart = new UnsafeList<int>(0, allocator);
            SectionParts = new UnsafeList<int>(0, allocator);
            ColumnStart = new UnsafeList<int>(0, allocator);
            Rows = new UnsafeList<int>(0, allocator);
            EliminationParent = new UnsafeList<int>(0, allocator);
            UpdateStart = new UnsafeList<int>(0, allocator);
            UpdateTarget = new UnsafeList<int>(0, allocator);
            RhsTarget = new UnsafeList<int>(0, allocator);
            LinkStart = new UnsafeList<int>(0, allocator);
            Links = new UnsafeList<VesselLink>(0, allocator);
            IncidentStart = new UnsafeList<int>(0, allocator);
            Incident = new UnsafeList<int>(0, allocator);
            PartLink = new UnsafeList<int>(0, allocator);
            ClosureLink = new UnsafeList<int>(0, allocator);
            ChunkStart = new UnsafeList<int>(0, allocator);
            ChunkColumns = new UnsafeList<int>(0, allocator);
            TopColumns = new UnsafeList<int>(0, allocator);
            ChunkBufferStart = new UnsafeList<int>(0, allocator);
            BufferTarget = new UnsafeList<int>(0, allocator);
            ChunkRhsStart = new UnsafeList<int>(0, allocator);
            RhsBufferTarget = new UnsafeList<int>(0, allocator);
        }

        public void Dispose()
        {
            Header.Dispose();
            PartSection.Dispose();
            SectionPartStart.Dispose();
            SectionParts.Dispose();
            ColumnStart.Dispose();
            Rows.Dispose();
            EliminationParent.Dispose();
            UpdateStart.Dispose();
            UpdateTarget.Dispose();
            RhsTarget.Dispose();
            LinkStart.Dispose();
            Links.Dispose();
            IncidentStart.Dispose();
            Incident.Dispose();
            PartLink.Dispose();
            ClosureLink.Dispose();
            ChunkStart.Dispose();
            ChunkColumns.Dispose();
            TopColumns.Dispose();
            ChunkBufferStart.Dispose();
            BufferTarget.Dispose();
            ChunkRhsStart.Dispose();
            RhsBufferTarget.Dispose();
        }
    }
}
