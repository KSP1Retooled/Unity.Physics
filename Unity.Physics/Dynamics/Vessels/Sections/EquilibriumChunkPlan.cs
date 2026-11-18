using System;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;

namespace Unity.Physics.Dynamics.Vessels.Sections
{
    /// <summary>
    /// Divides the factor into independent chunks and a top. Each chunk contributes to the top through its own slots.
    /// The top adds those slots in a fixed order so that lane assignments cannot change the result.
    /// </summary>
    internal struct EquilibriumChunkPlan : IDisposable
    {
        /// <summary>
        /// The start of each chunk's range in ChunkColumns. Columns are in ascending order within each chunk and in the top.
        /// </summary>
        public UnsafeList<int> ChunkStart;

        public UnsafeList<int> ChunkColumns;
        public UnsafeList<int> TopColumns;

        /// <summary>
        /// For each entry of the factor's structure, where the forward substitution's update goes: a row, or Count plus a
        /// buffer slot. Chunk c's slots are ChunkRhsStart[c] .. ChunkRhsStart[c + 1], and slot b adds to row RhsBufferTarget[b].
        /// </summary>
        public UnsafeList<int> RhsTarget;

        public UnsafeList<int> ChunkRhsStart;
        public UnsafeList<int> RhsBufferTarget;

        public int ChunkCount => ChunkStart.Length - 1;
        public int SlotCount => RhsBufferTarget.Length;

        public void Create()
        {
            ChunkStart = new UnsafeList<int>(0, Allocator.Persistent);
            ChunkColumns = new UnsafeList<int>(0, Allocator.Persistent);
            TopColumns = new UnsafeList<int>(0, Allocator.Persistent);
            RhsTarget = new UnsafeList<int>(0, Allocator.Persistent);
            ChunkRhsStart = new UnsafeList<int>(0, Allocator.Persistent);
            RhsBufferTarget = new UnsafeList<int>(0, Allocator.Persistent);
        }

        public void Dispose()
        {
            if (ChunkStart.IsCreated)
            {
                ChunkStart.Dispose();
                ChunkColumns.Dispose();
                TopColumns.Dispose();
                RhsTarget.Dispose();
                ChunkRhsStart.Dispose();
                RhsBufferTarget.Dispose();
            }
        }

        /// <summary>The target number of chunks, independent of the number of lanes.</summary>
        public const int ChunkTarget = 16;

        /// <summary>
        /// Cuts the solve into chunks of whole subtrees. A column weighs one plus the number of rows below its diagonal.
        /// A subtree within the target weight can join a chunk. The heavier columns go in the top.
        /// </summary>
        public void Build(int count, NativeArray<int> columnStart, NativeList<int> rows, NativeArray<int> parentOf)
        {
            var weight = new NativeArray<double>(math.max(1, count), Allocator.Temp);
            var subtree = new NativeArray<int>(math.max(1, count), Allocator.Temp);
            double total = 0.0;
            for (int s = 0; s < count; s++)
            {
                double own = 1 + columnStart[s + 1] - columnStart[s];
                weight[s] += own;
                subtree[s] += 1;
                total += own;
                if (parentOf[s] >= 0)
                {
                    weight[parentOf[s]] += weight[s];
                    subtree[parentOf[s]] += subtree[s];
                }
            }

            double target = total / ChunkTarget;
            var chunkOf = new NativeArray<int>(math.max(1, count), Allocator.Temp);
            ChunkStart.Clear();
            ChunkColumns.Clear();
            TopColumns.Clear();
            double accumulated = 0.0;
            bool open = false;
            for (int s = 0; s < count; s++)
            {
                chunkOf[s] = -1;
            }

            for (int s = 0; s < count; s++)
            {
                if (weight[s] > target)
                {
                    TopColumns.Add(s);
                    continue;
                }

                int parent = parentOf[s];
                if (parent >= 0 && weight[parent] <= target)
                {
                    continue;
                }

                // s roots a chunk subtree: columns s - subtree + 1 to s.
                if (!open)
                {
                    ChunkStart.Add(ChunkColumns.Length);
                    open = true;
                    accumulated = 0.0;
                }

                int chunk = ChunkStart.Length - 1;
                for (int c = s - subtree[s] + 1; c <= s; c++)
                {
                    ChunkColumns.Add(c);
                    chunkOf[c] = chunk;
                }

                accumulated += weight[s];
                if (accumulated >= target)
                {
                    open = false;
                }
            }

            ChunkStart.Add(ChunkColumns.Length);
            int chunks = ChunkStart.Length - 1;

            // Where each structure entry's forward update goes: a chunk's updates to the top's rows go to its slots.
            RhsTarget.Resize(math.max(1, rows.Length));
            ChunkRhsStart.Clear();
            RhsBufferTarget.Clear();
            var slotOf = new NativeArray<int>(math.max(1, count), Allocator.Temp);
            var slotChunk = new NativeArray<int>(math.max(1, count), Allocator.Temp);
            for (int s = 0; s < count; s++)
            {
                slotChunk[s] = -1;
            }

            for (int chunk = 0; chunk < chunks; chunk++)
            {
                ChunkRhsStart.Add(RhsBufferTarget.Length);
                for (int k = ChunkStart[chunk]; k < ChunkStart[chunk + 1]; k++)
                {
                    int s = ChunkColumns[k];
                    for (int e = columnStart[s]; e < columnStart[s + 1]; e++)
                    {
                        int row = rows[e];
                        if (chunkOf[row] == chunk)
                        {
                            RhsTarget[e] = row;
                            continue;
                        }

                        if (slotChunk[row] != chunk)
                        {
                            slotChunk[row] = chunk;
                            slotOf[row] = RhsBufferTarget.Length;
                            RhsBufferTarget.Add(row);
                        }

                        RhsTarget[e] = count + slotOf[row];
                    }
                }
            }

            ChunkRhsStart.Add(RhsBufferTarget.Length);
            for (int k = 0; k < TopColumns.Length; k++)
            {
                int s = TopColumns[k];
                for (int e = columnStart[s]; e < columnStart[s + 1]; e++)
                {
                    RhsTarget[e] = rows[e];
                }
            }
        }
    }
}
