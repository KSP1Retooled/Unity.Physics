using Unity.Mathematics;
using Unity.Numerics.Kernels;

namespace Unity.Physics.Dynamics.Vessels.Sections
{
    internal static unsafe class SectionEquilibriumSolve
    {
        /// <summary>Solves the whole system on the calling thread, from the right-hand side already in Work.</summary>
        public static void Solve(SectionEquilibrium* entry)
        {
            for (int c = 0; c < entry->Chunks.ChunkCount; c++)
            {
                ForwardChunk(entry, c, false);
            }

            Top(entry, false);
            for (int c = 0; c < entry->Chunks.ChunkCount; c++)
            {
                BackChunk(entry, c);
            }
        }

        /// <summary>
        /// Chunk c's forward substitution. With <paramref name="fromSum"/>, it first takes its columns' right-hand side as the
        /// mean of the section's summed loads.
        /// </summary>
        public static void ForwardChunk(SectionEquilibrium* entry, int chunk, bool fromSum)
        {
            double* y = entry->Work.Ptr;
            int count = entry->Count;
            for (int b = entry->Chunks.ChunkRhsStart[chunk]; b < entry->Chunks.ChunkRhsStart[chunk + 1]; b++)
            {
                default(double6).Store(y + 6 * (count + b));
            }

            int first = entry->Chunks.ChunkStart[chunk], last = entry->Chunks.ChunkStart[chunk + 1];
            if (fromSum)
            {
                double mean = 1.0 / entry->Summed;
                for (int k = first; k < last; k++)
                {
                    int s = entry->Chunks.ChunkColumns[k];
                    (mean * double6.Load(entry->Sum.Ptr + 6 * s)).Store(y + 6 * s);
                }
            }

            for (int k = first; k < last; k++)
            {
                ForwardColumn(entry, entry->Chunks.ChunkColumns[k], y);
            }
        }

        /// <summary>The top's share: the chunks' slots in order, then its forward and backward substitution.</summary>
        public static void Top(SectionEquilibrium* entry, bool fromSum)
        {
            double* y = entry->Work.Ptr;
            int count = entry->Count;
            if (fromSum)
            {
                double mean = 1.0 / entry->Summed;
                for (int k = 0; k < entry->Chunks.TopColumns.Length; k++)
                {
                    int s = entry->Chunks.TopColumns[k];
                    (mean * double6.Load(entry->Sum.Ptr + 6 * s)).Store(y + 6 * s);
                }
            }

            for (int b = 0; b < entry->Chunks.SlotCount; b++)
            {
                double* to = y + 6 * entry->Chunks.RhsBufferTarget[b];
                (double6.Load(to) + double6.Load(y + 6 * (count + b))).Store(to);
            }

            for (int k = 0; k < entry->Chunks.TopColumns.Length; k++)
            {
                ForwardColumn(entry, entry->Chunks.TopColumns[k], y);
            }

            for (int k = entry->Chunks.TopColumns.Length - 1; k >= 0; k--)
            {
                BackColumn(entry, entry->Chunks.TopColumns[k], y);
            }
        }

        /// <summary>Chunk c's backward substitution, after the top's.</summary>
        public static void BackChunk(SectionEquilibrium* entry, int chunk)
        {
            double* y = entry->Work.Ptr;
            for (int k = entry->Chunks.ChunkStart[chunk + 1] - 1; k >= entry->Chunks.ChunkStart[chunk]; k--)
            {
                BackColumn(entry, entry->Chunks.ChunkColumns[k], y);
            }
        }

        static void ForwardColumn(SectionEquilibrium* entry, int s, double* y)
        {
            double6 ys = double6.Load(y + 6 * s);
            double* blocks = entry->Blocks.Ptr;
            int* target = entry->Chunks.RhsTarget.Ptr;
            for (int e = entry->ColumnStart[s]; e < entry->ColumnStart[s + 1]; e++)
            {
                double* to = y + 6 * target[e];
                (double6.Load(to) - ((double6x6*)(blocks + Block6x6.Size * e))->TransposeMul(ys)).Store(to);
            }
        }

        static void BackColumn(SectionEquilibrium* entry, int s, double* y)
        {
            double6 x = ((double6x6*)(entry->Inverse.Ptr + Block6x6.Size * s))->Mul(double6.Load(y + 6 * s));
            double* blocks = entry->Blocks.Ptr;
            int* rows = entry->Rows.Ptr;
            for (int e = entry->ColumnStart[s]; e < entry->ColumnStart[s + 1]; e++)
            {
                x -= ((double6x6*)(blocks + Block6x6.Size * e))->Mul(double6.Load(y + 6 * rows[e]));
            }

            x.Store(y + 6 * s);
        }
    }
}
