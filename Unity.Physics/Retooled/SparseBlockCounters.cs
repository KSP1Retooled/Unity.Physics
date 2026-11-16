using System.Threading;
using Unity.Burst;
using Unity.Mathematics;

namespace Unity.Physics
{
    /// <summary>
    /// Counts the sizes of the dense blocks that <see cref="SparseConstraintSolver"/> factors, so that a bench can show
    /// which sizes its dense routines run on. Counting is off unless <see cref="Enabled"/> is set, because every count
    /// is an atomic addition on a line that all lanes share. Tests and benches read the counts between steps.
    /// </summary>
    public unsafe struct SparseBlockCounters
    {
        /// <summary>Sizes from 1 to <see cref="Largest"/> - 1 have their own entry; larger ones share the last entry.</summary>
        public const int Largest = 16;

        /// <summary>The nodes that a factor eliminated, by the number of unknowns in the node.</summary>
        public fixed long NodeSizes[Largest];

        /// <summary>The nodes that a factor eliminated, by how many later nodes they are joined to.</summary>
        public fixed long NeighbourCounts[Largest];

        /// <summary>
        /// The blocks below a node's diagonal block that a factor solved against that block, by their rows (the
        /// neighbour's size) and columns (the node's size): entry rows * Largest + columns.
        /// </summary>
        public fixed long BlockShapes[Largest * Largest];

        /// <summary>The factors that the counts above come from.</summary>
        public long Factorizations;

        static readonly SharedStatic<SparseBlockCounters> Shared = SharedStatic<SparseBlockCounters>.GetOrCreate<SparseBlockCounters>();
        static readonly SharedStatic<int> SharedEnabled = SharedStatic<int>.GetOrCreate<SparseBlockCounters, int>();

        /// <summary>Whether the solver counts. Set it on the main thread between steps.</summary>
        public static bool Enabled
        {
            get => SharedEnabled.Data != 0;
            set => SharedEnabled.Data = value ? 1 : 0;
        }

        /// <summary>The counts so far. Read them on the main thread between steps.</summary>
        public static SparseBlockCounters Current => Shared.Data;

        public static void Reset()
        {
            Shared.Data = default;
        }

        internal static void AddFactorization()
        {
            Interlocked.Increment(ref Shared.Data.Factorizations);
        }

        internal static void AddNode(int size, int neighbours)
        {
            ref SparseBlockCounters data = ref Shared.Data;
            Interlocked.Increment(ref data.NodeSizes[Bin(size)]);
            Interlocked.Increment(ref data.NeighbourCounts[Bin(neighbours)]);
        }

        internal static void AddBlock(int rows, int columns)
        {
            Interlocked.Increment(ref Shared.Data.BlockShapes[Bin(rows) * Largest + Bin(columns)]);
        }

        static int Bin(int size) => math.clamp(size, 0, Largest - 1);
    }
}
