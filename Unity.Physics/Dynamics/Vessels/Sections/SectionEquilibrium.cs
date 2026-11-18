using System;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;

namespace Unity.Physics.Dynamics.Vessels.Sections
{
    /// <summary>
    /// The equilibrium system of one rigid section that has closures inside it, factored once.
    /// <para>
    /// The unknowns are the small motions of the parts whose edge lies on the loop of some closure in the section. Every
    /// other edge is a bridge: cutting it splits the section in two, so it carries its child's whole subtree whatever the
    /// stiffnesses are, and the tree method gives its load exactly. A part whose own edge is a bridge, and the section's
    /// first part, is held fixed. The links are the edges on the loops and the closures. Each link has the stiffness that
    /// the live solve would give it if every part were its own section.
    /// </para>
    /// </summary>
    internal struct SectionEquilibrium : IDisposable
    {
        /// <summary>What the system was built for: the vessel, its structure version, the time step and the parts.</summary>
        public ulong Id;

        public uint Version;
        public double TimeStep;
        public UnsafeList<int> KeyParts;

        /// <summary>The number of unknowns, six numbers each.</summary>
        public int Count;

        /// <summary>For the unknown at each position of the elimination order, its part's place in the section's list of parts.</summary>
        public UnsafeList<int> Node;

        /// <summary>The factor's structure: each column's later rows, ascending, from ColumnStart[j].</summary>
        public UnsafeList<int> ColumnStart;

        public UnsafeList<int> Rows;

        /// <summary>The inverse of each pivot block, and inv(D) U for each block off the diagonal, 36 doubles each.</summary>
        public UnsafeList<double> Inverse;

        public UnsafeList<double> Blocks;

        public UnsafeList<EquilibriumLink> Links;

        /// <summary>How many pivots the factor had to clamp. It should be zero.</summary>
        public int ClampedPivots;

        public EquilibriumChunkPlan Chunks;

        /// <summary>For each unknown, by position, the unknowns of its children whose edges are on loops, from ChildStart[j].</summary>
        public UnsafeList<int> ChildStart;

        public UnsafeList<int> Children;

        /// <summary>The solve's numbers, 6 for each unknown and then 6 for each buffer slot.</summary>
        public UnsafeList<double> Work;

        /// <summary>Makes the lists that the factor and the solve fill.</summary>
        public void CreateSolveLists()
        {
            Chunks.Create();
            ChildStart = new UnsafeList<int>(0, Allocator.Persistent);
            Children = new UnsafeList<int>(0, Allocator.Persistent);
            Work = new UnsafeList<double>(0, Allocator.Persistent);
        }


        /// <summary>
        /// For the forest path, which builds its systems from rows: each unknown's position in the elimination order, the
        /// body of each unknown, each body's unknown or -1, the section's first body, and the links. A link is a pair,
        /// the unknowns of its child and parent sides or -1, and 1 for a closure or 0 for a tree pair.
        /// </summary>
        public UnsafeList<int> Position;

        public UnsafeList<int> Bodies;
        public UnsafeList<int> BodyMap;
        public int FirstBody;
        public UnsafeList<int4> PairLinks;

        /// <summary>The sum of each unknown's load over a step's substeps so far, and how many substeps it holds.</summary>
        public UnsafeList<double> Sum;

        public int Summed;

        /// <summary>The unknown that a body moves with, or -1 when the body is held fixed or isn't in the system.</summary>
        public int BodyUnknown(int body) => body < BodyMap.Length ? BodyMap[body] : -1;

        /// <summary>Whether a section of the current structure uses this system, while the systems are matched.</summary>
        public bool Claimed;

        /// <summary>The last match of the systems to a structure in which a section used this system.</summary>
        public long LastUsed;

        public void Dispose()
        {
            Free(ref KeyParts);
            Free(ref Node);
            Free(ref ColumnStart);
            Free(ref Rows);
            Free(ref Inverse);
            Free(ref Blocks);
            Free(ref Links);
            Chunks.Dispose();
            Free(ref ChildStart);
            Free(ref Children);
            Free(ref Work);
            Free(ref Sum);
            Free(ref Position);
            Free(ref Bodies);
            Free(ref BodyMap);
            Free(ref PairLinks);
        }

        static void Free<T>(ref UnsafeList<T> list) where T : unmanaged
        {
            if (list.IsCreated)
            {
                list.Dispose();
            }
        }
    }
}
