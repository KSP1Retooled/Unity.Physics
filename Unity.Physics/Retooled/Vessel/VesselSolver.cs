using System;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;

namespace Unity.Physics
{
    /// <summary>
    /// Solves one vessel per step: a tree of parts whose edges are locked, yielding, free about one axis or broken, plus
    /// closures between parts. Parts joined by locked edges move as one rigid section; the solve is an implicit Euler step
    /// over sections in body-only form, A = M + J^T E^-1 J, factored by block elimination leaves-inward in double
    /// precision. See SECTION-SOLVER-SPEC in the vessel solver study for the design.
    ///
    /// The caller owns the part, edge and closure arrays and fills them each step; the solver reads them, writes the
    /// parts' new poses and velocities back, and leaves every edge's load and next state in <see cref="EdgeLoads"/> and
    /// <see cref="NextState"/>. The structure (sections, order, symbolic factor) is rebuilt only when edge states change.
    ///
    /// A step runs as five jobs: section properties (parallel over sections), the factor by chunks (parallel), the top of
    /// the elimination tree (one job), the backward substitution by chunks (parallel), and the loads, state tests and
    /// integration (parallel over sections).
    /// </summary>
    public unsafe class VesselSolver : IDisposable
    {
        public VesselSolverSettings Settings;

        /// <summary>
        /// Stops a step once it has the sections' new velocities in <see cref="Solution"/>: no loads, no state tests, and the
        /// parts are neither moved nor given the new velocities, which leaves integrating them to the caller.
        /// </summary>
        public bool VelocityOnly;

        readonly NativeArray<VesselPart> m_Parts;
        readonly NativeArray<VesselEdge> m_Edges;
        readonly NativeArray<VesselClosure> m_Closures;

        VesselStructure* m_Structure;
        internal ref VesselStructure Structure => ref *m_Structure;
        internal NativeList<VesselSection> Sections;
        internal NativeList<double> Blocks;
        internal NativeList<double> Inverse;
        internal NativeList<double> Rhs;
        internal NativeList<double> Solution;
        internal NativeList<double> ExtraDiagonal;
        internal NativeList<VesselRows> LinkRows;
        internal NativeList<double> SectionResidual;
        NativeArray<double3> m_PartForce;
        NativeArray<double3> m_PartTorque;
        NativeArray<double3> m_ChildPoint;
        NativeArray<int> m_Counters;

        /// <summary>Each tree edge's load over the last step, indexed by its child part.</summary>
        public NativeArray<VesselEdgeLoad> EdgeLoads;

        /// <summary>Each closure's load over the last step, zero for one inert inside a section.</summary>
        public NativeArray<VesselEdgeLoad> ClosureLoads;

        /// <summary>Each tree edge's state after the last step's tests.</summary>
        public NativeArray<VesselEdgeState> NextState;

        bool m_NeedsRebuild = true;
        bool m_UseExtraDiagonal;
        // The jobs share one context through a pointer: a job may not hold the raw pointers the context is made of.
        VesselContext* m_Context;

        internal ref VesselContext Context => ref *m_Context;

        public VesselSolver(NativeArray<VesselPart> parts, NativeArray<VesselEdge> edges, NativeArray<VesselClosure> closures,
            VesselSolverSettings settings)
        {
            m_Parts = parts;
            m_Edges = edges;
            m_Closures = closures;
            Settings = settings;
            m_Context = (VesselContext*)UnsafeUtility.Malloc(sizeof(VesselContext), 16, Allocator.Persistent);
            *m_Context = default;
            m_Structure = (VesselStructure*)UnsafeUtility.Malloc(sizeof(VesselStructure), 16, Allocator.Persistent);
            *m_Structure = new VesselStructure(Allocator.Persistent);
            Sections = new NativeList<VesselSection>(Allocator.Persistent);
            Blocks = new NativeList<double>(Allocator.Persistent);
            Inverse = new NativeList<double>(Allocator.Persistent);
            Rhs = new NativeList<double>(Allocator.Persistent);
            Solution = new NativeList<double>(Allocator.Persistent);
            ExtraDiagonal = new NativeList<double>(Allocator.Persistent);
            LinkRows = new NativeList<VesselRows>(Allocator.Persistent);
            SectionResidual = new NativeList<double>(Allocator.Persistent);
            m_PartForce = new NativeArray<double3>(parts.Length, Allocator.Persistent);
            m_PartTorque = new NativeArray<double3>(parts.Length, Allocator.Persistent);
            m_ChildPoint = new NativeArray<double3>(parts.Length, Allocator.Persistent);
            m_Counters = new NativeArray<int>(VesselContext.CounterCount, Allocator.Persistent);
            EdgeLoads = new NativeArray<VesselEdgeLoad>(parts.Length, Allocator.Persistent);
            ClosureLoads = new NativeArray<VesselEdgeLoad>(math.max(1, closures.Length), Allocator.Persistent);
            NextState = new NativeArray<VesselEdgeState>(parts.Length, Allocator.Persistent);
        }

        public void Dispose()
        {
            UnsafeUtility.Free(m_Context, Allocator.Persistent);
            Structure.Dispose();
            UnsafeUtility.Free(m_Structure, Allocator.Persistent);
            Sections.Dispose();
            Blocks.Dispose();
            Inverse.Dispose();
            Rhs.Dispose();
            Solution.Dispose();
            ExtraDiagonal.Dispose();
            LinkRows.Dispose();
            SectionResidual.Dispose();
            m_PartForce.Dispose();
            m_PartTorque.Dispose();
            m_ChildPoint.Dispose();
            m_Counters.Dispose();
            EdgeLoads.Dispose();
            ClosureLoads.Dispose();
            NextState.Dispose();
        }

        public int SectionCount => Structure.Header[VesselStructure.SectionCount];
        public int LinkCount => Structure.Header[VesselStructure.LinkCount];
        public int ChunkCount => Structure.Header[VesselStructure.ChunkCount];
        public int TopCount => Structure.Header[VesselStructure.TopCount];
        public int HubCount => Structure.Header[VesselStructure.HubCount];
        public int FillCount => Structure.Header[VesselStructure.FillCount];
        public int MaxStructure => Structure.Header[VesselStructure.MaxStructure];
        public int OffDiagonalCount => Structure.Header[VesselStructure.OffDiagonalCount];

        /// <summary>Whether any edge's state test asked for a change in the last step.</summary>
        public bool StateChanged => m_Counters[VesselContext.StateChanged] != 0;

        /// <summary>How many diagonal blocks have needed a clamped pivot since the solver was made.</summary>
        public int ClampedPivots => m_Counters[VesselContext.ClampedPivots];

        /// <summary>The section each part belongs to, valid after a rebuild.</summary>
        public NativeArray<int> PartSection
        {
            get
            {
                var array = NativeArrayUnsafeUtility.ConvertExistingDataToNativeArray<int>(Structure.PartSection.Ptr, Structure.PartSection.Length, Allocator.None);
#if ENABLE_UNITY_COLLECTIONS_CHECKS
                NativeArrayUnsafeUtility.SetAtomicSafetyHandle(ref array, AtomicSafetyHandle.GetTempMemoryHandle());
#endif
                return array;
            }
        }

        /// <summary>Tells the solver that edge states (or closures) changed, so the next step rebuilds the structure.</summary>
        public void MarkStructureChanged() => m_NeedsRebuild = true;

        /// <summary>
        /// Extra 6x6 terms, one per section (row by row, linear then angular), added to the diagonal before the factor,
        /// as a boundary block's coupling would be. Sections are numbered by the last rebuild; pass false to stop.
        /// </summary>
        public NativeArray<double> UseExtraDiagonal(bool use)
        {
            m_UseExtraDiagonal = use;
            ExtraDiagonal.Resize(VesselBlocks.Size * math.max(1, SectionCount), NativeArrayOptions.ClearMemory);
            Context.ExtraDiagonal = use ? ExtraDiagonal.GetUnsafePtr() : null;
            return ExtraDiagonal.AsArray();
        }

        /// <summary>Rebuilds the structure from the edges' current states, on the calling thread, in Burst.</summary>
        public void Rebuild()
        {
            new BuildVesselStructureJob
            {
                Parts = m_Parts,
                Edges = m_Edges,
                Closures = m_Closures,
                HubClosures = Settings.HubClosures,
                ChunkTarget = Settings.ChunkTarget,
                MinimumDegree = Settings.MinimumDegree,
                Structure = m_Structure,
            }.Run();

            int sections = SectionCount;
            int blocks = sections + OffDiagonalCount + Structure.Header[VesselStructure.BufferBlockCount];
            int rhs = sections + Structure.Header[VesselStructure.BufferRhsCount];
            Sections.ResizeUninitialized(sections);
            Blocks.ResizeUninitialized(VesselBlocks.Size * blocks);
            Inverse.ResizeUninitialized(VesselBlocks.Size * sections);
            Rhs.ResizeUninitialized(6 * rhs);
            Solution.ResizeUninitialized(6 * sections);
            LinkRows.ResizeUninitialized(math.max(1, LinkCount));
            SectionResidual.ResizeUninitialized(sections);
            if (m_UseExtraDiagonal)
            {
                ExtraDiagonal.Resize(VesselBlocks.Size * sections, NativeArrayOptions.ClearMemory);
            }

            Context = MakeContext();
            m_NeedsRebuild = false;
        }

        VesselContext MakeContext()
        {
            return new VesselContext
            {
                Parts = (VesselPart*)m_Parts.GetUnsafePtr(),
                Edges = (VesselEdge*)m_Edges.GetUnsafePtr(),
                Closures = (VesselClosure*)m_Closures.GetUnsafeReadOnlyPtr(),
                Sections = Sections.GetUnsafePtr(),
                Blocks = Blocks.GetUnsafePtr(),
                Inverse = Inverse.GetUnsafePtr(),
                Rhs = Rhs.GetUnsafePtr(),
                Solution = Solution.GetUnsafePtr(),
                ExtraDiagonal = m_UseExtraDiagonal ? ExtraDiagonal.GetUnsafePtr() : null,
                LinkRows = LinkRows.GetUnsafePtr(),
                PartForce = (double3*)m_PartForce.GetUnsafePtr(),
                PartTorque = (double3*)m_PartTorque.GetUnsafePtr(),
                ChildPoint = (double3*)m_ChildPoint.GetUnsafePtr(),
                SectionResidual = SectionResidual.GetUnsafePtr(),
                EdgeLoads = (VesselEdgeLoad*)EdgeLoads.GetUnsafePtr(),
                ClosureLoads = (VesselEdgeLoad*)ClosureLoads.GetUnsafePtr(),
                NextState = (VesselEdgeState*)NextState.GetUnsafePtr(),
                Counters = (int*)m_Counters.GetUnsafePtr(),
                PartSection = Structure.PartSection.Ptr,
                SectionPartStart = Structure.SectionPartStart.Ptr,
                SectionParts = Structure.SectionParts.Ptr,
                ColumnStart = Structure.ColumnStart.Ptr,
                Rows = Structure.Rows.Ptr,
                UpdateStart = Structure.UpdateStart.Ptr,
                UpdateTarget = Structure.UpdateTarget.Ptr,
                RhsTarget = Structure.RhsTarget.Ptr,
                LinkStart = Structure.LinkStart.Ptr,
                Links = Structure.Links.Ptr,
                IncidentStart = Structure.IncidentStart.Ptr,
                Incident = Structure.Incident.Ptr,
                PartLink = Structure.PartLink.Ptr,
                ChunkStart = Structure.ChunkStart.Ptr,
                ChunkColumns = Structure.ChunkColumns.Ptr,
                TopColumns = Structure.TopColumns.Ptr,
                ChunkBufferStart = Structure.ChunkBufferStart.Ptr,
                BufferTarget = Structure.BufferTarget.Ptr,
                ChunkRhsStart = Structure.ChunkRhsStart.Ptr,
                RhsBufferTarget = Structure.RhsBufferTarget.Ptr,
                EliminationParent = Structure.EliminationParent.Ptr,
                SectionCount = SectionCount,
                OffDiagonalCount = OffDiagonalCount,
                ChunkCount = ChunkCount,
                TopCount = TopCount,
                MaxStructure = MaxStructure,
                TimeStep = Settings.TimeStep,
                HardFrequency = 2.0 * math.PI_DBL / Settings.HardPeriod,
                HardDampingRatio = Settings.HardDampingRatio,
                Centripetal = Settings.CentripetalLoads,
            };
        }

        /// <summary>
        /// Takes the state tests' results as the edges' states. An edge that starts to yield takes its current pose as
        /// rest, so its spring starts undeflected. Returns whether anything changed; the next step then rebuilds.
        /// </summary>
        public bool ApplyNextStates()
        {
            if (!StateChanged)
            {
                return false;
            }

            var edges = m_Edges;
            for (int i = 0; i < edges.Length; i++)
            {
                VesselEdge edge = edges[i];
                if (edge.State == NextState[i])
                {
                    continue;
                }

                if (edge.State == VesselEdgeState.Locked && NextState[i] == VesselEdgeState.Yielding)
                {
                    edge.Rest = VesselContext.Multiply(VesselContext.Conjugate(m_Parts[edge.Parent].Orientation), m_Parts[i].Orientation);
                }

                edge.State = NextState[i];
                edges[i] = edge;
            }

            m_Counters[VesselContext.StateChanged] = 0;
            m_NeedsRebuild = true;
            return true;
        }

        /// <summary>
        /// One step. With <paramref name="threads"/> false every job runs on the calling thread; otherwise the jobs are
        /// scheduled on the worker threads and the caller waits, helping, for the last. A pending rebuild runs first.
        /// </summary>
        public void Step(bool threads)
        {
            if (m_NeedsRebuild)
            {
                Rebuild();
            }

            Context.TimeStep = Settings.TimeStep;
            m_Counters[VesselContext.StateChanged] = 0;
            if (threads)
            {
                Schedule(default).Complete();
            }
            else
            {
                RunStages(null);
            }
        }

        /// <summary>
        /// The step's five jobs, chained. The structure must be current (call <see cref="Rebuild"/> after edge states
        /// change), since a rebuild can move the arrays the jobs point into.
        /// </summary>
        public JobHandle Schedule(JobHandle dependency)
        {
            VesselContext* context = m_Context;
            JobHandle handle = new SectionJob { Context = context }.Schedule(context->SectionCount, 32, dependency);
            handle = new FactorChunkJob { Context = context }.Schedule(context->ChunkCount, 1, handle);
            handle = new TopJob { Context = context }.Schedule(handle);
            handle = new SubstituteChunkJob { Context = context }.Schedule(context->ChunkCount, 1, handle);
            if (VelocityOnly)
            {
                return handle;
            }

            return new FinishJob { Context = context }.Schedule(context->SectionCount, 16, handle);
        }

        /// <summary>
        /// The five stages on the calling thread, each timed into <paramref name="ticks"/> (five Stopwatch tick counts)
        /// when it is given.
        /// </summary>
        public void RunStages(long[] ticks)
        {
            VesselContext* context = m_Context;
            long start = System.Diagnostics.Stopwatch.GetTimestamp();
            new SectionJob { Context = context }.Run(context->SectionCount);
            long a = System.Diagnostics.Stopwatch.GetTimestamp();
            new FactorChunkJob { Context = context }.Run(context->ChunkCount);
            long b = System.Diagnostics.Stopwatch.GetTimestamp();
            new TopJob { Context = context }.Run();
            long c = System.Diagnostics.Stopwatch.GetTimestamp();
            new SubstituteChunkJob { Context = context }.Run(context->ChunkCount);
            long d = System.Diagnostics.Stopwatch.GetTimestamp();
            if (!VelocityOnly)
            {
                new FinishJob { Context = context }.Run(context->SectionCount);
            }

            long e = System.Diagnostics.Stopwatch.GetTimestamp();
            if (ticks != null)
            {
                ticks[0] = a - start;
                ticks[1] = b - a;
                ticks[2] = c - b;
                ticks[3] = d - c;
                ticks[4] = e - d;
            }
        }

        /// <summary>
        /// Solves against the last step's factor for another right-hand side (one 6-vector per section, linear then
        /// angular), in place, on the calling thread in Burst. A sparse right-hand side still walks every column.
        /// </summary>
        public void SolveWithFactor(NativeArray<double> rhs)
        {
            new SolveJob { Context = m_Context, Rhs = (double*)rhs.GetUnsafePtr() }.Run();
        }

        [BurstCompile(DisableSafetyChecks = true, CompileSynchronously = true)]
        struct SectionJob : IJobParallelFor
        {
            [NativeDisableUnsafePtrRestriction] public VesselContext* Context;

            public void Execute(int s) => Context->ComputeSection(s);
        }

        [BurstCompile(DisableSafetyChecks = true, CompileSynchronously = true)]
        struct FactorChunkJob : IJobParallelFor
        {
            [NativeDisableUnsafePtrRestriction] public VesselContext* Context;

            public void Execute(int chunk)
            {
                var scratch = new NativeArray<double>(VesselBlocks.Size * math.max(1, Context->MaxStructure), Allocator.Temp,
                    NativeArrayOptions.UninitializedMemory);
                Context->FactorChunk(chunk, (double*)scratch.GetUnsafePtr());
            }
        }

        [BurstCompile(DisableSafetyChecks = true, CompileSynchronously = true)]
        struct TopJob : IJob
        {
            [NativeDisableUnsafePtrRestriction] public VesselContext* Context;

            public void Execute()
            {
                var scratch = new NativeArray<double>(VesselBlocks.Size * math.max(1, Context->MaxStructure), Allocator.Temp,
                    NativeArrayOptions.UninitializedMemory);
                Context->SolveTop((double*)scratch.GetUnsafePtr());
            }
        }

        [BurstCompile(DisableSafetyChecks = true, CompileSynchronously = true)]
        struct SubstituteChunkJob : IJobParallelFor
        {
            [NativeDisableUnsafePtrRestriction] public VesselContext* Context;

            public void Execute(int chunk) => Context->SubstituteChunk(chunk);
        }

        [BurstCompile(DisableSafetyChecks = true, CompileSynchronously = true)]
        struct FinishJob : IJobParallelFor
        {
            [NativeDisableUnsafePtrRestriction] public VesselContext* Context;

            public void Execute(int s) => Context->FinishSection(s);
        }

        [BurstCompile(DisableSafetyChecks = true, CompileSynchronously = true)]
        struct SolveJob : IJob
        {
            [NativeDisableUnsafePtrRestriction] public VesselContext* Context;

            [NativeDisableUnsafePtrRestriction]
            public double* Rhs;

            public void Execute() => Context->Solve(Rhs);
        }
    }
}
