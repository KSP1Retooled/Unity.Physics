using System;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;

namespace Unity.Physics
{
    /// <summary>
    /// Solve a vessel tree with closures using implicit Euler over rigid sections:
    /// A = M + J^T*E^-1*J, factored in double precision.
    /// The caller owns and updates input arrays; the solver writes part motion, EdgeLoads and NextState.
    /// Rebuild structure when edge states change. Serial and job paths use the same kernels.
    /// </summary>
    public unsafe class VesselSolver : IDisposable
    {
        public VesselSolverSettings Settings;

        /// <summary>
        /// Stop after computing section velocities in Solution, leaving loads, state tests and integration to the
        /// caller.
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

        // The static systems of the sections that have closures inside them. A rebuild factors them again.
        VesselStatics* m_Statics;
        uint m_StaticsVersion;

        internal ref VesselStatics Statics => ref *m_Statics;

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
            StrutLoads.Initialize();
            m_Statics = (VesselStatics*)UnsafeUtility.Malloc(sizeof(VesselStatics), 16, Allocator.Persistent);
            *m_Statics = VesselStatics.Create(Allocator.Persistent);
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
            m_Statics->Dispose();
            UnsafeUtility.Free(m_Statics, Allocator.Persistent);
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
        /// Optional row-major 6 by 6 additions to section diagonals, indexed by the last rebuild.
        /// Pass false to disable.
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

            // A solver holds one vessel, whose parts and closures may be anything after a rebuild, so the systems of its
            // sections are never used again: each rebuild has a version of its own.
            m_StaticsVersion++;
            new PrepareStaticsJob
            {
                Context = m_Context,
                Statics = m_Statics,
                PartCount = m_Parts.Length,
                ClosureCount = m_Closures.Length,
                Version = m_StaticsVersion,
                TimeStep = Settings.TimeStep,
            }.Run();
            Context.Statics = m_Statics;
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
        /// Apply edge state tests. Newly yielding edges take their current pose as rest.
        /// Return whether the next step requires a rebuild.
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
                    edge.Rest = VectorMath.MultiplyQuaternions(VectorMath.ConjugateQuaternion(m_Parts[edge.Parent].Orientation), m_Parts[i].Orientation);
                }

                edge.State = NextState[i];
                edges[i] = edge;
            }

            m_Counters[VesselContext.StateChanged] = 0;
            m_NeedsRebuild = true;
            return true;
        }

        /// <summary>
        /// Run one step, rebuilding first if needed. With threads false, run jobs on the caller;
        /// otherwise schedule workers and wait while helping.
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
        /// Schedule the five step jobs. Rebuild after edge state changes before calling, since arrays may move.
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
        /// Solve another six-vector-per-section right-hand side in place against the last factor, on the calling
        /// thread.
        /// </summary>
        public void SolveWithFactor(NativeArray<double> rhs)
        {
            new SolveJob { Context = m_Context, Rhs = (double*)rhs.GetUnsafePtr() }.Run();
        }

        [BurstCompile(DisableSafetyChecks = true, CompileSynchronously = true)]
        struct PrepareStaticsJob : IJob
        {
            [NativeDisableUnsafePtrRestriction] public VesselContext* Context;
            [NativeDisableUnsafePtrRestriction] public VesselStatics* Statics;
            public int PartCount;
            public int ClosureCount;
            public uint Version;
            public double TimeStep;

            public void Execute()
            {
                var id = new NativeArray<ulong>(math.max(1, PartCount), Allocator.Temp);
                var version = new NativeArray<uint>(math.max(1, PartCount), Allocator.Temp);
                var index = new NativeArray<int>(math.max(1, PartCount), Allocator.Temp);
                for (int i = 0; i < PartCount; i++)
                {
                    version[i] = Version;
                    index[i] = i;
                }

                Statics->Prepare(*Context, PartCount, ClosureCount, (ulong*)id.GetUnsafePtr(), (uint*)version.GetUnsafePtr(),
                    (int*)index.GetUnsafePtr(), TimeStep);
            }
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
