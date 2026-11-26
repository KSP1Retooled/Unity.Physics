using System.Runtime.CompilerServices;
using Unity.Physics.Dynamics.Solver.Islands.Data;
using System;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using Unity.Numerics.Linear.Solvers.Lowlevel;
using Unity.Physics.Base.Threading;
using Unity.Physics.Dynamics.Solver.Sparse;

namespace Unity.Physics.Dynamics.Solver.Vessels
{
    internal static partial class VesselIslandSolver
    {
        /// <summary>
        /// Gather sparse blocks and couplings, numbering bodies in Jacobian order.
        /// headerRows gives each Jacobian's row range.
        /// </summary>
        internal static unsafe void GatherSparseBlocks(ref SparseSetup setup, IslandHeader* islandHeaders, int* headerRows, int headerCount, int rows,
            in NativeArray<MotionVelocity> motionVelocities, WorkTeam* team)
        {
            // Number bodies serially in Jacobian order; split Jacobian reads and list writes into disjoint task
            // ranges.
            int taskCount = (headerCount + JacobiansPerSummaryTask - 1) / JacobiansPerSummaryTask;
            var summaries = new NativeArray<JacobianSummary>(math.max(1, headerCount), Allocator.Temp, NativeArrayOptions.UninitializedMemory);
            var taskBlocks = new NativeArray<int>(math.max(1, taskCount), Allocator.Temp, NativeArrayOptions.UninitializedMemory);
            var taskCouplings = new NativeArray<int>(math.max(1, taskCount), Allocator.Temp, NativeArrayOptions.UninitializedMemory);
            var summarizing = new JacobianSummaries
            {
                Headers = islandHeaders,
                HeaderRows = headerRows,
                Count = headerCount,
                Rows = rows,
                MotionVelocities = (MotionVelocity*)motionVelocities.GetUnsafeReadOnlyPtr(),
                MotionCount = motionVelocities.Length,
                Summaries = (JacobianSummary*)summaries.GetUnsafePtr(),
                TaskBlocks = (int*)taskBlocks.GetUnsafePtr(),
                TaskCouplings = (int*)taskCouplings.GetUnsafePtr(),
                CompactBodies = (int*)setup.CompactBodies.GetUnsafePtr(),
            };

            WorkTeam.RunAll<DirectTaskDispatch>(team, &summarizing, IslandTaskKind.SummarizeJacobiansKind, taskCount, null);

            for (int h = 0; h < headerCount; h++)
            {
                if (summaries[h].Rows > 0)
                {
                    setup.HoldsPositions |= summaries[h].HoldsPosition;
                    setup.Number(summaries[h].BodyA);
                    setup.Number(summaries[h].BodyB);
                }
            }

            // From each task's counts to where its blocks and couplings start.
            int blockCount = 0;
            int couplingCount = 0;
            for (int task = 0; task < taskCount; task++)
            {
                int ownBlocks = taskBlocks[task];
                int ownCouplings = taskCouplings[task];
                taskBlocks[task] = blockCount;
                taskCouplings[task] = couplingCount;
                blockCount += ownBlocks;
                couplingCount += ownCouplings;
            }

            setup.Blocks.ResizeUninitialized(blockCount);
            setup.Headers.ResizeUninitialized(blockCount);
            setup.Couplings.ResizeUninitialized(couplingCount);
            summarizing.Blocks = setup.Blocks.GetUnsafePtr();
            summarizing.HeaderPointers = setup.Headers.GetUnsafePtr();
            summarizing.Couplings = setup.Couplings.GetUnsafePtr();
            WorkTeam.RunAll<DirectTaskDispatch>(team, &summarizing, IslandTaskKind.FillBlocksKind, taskCount, null);

            summaries.Dispose();
            taskBlocks.Dispose();
            taskCouplings.Dispose();
        }

        /// <summary>The blocks of a sparse solve, gathered from an island's Jacobians in their order.</summary>
        internal unsafe struct SparseSetup
        {
            public NativeList<SparseConstraintSolver.ConstraintBlock> Blocks;
            public NativeList<BoundCoupling> Couplings;

            /// <summary>Each block's Jacobian, which stays where it is in the stream for the whole step.</summary>
            public NativeList<IntPtr> Headers;

            /// <summary>Each of the island's moving bodies by its index in the world, in the order the blocks reach them.</summary>
            public NativeList<int> WorldBodies;

            /// <summary>Where each moving body sits among the island's, by its index in the world, or -1.</summary>
            public NativeArray<int> CompactBodies;

            public bool HoldsPositions;

            /// <summary>
            /// Gives back the index of the island's bodies: sets the island's entries in the shared index to -1 again, or
            /// frees an index of the island's own. Call it before <see cref="WorldBodies"/> is freed.
            /// </summary>
            public void ReleaseCompactBodies(bool shared)
            {
                if (!shared)
                {
                    CompactBodies.Dispose();
                    return;
                }

                for (int i = 0; i < WorldBodies.Length; i++)
                {
                    CompactBodies[WorldBodies[i]] = -1;
                }
            }

            /// <summary>Where a moving body sits among the island's, numbering it if it is new, or -1 for none.</summary>
            public int Number(int bodyIndex)
            {
                if (bodyIndex < 0)
                {
                    return -1;
                }

                int compact = CompactBodies[bodyIndex];
                if (compact < 0)
                {
                    compact = WorldBodies.Length;
                    CompactBodies[bodyIndex] = compact;
                    WorldBodies.Add(bodyIndex);
                }

                return compact;
            }
        }

        /// <summary>What a sparse solve takes from one Jacobian, which needs nothing from the others.</summary>
        internal struct JacobianSummary
        {
            /// <summary>The Jacobian's bodies by their index in the world, or -1 for one that doesn't move.</summary>
            public int BodyA;
            public int BodyB;

            public int Rows;

            /// <summary>The contact's coefficient of friction, or 0 for a joint or a contact without friction rows.</summary>
            public float Friction;

            /// <summary>
            /// The normal impulse the contact holds already: earlier substeps' in its total, and this one's warm start, if it
            /// has one, in its points.
            /// </summary>
            public float HeldImpulse;

            public bool HoldsPosition;
        }

        internal static unsafe JacobianSummary Summarize(ref JacobianHeader header, int rows, MotionVelocity* motionVelocities,
            int motionCount)
        {
            var summary = new JacobianSummary { Rows = rows };
            if (rows == 0)
            {
                return summary;
            }

            summary.BodyA = Moving(header.BodyPair.BodyIndexA, motionVelocities, motionCount);
            summary.BodyB = Moving(header.BodyPair.BodyIndexB, motionVelocities, motionCount);
            summary.HoldsPosition = header.Type == JacobianType.LinearLimit || header.Type == JacobianType.AngularLimit1D
                || (header.Type == JacobianType.AngularLimit2D && header.AccessBaseJacobian<AngularLimit2DJacobian>().IsSwingLock);
            if (header.Type == JacobianType.Contact)
            {
                ref ContactJacobian contact = ref header.AccessBaseJacobian<ContactJacobian>();
                summary.Friction = math.max(contact.CoefficientOfFriction, 0.0f);
                if (summary.Friction > 0)
                {
                    float held = contact.NormalImpulseOverSubsteps;
                    for (int i = 0; i < contact.BaseJacobian.NumContacts; i++)
                    {
                        held += header.AccessAngularJacobian(i).Jac.Impulse;
                    }

                    summary.HeldImpulse = held;
                }
            }

            return summary;
        }

        internal static unsafe int Moving(int bodyIndex, MotionVelocity* motionVelocities, int motionCount)
        {
            return bodyIndex < motionCount && !motionVelocities[bodyIndex].IsKinematic ? bodyIndex : -1;
        }

        /// <summary>A contact's friction rows, which follow its normal rows and are bounded by them.</summary>
        static BoundCoupling FrictionCoupling(in JacobianSummary summary, int firstRow)
        {
            int normals = summary.Rows - 3;
            return new BoundCoupling
            {
                FirstRow = firstRow + normals,
                RowCount = 3,
                FirstSourceRow = firstRow,
                SourceRowCount = normals,
                Factor = summary.Friction,
                HeldSourceImpulse = summary.HeldImpulse,
            };
        }

        const int JacobiansPerSummaryTask = 256;

        /// <summary>
        /// What gathering a sparse solve's blocks from an island's Jacobians reads and writes, shared by the lanes that split
        /// the Jacobians. Each task covers the same Jacobians in both stages.
        /// </summary>
        unsafe struct JacobianSummaries
        {
            public IslandHeader* Headers;

            /// <summary>Each Jacobian's first row; a Jacobian's rows end where the next one's start.</summary>
            public int* HeaderRows;

            public int Count;
            public int Rows;
            public MotionVelocity* MotionVelocities;
            public int MotionCount;
            public JacobianSummary* Summaries;

            /// <summary>
            /// Each task's count of blocks and of couplings once it has summarized, and by the time the lists are filled,
            /// where its first ones go.
            /// </summary>
            public int* TaskBlocks;
            public int* TaskCouplings;

            /// <summary>Where each moving body sits among the island's, by its index in the world, once all are numbered.</summary>
            public int* CompactBodies;

            public SparseConstraintSolver.ConstraintBlock* Blocks;
            public IntPtr* HeaderPointers;
            public BoundCoupling* Couplings;
        }

        internal static unsafe void SummarizeJacobiansTask(void* context, int task)
        {
            var summaries = (JacobianSummaries*)context;
            int end = math.min((task + 1) * JacobiansPerSummaryTask, summaries->Count);
            int blocks = 0;
            int couplings = 0;
            for (int h = task * JacobiansPerSummaryTask; h < end; h++)
            {
                int rowEnd = h + 1 < summaries->Count ? summaries->HeaderRows[h + 1] : summaries->Rows;
                JacobianSummary summary = Summarize(ref *summaries->Headers[h].Header, rowEnd - summaries->HeaderRows[h],
                    summaries->MotionVelocities, summaries->MotionCount);
                summaries->Summaries[h] = summary;
                if (summary.Rows > 0)
                {
                    blocks++;
                    couplings += summary.Friction > 0 ? 1 : 0;
                }
            }

            summaries->TaskBlocks[task] = blocks;
            summaries->TaskCouplings[task] = couplings;
        }

        internal static unsafe void FillBlocksTask(void* context, int task)
        {
            var summaries = (JacobianSummaries*)context;
            int end = math.min((task + 1) * JacobiansPerSummaryTask, summaries->Count);
            int block = summaries->TaskBlocks[task];
            int coupling = summaries->TaskCouplings[task];
            for (int h = task * JacobiansPerSummaryTask; h < end; h++)
            {
                JacobianSummary summary = summaries->Summaries[h];
                if (summary.Rows == 0)
                {
                    continue;
                }

                int firstRow = summaries->HeaderRows[h];
                if (summary.Friction > 0)
                {
                    summaries->Couplings[coupling++] = FrictionCoupling(summary, firstRow);
                }

                summaries->Blocks[block] = new SparseConstraintSolver.ConstraintBlock
                {
                    BodyA = summary.BodyA < 0 ? -1 : summaries->CompactBodies[summary.BodyA],
                    BodyB = summary.BodyB < 0 ? -1 : summaries->CompactBodies[summary.BodyB],
                    FirstRow = firstRow,
                    RowCount = summary.Rows,
                };
                summaries->HeaderPointers[block] = (IntPtr)summaries->Headers[h].Header;
                block++;
            }
        }

        const int BodiesPerSetupTask = 64;

        /// <summary>What finding a sparse solve's mass matrices reads and writes, shared by the lanes that split them.</summary>
        unsafe struct BodySetup
        {
            public int* WorldBodies;
            public int Count;
            public MotionVelocity* MotionVelocities;
            public MotionData* MotionDatas;
            public float Timestep;
            public bool Gyroscopic;
            public SparseConstraintSolver.Body* Bodies;
        }

        /// <summary>
        /// One task of finding the mass matrices of a sparse solve's bodies. The solver divides by neither mass nor inertia, so
        /// an axis with none needs a value that stands for infinite, and this one is far past any body.
        /// </summary>
        internal static unsafe void SetupBodiesTask(void* context, int task)
        {
            const float smallestInverse = 1e-12f;
            var setup = (BodySetup*)context;
            int end = math.min((task + 1) * BodiesPerSetupTask, setup->Count);
            for (int compact = task * BodiesPerSetupTask; compact < end; compact++)
            {
                int body = setup->WorldBodies[compact];
                MotionVelocity motion = setup->MotionVelocities[body];
                motion.InverseInertia = math.max(motion.InverseInertia, smallestInverse);
                ComputeInertiaTensor(motion, setup->MotionDatas[body], setup->Timestep, setup->Gyroscopic, out float3x3 inertia, out _);
                setup->Bodies[compact] = new SparseConstraintSolver.Body
                {
                    Mass = 1.0 / math.max(motion.InverseMass, smallestInverse),
                    Inertia = new double3x3(inertia),
                };
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static void ComputeInertiaTensor(in MotionVelocity motionVelocity, in MotionData motionData, float timestep,
            bool enableGyroscopicTorque, out float3x3 I, out float3x3 invI)
        {
            // Principal components of inertia and inverse inertia
            var invInertiaPCs = motionVelocity.InverseInertia;
            var inertiaPCs = 1 / invInertiaPCs;

            float3x3 inertiaTensor;
            float3x3 invInertiaTensor;
            if (enableGyroscopicTorque)
            {
                var I0 = new float3x3(inertiaPCs.x, 0, 0,
                    0, inertiaPCs.y, 0,
                    0, 0, inertiaPCs.z);
                var invI0 = new float3x3(invInertiaPCs.x, 0, 0,
                    0, invInertiaPCs.y, 0,
                    0, 0, invInertiaPCs.z);

                // Add mass stabilization term from Claude Lacoursiere's PhD thesis to inertia tensor in order to
                // ensure a strictly dissipative integration scheme:
                //      I = I_0 - h^2 * \hat(L) * I_0^-1 * \hat(L),
                // where L corresponds to the angular momentum I_0 * w of the rigid body.
                // We perform this computation in body space and then bring the final matrix to world space.
                float3 L = inertiaPCs * motionVelocity.AngularVelocity; // Warning: don't use mul here so that we get
                                                                        // the Hadamard product and not the dot product,
                                                                        // as desired!
                var hatL = DoubleMath.Cross(L);
                var stabilizationTerm = timestep * timestep * math.mul(math.mul(hatL, invI0), hatL);
                inertiaTensor = I0 - stabilizationTerm;
                invInertiaTensor = math.inverse(inertiaTensor);
            }
            else
            {
                inertiaTensor = new float3x3(inertiaPCs.x, 0, 0,
                    0, inertiaPCs.y, 0,
                    0, 0, inertiaPCs.z);
                invInertiaTensor = new float3x3(invInertiaPCs.x, 0, 0,
                    0, invInertiaPCs.y, 0,
                    0, 0, invInertiaPCs.z);
            }

            var worldTransform = motionData.WorldFromMotion;
            float3x3 worldRotM = new float3x3(worldTransform.rot);

            var inertiaTensorWorld = math.mul(math.mul(worldRotM, inertiaTensor), math.transpose(worldRotM));
            var invInertiaTensorWorld = math.mul(math.mul(worldRotM, invInertiaTensor),
                math.transpose(worldRotM));

            I = JacobianUtilities.BuildSymmetricMatrix(inertiaTensorWorld);
            invI = JacobianUtilities.BuildSymmetricMatrix(invInertiaTensorWorld);
        }
    }
}
