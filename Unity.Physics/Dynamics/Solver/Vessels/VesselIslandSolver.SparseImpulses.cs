using static Unity.Physics.Solver;
using Unity.Physics.Dynamics.Solver.Islands.Data;
using System;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using Unity.Physics.Base.Threading;
using Unity.Physics.Dynamics.Solver.Sparse;
using Unity.Profiling;

namespace Unity.Physics.Dynamics.Solver.Vessels
{
    internal static partial class VesselIslandSolver
    {
        static readonly ProfilerMarker ApplyImpulsesMarker = new ProfilerMarker(ProfilerCategory.Physics, "Physics.Direct.ApplyImpulses");

        /// <summary>
        /// Applies a sparse solve's impulses: each body takes all its rows' at once, and each Jacobian keeps what its events
        /// read. Neither writes what another body or Jacobian does, so both split among a team's lanes.
        /// </summary>
        internal static unsafe void ApplySparseImpulses(in SparseConstraintSolver.Problem problem, SparseConstraintSolver* solver,
            float3* bodyLinear, float3* bodyAngular, int* worldBodies, in NativeList<IntPtr> headers, float* impulses,
            in NativeArray<MotionVelocity> motionVelocities, in NativeArray<MotionData> motionDatas, WorkTeam* team)
        {
            using ProfilerMarker.AutoScope span = ApplyImpulsesMarker.Auto();

            var application = new ImpulseApplication
            {
                Problem = problem,
                Solver = solver,
                BodyLinear = bodyLinear,
                BodyAngular = bodyAngular,
                Impulses = impulses,
                Headers = (JacobianHeader**)headers.GetUnsafeReadOnlyPtr(),
                WorldBodies = worldBodies,
                MotionVelocities = (MotionVelocity*)motionVelocities.GetUnsafePtr(),
                MotionDatas = (MotionData*)motionDatas.GetUnsafeReadOnlyPtr(),
                BodyTasks = (problem.BodyCount + BodiesPerApplyTask - 1) / BodiesPerApplyTask,
            };

            int headerTasks = (problem.BlockCount + DriftBlocksPerTask - 1) / DriftBlocksPerTask;
            WorkTeam.RunAll<DirectTaskDispatch>(team, &application, IslandTaskKind.ApplyImpulsesKind, application.BodyTasks + headerTasks, null);
        }

        const int BodiesPerApplyTask = 64;

        /// <summary>What applying a sparse solve's impulses reads and writes, shared by the lanes that split it.</summary>
        unsafe struct ImpulseApplication
        {
            public SparseConstraintSolver.Problem Problem;
            public SparseConstraintSolver* Solver;

            /// <summary>Each body's impulse, for an application without a solver to ask.</summary>
            public float3* BodyLinear;
            public float3* BodyAngular;
            public float* Impulses;
            public JacobianHeader** Headers;
            public int* WorldBodies;
            public MotionVelocity* MotionVelocities;
            public MotionData* MotionDatas;

            /// <summary>The tasks before this apply to bodies, and the rest record Jacobians' impulses.</summary>
            public int BodyTasks;
        }

        /// <summary>One task of <see cref="ApplySparseImpulses"/>.</summary>
        internal static unsafe void ApplyImpulsesTask(void* context, int task)
        {
            var application = (ImpulseApplication*)context;
            if (task < application->BodyTasks)
            {
                int end = math.min((task + 1) * BodiesPerApplyTask, application->Problem.BodyCount);
                for (int compact = task * BodiesPerApplyTask; compact < end; compact++)
                {
                    int body = application->WorldBodies[compact];
                    float3 linear, angular;
                    if (application->Solver != null)
                    {
                        application->Solver->BodyImpulse(application->Problem, application->Impulses, compact, out linear, out angular);
                    }
                    else
                    {
                        linear = application->BodyLinear[compact];
                        angular = application->BodyAngular[compact];
                    }

                    ref MotionVelocity velocity = ref application->MotionVelocities[body];
                    velocity.ApplyLinearImpulse(linear);

                    // An angular impulse goes in in motion space, the body's own frame.
                    quaternion motionFromWorld = math.conjugate(application->MotionDatas[body].WorldFromMotion.rot);
                    velocity.ApplyAngularImpulse(math.rotate(motionFromWorld, angular));
                }

                return;
            }

            int headerTask = task - application->BodyTasks;
            int endBlock = math.min((headerTask + 1) * DriftBlocksPerTask, application->Problem.BlockCount);
            for (int block = headerTask * DriftBlocksPerTask; block < endBlock; block++)
            {
                SparseConstraintSolver.ConstraintBlock rows = application->Problem.Blocks[block];
                RecordImpulses(ref *application->Headers[block], application->Problem, rows, application->Impulses);
            }
        }

        /// <summary>An array over memory another lane owns, for code that takes arrays.</summary>
        internal static unsafe NativeArray<T> View<T>(T* data, int length) where T : unmanaged
        {
            var array = NativeArrayUnsafeUtility.ConvertExistingDataToNativeArray<T>(data, length, Allocator.None);
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            NativeArrayUnsafeUtility.SetAtomicSafetyHandle(ref array, AtomicSafetyHandle.GetTempMemoryHandle());
#endif
            return array;
        }

        /// <summary>
        /// Keeps what the events and the next substep read of one Jacobian's solved impulses. A Jacobian writes only its own
        /// data, so Jacobians can do this on any thread.
        /// </summary>
        static unsafe void RecordImpulses(ref JacobianHeader header, in SparseConstraintSolver.Problem problem,
            in SparseConstraintSolver.ConstraintBlock block, float* impulses)
        {
            if (header.Type == JacobianType.Contact)
            {
                HandOverContactImpulses(ref header, impulses + block.FirstRow, ContactRowReach);
            }
            else if ((header.Flags & JacobianFlags.EnableImpulseEvents) != 0)
            {
                // Each row contributes its impulse along its own direction. Report body A's share, using
                // the opposite of body B's share when A is held in place. A linear event omits the moments
                // about the bodies' centres of mass that come from the distance to the joint anchor.
                bool useBodyA = block.BodyA >= 0;
                int side = useBodyA ? 0 : SparseConstraintSolver.ColumnsPerRow / 2;
                int component = header.Type == JacobianType.LinearLimit ? 0 : 3;
                float sign = useBodyA ? 1.0f : -1.0f;
                float3 impulse = float3.zero;
                for (int row = block.FirstRow; row < block.FirstRow + block.RowCount; row++)
                {
                    float* axis = problem.Jacobian + row * SparseConstraintSolver.ColumnsPerRow + side + component;
                    impulse += (sign * impulses[row]) * new float3(axis[0], axis[1], axis[2]);
                }

                header.AccessImpulseEventSolverData().AccumulatedImpulse += impulse;
            }
        }


        /// <summary>
        /// Store normal impulse per contact point and accumulate step-wide normal/friction totals for later substep
        /// bounds.
        /// </summary>
        static unsafe void HandOverContactImpulses(ref JacobianHeader header, float* lambda, float collisionTolerance)
        {
            ref var contactJacobian = ref header.AccessBaseJacobian<ContactJacobian>();
            float sum = 0.0f;
            float pushed = 0.0f;
            int normalRow = 0;
            for (int i = 0; i < contactJacobian.BaseJacobian.NumContacts; ++i)
            {
                ref var point = ref header.AccessAngularJacobian(i);
                if (point.ContactDistance <= collisionTolerance)
                {
                    // The point already holds its warm start, which the events count apart from this.
                    float solved = lambda[normalRow++];
                    point.Jac.Impulse = math.max(point.Jac.Impulse + solved, 0.0f);
                    sum += solved;
                }

                // A point without a row still pushed with its warm start.
                pushed += point.Jac.Impulse;
            }

            contactJacobian.SumImpulsesOverSubsteps += sum;
            contactJacobian.NormalImpulseOverSubsteps += pushed;
            if (contactJacobian.CoefficientOfFriction > 0 && normalRow > 0)
            {
                contactJacobian.FrictionImpulseOverSubsteps += new float3(lambda[normalRow], lambda[normalRow + 1], lambda[normalRow + 2]);
            }
        }
    }
}
