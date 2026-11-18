using System;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using Unity.Physics.Base.Threading;
using Unity.Physics.Dynamics.Solver.Sparse;
using Unity.Profiling;

namespace Unity.Physics
{
    public static partial class Solver
    {
        /// <summary>
        /// Applies a sparse solve's impulses: each body takes all its rows' at once, and each Jacobian keeps what its events
        /// read. Neither writes what another body or Jacobian does, so both split among a team's lanes.
        /// </summary>
        internal static unsafe void ApplySparseImpulses(in SparseConstraintSolver.Problem problem, SparseConstraintSolver* solver,
            float3* bodyLinear, float3* bodyAngular, int* worldBodies, in NativeList<IntPtr> headers, float* impulses,
            in NativeArray<MotionVelocity> motionVelocities, in NativeArray<MotionData> motionDatas, bool sparseContacts,
            WorkTeam* team)
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
                SparseContacts = sparseContacts,
                BodyTasks = (problem.BodyCount + BodiesPerApplyTask - 1) / BodiesPerApplyTask,
            };

            int headerTasks = (problem.BlockCount + DriftBlocksPerTask - 1) / DriftBlocksPerTask;
            WorkTeam.RunAll(team, &application, WorkTeam.ApplyImpulsesKind, application.BodyTasks + headerTasks, null);
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
            public bool SparseContacts;

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
                RecordImpulses(ref *application->Headers[block], application->Impulses + rows.FirstRow, rows.RowCount,
                    application->SparseContacts);
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
    }
}
