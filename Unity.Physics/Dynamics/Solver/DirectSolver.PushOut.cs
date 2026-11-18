using static Unity.Physics.Dynamics.Solver.Forest.ForestSolver;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using Unity.Numerics.Linear.Solvers.Lowlevel;
using Unity.Physics.Dynamics.Solver.Sparse;
using Unity.Profiling;

namespace Unity.Physics
{
    public static partial class Solver
    {
        /// <summary>
        /// Correct deep penetration without adding velocity, propagating motion through joints.
        /// Use a separate push-out solve with unilateral contacts and no friction so that the main solver state is
        /// preserved.
        /// </summary>
        static unsafe void SolvePushOut(in SparseConstraintSolver.Problem problem, JacobianHeader** headers,
            in NativeList<int> worldBodies, float* recovery, in NativeArray<MotionVelocity> motionVelocities,
            in NativeArray<MotionData> motionDatas, float timestep, float fastest, ForestRun* forest = null)
        {
            int rows = problem.RowCount;
            bool pushing = false;
            for (int row = 0; row < rows && !pushing; row++)
            {
                pushing = recovery[row] > 0.0f;
            }

            if (!pushing)
            {
                return;
            }

            using ProfilerMarker.AutoScope span = PushOutMarker.Auto();

            var values = new NativeArray<float>(rows * 4, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
            float* rightHandSide = (float*)values.GetUnsafePtr();
            float* lower = rightHandSide + rows;
            float* upper = lower + rows;
            float* push = upper + rows;
            for (int row = 0; row < rows; row++)
            {
                bool contact = recovery[row] >= 0.0f;
                rightHandSide[row] = contact ? recovery[row] : 0.0f;
                lower[row] = contact ? 0.0f : problem.Lower[row];
                upper[row] = problem.Upper[row];
            }

            for (int coupling = 0; coupling < problem.CouplingCount; coupling++)
            {
                BoundCoupling friction = problem.Couplings[coupling];
                for (int row = friction.FirstRow; row < friction.FirstRow + friction.RowCount; row++)
                {
                    lower[row] = 0.0f;
                    upper[row] = 0.0f;
                }
            }

            SparseConstraintSolver.Problem pushOut = problem;
            pushOut.RightHandSide = rightHandSide;
            pushOut.Lower = lower;
            pushOut.Upper = upper;
            pushOut.InitialStates = null;
            pushOut.CouplingCount = 0;
            pushOut.Couplings = null;

            // Skip unconverged push-out rather than applying a large clamped correction.
            // Forest push-out reuses the main solve's factor and boundary block with new bounds.
            var solver = default(SparseConstraintSolver);
            var bodyImpulses = new NativeArray<float3>(forest != null ? 2 * worldBodies.Length : 0, Allocator.Temp,
                NativeArrayOptions.UninitializedMemory);
            bool solved;
            if (forest != null)
            {
                ForestRun pushRun = *forest;
                pushRun.Problem = pushOut;
                pushRun.BaseVelocity = default;
                pushRun.Bounded = true;
                solved = SolveForest(ref pushRun, rightHandSide, push);
                float3* linearImpulses = (float3*)bodyImpulses.GetUnsafePtr();
                SettleForest(ref pushRun, pushOut, push, linearImpulses, linearImpulses + worldBodies.Length, true);
            }
            else
            {
                solver = new SparseConstraintSolver(Allocator.Temp);
                solved = solver.Solve(pushOut, push, true);
            }

            if (!solved)
            {
                if (forest == null)
                {
                    solver.Dispose();
                }

                bodyImpulses.Dispose();
                values.Dispose();
                return;
            }

            // Contacts that wedge a body push it hard along whatever way out they leave, faster than any one of them asks.
            // Scaling the whole push, not each body's, keeps jointed bodies where their joints hold them.
            var datas = (MotionData*)motionDatas.GetUnsafePtr();
            var linears = new NativeArray<float3>(worldBodies.Length, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
            var spins = new NativeArray<float3>(worldBodies.Length, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
            float fastestPoint = 0.0f;
            for (int compact = 0; compact < worldBodies.Length; compact++)
            {
                int body = worldBodies[compact];
                MotionVelocity velocity = motionVelocities[body];
                float3 linear, angular;
                if (forest != null)
                {
                    linear = bodyImpulses[compact];
                    angular = bodyImpulses[worldBodies.Length + compact];
                }
                else
                {
                    solver.BodyImpulse(pushOut, push, compact, out linear, out angular);
                }

                // An angular impulse acts in motion space, the body's own frame.
                linears[compact] = linear * velocity.InverseMass;
                spins[compact] = math.rotate(math.conjugate(datas[body].WorldFromMotion.rot), angular) * velocity.InverseInertia;
                float speed = math.length(linears[compact]) + math.length(spins[compact]) * velocity.AngularExpansionFactor;
                fastestPoint = math.max(fastestPoint, speed);
            }

            float scale = fastestPoint > fastest ? fastest / fastestPoint : 1.0f;
            MovePushedContacts(problem, headers, worldBodies, datas, linears, spins, scale * timestep);
            for (int compact = 0; compact < worldBodies.Length; compact++)
            {
                ref RigidTransform pose = ref datas[worldBodies[compact]].WorldFromMotion;
                pose.pos += linears[compact] * (scale * timestep);
                Integrator.IntegrateOrientation(ref pose.rot, spins[compact] * scale, timestep);
            }

            linears.Dispose();
            spins.Dispose();
            if (forest == null)
            {
                solver.Dispose();
            }

            bodyImpulses.Dispose();
            values.Dispose();
        }

        /// <summary>
        /// Update contact distances for the position changes from SolvePushOut before the next substep.
        /// Rows absent from the solve retain their distances.
        /// </summary>
        static unsafe void MovePushedContacts(in SparseConstraintSolver.Problem problem, JacobianHeader** headers,
            in NativeList<int> worldBodies, MotionData* datas, in NativeArray<float3> linears, in NativeArray<float3> spins,
            float duration)
        {
            for (int block = 0; block < problem.BlockCount; block++)
            {
                ref JacobianHeader header = ref *headers[block];
                if (header.Type != JacobianType.Contact)
                {
                    continue;
                }

                SparseConstraintSolver.ConstraintBlock rows = problem.Blocks[block];
                int row = rows.FirstRow;
                int points = header.AccessBaseJacobian<ContactJacobian>().BaseJacobian.NumContacts;
                for (int i = 0; i < points; i++)
                {
                    // Rows are built for the points within reach, in order; see the contact rows of BuildJacobianRows.
                    ref ContactJacAngAndVelToReachCp point = ref header.AccessAngularJacobian(i);
                    if (point.ContactDistance > ContactRowReach)
                    {
                        continue;
                    }

                    float* columns = problem.Jacobian + row * SparseConstraintSolver.ColumnsPerRow;
                    float rate = SideRate(columns, rows.BodyA, worldBodies, datas, linears, spins)
                        + SideRate(columns + SparseConstraintSolver.ColumnsPerRow / 2, rows.BodyB, worldBodies, datas, linears, spins);
                    point.ContactDistance += rate * duration;
                    row++;
                }
            }
        }

        /// <summary>How fast one side's body moves a row along, the Jacobian's angular columns being in world space.</summary>
        static unsafe float SideRate(float* columns, int compact, in NativeList<int> worldBodies, MotionData* datas,
            in NativeArray<float3> linears, in NativeArray<float3> spins)
        {
            if (compact < 0)
            {
                return 0.0f;
            }

            float3 spin = math.rotate(datas[worldBodies[compact]].WorldFromMotion.rot, spins[compact]);
            return math.dot(new float3(columns[0], columns[1], columns[2]), linears[compact])
                + math.dot(new float3(columns[3], columns[4], columns[5]), spin);
        }
    }
}
