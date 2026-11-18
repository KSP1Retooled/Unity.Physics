using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using Unity.Numerics.Linear.Solvers.Lowlevel;

namespace Unity.Physics.Dynamics.Solver.Sparse
{
    internal unsafe partial struct SparseConstraintSolver
    {
        private const int FinishingSweeps = 12;

        private struct RowResponse
        {
            internal double3 LinearA;
            internal double3 AngularA;
            internal double3 LinearB;
            internal double3 AngularB;
            internal double InverseDiagonal;
        }

        /// <summary>
        /// Start again from zero after pivoting runs out of iterations. Clamping the last pivot's impulses can
        /// remove the rows that balanced them. Recompute each impulse against the motion from the other rows.
        /// </summary>
        private void FinishBySweeps(in Problem problem, float* impulses)
        {
            var inverseInertia = new NativeArray<double3x3>(problem.BodyCount, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
            var responses = new NativeArray<RowResponse>(problem.RowCount, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
            var linear = new NativeArray<double3>(problem.BodyCount, Allocator.Temp);
            var angular = new NativeArray<double3>(problem.BodyCount, Allocator.Temp);
            var firstCoupledRow = new NativeArray<int>(problem.RowCount, Allocator.Temp);

            for (int body = 0; body < problem.BodyCount; body++)
            {
                inverseInertia[body] = math.inverse(problem.Bodies[body].Inertia);
            }

            for (int i = 0; i < problem.CouplingCount; i++)
            {
                firstCoupledRow[problem.Couplings[i].FirstRow] = i + 1;
            }

            UnsafeUtility.MemClear(impulses, (long)problem.RowCount * sizeof(float));
            for (int blockIndex = 0; blockIndex < problem.BlockCount; blockIndex++)
            {
                ConstraintBlock block = problem.Blocks[blockIndex];
                for (int row = block.FirstRow; row < block.FirstRow + block.RowCount; row++)
                {
                    lower[row] = problem.Lower[row];
                    upper[row] = problem.Upper[row];
                    float* columns = problem.Jacobian + row * ColumnsPerRow;
                    var response = new RowResponse();
                    double diagonal = problem.Epsilon[row];
                    diagonal += ResponseOfSide(problem, inverseInertia, block.BodyA, columns,
                        out response.LinearA, out response.AngularA);
                    diagonal += ResponseOfSide(problem, inverseInertia, block.BodyB, columns + SideColumns,
                        out response.LinearB, out response.AngularB);
                    response.InverseDiagonal = diagonal > 0.0 ? 1.0 / diagonal : 0.0;
                    responses[row] = response;
                }
            }

            for (int sweep = 0; sweep < FinishingSweeps; sweep++)
            {
                for (int blockIndex = 0; blockIndex < problem.BlockCount; blockIndex++)
                {
                    ConstraintBlock block = problem.Blocks[blockIndex];
                    for (int row = block.FirstRow; row < block.FirstRow + block.RowCount; row++)
                    {
                        int coupling = firstCoupledRow[row] - 1;
                        if (coupling >= 0)
                        {
                            SetProjectedBounds(problem, impulses, problem.Couplings[coupling]);
                        }

                        float* columns = problem.Jacobian + row * ColumnsPerRow;
                        double speed = SideSpeed(block.BodyA, columns, linear, angular)
                            + SideSpeed(block.BodyB, columns + SideColumns, linear, angular);
                        double residual = problem.RightHandSide[row] - speed - problem.Epsilon[row] * (double)impulses[row];
                        RowResponse response = responses[row];
                        float impulse = (float)math.clamp(impulses[row] + residual * response.InverseDiagonal, lower[row], upper[row]);
                        double change = (double)impulse - impulses[row];
                        impulses[row] = impulse;
                        AddProjectedVelocity(block.BodyA, response.LinearA * change, response.AngularA * change, linear, angular);
                        AddProjectedVelocity(block.BodyB, response.LinearB * change, response.AngularB * change, linear, angular);
                    }
                }
            }

            for (int i = 0; i < problem.CouplingCount; i++)
            {
                BoundCoupling coupling = problem.Couplings[i];
                SetProjectedBounds(problem, impulses, coupling);
                for (int row = coupling.FirstRow; row < coupling.FirstRow + coupling.RowCount; row++)
                {
                    impulses[row] = math.clamp(impulses[row], lower[row], upper[row]);
                }
            }

            firstCoupledRow.Dispose();
            angular.Dispose();
            linear.Dispose();
            responses.Dispose();
            inverseInertia.Dispose();
        }

        private void SetProjectedBounds(in Problem problem, float* impulses, in BoundCoupling coupling)
        {
            float sum = coupling.HeldSourceImpulse;
            for (int row = coupling.FirstSourceRow; row < coupling.FirstSourceRow + coupling.SourceRowCount; row++)
            {
                sum += impulses[row];
            }

            float bound = math.max(sum, 0f) * coupling.Factor;
            for (int row = coupling.FirstRow; row < coupling.FirstRow + coupling.RowCount; row++)
            {
                lower[row] = problem.Lower[row] - bound;
                upper[row] = problem.Upper[row] + bound;
            }
        }

        private static double ResponseOfSide(in Problem problem, NativeArray<double3x3> inverseInertia, int body,
            float* columns, out double3 linear, out double3 angular)
        {
            linear = double3.zero;
            angular = double3.zero;
            if (body < 0)
            {
                return 0.0;
            }

            var axis = new double3(columns[0], columns[1], columns[2]);
            var moment = new double3(columns[3], columns[4], columns[5]);
            linear = axis / problem.Bodies[body].Mass;
            angular = math.mul(inverseInertia[body], moment);
            return math.dot(axis, linear) + math.dot(moment, angular);
        }

        private static double SideSpeed(int body, float* columns, NativeArray<double3> linear, NativeArray<double3> angular)
        {
            return body < 0 ? 0.0
                : math.dot(new double3(columns[0], columns[1], columns[2]), linear[body])
                + math.dot(new double3(columns[3], columns[4], columns[5]), angular[body]);
        }

        private static void AddProjectedVelocity(int body, double3 deltaLinear, double3 deltaAngular,
            NativeArray<double3> linear, NativeArray<double3> angular)
        {
            if (body >= 0)
            {
                linear[body] += deltaLinear;
                angular[body] += deltaAngular;
            }
        }
    }
}
