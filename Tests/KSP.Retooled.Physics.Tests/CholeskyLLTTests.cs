using NUnit.Framework;
using Unity.Mathematics;
using Unity.Numerics.Kernels;

namespace KSP.Retooled.Physics.Tests
{
    /// <summary>
    /// Tests for Cholesky factorization, solves, insertion and deletion, covering the base case and recursive path.
    /// </summary>
    public unsafe class CholeskyLLTTests
    {
        private const double Tolerance = 1e-9;

        /// <summary>Create a random n by n positive definite matrix A = J*J^T + 0.01*I in column-major order.</summary>
        private static double[] Matrix(int n, uint seed)
        {
            var random = new Random(seed);
            int dofs = n / 2 + 3;
            var jacobian = new double[n * dofs];
            for (int k = 0; k < jacobian.Length; k++)
            {
                jacobian[k] = random.NextDouble(-1.0, 1.0);
            }

            var a = new double[n * n];
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    double sum = i == j ? 1e-2 : 0.0;
                    for (int k = 0; k < dofs; k++)
                    {
                        sum += jacobian[i * dofs + k] * jacobian[j * dofs + k];
                    }

                    a[i + j * n] = sum;
                }
            }

            return a;
        }

        /// <summary>Factor the principal submatrix selected by index, with leading dimension ld.</summary>
        private static double[] FreshFactor(double[] a, int n, int[] index, int ld)
        {
            int m = index.Length;
            var l = new double[ld * ld];
            for (int j = 0; j < m; j++)
            {
                for (int i = j; i < m; i++)
                {
                    l[i + j * ld] = a[index[i] + index[j] * n];
                }
            }

            fixed (double* lp = l)
            {
                CholeskyLLT.Factor(m, lp, ld);
            }

            return l;
        }

        /// <summary>Maximum absolute difference in the lower triangles, divided by the largest absolute entry of y.</summary>
        private static double Difference(double[] x, double[] y, int m, int ld)
        {
            double largest = 0.0, size = 1e-300;
            for (int j = 0; j < m; j++)
            {
                for (int i = j; i < m; i++)
                {
                    largest = math.max(largest, math.abs(x[i + j * ld] - y[i + j * ld]));
                    size = math.max(size, math.abs(y[i + j * ld]));
                }
            }

            return largest / size;
        }

        private static readonly int[] Sizes = { 1, 3, 4, 5, 9, 16, 17, 33, 64, 100, 119 };

        [Test]
        public void TheFactorTimesItsTransposeIsTheMatrix([ValueSource(nameof(Sizes))] int n)
        {
            double[] a = Matrix(n, (uint)n + 1);
            int[] index = new int[n];
            for (int i = 0; i < n; i++)
            {
                index[i] = i;
            }

            double[] l = FreshFactor(a, n, index, n);
            double largest = 0.0, size = 0.0;
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j <= i; j++)
                {
                    double sum = 0.0;
                    for (int k = 0; k <= j; k++)
                    {
                        sum += l[i + k * n] * l[j + k * n];
                    }

                    largest = math.max(largest, math.abs(sum - a[i + j * n]));
                    size = math.max(size, math.abs(a[i + j * n]));
                }
            }

            Assert.That(largest, Is.LessThan(Tolerance * size), $"n = {n}");

            // Check the solve by computing the residual A*x - b.
            var random = new Random(7);
            var b = new double[n];
            var x = new double[n];
            for (int i = 0; i < n; i++)
            {
                b[i] = random.NextDouble(-1.0, 1.0);
                x[i] = b[i];
            }

            fixed (double* lp = l, xp = x)
            {
                CholeskyLLT.Solve(n, lp, n, xp);
            }

            double residual = 0.0, scale = 0.0;
            for (int i = 0; i < n; i++)
            {
                double sum = 0.0;
                for (int j = 0; j < n; j++)
                {
                    sum += a[i + j * n] * x[j];
                    scale = math.max(scale, math.abs(a[i + j * n] * x[j]));
                }

                residual = math.max(residual, math.abs(sum - b[i]));
            }

            Assert.That(residual, Is.LessThan(Tolerance * math.max(1.0, scale)), $"n = {n}: solve");
        }

        /// <summary>
        /// Compare insertion at the start, middle and end with a fresh factorization of the larger matrix.
        /// </summary>
        [Test]
        public void InsertingRowsGivesTheFactorOfTheLargerMatrix([Values(5, 40, 100)] int n, [Values(1, 3, 9)] int r,
            [Values(0, 0.5, 1.0)] double where)
        {
            int total = n + r;
            double[] a = Matrix(total, (uint)(n * 31 + r));
            int p = (int)(where * n);

            // Exclude rows and columns p through p + r - 1 to form the original matrix.
            var oldIndex = new int[n];
            for (int i = 0, k = 0; i < total; i++)
            {
                if (i < p || i >= p + r)
                {
                    oldIndex[k++] = i;
                }
            }

            var newIndex = new int[total];
            for (int i = 0; i < total; i++)
            {
                newIndex[i] = i;
            }

            int ld = total + 3;
            double[] l = FreshFactor(a, total, oldIndex, ld);
            var columns = new double[total * r];
            for (int c = 0; c < r; c++)
            {
                for (int i = 0; i < total; i++)
                {
                    columns[i + c * total] = a[i + (p + c) * total];
                }
            }

            var scratch = new double[CholeskyLLT.InsertScratch(n, p, r) + 1];
            fixed (double* lp = l, bp = columns, sp = scratch)
            {
                CholeskyLLT.InsertRowsAndColumns(n, lp, ld, p, r, bp, total, sp);
            }

            double[] fresh = FreshFactor(a, total, newIndex, ld);
            Assert.That(Difference(l, fresh, total, ld), Is.LessThan(Tolerance), $"n = {n}, r = {r}, p = {p}");
        }

        /// <summary>Compare deletion with a fresh factorization of the retained principal submatrix.</summary>
        [Test]
        public void DeletingRowsGivesTheFactorOfTheSmallerMatrix([Values(6, 40, 119)] int n, [Values(1, 2, 5)] int r,
            [Values(1u, 2u, 3u)] uint seed)
        {
            double[] a = Matrix(n, seed * 101 + (uint)n);
            var random = new Random(seed);
            var deleted = new System.Collections.Generic.SortedSet<int>();
            if (seed == 1)
            {
                // Delete the first r rows and columns to exercise an update of the whole remaining factor.
                for (int k = 0; k < r; k++)
                {
                    deleted.Add(k);
                }
            }
            else
            {
                while (deleted.Count < r)
                {
                    deleted.Add(random.NextInt(0, n));
                }
            }

            var all = new int[n];
            for (int i = 0; i < n; i++)
            {
                all[i] = i;
            }

            var keptIndex = new int[n - r];
            for (int i = 0, k = 0; i < n; i++)
            {
                if (!deleted.Contains(i))
                {
                    keptIndex[k++] = i;
                }
            }

            int ld = n + 1;
            double[] l = FreshFactor(a, n, all, ld);
            var indices = new int[r];
            deleted.CopyTo(indices);
            var scratch = new double[CholeskyLLT.DeleteScratch(n, indices[0], r) + 1];
            fixed (double* lp = l, sp = scratch)
            fixed (int* ip = indices)
            {
                CholeskyLLT.DeleteRowsAndColumns(n, lp, ld, ip, r, sp);
            }

            double[] fresh = FreshFactor(a, n, keptIndex, ld);
            Assert.That(Difference(l, fresh, n - r, ld), Is.LessThan(Tolerance), $"n = {n}, deleted {string.Join(",", indices)}");
        }
    }
}
