using NUnit.Framework;
using Unity.Mathematics;
using Unity.Numerics.Kernels;
using static Unity.Numerics.Kernels.Tests.StridedKernelTests;

namespace Unity.Numerics.Kernels.Tests
{
    /// <summary>
    /// Compare float BLAS and Cholesky kernels with double reference loops across strided layouts, vector tails and
    /// tile edges.
    /// </summary>
    public unsafe class FloatKernelTests
    {
        private static float[] StoreFloat(double[,] m, Place place)
        {
            var data = new float[place.Length];
            for (int i = 0; i < data.Length; i++)
            {
                data[i] = float.NaN;
            }

            for (int i = 0; i < m.GetLength(0); i++)
            {
                for (int j = 0; j < m.GetLength(1); j++)
                {
                    data[place.Index(i, j)] = (float)m[i, j];
                }
            }

            return data;
        }

        /// <summary>Stores a lower triangular matrix with NaN above its diagonal, and on it for a unit diagonal.</summary>
        private static float[] StoreFloatTriangle(double[,] l, Place place, bool unitDiagonal)
        {
            float[] data = StoreFloat(l, place);
            int n = l.GetLength(0);
            for (int i = 0; i < n; i++)
            {
                for (int j = unitDiagonal ? i : i + 1; j < n; j++)
                {
                    data[place.Index(i, j)] = float.NaN;
                }
            }

            return data;
        }

        private static double[,] LoadFloat(float[] data, Place place, int rows, int columns)
        {
            var m = new double[rows, columns];
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < columns; j++)
                {
                    m[i, j] = data[place.Index(i, j)];
                }
            }

            return m;
        }

        /// <summary>The matrix as floats and back, so that the reference starts from the same entries as the kernel.</summary>
        private static double[,] Rounded(double[,] m)
        {
            var r = new double[m.GetLength(0), m.GetLength(1)];
            for (int i = 0; i < m.GetLength(0); i++)
            {
                for (int j = 0; j < m.GetLength(1); j++)
                {
                    r[i, j] = (float)m[i, j];
                }
            }

            return r;
        }

        [Test]
        public void DotAndAxpyMatchPlainLoops([Values(1, 3, 4, 7, 8, 13, 31, 32, 45, 77)] int count, [Values(1, 2)] int stride)
        {
            var random = new Unity.Mathematics.Random((uint)(count * 3 + stride));
            var x = new float[count * stride];
            var y = new float[count * stride];
            for (int i = 0; i < x.Length; i++)
            {
                x[i] = random.NextFloat(-1f, 1f);
                y[i] = random.NextFloat(-1f, 1f);
            }

            double dot = 0.0;
            var expected = new double[y.Length];
            for (int i = 0; i < y.Length; i++)
            {
                expected[i] = y[i];
            }

            for (int k = 0; k < count; k++)
            {
                dot += (double)x[k * stride] * y[k * stride];
                expected[k * stride] += 0.5 * x[k * stride];
            }

            fixed (float* xp = x, yp = y)
            {
                Assert.That(Blas.Dot(count, xp, stride, yp, stride), Is.EqualTo(dot).Within(1e-5));
                Blas.Axpy(count, 0.5f, xp, stride, yp, stride);
            }

            for (int i = 0; i < y.Length; i++)
            {
                Assert.That(y[i], Is.EqualTo(expected[i]).Within(1e-6), $"entry {i}");
            }

            fixed (float* yp = y)
            {
                Blas.Scale(count, -2f, yp, stride);
            }

            for (int k = 0; k < count; k++)
            {
                Assert.That(y[k * stride], Is.EqualTo(-2.0 * expected[k * stride]).Within(1e-5), $"scaled entry {k}");
            }
        }

        [Test]
        public void TheMatrixVectorProductMatchesPlainLoops([Values(1, 7, 13, 40)] int m, [Values(1, 6, 17)] int n, [Values] Layout layout)
        {
            double[,] a = Rounded(RandomMatrix(m, n, (uint)(m * 5 + n)));
            double[,] x = Rounded(RandomMatrix(n, 1, 4));
            Place place = Place.Of(layout, m, n);
            float[] stored = StoreFloat(a, place);
            var xf = new float[n];
            var y = new float[m];
            for (int j = 0; j < n; j++)
            {
                xf[j] = (float)x[j, 0];
            }

            fixed (float* ap = stored, xp = xf, yp = y)
            {
                Blas.Gemv(2f, place.View(ap, m, n), xp, 1, yp, 1);
            }

            double[,] product = Multiply(a, x);
            for (int i = 0; i < m; i++)
            {
                Assert.That(y[i], Is.EqualTo(2.0 * product[i, 0]).Within(1e-4), $"entry {i}");
            }
        }

        /// <summary>The float product with each layout of A and C, which includes A transposed (the copied path).</summary>
        [Test]
        public void TheProductMatchesPlainLoops([Values(1, 15, 16, 17, 37)] int m, [Values(1, 6, 13)] int n, [Values(1, 9, 300)] int k,
            [Values] Layout aLayout, [Values(Layout.ColumnMajor, Layout.Transposed)] Layout bLayout,
            [Values(Layout.ColumnMajor, Layout.Transposed, Layout.EveryOtherRow)] Layout cLayout)
        {
            double[,] a = Rounded(RandomMatrix(m, k, (uint)(m + 3 * n + 11 * k)));
            double[,] b = Rounded(RandomMatrix(k, n, (uint)(m + 5 * n + 7 * k)));
            double[,] c = Rounded(RandomMatrix(m, n, (uint)(2 * m + n + k)));
            Place ap = Place.Of(aLayout, m, k), bp = Place.Of(bLayout, k, n), cp = Place.Of(cLayout, m, n);
            float[] aData = StoreFloat(a, ap), bData = StoreFloat(b, bp), cData = StoreFloat(c, cp);
            fixed (float* aPtr = aData, bPtr = bData, cPtr = cData)
            {
                Blas.Gemm(1.5f, ap.View(aPtr, m, k), bp.View(bPtr, k, n), cp.View(cPtr, m, n));
            }

            double[,] product = Multiply(a, b);
            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    c[i, j] += 1.5 * product[i, j];
                }
            }

            AssertNear(c, LoadFloat(cData, cp, m, n), 2e-5 * k, "C");
        }

        [Test]
        public void TheLowerTriangleProductMatchesPlainLoops([Values(1, 7, 16, 17, 40)] int n, [Values(1, 9)] int k,
            [Values(Layout.ColumnMajor, Layout.Transposed)] Layout layout)
        {
            double[,] a = Rounded(RandomMatrix(n, k, (uint)(n * 3 + k)));
            double[,] c = Rounded(RandomMatrix(n, n, (uint)(n * 5 + k)));
            Place ap = Place.Of(layout, n, k), cp = Place.Of(layout, n, n);
            float[] aData = StoreFloat(a, ap), cData = StoreFloat(c, cp);
            fixed (float* aPtr = aData, cPtr = cData)
            {
                Blas.SyrkLower(-1f, ap.View(aPtr, n, k), cp.View(cPtr, n, n));
            }

            double[,] product = Multiply(a, Transpose(a));
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j <= i; j++)
                {
                    c[i, j] -= product[i, j];
                }
            }

            // The entries above the diagonal keep their values.
            AssertNear(c, LoadFloat(cData, cp, n, n), 1e-4, "C");
        }

        [Test]
        public void TheTriangularSolvesMatchPlainLoops([Values(1, 5, 8, 9, 20, 33, 70)] int n, [Values(1, 5)] int nrhs,
            [Values(Layout.ColumnMajor, Layout.Transposed, Layout.EveryOtherRow)] Layout lLayout,
            [Values(Layout.ColumnMajor, Layout.Transposed)] Layout bLayout, [Values] bool unitDiagonal)
        {
            double[,] l = Lower(n, (uint)(n * 7 + nrhs));
            for (int i = 0; i < n; i++)
            {
                // Smaller entries below the diagonal keep the solve well conditioned in floats.
                for (int j = 0; j < i; j++)
                {
                    l[i, j] *= 2.0 / n;
                }

                if (unitDiagonal)
                {
                    l[i, i] = 1.0;
                }
            }

            l = Rounded(l);
            double[,] x = RandomMatrix(n, nrhs, (uint)(n + nrhs));
            Place lp = Place.Of(lLayout, n, n), bp = Place.Of(bLayout, n, nrhs);
            float[] lData = StoreFloatTriangle(l, lp, unitDiagonal);

            double[,] lx = Multiply(l, x), ux = Multiply(Transpose(l), x);
            float[] lower = StoreFloat(lx, bp);
            float[] upper = StoreFloat(ux, bp);
            var v = new float[n];
            var w = new float[n];
            for (int i = 0; i < n; i++)
            {
                v[i] = (float)lx[i, 0];
                w[i] = (float)ux[i, 0];
            }

            fixed (float* lPtr = lData, lowerPtr = lower, upperPtr = upper, vp = v, wp = w)
            {
                MatrixView<float> lv = lp.View(lPtr, n, n);
                Blas.TrsmLeftLower(lv, bp.View(lowerPtr, n, nrhs), unitDiagonal);
                Blas.TrsmLeftUpper(lv.Transpose, bp.View(upperPtr, n, nrhs), unitDiagonal);
                Blas.TrsvLower(lv, vp, 1, unitDiagonal);
                Blas.TrsvUpper(lv.Transpose, wp, 1, unitDiagonal);
            }

            AssertNear(x, LoadFloat(lower, bp, n, nrhs), 1e-4, "lower");
            AssertNear(x, LoadFloat(upper, bp, n, nrhs), 1e-4, "upper");
            for (int i = 0; i < n; i++)
            {
                Assert.That(v[i], Is.EqualTo(x[i, 0]).Within(1e-4), $"vector lower: entry {i}");
                Assert.That(w[i], Is.EqualTo(x[i, 0]).Within(1e-4), $"vector upper: entry {i}");
            }
        }

        [Test]
        public void TheTriangularProductsMatchPlainLoops([Values(1, 5, 8, 9, 20, 33, 70)] int n, [Values(1, 5)] int nrhs,
            [Values(Layout.ColumnMajor, Layout.Transposed, Layout.EveryOtherRow)] Layout lLayout,
            [Values(Layout.ColumnMajor, Layout.Transposed)] Layout bLayout, [Values] bool unitDiagonal)
        {
            double[,] l = Lower(n, (uint)(n * 5 + nrhs));
            if (unitDiagonal)
            {
                for (int i = 0; i < n; i++)
                {
                    l[i, i] = 1.0;
                }
            }

            l = Rounded(l);
            double[,] x = Rounded(RandomMatrix(n, nrhs, (uint)(n + 2 * nrhs)));
            Place lp = Place.Of(lLayout, n, n), bp = Place.Of(bLayout, n, nrhs);
            float[] lData = StoreFloatTriangle(l, lp, unitDiagonal);
            float[] lower = StoreFloat(x, bp);
            float[] upper = StoreFloat(x, bp);
            fixed (float* lPtr = lData, lowerPtr = lower, upperPtr = upper)
            {
                MatrixView<float> lv = lp.View(lPtr, n, n);
                Blas.TrmmLeftLower(lv, bp.View(lowerPtr, n, nrhs), unitDiagonal);
                Blas.TrmmLeftUpper(lv.Transpose, bp.View(upperPtr, n, nrhs), unitDiagonal);
            }

            AssertNear(Multiply(l, x), LoadFloat(lower, bp, n, nrhs), 1e-5 * n, "lower");
            AssertNear(Multiply(Transpose(l), x), LoadFloat(upper, bp, n, nrhs), 1e-5 * n, "upper");
        }

        [Test]
        public void TheFactorAndItsSolveMatchPlainLoops([Values(1, 5, 8, 31, 32, 33, 64, 100)] int n,
            [Values(Layout.ColumnMajor, Layout.Transposed, Layout.EveryOtherRow)] Layout layout)
        {
            double[,] a = PositiveDefinite(n, (uint)(n + 40));
            for (int i = 0; i < n; i++)
            {
                // A larger diagonal keeps the matrix well conditioned in floats.
                a[i, i] += n;
            }

            a = Rounded(a);
            Place place = Place.Of(layout, n, n);
            float[] data = StoreFloat(a, place);
            fixed (float* p = data)
            {
                Assert.That(CholeskyLLT.TryFactor(place.View(p, n, n)), Is.EqualTo(-1));
            }

            double[,] l = LoadFloat(data, place, n, n);
            for (int i = 0; i < n; i++)
            {
                for (int j = i + 1; j < n; j++)
                {
                    // The factor never writes above the diagonal.
                    Assert.That(l[i, j], Is.EqualTo(a[i, j]), $"entry ({i}, {j}) above the diagonal");
                    l[i, j] = 0.0;
                }
            }

            AssertNear(a, Multiply(l, Transpose(l)), 1e-5 * n, "L L^T");

            double[,] x = RandomMatrix(n, 1, 3);
            double[,] b = Multiply(a, x);
            var v = new float[2 * n];
            for (int i = 0; i < n; i++)
            {
                v[2 * i] = (float)b[i, 0];
            }

            fixed (float* p = data, vp = v)
            {
                CholeskyLLT.Solve(place.View(p, n, n), vp, 2);
            }

            for (int i = 0; i < n; i++)
            {
                Assert.That(v[2 * i], Is.EqualTo(x[i, 0]).Within(1e-4), $"solve: entry {i}");
            }
        }

        [Test]
        public void TryFactorGivesTheFirstPivotThatIsNotPositive([Values(6, 50)] int n, [Values(Layout.ColumnMajor, Layout.Transposed)] Layout layout)
        {
            double[,] a = PositiveDefinite(n, 8);
            int bad = n - 2;
            for (int j = 0; j < n; j++)
            {
                a[bad, j] = a[0, j];
                a[j, bad] = a[j, 0];
            }

            a[bad, bad] = a[0, 0] * 0.5;
            Place place = Place.Of(layout, n, n);
            float[] data = StoreFloat(a, place);
            fixed (float* p = data)
            {
                Assert.That(CholeskyLLT.TryFactor(place.View(p, n, n)), Is.EqualTo(bad));
            }
        }
    }
}
