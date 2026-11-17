using NUnit.Framework;
using Unity.Mathematics;
using Unity.Numerics.Kernels;

namespace Unity.Numerics.Kernels.Tests
{
    /// <summary>
    /// Compare double BLAS and Cholesky kernels with scalar references on padded column-major, transposed,
    /// submatrix and noncontiguous strided views.
    /// </summary>
    public unsafe class StridedKernelTests
    {
        public enum Layout
        {
            ColumnMajor,
            Transposed,
            Block,
            EveryOtherRow,
        }

        /// <summary>Where a matrix of the given layout sits in an array of doubles, and how big that array is.</summary>
        internal struct Place
        {
            public int Offset;
            public int RowStride;
            public int ColumnStride;
            public int Length;

            public static Place Of(Layout layout, int rows, int columns)
            {
                rows = math.max(rows, 1);
                columns = math.max(columns, 1);
                switch (layout)
                {
                    case Layout.ColumnMajor:
                        return new Place { Offset = 0, RowStride = 1, ColumnStride = rows + 3, Length = (rows + 3) * columns };
                    case Layout.Transposed:
                        return new Place { Offset = 0, RowStride = columns + 2, ColumnStride = 1, Length = (columns + 2) * rows };
                    case Layout.Block:
                        // Rows 3 on and columns 2 on of a matrix with 5 more rows and 2 more columns.
                        int ld = rows + 5;
                        return new Place { Offset = 3 + 2 * ld, RowStride = 1, ColumnStride = ld, Length = ld * (columns + 2) };
                    default:
                        int tall = 2 * rows + 1;
                        return new Place { Offset = 1, RowStride = 2, ColumnStride = tall, Length = tall * columns + 1 };
                }
            }

            public int Index(int i, int j) => Offset + i * RowStride + j * ColumnStride;

            public MatrixView<double> View(double* data, int rows, int columns)
            {
                return new MatrixView<double>(data + Offset, rows, columns, RowStride, ColumnStride);
            }

            public MatrixView<float> View(float* data, int rows, int columns)
            {
                return new MatrixView<float>(data + Offset, rows, columns, RowStride, ColumnStride);
            }
        }

        /// <summary>
        /// Create a matrix view with NaN padding to detect out-of-range kernel reads.
        /// </summary>
        internal static double[] Store(double[,] m, Place place)
        {
            var data = new double[place.Length];
            for (int i = 0; i < data.Length; i++)
            {
                data[i] = double.NaN;
            }

            for (int i = 0; i < m.GetLength(0); i++)
            {
                for (int j = 0; j < m.GetLength(1); j++)
                {
                    data[place.Index(i, j)] = m[i, j];
                }
            }

            return data;
        }

        /// <summary>
        /// Fill unused triangular entries, including unit diagonals, with NaN to detect accidental reads.
        /// </summary>
        internal static double[] StoreTriangle(double[,] l, Place place, bool unitDiagonal)
        {
            double[] data = Store(l, place);
            int n = l.GetLength(0);
            for (int i = 0; i < n; i++)
            {
                for (int j = unitDiagonal ? i : i + 1; j < n; j++)
                {
                    data[place.Index(i, j)] = double.NaN;
                }
            }

            return data;
        }

        internal static double[,] Load(double[] data, Place place, int rows, int columns)
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

        internal static double[,] RandomMatrix(int rows, int columns, uint seed)
        {
            var random = new Unity.Mathematics.Random(seed * 7919 + 1);
            var m = new double[rows, columns];
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < columns; j++)
                {
                    m[i, j] = random.NextDouble(-1.0, 1.0);
                }
            }

            return m;
        }

        /// <summary>A lower triangular matrix with a diagonal between 1 and 2 and zeros above it.</summary>
        internal static double[,] Lower(int n, uint seed)
        {
            double[,] l = RandomMatrix(n, n, seed);
            for (int i = 0; i < n; i++)
            {
                l[i, i] = 1.0 + math.abs(l[i, i]);
                for (int j = i + 1; j < n; j++)
                {
                    l[i, j] = 0.0;
                }
            }

            return l;
        }

        /// <summary>A symmetric positive definite matrix, J J^T plus a small diagonal.</summary>
        internal static double[,] PositiveDefinite(int n, uint seed)
        {
            double[,] jacobian = RandomMatrix(n, n / 2 + 3, seed);
            var a = new double[n, n];
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    double sum = i == j ? 1e-2 : 0.0;
                    for (int k = 0; k < jacobian.GetLength(1); k++)
                    {
                        sum += jacobian[i, k] * jacobian[j, k];
                    }

                    a[i, j] = sum;
                }
            }

            return a;
        }

        internal static double[,] Multiply(double[,] a, double[,] b)
        {
            var c = new double[a.GetLength(0), b.GetLength(1)];
            for (int i = 0; i < a.GetLength(0); i++)
            {
                for (int j = 0; j < b.GetLength(1); j++)
                {
                    double sum = 0.0;
                    for (int p = 0; p < a.GetLength(1); p++)
                    {
                        sum += a[i, p] * b[p, j];
                    }

                    c[i, j] = sum;
                }
            }

            return c;
        }

        internal static double[,] Transpose(double[,] a)
        {
            var t = new double[a.GetLength(1), a.GetLength(0)];
            for (int i = 0; i < a.GetLength(0); i++)
            {
                for (int j = 0; j < a.GetLength(1); j++)
                {
                    t[j, i] = a[i, j];
                }
            }

            return t;
        }

        /// <summary>Checks the entries for which <paramref name="which"/> is true, or all of them if it is null.</summary>
        internal static void AssertNear(double[,] expected, double[,] actual, double tolerance, string what,
            System.Func<int, int, bool> which = null)
        {
            for (int i = 0; i < expected.GetLength(0); i++)
            {
                for (int j = 0; j < expected.GetLength(1); j++)
                {
                    if (which == null || which(i, j))
                    {
                        Assert.That(actual[i, j], Is.EqualTo(expected[i, j]).Within(tolerance), $"{what}: entry ({i}, {j})");
                    }
                }
            }
        }

        private static readonly int[] Sizes = { 1, 3, 7, 13, 21, 40 };

        [Test]
        public void DotAndAxpyTakeAnyStride([Values(1, 5, 23)] int count, [Values(1, 3, -2)] int incx, [Values(1, 2)] int incy)
        {
            var random = new Unity.Mathematics.Random((uint)(count * 13 + incx * 3 + incy + 50));
            int span = count * 3 + 1;
            var x = new double[span];
            var y = new double[span];
            for (int i = 0; i < span; i++)
            {
                x[i] = random.NextDouble(-1.0, 1.0);
                y[i] = random.NextDouble(-1.0, 1.0);
            }

            // A negative stride starts at the far end, as faer's reversed views do.
            int xStart = incx < 0 ? (count - 1) * -incx : 0;
            double expectedDot = 0.0;
            var expectedY = (double[])y.Clone();
            for (int k = 0; k < count; k++)
            {
                expectedDot += x[xStart + k * incx] * y[k * incy];
                expectedY[k * incy] += 0.75 * x[xStart + k * incx];
            }

            fixed (double* xp = x, yp = y)
            {
                Assert.That(Blas.Dot(count, xp + xStart, incx, yp, incy), Is.EqualTo(expectedDot).Within(1e-12));
                Blas.Axpy(count, 0.75, xp + xStart, incx, yp, incy);
            }

            for (int i = 0; i < span; i++)
            {
                Assert.That(y[i], Is.EqualTo(expectedY[i]).Within(1e-12), $"entry {i}");
            }
        }

        [Test]
        public void TheMatrixVectorProductTakesAnyLayout([ValueSource(nameof(Sizes))] int m, [Values(1, 6, 17)] int n,
            [Values] Layout layout, [Values(1, 3)] int incx)
        {
            double[,] a = RandomMatrix(m, n, (uint)(m * 31 + n));
            Place place = Place.Of(layout, m, n);
            double[] stored = Store(a, place);
            var x = new double[n * incx];
            var y = new double[m * 2];
            var random = new Unity.Mathematics.Random(3);
            for (int i = 0; i < x.Length; i++)
            {
                x[i] = random.NextDouble(-1.0, 1.0);
            }

            for (int i = 0; i < y.Length; i++)
            {
                y[i] = random.NextDouble(-1.0, 1.0);
            }

            var expected = (double[])y.Clone();
            for (int i = 0; i < m; i++)
            {
                double sum = 0.0;
                for (int j = 0; j < n; j++)
                {
                    sum += a[i, j] * x[j * incx];
                }

                expected[i * 2] += -1.5 * sum;
            }

            fixed (double* ap = stored, xp = x, yp = y)
            {
                Blas.Gemv(-1.5, place.View(ap, m, n), xp, incx, yp, 2);
            }

            for (int i = 0; i < y.Length; i++)
            {
                Assert.That(y[i], Is.EqualTo(expected[i]).Within(1e-12), $"entry {i}");
            }

            // The transposed product, on the same storage.
            var z = new double[n];
            var expectedZ = new double[n];
            var w = new double[m];
            for (int i = 0; i < m; i++)
            {
                w[i] = random.NextDouble(-1.0, 1.0);
            }

            for (int j = 0; j < n; j++)
            {
                for (int i = 0; i < m; i++)
                {
                    expectedZ[j] += a[i, j] * w[i];
                }
            }

            fixed (double* ap = stored, wp = w, zp = z)
            {
                Blas.Gemv(1.0, place.View(ap, m, n).Transpose, wp, 1, zp, 1);
            }

            for (int j = 0; j < n; j++)
            {
                Assert.That(z[j], Is.EqualTo(expectedZ[j]).Within(1e-12), $"transposed: entry {j}");
            }
        }

        /// <summary>
        /// Test C += alpha*A*B across all operand layouts, including transposed and noncontiguous views.
        /// </summary>
        [Test]
        public void TheProductTakesAnyLayout([Values(1, 8, 13, 21)] int m, [Values(1, 6, 7)] int n, [Values(1, 9)] int k,
            [Values] Layout aLayout, [Values(Layout.ColumnMajor, Layout.Transposed)] Layout bLayout, [Values] Layout cLayout)
        {
            double[,] a = RandomMatrix(m, k, (uint)(m + 3 * n + 11 * k));
            double[,] b = RandomMatrix(k, n, (uint)(m + 5 * n + 7 * k));
            double[,] c = RandomMatrix(m, n, (uint)(2 * m + n + k));
            Place ap = Place.Of(aLayout, m, k), bp = Place.Of(bLayout, k, n), cp = Place.Of(cLayout, m, n);
            double[] aData = Store(a, ap), bData = Store(b, bp), cData = Store(c, cp);
            double[,] product = Multiply(a, b);
            fixed (double* aPtr = aData, bPtr = bData, cPtr = cData)
            {
                Blas.Gemm(-0.5, ap.View(aPtr, m, k), bp.View(bPtr, k, n), cp.View(cPtr, m, n));
            }

            double[,] result = Load(cData, cp, m, n);
            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    c[i, j] += -0.5 * product[i, j];
                }
            }

            AssertNear(c, result, 1e-12, "C");
        }

        /// <summary>A product deep enough that the copied path takes more than one panel of the inner dimension.</summary>
        [Test]
        public void TheProductWithATransposedAndALongInnerDimensionMatchesPlainLoops()
        {
            int m = 19, n = 7, k = 300;
            double[,] a = RandomMatrix(m, k, 1);
            double[,] b = RandomMatrix(k, n, 2);
            var c = new double[m, n];
            Place ap = Place.Of(Layout.Transposed, m, k), bp = Place.Of(Layout.ColumnMajor, k, n), cp = Place.Of(Layout.ColumnMajor, m, n);
            double[] aData = Store(a, ap), bData = Store(b, bp), cData = Store(c, cp);
            fixed (double* aPtr = aData, bPtr = bData, cPtr = cData)
            {
                Blas.Gemm(1.0, ap.View(aPtr, m, k), bp.View(bPtr, k, n), cp.View(cPtr, m, n));
            }

            AssertNear(Multiply(a, b), Load(cData, cp, m, n), 1e-11, "C");
        }

        [Test]
        public void TheLowerTriangleProductTakesAnyLayout([ValueSource(nameof(Sizes))] int n, [Values(1, 9)] int k,
            [Values] Layout aLayout, [Values] Layout cLayout)
        {
            double[,] a = RandomMatrix(n, k, (uint)(n * 3 + k));
            double[,] c = RandomMatrix(n, n, (uint)(n * 5 + k));
            Place ap = Place.Of(aLayout, n, k), cp = Place.Of(cLayout, n, n);
            double[] aData = Store(a, ap), cData = Store(c, cp);
            fixed (double* aPtr = aData, cPtr = cData)
            {
                Blas.SyrkLower(-1.0, ap.View(aPtr, n, k), cp.View(cPtr, n, n));
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
            AssertNear(c, Load(cData, cp, n, n), 1e-12, "C");
        }

        [Test]
        public void TheTriangularSolvesTakeAnyLayout([ValueSource(nameof(Sizes))] int n, [Values(1, 5)] int nrhs,
            [Values] Layout lLayout, [Values(Layout.ColumnMajor, Layout.Transposed, Layout.EveryOtherRow)] Layout bLayout,
            [Values] bool unitDiagonal)
        {
            double[,] l = Lower(n, (uint)(n * 7 + nrhs));
            if (unitDiagonal)
            {
                for (int i = 0; i < n; i++)
                {
                    l[i, i] = 1.0;
                }
            }

            double[,] x = RandomMatrix(n, nrhs, (uint)(n + nrhs));
            Place lp = Place.Of(lLayout, n, n), bp = Place.Of(bLayout, n, nrhs);
            double[] lData = StoreTriangle(l, lp, unitDiagonal);

            // B = L X, so the lower solve gives X back; B = L^T X, so the solve with the upper triangle L^T does.
            double[] lower = Store(Multiply(l, x), bp);
            double[] upper = Store(Multiply(Transpose(l), x), bp);
            fixed (double* lPtr = lData, lowerPtr = lower, upperPtr = upper)
            {
                MatrixView<double> lv = lp.View(lPtr, n, n);
                Blas.TrsmLeftLower(lv, bp.View(lowerPtr, n, nrhs), unitDiagonal);
                Blas.TrsmLeftUpper(lv.Transpose, bp.View(upperPtr, n, nrhs), unitDiagonal);
            }

            AssertNear(x, Load(lower, bp, n, nrhs), 1e-10, "lower");
            AssertNear(x, Load(upper, bp, n, nrhs), 1e-10, "upper");

            // The vector solves, on the first column, with entries 3 apart.
            var v = new double[3 * n];
            var w = new double[3 * n];
            double[,] lx = Multiply(l, x), ux = Multiply(Transpose(l), x);
            for (int i = 0; i < n; i++)
            {
                v[3 * i] = lx[i, 0];
                w[3 * i] = ux[i, 0];
            }

            fixed (double* lPtr = lData, vp = v, wp = w)
            {
                MatrixView<double> lv = lp.View(lPtr, n, n);
                Blas.TrsvLower(lv, vp, 3, unitDiagonal);
                Blas.TrsvUpper(lv.Transpose, wp, 3, unitDiagonal);
            }

            for (int i = 0; i < n; i++)
            {
                Assert.That(v[3 * i], Is.EqualTo(x[i, 0]).Within(1e-10), $"vector lower: entry {i}");
                Assert.That(w[3 * i], Is.EqualTo(x[i, 0]).Within(1e-10), $"vector upper: entry {i}");
            }
        }

        /// <summary>The triangular products B := L B and B := L^T B take any layout and never read outside the triangle.</summary>
        [Test]
        public void TheTriangularProductsTakeAnyLayout([ValueSource(nameof(Sizes))] int n, [Values(1, 5)] int nrhs,
            [Values] Layout lLayout, [Values(Layout.ColumnMajor, Layout.Transposed, Layout.EveryOtherRow)] Layout bLayout,
            [Values] bool unitDiagonal)
        {
            double[,] l = Lower(n, (uint)(n * 3 + nrhs));
            if (unitDiagonal)
            {
                for (int i = 0; i < n; i++)
                {
                    l[i, i] = 1.0;
                }
            }

            double[,] x = RandomMatrix(n, nrhs, (uint)(n * 2 + nrhs));
            Place lp = Place.Of(lLayout, n, n), bp = Place.Of(bLayout, n, nrhs);
            double[] lData = StoreTriangle(l, lp, unitDiagonal);
            double[] lower = Store(x, bp);
            double[] upper = Store(x, bp);
            fixed (double* lPtr = lData, lowerPtr = lower, upperPtr = upper)
            {
                MatrixView<double> lv = lp.View(lPtr, n, n);
                Blas.TrmmLeftLower(lv, bp.View(lowerPtr, n, nrhs), unitDiagonal);
                Blas.TrmmLeftUpper(lv.Transpose, bp.View(upperPtr, n, nrhs), unitDiagonal);
            }

            AssertNear(Multiply(l, x), Load(lower, bp, n, nrhs), 1e-12 * n, "lower");
            AssertNear(Multiply(Transpose(l), x), Load(upper, bp, n, nrhs), 1e-12 * n, "upper");
        }

        [Test]
        public void TheFactorAndItsSolvesTakeAnyLayout([Values(1, 5, 16, 17, 40, 70)] int n, [Values] Layout layout)
        {
            double[,] a = PositiveDefinite(n, (uint)n);
            Place place = Place.Of(layout, n, n);
            double[] data = Store(a, place);
            double[] scratchUpper = Store(a, place);
            int failed;
            fixed (double* p = data, q = scratchUpper)
            {
                failed = CholeskyLLT.TryFactor(place.View(p, n, n));

                // A factor that may write above the diagonal gives the same lower triangle, entry for entry.
                CholeskyLLT.Factor(place.View(q, n, n), keepUpper: false);
            }

            Assert.That(failed, Is.EqualTo(-1));
            double[,] l = Load(data, place, n, n);
            double[,] fast = Load(scratchUpper, place, n, n);
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j <= i; j++)
                {
                    Assert.That(fast[i, j], Is.EqualTo(l[i, j]), $"keepUpper false: entry ({i}, {j})");
                }
            }

            for (int i = 0; i < n; i++)
            {
                for (int j = i + 1; j < n; j++)
                {
                    // The factor never writes above the diagonal.
                    Assert.That(l[i, j], Is.EqualTo(a[i, j]), $"entry ({i}, {j}) above the diagonal");
                    l[i, j] = 0.0;
                }
            }

            AssertNear(a, Multiply(l, Transpose(l)), 1e-10 * n, "L L^T");

            double[,] x = RandomMatrix(n, 3, 9);
            double[] b = Store(Multiply(a, x), Place.Of(Layout.Transposed, n, 3));
            fixed (double* p = data, bPtr = b)
            {
                CholeskyLLT.Solve(place.View(p, n, n), Place.Of(Layout.Transposed, n, 3).View(bPtr, n, 3));
            }

            AssertNear(x, Load(b, Place.Of(Layout.Transposed, n, 3), n, 3), 1e-7, "solve");
        }

        /// <summary>TryFactor stops at the first pivot that is not positive and gives its index, as faer does.</summary>
        [Test]
        public void TryFactorGivesTheFirstPivotThatIsNotPositive([Values(4, 30)] int n, [Values] Layout layout)
        {
            double[,] a = PositiveDefinite(n, 5);
            int bad = n - 3;
            // Make row and column bad a copy of row and column 0, so the matrix is singular there.
            for (int j = 0; j < n; j++)
            {
                a[bad, j] = a[0, j];
                a[j, bad] = a[j, 0];
            }

            a[bad, bad] = a[0, 0] * 0.5;
            Place place = Place.Of(layout, n, n);
            double[] data = Store(a, place);
            fixed (double* p = data)
            {
                Assert.That(CholeskyLLT.TryFactor(place.View(p, n, n)), Is.EqualTo(bad));
            }
        }

        [Test]
        public void TheFactorsChangesTakeAnyLayout([Values(Layout.ColumnMajor, Layout.Transposed, Layout.EveryOtherRow)] Layout layout,
            [Values(9, 40)] int n)
        {
            int r = 3;
            int total = n + r;
            double[,] a = PositiveDefinite(total, 77);
            Place place = Place.Of(layout, total, total);

            // Delete rows 1, 4 and n - 1 from the factor of the whole matrix.
            int[] deleted = { 1, 4, total - 2 };
            double[] data = Store(a, place);
            var scratch = new double[CholeskyLLT.DeleteScratch(total, deleted[0], r) + 1];
            fixed (double* p = data, s = scratch)
            fixed (int* indices = deleted)
            {
                CholeskyLLT.Factor(place.View(p, total, total));
                CholeskyLLT.DeleteRowsAndColumns(place.View(p, total, total), indices, r, s);
            }

            var kept = new System.Collections.Generic.List<int>();
            for (int i = 0; i < total; i++)
            {
                if (System.Array.IndexOf(deleted, i) < 0)
                {
                    kept.Add(i);
                }
            }

            var small = new double[n, n];
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    small[i, j] = a[kept[i], kept[j]];
                }
            }

            double[,] l = Load(data, place, n, n);
            for (int i = 0; i < n; i++)
            {
                for (int j = i + 1; j < n; j++)
                {
                    l[i, j] = 0.0;
                }
            }

            AssertNear(small, Multiply(l, Transpose(l)), 1e-9, "after the deletion");

            // Insert them again at place 2, as rows 2 to 4 of the new matrix.
            int at = 2;
            var order = new System.Collections.Generic.List<int>(kept);
            order.InsertRange(at, deleted);
            var columns = new double[total, r];
            for (int i = 0; i < total; i++)
            {
                for (int c = 0; c < r; c++)
                {
                    columns[i, c] = a[order[i], deleted[c]];
                }
            }

            Place columnsPlace = Place.Of(layout, total, r);
            double[] columnsData = Store(columns, columnsPlace);
            var insertScratch = new double[CholeskyLLT.InsertScratch(n, at, r) + 1];
            fixed (double* p = data, bPtr = columnsData, s = insertScratch)
            {
                CholeskyLLT.InsertRowsAndColumns(n, place.View(p, total, total), at, r, columnsPlace.View(bPtr, total, r), s);
            }

            var large = new double[total, total];
            for (int i = 0; i < total; i++)
            {
                for (int j = 0; j < total; j++)
                {
                    large[i, j] = a[order[i], order[j]];
                }
            }

            l = Load(data, place, total, total);
            for (int i = 0; i < total; i++)
            {
                for (int j = i + 1; j < total; j++)
                {
                    l[i, j] = 0.0;
                }
            }

            AssertNear(large, Multiply(l, Transpose(l)), 1e-9, "after the insertion");
        }
    }
}
