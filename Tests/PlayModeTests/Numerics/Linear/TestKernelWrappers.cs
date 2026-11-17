using NUnit.Framework;
using Unity.Collections;
using Unity.Mathematics;
using Unity.Numerics.Linear.Dense.Primitives;
using Unity.Numerics.Memory;

namespace Unity.Numerics.Linear.Tests
{
    /// <summary>
    /// Compare Matrix/Vector kernel wrappers with scalar references across transposes, sides, triangles,
    /// submatrices and strided rows.
    /// Fill unused triangles to detect accidental reads.
    /// </summary>
    public unsafe class TestKernelWrappers
    {
        static Matrix RandomMatrix(MemoryManager heap, int rows, int columns, uint seed)
        {
            var random = new Mathematics.Random(seed * 977 + 3);
            var m = heap.Matrix(rows, columns);
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < columns; j++)
                {
                    m[i, j] = random.NextFloat(-1f, 1f);
                }
            }

            return m;
        }

        static double[,] Read(in Matrix m)
        {
            var r = new double[m.NumRows, m.NumCols];
            for (int i = 0; i < m.NumRows; i++)
            {
                for (int j = 0; j < m.NumCols; j++)
                {
                    r[i, j] = m[i, j];
                }
            }

            return r;
        }

        static double[,] Apply(double[,] a, bool transpose)
        {
            if (!transpose)
            {
                return a;
            }

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

        static double[,] Multiply(double[,] a, double[,] b)
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

        static void AssertNear(double[,] expected, in Matrix actual, double tolerance, string what)
        {
            for (int i = 0; i < expected.GetLength(0); i++)
            {
                for (int j = 0; j < expected.GetLength(1); j++)
                {
                    Assert.That(actual[i, j], Is.EqualTo(expected[i, j]).Within(tolerance), $"{what}: entry ({i}, {j})");
                }
            }
        }

        /// <summary>C := alpha op(A) op(B) + beta C, for each transpose of A and B, on blocks of larger matrices.</summary>
        [Test]
        public void TheMatrixProductMatchesPlainLoops([Values(1, 9, 20)] int m, [Values(1, 7)] int n, [Values(1, 13)] int k,
            [Values] bool transposeA, [Values] bool transposeB, [Values(0f, 1f, 0.5f)] float beta)
        {
            using (var heap = new MemoryManager(Allocator.Persistent))
            {
                using var bigA = RandomMatrix(heap, 25, 25, 1);
                using var bigB = RandomMatrix(heap, 25, 25, 2);
                using var bigC = RandomMatrix(heap, 25, 25, 3);
                using var a = transposeA ? bigA.Submatrix(2, 1, k, m) : bigA.Submatrix(2, 1, m, k);
                using var b = transposeB ? bigB.Submatrix(1, 3, n, k) : bigB.Submatrix(1, 3, k, n);
                using var c = bigC.Submatrix(3, 2, m, n);
                double[,] expected = Multiply(Apply(Read(a), transposeA), Apply(Read(b), transposeB));
                double[,] old = Read(c);
                for (int i = 0; i < m; i++)
                {
                    for (int j = 0; j < n; j++)
                    {
                        expected[i, j] = 1.5 * expected[i, j] + beta * old[i, j];
                    }
                }

                c.ScaleAndAddProduct(transposeA ? Linear.Op.Transpose : Linear.Op.None,
                    transposeB ? Linear.Op.Transpose : Linear.Op.None, 1.5f, a, b, beta);
                AssertNear(expected, c, 1e-5 * k, "C");
            }
        }

        /// <summary>
        /// B := alpha op(T) B or alpha B op(T), and the solves op(T) X = alpha B and X op(T) = alpha B, for each side,
        /// triangle, transpose and kind of diagonal.
        /// </summary>
        [Test]
        public void TheTriangularProductsAndSolvesMatchPlainLoops([Values(1, 6, 19)] int m, [Values(1, 4, 11)] int n,
            [Values(0, 1)] int side, [Values(0, 1)] int type, [Values] bool transpose, [Values] bool unit)
        {
            var sideValue = side == 0 ? Side.Left : Side.Right;
            var typeValue = type == 0 ? TriangularType.Lower : TriangularType.Upper;
            var opValue = transpose ? Linear.Op.Transpose : Linear.Op.None;
            var diagValue = unit ? DiagonalType.Unit : DiagonalType.Explicit;
            int size = sideValue == Side.Left ? m : n;
            using (var heap = new MemoryManager(Allocator.Persistent))
            {
                using var bigA = RandomMatrix(heap, 25, 25, (uint)(size + 5));
                using var a = bigA.Submatrix(1, 2, size, size);
                // A well conditioned triangle: a diagonal between 1 and 2 and small entries off it. The other
                // triangle keeps its larger random values.
                var t = new double[size, size];
                for (int i = 0; i < size; i++)
                {
                    a[i, i] = 1f + math.abs(a[i, i]);
                    for (int j = 0; j < size; j++)
                    {
                        bool inside = typeValue == TriangularType.Lower ? i > j : i < j;
                        if (inside)
                        {
                            a[i, j] *= 1f / size;
                            t[i, j] = a[i, j];
                        }
                    }

                    t[i, i] = unit ? 1.0 : a[i, i];
                }

                double[,] op = Apply(t, transpose);
                using var bigB = RandomMatrix(heap, 25, 25, (uint)(m * 3 + n));
                using var b = bigB.Submatrix(2, 3, m, n);
                double[,] original = Read(b);
                double[,] product = sideValue == Side.Left ? Multiply(op, original) : Multiply(original, op);
                b.MultiplyByTriangular(sideValue, typeValue, opValue, diagValue, 0.5f, a);
                for (int i = 0; i < m; i++)
                {
                    for (int j = 0; j < n; j++)
                    {
                        product[i, j] *= 0.5;
                    }
                }

                AssertNear(product, b, 1e-5 * size, "product");

                // The solve with alpha 2 undoes the product with alpha 0.5, so it gives the original back.
                b.SolveGeneralizedTriangular(sideValue, typeValue, opValue, diagValue, 2f, a);
                AssertNear(original, b, 1e-4, "solve");
            }
        }

        /// <summary>
        /// Verify triangular products fall back correctly for disjoint blocks whose address ranges overlap.
        /// </summary>
        [Test]
        public void TheTriangularProductsMatchPlainLoopsWhenTheyShareStorage([Values(6, 19)] int m, [Values(4, 11)] int n,
            [Values(0, 1)] int side, [Values(0, 1)] int type, [Values] bool transpose, [Values] bool unit)
        {
            var sideValue = side == 0 ? Side.Left : Side.Right;
            var typeValue = type == 0 ? TriangularType.Lower : TriangularType.Upper;
            var opValue = transpose ? Linear.Op.Transpose : Linear.Op.None;
            var diagValue = unit ? DiagonalType.Unit : DiagonalType.Explicit;
            int size = sideValue == Side.Left ? m : n;
            using (var heap = new MemoryManager(Allocator.Persistent))
            {
                // T takes rows 0 to 18 at most, and B takes rows 25 to 43 at most. A column of the big matrix holds
                // rows of both, so the address ranges of T and B overlap.
                using var big = RandomMatrix(heap, 50, 50, (uint)(size + 7));
                using var a = big.Submatrix(0, 0, size, size);
                using var b = big.Submatrix(25, 1, m, n);
                Assert.That(Matrix.TryView(a, out var aView) && Matrix.TryView(b, out var bView) && Matrix.Overlaps(aView, bView),
                    "The address ranges of T and B must overlap, so that the routine runs its own loops");

                var t = new double[size, size];
                for (int i = 0; i < size; i++)
                {
                    a[i, i] = 1f + math.abs(a[i, i]);
                    for (int j = 0; j < size; j++)
                    {
                        bool inside = typeValue == TriangularType.Lower ? i > j : i < j;
                        if (inside)
                        {
                            a[i, j] *= 1f / size;
                            t[i, j] = a[i, j];
                        }
                    }

                    t[i, i] = unit ? 1.0 : a[i, i];
                }

                double[,] op = Apply(t, transpose);
                double[,] original = Read(b);
                double[,] product = sideValue == Side.Left ? Multiply(op, original) : Multiply(original, op);
                b.MultiplyByTriangular(sideValue, typeValue, opValue, diagValue, 0.5f, a);
                for (int i = 0; i < m; i++)
                {
                    for (int j = 0; j < n; j++)
                    {
                        product[i, j] *= 0.5;
                    }
                }

                AssertNear(product, b, 1e-5 * size, "product");
            }
        }

        /// <summary>y := alpha op(A) x + beta y, x := op(T) x and M += alpha x y^T, with vectors that are rows of a matrix.</summary>
        [Test]
        public void TheMatrixVectorRoutinesMatchPlainLoops([Values(1, 6, 21)] int m, [Values(1, 9)] int n, [Values] bool transpose,
            [Values(0f, 0.5f)] float beta)
        {
            using (var heap = new MemoryManager(Allocator.Persistent))
            {
                using var bigA = RandomMatrix(heap, 25, 25, 7);
                using var a = bigA.Submatrix(1, 1, m, n);
                using var vectors = RandomMatrix(heap, 4, 25, 8);
                var x = vectors.Rows[0].Subvector(0, transpose ? m : n);
                var y = vectors.Rows[1].Subvector(0, transpose ? n : m);
                double[,] av = Read(a);
                var expected = new double[y.Dimension];
                for (int i = 0; i < y.Dimension; i++)
                {
                    double sum = 0.0;
                    for (int j = 0; j < x.Dimension; j++)
                    {
                        sum += (transpose ? av[j, i] : av[i, j]) * x[j];
                    }

                    expected[i] = 2.0 * sum + beta * y[i];
                }

                y.ScaleAndAddProduct(a, transpose ? Linear.Op.Transpose : Linear.Op.None, x, 2f, beta);
                for (int i = 0; i < y.Dimension; i++)
                {
                    Assert.That(y[i], Is.EqualTo(expected[i]).Within(1e-5 * m * n), $"gemv: entry {i}");
                }

                // M += alpha u w^T, with u and w rows of another matrix.
                var u = vectors.Rows[2].Subvector(0, m);
                var w = vectors.Rows[3].Subvector(0, n);
                double[,] outer = Read(a);
                for (int i = 0; i < m; i++)
                {
                    for (int j = 0; j < n; j++)
                    {
                        outer[i, j] += -0.25 * u[i] * w[j];
                    }
                }

                a.AddOuterProduct(u, w, -0.25f);
                AssertNear(outer, a, 1e-5, "outer product");

                // x := op(T) x, for the lower triangle of the square block of A.
                int size = math.min(m, n);
                using var square = bigA.Submatrix(1, 1, size, size);
                var v = vectors.Rows[0].Subvector(0, size);
                double[,] tv = Read(square);
                var product = new double[size];
                for (int i = 0; i < size; i++)
                {
                    for (int j = 0; j < size; j++)
                    {
                        // Entry (i, j) of L is tv[i, j] for j <= i; entry (i, j) of L^T is tv[j, i] for j >= i.
                        product[i] += (transpose ? (j >= i ? tv[j, i] : 0.0) : (j <= i ? tv[i, j] : 0.0)) * v[j];
                    }
                }

                v.MultiplyTriangular(square, transpose ? Linear.Op.Transpose : Linear.Op.None,
                    TriangularType.Lower, DiagonalType.Explicit);
                for (int i = 0; i < size; i++)
                {
                    Assert.That(v[i], Is.EqualTo(product[i]).Within(1e-5 * size), $"trmv: entry {i}");
                }
            }
        }

        /// <summary>Dot, AddScaled and Scale on rows, whose stride is the number of rows, and on columns.</summary>
        [Test]
        public void TheVectorRoutinesMatchPlainLoops([Values(1, 7, 40)] int length)
        {
            using (var heap = new MemoryManager(Allocator.Persistent))
            {
                using var m = RandomMatrix(heap, 3, length, 11);
                using var column = RandomMatrix(heap, length, 1, 12);
                var row = m.Rows[1];
                var col = column.Cols[0];
                double dot = 0.0;
                var sum = new double[length];
                for (int i = 0; i < length; i++)
                {
                    dot += (double)row[i] * col[i];
                    sum[i] = 2.0 * (col[i] + 0.5 * row[i]);
                }

                Assert.That(row.Dot(col), Is.EqualTo(dot).Within(1e-5 * length));
                col.AddScaled(row, 0.5f);
                col.Scale(2f);
                for (int i = 0; i < length; i++)
                {
                    Assert.That(col[i], Is.EqualTo(sum[i]).Within(1e-5), $"entry {i}");
                }
            }
        }

        /// <summary>Both Cholesky factors give L L^T = A, report a singular row at the right place, and keep A's upper triangle.</summary>
        [Test]
        public void TheCholeskyFactorsMatchPlainLoops([Values(1, 5, 33, 70)] int n)
        {
            using (var heap = new MemoryManager(Allocator.Persistent))
            {
                using var j = RandomMatrix(heap, n, n, (uint)n);
                using var a = heap.Matrix(n, n);
                a.ScaleAndAddProduct(Linear.Op.None, Linear.Op.Transpose, 1f, j, j, 0f);
                for (int i = 0; i < n; i++)
                {
                    a[i, i] += n;
                }

                double[,] av = Read(a);
                using var l = heap.Matrix(n, n);
                Cholesky.Factor(a, l, out int singularRow);
                Assert.That(singularRow, Is.EqualTo(-1));
                Cholesky.Factor(a, out singularRow);
                Assert.That(singularRow, Is.EqualTo(-1));
                for (int r = 0; r < n; r++)
                {
                    for (int c = 0; c < n; c++)
                    {
                        if (c > r)
                        {
                            Assert.That(a[r, c], Is.EqualTo((float)av[r, c]), $"upper triangle ({r}, {c})");
                            Assert.That(l[r, c], Is.EqualTo(0f), $"L above the diagonal ({r}, {c})");
                            continue;
                        }

                        Assert.That(a[r, c], Is.EqualTo(l[r, c]).Within(1e-4), $"in place against out of place ({r}, {c})");
                        double sum = 0.0;
                        for (int p = 0; p <= c; p++)
                        {
                            sum += (double)l[r, p] * l[c, p];
                        }

                        Assert.That(sum, Is.EqualTo(av[r, c]).Within(1e-4 * n), $"L L^T ({r}, {c})");
                    }
                }

                if (n > 3)
                {
                    // Row 2 a copy of row 0 with half its diagonal makes the third pivot negative.
                    using var bad = heap.Matrix(n, n);
                    for (int r = 0; r < n; r++)
                    {
                        for (int c = 0; c < n; c++)
                        {
                            bad[r, c] = (float)av[r, c];
                        }
                    }

                    for (int c = 0; c < n; c++)
                    {
                        bad[2, c] = bad[0, c];
                        bad[c, 2] = bad[c, 0];
                    }

                    bad[2, 2] = bad[0, 0] * 0.5f;
                    Cholesky.Factor(bad, l, out singularRow);
                    Assert.That(singularRow, Is.EqualTo(2));
                    Cholesky.Factor(bad, out singularRow);
                    Assert.That(singularRow, Is.EqualTo(2));
                }
            }
        }
    }
}
