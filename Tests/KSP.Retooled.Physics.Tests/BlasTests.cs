using NUnit.Framework;
using Unity.Mathematics;
using Unity.Numerics.Kernels;

namespace KSP.Retooled.Physics.Tests
{
    /// <summary>
    /// Compare float and double matrix products with scalar reference loops across full tiles and edges.
    /// Allow rounding differences from FMA and accumulation order.
    /// </summary>
    public unsafe class BlasTests
    {
        private static readonly int[] Sizes = { 1, 3, 7, 8, 13, 16, 17, 33 };

        [Test]
        public void TheDoubleProductsMatchPlainLoops([ValueSource(nameof(Sizes))] int m, [Values(1, 5, 6, 13)] int n, [Values(1, 9)] int k)
        {
            var random = new Random((uint)(m * 131 + n * 7 + k));
            int lda = m + 2, ldb = math.max(k, n) + 1, ldc = m + 3;
            var a = new double[lda * k];
            var b = new double[ldb * math.max(k, n)];
            for (int i = 0; i < a.Length; i++)
            {
                a[i] = random.NextDouble(-1.0, 1.0);
            }

            for (int i = 0; i < b.Length; i++)
            {
                b[i] = random.NextDouble(-1.0, 1.0);
            }

            for (int transposed = 0; transposed < 2; transposed++)
            {
                var c = new double[ldc * n];
                var expected = new double[ldc * n];
                for (int i = 0; i < c.Length; i++)
                {
                    c[i] = expected[i] = random.NextDouble(-1.0, 1.0);
                }

                for (int j = 0; j < n; j++)
                {
                    for (int i = 0; i < m; i++)
                    {
                        double sum = 0.0;
                        for (int p = 0; p < k; p++)
                        {
                            sum += a[i + p * lda] * (transposed == 0 ? b[p + j * ldb] : b[j + p * ldb]);
                        }

                        expected[i + j * ldc] += -0.5 * sum;
                    }
                }

                fixed (double* ap = a, bp = b, cp = c)
                {
                    if (transposed == 0)
                    {
                        Blas.GemmNN(m, n, k, -0.5, ap, lda, bp, ldb, cp, ldc);
                    }
                    else
                    {
                        Blas.GemmNT(m, n, k, -0.5, ap, lda, bp, ldb, cp, ldc);
                    }
                }

                for (int i = 0; i < c.Length; i++)
                {
                    Assert.That(c[i], Is.EqualTo(expected[i]).Within(1e-12), $"entry {i}, transposed {transposed}");
                }
            }
        }

        [Test]
        public void TheFloatProductsMatchPlainLoops([ValueSource(nameof(Sizes))] int m, [Values(1, 5, 6, 13)] int n, [Values(1, 9)] int k)
        {
            var random = new Random((uint)(m * 131 + n * 7 + k + 5));
            int lda = m + 2, ldb = math.max(k, n) + 1, ldc = m + 3;
            var a = new float[lda * k];
            var b = new float[ldb * math.max(k, n)];
            for (int i = 0; i < a.Length; i++)
            {
                a[i] = random.NextFloat(-1f, 1f);
            }

            for (int i = 0; i < b.Length; i++)
            {
                b[i] = random.NextFloat(-1f, 1f);
            }

            for (int transposed = 0; transposed < 2; transposed++)
            {
                var c = new float[ldc * n];
                var expected = new double[ldc * n];
                for (int i = 0; i < c.Length; i++)
                {
                    c[i] = random.NextFloat(-1f, 1f);
                    expected[i] = c[i];
                }

                for (int j = 0; j < n; j++)
                {
                    for (int i = 0; i < m; i++)
                    {
                        double sum = 0.0;
                        for (int p = 0; p < k; p++)
                        {
                            sum += a[i + p * lda] * (double)(transposed == 0 ? b[p + j * ldb] : b[j + p * ldb]);
                        }

                        expected[i + j * ldc] += 1.5 * sum;
                    }
                }

                fixed (float* ap = a, bp = b, cp = c)
                {
                    if (transposed == 0)
                    {
                        Blas.GemmNN(m, n, k, 1.5f, ap, lda, bp, ldb, cp, ldc);
                    }
                    else
                    {
                        Blas.GemmNT(m, n, k, 1.5f, ap, lda, bp, ldb, cp, ldc);
                    }
                }

                for (int i = 0; i < c.Length; i++)
                {
                    Assert.That(c[i], Is.EqualTo(expected[i]).Within(1e-5), $"entry {i}, transposed {transposed}");
                }
            }
        }

        /// <summary>The lower-triangle product C += alpha A A^T matches plain loops on and below the diagonal.</summary>
        [Test]
        public void TheLowerTriangleProductMatchesPlainLoops([ValueSource(nameof(Sizes))] int n, [Values(1, 9)] int k)
        {
            var random = new Random((uint)(n * 17 + k));
            int lda = n + 1, ldc = n + 2;
            var a = new double[lda * k];
            var c = new double[ldc * n];
            for (int i = 0; i < a.Length; i++)
            {
                a[i] = random.NextDouble(-1.0, 1.0);
            }

            for (int i = 0; i < c.Length; i++)
            {
                c[i] = random.NextDouble(-1.0, 1.0);
            }

            var original = (double[])c.Clone();
            fixed (double* ap = a, cp = c)
            {
                Blas.SyrkLowerNT(n, k, -1.0, ap, lda, cp, ldc);
            }

            for (int j = 0; j < n; j++)
            {
                for (int i = j; i < n; i++)
                {
                    double sum = 0.0;
                    for (int p = 0; p < k; p++)
                    {
                        sum += a[i + p * lda] * a[j + p * lda];
                    }

                    Assert.That(c[i + j * ldc], Is.EqualTo(original[i + j * ldc] - sum).Within(1e-12), $"entry ({i}, {j})");
                }
            }
        }
    }
}
