using System;
using Unity.Collections;

namespace Unity.Numerics.Linear.Dense.Primitives
{
    //
    // Level 2 BLAS equivalents
    //

    [GenerateTestsForBurstCompatibility]
    internal unsafe partial struct Vector
    {
        /// <summary>
        /// Compute v := alpha op(A) x + beta v,<br/>
        /// where v denotes this vector.<br/>
        /// Note: elements of A are accessed sequentially with a single pass through A.
        /// </summary>
        /// <param name="A">A matrix of appropriate dimensions</param>
        /// <param name="op">If set to <see cref="Op.Transpose"/>, A^T will be used instead of A</param>
        /// <param name="x">The vector to add</param>
        /// <param name="alpha">The multiplier of the matrix product</param>
        /// <param name="beta">The muliplier of the added vector</param>
        /// <remarks>BLAS equivalent: SGEMV</remarks>
        public void ScaleAndAddProduct(in Matrix A, Op op, in Vector x, float alpha, float beta)
        {
            // Validate input
            if (op == Op.None)
            {
                Assertions.AssertCompatibleDimensionsTranspose(A, this);
                Assertions.AssertCompatibleDimensions(A, x);
            }
            else if (op == Op.Transpose)
            {
                Assertions.AssertCompatibleDimensions(A, this);
                Assertions.AssertCompatibleDimensionsTranspose(A, x);
            }
            else
            {
                UnityEngine.Debug.LogError("Invalid operation in Vector.ScaleAndAddProduct()");
            }

            if (A.NumRows == 0 || A.NumCols == 0 || (alpha == 0.0f && (beta == 1.0)))
            {
                return;
            }

            // As in the reference BLAS, a beta of 0 sets this vector to 0 without reading it, so NaN in it doesn't
            // carry into the result.
            if (beta == 0.0f)
            {
                // The same entries as the loops below and Scale take, for either sign of the stride.
                float* first = KernelStart;
                for (int i = 0; i < Dimension; i++)
                {
                    first[(long)i * Stride] = 0f;
                }
            }
            else
            {
                Scale(beta);
            }

            if (alpha == 0.0f)
                return;

            // The kernel takes op(A) as a view, with the transpose as the same storage with the strides swapped. The
            // loops below are kept for storage that the kernel can't take, or where this vector shares storage with A
            // or x.
            if (Matrix.TryView(A, out var a) && !Matrix.Overlaps(a, AsColumn) && !Matrix.Overlaps(x.AsColumn, AsColumn))
            {
                Unity.Numerics.Kernels.Blas.Gemv(alpha, op == Op.None ? a : a.Transpose, x.KernelStart, x.Stride, KernelStart, Stride);
                return;
            }

            if (op == Op.None)
            {
                if (alpha == 1.0f)
                {
                    for (int i = 0; i < A.NumCols; i++)
                    {
                        AddScaled(A.Cols[i], x[i]);
                    }
                }
                else
                {
                    for (int i = 0; i < A.NumCols; i++)
                    {
                        AddScaled(A.Cols[i], x[i] * alpha);
                    }
                }
            }
            else
            {
                // These loops used to overwrite entry i with the product, which dropped beta times the old entry and
                // ignored this vector's stride. They now add the product to entry i, as the summary says.
                for (int i = 0; i < A.NumCols; i++)
                {
                    this[i] += alpha * A.Cols[i].Dot(x);
                }
            }
        }

        // Multiplies this vector by a triangular matrix
        // BLAS equivalent: STRMV
        public void MultiplyTriangular(in Matrix A, Op op, TriangularType type, DiagonalType diag)
        {
            // Validate input
            if (op == Op.None)
            {
                Assertions.AssertCompatibleDimensions(A, this);
            }
            else if (op == Op.Transpose)
            {
                Assertions.AssertCompatibleDimensionsTranspose(A, this);
            }
            else
            {
                UnityEngine.Debug.LogError("Invalid operation in Vector.MultiplyTriangular()");
            }

            if (A.NumRows != A.NumCols)
            {
                UnityEngine.Debug.LogError("Non-square matrix in Vector.MultiplyTriangular()");
            }

            if (A.NumRows == 0 || A.NumCols == 0 || Dimension == 0)
            {
                return;
            }

            // The kernel's triangular product, with this vector as a matrix of one column, unless this vector shares
            // storage with A.
            if (Matrix.TryView(A, out var a) && A.NumRows >= Dimension && A.NumCols >= Dimension && !Matrix.Overlaps(a, AsColumn))
            {
                bool lower = Matrix.TriangleForKernels(a, Dimension, Side.Left, type, op, out var triangle);
                if (lower)
                {
                    Unity.Numerics.Kernels.Blas.TrmmLeftLower(triangle, AsColumn, diag == DiagonalType.Unit);
                }
                else
                {
                    Unity.Numerics.Kernels.Blas.TrmmLeftUpper(triangle, AsColumn, diag == DiagonalType.Unit);
                }

                return;
            }

            // set up the vector starting point
            var start = data;
            if (Stride < 0)
            {
                start += (-Dimension + 1) * Stride;
            }

            // If not transposing, we can access the elements of A sequentially in a single pass
            if (op == Op.None)
            {
                // x <- A x
                if (type == TriangularType.Upper)
                {
                    if (Stride == 1)
                    {
                        for (int j = 0; j < Dimension; j++)
                        {
                            if (data[j] != 0)
                            {
                                float temp = data[j];
                                for (int i = 0; i < j; i++)
                                {
                                    data[i] += temp * A[i, j];
                                }

                                if (diag == DiagonalType.Explicit)
                                {
                                    data[j] *= A[j, j];
                                }
                            }
                        }
                    }
                    else
                    {
                        var subVec = start;
                        for (int j = 0; j < Dimension; j++, subVec += Stride)
                        {
                            if (*subVec != 0.0)
                            {
                                var temp = *subVec;
                                var dst = start;
                                for (int i = 0; i < j; i++, dst += Stride)
                                {
                                    *dst += temp * A[i, j];
                                }

                                if (diag == DiagonalType.Explicit)
                                {
                                    *subVec *= A[j, j];
                                }
                            }
                        }
                    }
                }
                else
                {
                    if (Stride == 1)
                    {
                        for (int j = Dimension - 1; 0 <= j; j--)
                        {
                            if (data[j] != 0.0)
                            {
                                var temp = data[j];
                                for (int i = A.NumRows - 1; j < i; i--)
                                {
                                    data[i] = data[i] + temp * A[i, j];
                                }

                                if (diag == DiagonalType.Explicit)
                                {
                                    data[j] = data[j] * A[j, j];
                                }
                            }
                        }
                    }
                    else
                    {
                        start += (Dimension - 1) * Stride;

                        var subVec = start;
                        for (int j = Dimension - 1; 0 <= j; j--, subVec -= Stride)
                        {
                            if (*subVec != 0)
                            {
                                var temp = *subVec;
                                var dst = start;
                                for (int i = Dimension - 1; j < i; i--, dst -= Stride)
                                {
                                    *dst += temp * A[i, j];
                                }

                                if (diag == DiagonalType.Explicit)
                                {
                                    *start *= A[j, j];
                                }
                            }
                        }
                    }
                }
            }
            // x <- A^T x
            else
            {
                if (type == TriangularType.Upper)
                {
                    if (Stride == 1)
                    {
                        for (int j = A.NumRows - 1; 0 <= j; j--)
                        {
                            var temp = data[j];
                            if (diag == DiagonalType.Explicit)
                            {
                                temp *= A[j, j];
                            }
                            for (int i = j - 1; 0 <= i; i--)
                            {
                                temp += A[i, j] * data[i];
                            }
                            data[j] = temp;
                        }
                    }
                    else
                    {
                        var subVec = start + (Dimension - 1) * Stride;
                        for (int j = A.NumRows - 1; 0 <= j; j--, subVec -= Stride)
                        {
                            var temp = *subVec;
                            var dst = subVec;

                            if (diag == DiagonalType.Explicit)
                            {
                                temp *= A[j, j];
                            }
                            for (int i = j - 1; 0 <= i; i--, dst -= Stride)
                            {
                                temp += A[i, j] * *dst;
                            }
                            *subVec = temp;
                        }
                    }
                }
                else
                {
                    if (Stride == 1)
                    {
                        for (int j = 0; j < A.NumRows; j++)
                        {
                            var temp = data[j];
                            if (diag == DiagonalType.Explicit)
                            {
                                temp *= A[j, j];
                            }
                            for (int i = j + 1; i < A.NumRows; i++)
                            {
                                temp += A[i, j] * data[i];
                            }
                            data[j] = temp;
                        }
                    }
                    else
                    {
                        var subVec = start;
                        for (int j = 0; j < Dimension; j++, subVec += Stride)
                        {
                            var temp = *subVec;
                            var dst = subVec;
                            if (diag == DiagonalType.Explicit)
                            {
                                temp *= A[j, j];
                            }
                            for (int i = j + 1; i < A.NumRows; i++, dst += Stride)
                            {
                                temp += A[i, j] * *dst;
                            }
                            *subVec = temp;
                        }
                    }
                }
            }
        }
    }

    [GenerateTestsForBurstCompatibility]
    internal unsafe partial struct Matrix
    {
        /// <summary>
        /// Compute M := M + alpha x y^T
        /// BLAS equivalent: SGER
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="alpha"></param>
        public void AddOuterProduct(in Vector x, in Vector y, float alpha)
        {
            if (NumRows == 0 || NumCols == 0 || alpha == 0)
            {
                return;
            }

            // The kernel's product with an inner dimension of 1: x as a matrix of one column times y as a matrix of
            // one row. The loops below are kept for storage that the kernel can't take, or that x or y shares.
            if (TryView(this, out var m) && x.Dimension == NumRows && y.Dimension == NumCols &&
                !Overlaps(m, x.AsColumn) && !Overlaps(m, y.AsColumn))
            {
                Unity.Numerics.Kernels.Blas.Gemm(alpha, x.AsColumn, y.AsRow, m);
                return;
            }

            if (alpha == 1.0f)
            {
                for (int i = 0; i < NumCols; i++)
                    Cols[i].AddScaled(x, y[i]);
            }
            else
            {
                for (int i = 0; i < NumCols; i++)
                    Cols[i].AddScaled(x, alpha * y[i]);
            }
        }
    }
}
