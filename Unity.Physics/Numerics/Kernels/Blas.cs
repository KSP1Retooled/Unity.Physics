namespace Unity.Numerics.Kernels
{
    /// <summary>
    /// BLAS-style routines on dense matrices of doubles and of floats, given as a <see cref="MatrixView{T}"/> or as raw
    /// pointers. Entry (i, j) of a matrix a that is stored column by column with leading dimension lda is
    /// a[i + j * lda]. The routines are written for the small dense matrices of the physics solver (up to a few
    /// hundred rows), which fit in the processor's caches, so the matrix products don't pack their inputs into buffers
    /// the way the gemm crate does for large matrices, except for an A whose columns are not contiguous.
    /// </summary>
    public static unsafe partial class Blas
    {
        /// <summary>Which entries of C a product computes.</summary>
        enum Triangle
        {
            All,

            /// <summary>The tiles that hold entries on or below the diagonal.</summary>
            Lower,

            /// <summary>The tiles that hold entries on or above the diagonal.</summary>
            Upper,
        }

        /// <summary>The triangle of C^T that holds the entries of the given triangle of C.</summary>
        static Triangle Flip(Triangle triangle)
        {
            return triangle == Triangle.Lower ? Triangle.Upper : triangle == Triangle.Upper ? Triangle.Lower : Triangle.All;
        }
    }
}
