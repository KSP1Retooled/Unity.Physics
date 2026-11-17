namespace Unity.Numerics.Kernels
{
    /// <summary>
    /// BLAS-style operations for dense float and double matrices, optimized for small matrices.
    /// Use MatrixView for arbitrary strides. Pointer overloads use column-major order: a[i + j * lda].
    /// </summary>
    public static unsafe partial class Blas
    {
        /// <summary>Region of C to update.</summary>
        enum Triangle
        {
            All,

            /// <summary>Tiles on or below the diagonal.</summary>
            Lower,

            /// <summary>Tiles on or above the diagonal.</summary>
            Upper,
        }

        /// <summary>Swap lower and upper triangles when transposing.</summary>
        static Triangle Flip(Triangle triangle)
        {
            return triangle == Triangle.Lower ? Triangle.Upper : triangle == Triangle.Upper ? Triangle.Lower : Triangle.All;
        }
    }
}
