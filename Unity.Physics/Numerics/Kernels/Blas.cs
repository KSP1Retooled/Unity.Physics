using Unity.Mathematics;

namespace Unity.Numerics.Kernels
{
    /// <summary>
    /// BLAS-style routines on double-precision matrices in column-major order, given as raw pointers with a leading
    /// dimension. Entry (i, j) of a matrix a with leading dimension lda is a[i + j * lda]. The routines are written for
    /// the small dense matrices of the physics solver (up to a few hundred rows), which fit in the processor's caches,
    /// so the matrix products don't pack their inputs into buffers the way the gemm crate does for large matrices.
    /// </summary>
    public static unsafe partial class Blas
    {
    }
}
