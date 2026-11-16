// The shape of faer 0.19.4's MatRef and MatMut (src/mat/matref.rs): a pointer, the numbers of rows and columns, and a
// stride for each. faer: https://github.com/sarah-quinones/faer-rs, MIT license, see Third Party Notices.md at the
// package root.

using System.Runtime.CompilerServices;

namespace Unity.Numerics.Kernels
{
    /// <summary>
    /// A view of a dense matrix of floats or doubles that the kernels read or write in place. Entry (i, j) is at
    /// <c>Data[i * RowStride + j * ColumnStride]</c>. A matrix that is stored column by column with a leading dimension
    /// ld has a row stride of 1 and a column stride of ld. Its transpose is the same memory with the two strides
    /// swapped, and a block of it is the same strides with the pointer moved to the block's first entry. Strides may be
    /// any value, including negative ones, as in faer. The kernels have a fast path for the layouts that they are
    /// written for, mostly a row stride of 1, and a plain path for every other layout.
    /// </summary>
    public unsafe struct MatrixView<T> where T : unmanaged
    {
        public T* Data;
        public int Rows;
        public int Columns;
        public int RowStride;
        public int ColumnStride;

        public MatrixView(T* data, int rows, int columns, int rowStride, int columnStride)
        {
            Data = data;
            Rows = rows;
            Columns = columns;
            RowStride = rowStride;
            ColumnStride = columnStride;
        }

        /// <summary>A matrix that is stored column by column, with <paramref name="leadingDimension"/> between columns.</summary>
        public static MatrixView<T> ColumnMajor(T* data, int rows, int columns, int leadingDimension)
        {
            return new MatrixView<T>(data, rows, columns, 1, leadingDimension);
        }

        /// <summary>A matrix that is stored row by row, with <paramref name="leadingDimension"/> between rows.</summary>
        public static MatrixView<T> RowMajor(T* data, int rows, int columns, int leadingDimension)
        {
            return new MatrixView<T>(data, rows, columns, leadingDimension, 1);
        }

        public ref T this[int row, int column]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => ref Data[(long)row * RowStride + (long)column * ColumnStride];
        }

        /// <summary>The address of entry (row, column).</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public T* At(int row, int column) => Data + (long)row * RowStride + (long)column * ColumnStride;

        /// <summary>The block of <paramref name="rows"/> by <paramref name="columns"/> entries that starts at (row, column).</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public MatrixView<T> Block(int row, int column, int rows, int columns)
        {
            return new MatrixView<T>(At(row, column), rows, columns, RowStride, ColumnStride);
        }

        /// <summary>The transpose: the same memory, with the rows and columns and their strides swapped.</summary>
        public MatrixView<T> Transpose => new MatrixView<T>(Data, Columns, Rows, ColumnStride, RowStride);

        /// <summary>
        /// Whether each column is a line of entries in memory. A matrix with one row counts, whatever its row stride,
        /// because the kernels never step from its first row to a second one.
        /// </summary>
        public bool ColumnsAreContiguous => RowStride == 1 || Rows <= 1;

        /// <summary>Whether each row is a line of entries in memory. A matrix with one column counts, as above.</summary>
        public bool RowsAreContiguous => ColumnStride == 1 || Columns <= 1;

        /// <summary>
        /// The distance between columns, for a kernel that takes a matrix with contiguous columns and a leading
        /// dimension. A matrix with one column gives its number of rows, so that the value is never smaller than that.
        /// </summary>
        public int LeadingDimension => Columns <= 1 ? Rows : ColumnStride;
    }
}
