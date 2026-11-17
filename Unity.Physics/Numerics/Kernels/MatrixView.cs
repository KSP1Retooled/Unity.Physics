// Based on faer 0.19.4 MatRef and MatMut (src/mat/matref.rs).
// faer: https://github.com/sarah-quinones/faer-rs, MIT license, see Third Party Notices.md at the package root.

using System.Runtime.CompilerServices;

namespace Unity.Numerics.Kernels
{
    /// <summary>
    /// A view of a dense matrix with arbitrary strides, including negative strides.
    /// Entry (i, j) is at <c>Data[i * RowStride + j * ColumnStride]</c>. Strides are measured in elements.
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

        /// <summary>Create a column-major view, with leadingDimension elements between columns.</summary>
        public static MatrixView<T> ColumnMajor(T* data, int rows, int columns, int leadingDimension)
        {
            return new MatrixView<T>(data, rows, columns, 1, leadingDimension);
        }

        /// <summary>Create a row-major view, with leadingDimension elements between rows.</summary>
        public static MatrixView<T> RowMajor(T* data, int rows, int columns, int leadingDimension)
        {
            return new MatrixView<T>(data, rows, columns, leadingDimension, 1);
        }

        public ref T this[int row, int column]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => ref Data[(long)row * RowStride + (long)column * ColumnStride];
        }

        /// <summary>Address of entry (row, column).</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public T* At(int row, int column) => Data + (long)row * RowStride + (long)column * ColumnStride;

        /// <summary>View of a rows by columns block starting at (row, column). Shares the original storage.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public MatrixView<T> Block(int row, int column, int rows, int columns)
        {
            return new MatrixView<T>(At(row, column), rows, columns, RowStride, ColumnStride);
        }

        /// <summary>Transposed view of the same storage.</summary>
        public MatrixView<T> Transpose => new MatrixView<T>(Data, Columns, Rows, ColumnStride, RowStride);

        /// <summary>True if entries within each column are contiguous. A single row is always contiguous.</summary>
        public bool ColumnsAreContiguous => RowStride == 1 || Rows <= 1;

        /// <summary>True if entries within each row are contiguous. A single column is always contiguous.</summary>
        public bool RowsAreContiguous => ColumnStride == 1 || Columns <= 1;

        /// <summary>
        /// Stride between columns, for use with contiguous columns.
        /// For a single column, return the number of rows.
        /// </summary>
        public int LeadingDimension => Columns <= 1 ? Rows : ColumnStride;
    }
}
