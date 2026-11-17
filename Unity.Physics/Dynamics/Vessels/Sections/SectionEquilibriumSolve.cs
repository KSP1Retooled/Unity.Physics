using Unity.Mathematics;
using Unity.Numerics.Kernels;

namespace Unity.Physics.Dynamics.Vessels.Sections
{
    internal static unsafe class SectionEquilibriumSolve
    {
        /// <summary>Solves A x = b in place against a section's factor, as VesselContext.Solve does.</summary>
        public static void Solve(SectionEquilibrium* entry, double* rhs)
        {
            int count = entry->Count;
            int* columnStart = entry->ColumnStart.Ptr;
            int* rows = entry->Rows.Ptr;
            double* blocks = entry->Blocks.Ptr;
            double* inverse = entry->Inverse.Ptr;
            for (int j = 0; j < count; j++)
            {
                double6 y = double6.Load(rhs + 6 * j);
                for (int e = columnStart[j]; e < columnStart[j + 1]; e++)
                {
                    double* to = rhs + 6 * rows[e];
                    (double6.Load(to) - ((double6x6*)(blocks + Block6x6.Size * e))->TransposeMul(y)).Store(to);
                }
            }

            for (int j = count - 1; j >= 0; j--)
            {
                double6 x = ((double6x6*)(inverse + Block6x6.Size * j))->Mul(double6.Load(rhs + 6 * j));
                for (int e = columnStart[j]; e < columnStart[j + 1]; e++)
                {
                    x -= ((double6x6*)(blocks + Block6x6.Size * e))->Mul(double6.Load(rhs + 6 * rows[e]));
                }

                x.Store(rhs + 6 * j);
            }
        }
    }
}
