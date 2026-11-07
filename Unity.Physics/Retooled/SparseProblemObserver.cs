using System;
using System.Runtime.InteropServices;
using Unity.Burst;

namespace Unity.Physics
{
    /// <summary>
    /// Shows each problem the direct solver finishes to managed code, so real ones can be saved and replayed offline. The solve
    /// runs in Burst on a worker thread, which reaches a managed method through a pointer in a SharedStatic. The observer runs
    /// on that thread and in the middle of the job, so anything slow it does shows up in the step.
    /// </summary>
    internal static unsafe class SparseProblemObserver
    {
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate void Callback(SparseConstraintSolver.Problem* problem, float* impulses, byte bounded);

        struct SlotKey
        {
        }

        static readonly SharedStatic<IntPtr> Slot = SharedStatic<IntPtr>.GetOrCreate<SlotKey>();

        // The pointer holds no reference to its delegate, which the collector would otherwise take.
        static Callback current;

        /// <summary>
        /// Starts calling <paramref name="callback"/>, or stops with null. A managed pointer dies with its domain, so whoever
        /// sets one has to clear it before an assembly reload.
        /// </summary>
        public static void Set(Callback callback)
        {
            current = callback;
            Slot.Data = callback == null ? IntPtr.Zero : Marshal.GetFunctionPointerForDelegate(callback);
        }

        public static void Report(SparseConstraintSolver.Problem* problem, float* impulses, bool bounded)
        {
            IntPtr pointer = Slot.Data;
            if (pointer != IntPtr.Zero)
            {
                new FunctionPointer<Callback>(pointer).Invoke(problem, impulses, bounded ? (byte)1 : (byte)0);
            }
        }
    }
}
