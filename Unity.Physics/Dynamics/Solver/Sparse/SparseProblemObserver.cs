using System;
using System.Runtime.InteropServices;
using Unity.Burst;

namespace Unity.Physics.Dynamics.Solver.Sparse
{
    /// <summary>
    /// Report completed direct-solver problems to managed observers for capture and replay.
    /// Callbacks run on the solving worker inside the job, adding to step time.
    /// </summary>
    internal static unsafe class SparseProblemObserver
    {
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate void Callback(SparseConstraintSolver.Problem* problem, float* impulses, byte bounded);

        struct SlotKey
        {
        }

        static readonly SharedStatic<IntPtr> Slot = SharedStatic<IntPtr>.GetOrCreate<SlotKey>();

        // Keep a managed reference to the delegate. The function pointer alone does not prevent garbage collection.
        static Callback current;

        /// <summary>
        /// Register an observer, or disable reporting with null.
        /// Clear the callback before domain unload invalidates its managed function pointer.
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
