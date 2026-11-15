using System;
using System.Runtime.InteropServices;
using Unity.Burst;

namespace Unity.Physics
{
    /// <summary>
    /// Exposes each problem completed by the direct solver to managed code, allowing real problems to be saved and
    /// replayed offline. The solver runs in Burst on a worker thread and calls a managed method through a pointer
    /// stored in a SharedStatic. The observer runs on that same thread during the job, so time spent in the observer
    /// adds to the step's duration.
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
        /// Registers <paramref name="callback"/> for subsequent reports, or disables reports if it is null. A pointer
        /// to a managed method becomes invalid when its domain is unloaded. The caller must therefore clear the
        /// callback before an assembly reload.
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
