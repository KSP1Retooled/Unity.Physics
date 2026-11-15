using System;
using System.Runtime.InteropServices;
using Unity.Burst;
using Unity.Profiling.LowLevel.Unsafe;

namespace Unity.Physics
{
    /// <summary>
    /// Blocks a thread until another changes a 32-bit word, through the system's address wait: WaitOnAddress on Windows, a
    /// private futex on Linux, __ulock_wait on macOS. A thread waiting this way gives its core back to the system, where a
    /// spinning one keeps it, and it wakes in a few microseconds. Waits may return early for no reason, so callers check the
    /// word again.
    ///
    /// Burst reaches the system only through a native plugin, so the backend that has one installs its functions here. Until
    /// then <see cref="Available"/> is false and callers spin instead.
    /// </summary>
    internal static unsafe class AddressWait
    {
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate void WaitFunction(int* word, int expected, int timeoutMicroseconds);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate void WakeFunction(int* word, int all);

        /// <summary>The longest a wait lasts before its caller looks again on its own.</summary>
        const int TimeoutMicroseconds = 1000;

        struct Functions
        {
            public FunctionPointer<WaitFunction> Wait;
            public FunctionPointer<WakeFunction> Wake;
            public int Installed;
        }

        struct FunctionsKey
        {
        }

        static readonly SharedStatic<Functions> s_Functions = SharedStatic<Functions>.GetOrCreate<FunctionsKey>();

        public static bool Available => s_Functions.Data.Installed != 0;

        /// <summary>Installs the native wait and wake, or with null pointers removes them.</summary>
        public static void Install(IntPtr wait, IntPtr wake)
        {
            ref Functions functions = ref s_Functions.Data;
            functions.Installed = 0;
            if (wait == IntPtr.Zero || wake == IntPtr.Zero)
            {
                return;
            }

            functions.Wait = new FunctionPointer<WaitFunction>(wait);
            functions.Wake = new FunctionPointer<WakeFunction>(wake);
            functions.Installed = 1;
        }

        /// <summary>Waits while the word still holds <paramref name="expected"/>, for about a millisecond at most.</summary>
        public static void Wait(int* word, int expected)
        {
            if (Available)
            {
                s_Functions.Data.Wait.Invoke(word, expected, TimeoutMicroseconds);
            }
        }

        public static void WakeOne(int* word)
        {
            if (Available)
            {
                s_Functions.Data.Wake.Invoke(word, 0);
            }
        }

        public static void WakeAll(int* word)
        {
            if (Available)
            {
                s_Functions.Data.Wake.Invoke(word, 1);
            }
        }

        /// <summary>
        /// How many <see cref="ProfilerUnsafeUtility.Timestamp"/> ticks make up the given microseconds. Reads the conversion,
        /// which Burst can't, so the main thread works these out and hands them to jobs.
        /// </summary>
        public static long TicksFromMicroseconds(long microseconds)
        {
            ProfilerUnsafeUtility.TimestampConversionRatio ratio = ProfilerUnsafeUtility.TimestampToNanosecondsConversionRatio;
            return microseconds * 1000 * ratio.Denominator / ratio.Numerator;
        }
    }
}
