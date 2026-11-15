using System;
using System.Runtime.InteropServices;
using Unity.Burst;
using Unity.Profiling.LowLevel.Unsafe;

namespace Unity.Physics
{
    /// <summary>
    /// Allows a thread to block while waiting for a 32-bit word to change. Uses WaitOnAddress on Windows, a private
    /// futex on Linux, or __ulock_wait on macOS. A blocked thread releases its core for other work, whereas a spinning
    /// thread keeps using it. Waking takes a few microseconds. Waits can return without a change to the word, so
    /// callers must check it again.
    ///
    /// Burst calls these system functions through a native plugin. A backend with such a plugin installs the functions
    /// here. Until then, <see cref="Available"/> is false and callers spin instead.
    /// </summary>
    internal static unsafe class AddressWait
    {
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate void WaitFunction(int* word, int expected, int timeoutMicroseconds);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate void WakeFunction(int* word, int all);

        /// <summary>The timeout passed to the native wait before the caller checks the word again.</summary>
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

        /// <summary>Installs native wait and wake functions. Disables them if either pointer is null.</summary>
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

        /// <summary>
        /// If native functions are installed, waits while the word equals <paramref name="expected"/> , with a timeout
        /// of about one millisecond. Otherwise, returns immediately.
        /// </summary>
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
        /// Converts microseconds to <see cref="ProfilerUnsafeUtility.Timestamp"/> ticks. Reading the conversion ratio
        /// is unavailable in Burst, so the main thread computes these durations and passes them to jobs.
        /// </summary>
        public static long TicksFromMicroseconds(long microseconds)
        {
            ProfilerUnsafeUtility.TimestampConversionRatio ratio = ProfilerUnsafeUtility.TimestampToNanosecondsConversionRatio;
            return microseconds * 1000 * ratio.Denominator / ratio.Numerator;
        }
    }
}
