using System;
using System.Runtime.InteropServices;
using Unity.Burst;
using Unity.Profiling.LowLevel.Unsafe;

namespace Unity.Physics
{
    /// <summary>
    /// Wait for a 32-bit word to change using native address waits. Callers must recheck the word after waking.
    /// Use installed plugin functions when available; otherwise callers spin.
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
        /// Convert microseconds to profiler timestamp ticks on the main thread, since Burst cannot read the
        /// conversion ratio.
        /// </summary>
        public static long TicksFromMicroseconds(long microseconds)
        {
            ProfilerUnsafeUtility.TimestampConversionRatio ratio = ProfilerUnsafeUtility.TimestampToNanosecondsConversionRatio;
            return microseconds * 1000 * ratio.Denominator / ratio.Numerator;
        }
    }
}
