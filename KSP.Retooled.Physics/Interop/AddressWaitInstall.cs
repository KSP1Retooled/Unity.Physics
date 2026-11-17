using System;
using Unity.Physics.Base.Threading;

namespace KSP.Retooled.Physics.Interop
{
    internal static unsafe class AddressWaitInstall
    {
        /// <summary>
        /// Install native address wait/wake functions. Without them, solver workers spin while waiting.
        /// </summary>
        internal static void InstallAddressWait()
        {
            IntPtr wait = IntPtr.Zero;
            IntPtr wake = IntPtr.Zero;
            try
            {
                NativeLibrary.RetooledPhysics_WaitFunctions(&wait, &wake);
            }
            catch (EntryPointNotFoundException)
            {
                return;
            }

            AddressWait.Install(wait, wake);
        }
    }
}
