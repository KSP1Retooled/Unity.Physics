using System.Runtime.InteropServices;
using System;
using Unity.Physics.Store.World;

namespace KSP.Retooled.Physics.Interop
{
    internal static unsafe class NativeLibrary
    {
        internal const string Library = "RetooledPhysics";

        [DllImport(Library)]
        internal static extern int RetooledPhysics_StructSize(int id);

        [DllImport(Library)]
        internal static extern void RetooledPhysics_Attach(ManagedCallbacks* callbacks);

        [DllImport(Library)]
        internal static extern void RetooledPhysics_Detach();

        [DllImport(Library)]
        internal static extern StoreGlobals* RetooledPhysics_Globals();

        [DllImport(Library)]
        internal static extern int RetooledPhysics_WorldCount();

        [DllImport(Library)]
        internal static extern WorldData* RetooledPhysics_GetWorld(int index);

        [DllImport(Library)]
        internal static extern void RetooledPhysics_SyncQueryWorlds();

        /// <summary>
        /// Returns a non-zero value on the thread that attached the plugin, which is the main thread.
        /// </summary>
        [DllImport(Library)]
        internal static extern int RetooledPhysics_OnMainThread();

        /// <summary>
        /// Get native address wait/wake functions, if the plugin provides them.
        /// </summary>
        [DllImport(Library)]
        internal static extern void RetooledPhysics_WaitFunctions(IntPtr* wait, IntPtr* wake);
    }
}
