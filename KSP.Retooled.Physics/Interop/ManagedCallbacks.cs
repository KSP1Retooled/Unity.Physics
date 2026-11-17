using System;
using System.Runtime.InteropServices;
using Unity.Physics.Store.Queries;
using Unity.Physics.Store.World;

namespace KSP.Retooled.Physics.Interop
{
    [StructLayout(LayoutKind.Sequential)]
    internal unsafe struct ManagedCallbacks
    {
        /// <summary>
        /// Matches Physics::QueryHitResultCollectorCallback. The return value is a C++ bool, so it occupies one byte.
        /// </summary>
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        internal delegate byte HitCollectorFunction(void* context, RayCastHit* hits, ulong count);

        /// <summary>Matches Physics::QueryShapesResultCollectorCallback.</summary>
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        internal delegate byte ShapeCollectorFunction(void* context, void** shapeUserDatas, ulong count);

        /// <summary>Matches Physics::BroadPhaseQueryShapeCollectorCallback.</summary>
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        internal delegate byte BroadphaseCollectorFunction(void* context, void* shape, void* body, void* shapeUserData, ushort geometryType);

        internal delegate void SimulateCallback(WorldData* world, float deltaTime);

        internal delegate int BuildQueryWorldCallback(WorldData* world);

        internal delegate void DestroyWorldCallback(WorldData* world);

        /// <summary>
        /// Burst command entry point. command is the plugin API ID; handle identifies the target object.
        /// </summary>
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        internal delegate void CommandFunction(uint command, void* handle, CommandArgs* args, StoreGlobals* globals);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        internal delegate void* ReallocateFunction(void* memory, ulong size);

        /// <summary>
        /// Represents plugin functions that take one pointer to free memory or to retain or release a record.
        /// </summary>
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        internal delegate void PointerFunction(void* pointer);

        public IntPtr simulate;
        public IntPtr buildQueryWorld;
        public IntPtr query;
        public IntPtr destroyWorld;
        public IntPtr cookMesh;
        public IntPtr validateMesh;
        public IntPtr bodyCommand;
        public IntPtr shapeCommand;
        public IntPtr worldCommand;
        public IntPtr jointCommand;
        public IntPtr vehicleCommand;
        public IntPtr helpRefit;

        /// <summary>Matches ManagedCallbacks::helpRefit: does the work that is left of a pending refit.</summary>
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        internal delegate void RefitFunction(void* refit);
    }
}
