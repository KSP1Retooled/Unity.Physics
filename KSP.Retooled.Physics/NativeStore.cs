using System;
using System.Runtime.InteropServices;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Physics.Base.Threading;
using Unity.Transforms;
using KSP.Retooled.Physics.Entities;
using Unity.Physics.Store.Bodies;
using Unity.Physics.Store.Joints;
using Unity.Physics.Store.Queries;
using Unity.Physics.Store.Shapes;
using Unity.Physics.Store.World;

namespace KSP.Retooled.Physics
{
    /// <summary>
    /// Entry points for the native store and checks for the layout shared with the plugin.
    /// </summary>
    internal static unsafe class NativeStore
    {
        internal const string Library = "RetooledPhysics";

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

        [StructLayout(LayoutKind.Sequential)]
        internal struct ManagedCallbacks
        {
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
        }

        /// <summary>Matches ManagedCallbacks::helpRefit: does the work that is left of a pending refit.</summary>
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        internal delegate void RefitFunction(void* refit);

        [DllImport(Library)]
        private static extern int RetooledPhysics_StructSize(int id);

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
        private static extern void RetooledPhysics_WaitFunctions(IntPtr* wait, IntPtr* wake);

        /// <summary>
        /// Install native address wait/wake functions. Without them, solver workers spin while waiting.
        /// </summary>
        internal static void InstallAddressWait()
        {
            IntPtr wait = IntPtr.Zero;
            IntPtr wake = IntPtr.Zero;
            try
            {
                RetooledPhysics_WaitFunctions(&wait, &wake);
            }
            catch (EntryPointNotFoundException)
            {
                return;
            }

            AddressWait.Install(wait, wake);
        }

        /// <summary>
        /// Return the first struct size, command count or stored layout mismatch that prevents attach, or null.
        /// </summary>
        internal static string FirstSizeMismatch()
        {
            int homeAlignment = UnsafeUtility.AlignOf<BodyHome>();
            int blockAlignment = UnsafeUtility.AlignOf<BodyHomeBlock>();
            if (sizeof(BodyHome) != sizeof(BodyHomeBlock) || homeAlignment != blockAlignment)
            {
                return $"BodyHome is {sizeof(BodyHome)} bytes with alignment {homeAlignment}. "
                    + $"BodyHomeBlock is {sizeof(BodyHomeBlock)} bytes with alignment {blockAlignment}.";
            }

            int[] sizes =
            {
                UnsafeUtility.GetFieldOffset(typeof(BodyData).GetField(nameof(BodyData.velocity))),
                sizeof(ShapeData),
                sizeof(MaterialData),
                sizeof(WorldData),
                sizeof(StoreGlobals),
                sizeof(QueryInput),
                sizeof(RayCastHit),
                sizeof(ManagedCallbacks),
                sizeof(MeshData),
                sizeof(CookInput),
                sizeof(CookOutput),
                UnsafeUtility.GetFieldOffset(typeof(Body).GetField(nameof(Body.home))),
                sizeof(Shape),
                sizeof(CommandArgs),
                (int)BodyCommand.Count,
                (int)ShapeCommand.Count,
                (int)WorldCommand.Count,
                sizeof(JointBreak),
                (int)JointCommand.Count,
                sizeof(HeightField),
                (int)VehicleCommand.Count,
            };

            string[] names =
            {
                nameof(BodyData),
                nameof(ShapeData),
                nameof(MaterialData),
                nameof(WorldData),
                nameof(StoreGlobals),
                nameof(QueryInput),
                nameof(RayCastHit),
                nameof(ManagedCallbacks),
                nameof(MeshData),
                nameof(CookInput),
                nameof(CookOutput),
                nameof(Body),
                nameof(Shape),
                nameof(CommandArgs),
                nameof(BodyCommand),
                nameof(ShapeCommand),
                nameof(WorldCommand),
                nameof(JointBreak),
                nameof(JointCommand),
                nameof(HeightField),
                nameof(VehicleCommand),
            };

            for (int i = 0; i < sizes.Length; i++)
            {
                int native = RetooledPhysics_StructSize(i);
                if (native != sizes[i])
                {
                    return $"{names[i]} is {sizes[i]} here and {native} in the plugin. Rebuild the plugin with `RetooledCli native build`";
                }
            }

            // The plugin reads a pose as BodyPose in Store.h: the position, the scale and the rotation.
            if (sizeof(LocalTransform) != 32 || UnsafeUtility.GetFieldOffset(typeof(LocalTransform).GetField(nameof(LocalTransform.Rotation))) != 16)
            {
                return $"{nameof(LocalTransform)} doesn't have the layout of BodyPose in the plugin";
            }

            StoreGlobals* globals = RetooledPhysics_Globals();
            if (globals->jointDataSize != 0 && globals->jointDataSize != sizeof(JointData))
            {
                return $"{nameof(JointData)} is {sizeof(JointData)} here and {globals->jointDataSize} in the store's joints. Restart Unity";
            }

            if (globals->bodyDataSize != 0 && (globals->bodyDataSize != sizeof(BodyData) || globals->bodySize != sizeof(Body)))
            {
                return $"Body records are {sizeof(BodyData)} and {sizeof(Body)} bytes here and {globals->bodyDataSize} and {globals->bodySize} in the store. Restart Unity";
            }

            globals->jointDataSize = (uint)sizeof(JointData);
            globals->bodyDataSize = (uint)sizeof(BodyData);
            globals->bodySize = (uint)sizeof(Body);
            return null;
        }
    }
}
