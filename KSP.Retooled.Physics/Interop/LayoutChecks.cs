using Unity.Collections.LowLevel.Unsafe;
using Unity.Transforms;
using KSP.Retooled.Physics.Entities;
using Unity.Physics.Store.Bodies;
using Unity.Physics.Store.Joints;
using Unity.Physics.Store.Queries;
using Unity.Physics.Store.Shapes;
using Unity.Physics.Store.World;

namespace KSP.Retooled.Physics.Interop
{
    internal static unsafe class LayoutChecks
    {
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
                int native = NativeLibrary.RetooledPhysics_StructSize(i);
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

            StoreGlobals* globals = NativeLibrary.RetooledPhysics_Globals();
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
