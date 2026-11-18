using Unity.Profiling;
using KSP.Retooled.Physics.Entities;
using KSP.Retooled.Physics.Host;
using KSP.Retooled.Physics.Terrain;
using KSP.Retooled.Physics.Interop;
using System;
using AOT;
using Unity.Burst;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using UnityEngine;
using Unity.Physics.Store.Bodies;
using Unity.Physics.Store.Memory;
using Unity.Physics.Store.Shapes;
using Unity.Physics.Store.Versions;
using Unity.Physics.Store.World;
using Unity.Physics.Stepping;
using Unity.Physics.Collision.Colliders.Building;
using Unity.Physics.Collision.Filter.Groups;
using Unity.Physics.Dynamics.Bodies;

namespace KSP.Retooled.Physics.Commands
{
    /// <summary>
    /// Handle shape commands outside simulation. Unity may also set poses from workers using existing body row
    /// pointers.
    /// </summary>
    [BurstCompile]
    internal static unsafe class ShapeCommands
    {
        private static readonly ProfilerMarker ShapeCommandMarker = new ProfilerMarker(ProfilerCategory.Physics, "Physics.Commands.Shape");
        private static readonly ProfilerMarker CreateShapeMarker = new ProfilerMarker(ProfilerCategory.Physics, "Physics.Commands.CreateShape");
        private static readonly ProfilerMarker DestroyShapeMarker = new ProfilerMarker(ProfilerCategory.Physics, "Physics.Commands.DestroyShape");
        private static readonly ProfilerMarker SetGeometryMarker = new ProfilerMarker(ProfilerCategory.Physics, "Physics.Commands.SetGeometry");

        [BurstCompile(DisableSafetyChecks = true)]
        [MonoPInvokeCallback(typeof(ManagedCallbacks.CommandFunction))]
        internal static void Run(uint command, void* handle, CommandArgs* args, StoreGlobals* globals)
        {
            // A span costs about as much as a read, so reads run without the category's span, as in BodyCommands.
            if (IsRead((ShapeCommand)command))
            {
                Dispatch(command, handle, args, globals);
                return;
            }

            using ProfilerMarker.AutoScope span = ShapeCommandMarker.Auto();
            Dispatch(command, handle, args, globals);
        }

        /// <summary>Whether the command only reads the shape. Reads run without the category's span.</summary>
        internal static bool IsRead(ShapeCommand command)
        {
            switch (command)
            {
                case ShapeCommand.GetActorDescriptor:
                case ShapeCommand.GetUserData:
                case ShapeCommand.GetFlags:
                case ShapeCommand.GetFilterData:
                case ShapeCommand.GetLocalPose:
                case ShapeCommand.GetPose:
                case ShapeCommand.GetWorldBounds:
                case ShapeCommand.GetBoundsAtPose:
                case ShapeCommand.ShouldIgnoreCollision:
                case ShapeCommand.GetGeometryType:
                case ShapeCommand.GetGeometry:
                    return true;
                default:
                    return false;
            }
        }

        private static void Dispatch(uint command, void* handle, CommandArgs* args, StoreGlobals* globals)
        {
            bool ranInMono = false;
            RunInMono(command, handle, args, globals, ref ranInMono);
            if (!ranInMono)
            {
                Execute((ShapeCommand)command, (Shape*)handle, args, globals);
            }
        }

        /// <summary>
        /// Burst excludes this method, so it runs only in Mono. Catch exceptions here so that they do not leave the
        /// managed callback.
        /// </summary>
        [BurstDiscard]
        private static void RunInMono(uint command, void* handle, CommandArgs* args, StoreGlobals* globals, ref bool ranInMono)
        {
            ranInMono = true;
            BackendCounters.NoteManagedCommand();
            try
            {
                Execute((ShapeCommand)command, (Shape*)handle, args, globals);
            }
            catch (Exception e)
            {
                Debug.LogException(e);
            }
        }

        private static void Execute(ShapeCommand command, Shape* shape, CommandArgs* a, StoreGlobals* globals)
        {
            switch (command)
            {
                case ShapeCommand.CreateShape:
                    a->pointer = Create(a, globals);
                    break;
                case ShapeCommand.DestroyShape:
                    Destroy(shape, globals);
                    break;
                case ShapeCommand.GetUserData:
                    a->pointer = shape->data.userData;
                    break;
                case ShapeCommand.GetFlags:
                    a->flags = shape->data.flags;
                    break;
                case ShapeCommand.SetFlags:
                    if (((shape->data.flags ^ a->flags) & StoreConstants.ShapeFlagTrigger) != 0)
                    {
                        PairFilterVersion.Changed();
                    }

                    shape->data.flags = a->flags;
                    TouchBody(shape);
                    break;
                case ShapeCommand.SetFilterData:
                    SetFilter(shape, a);
                    break;
                case ShapeCommand.SetMaterial:
                    SetMaterial(shape, (MaterialData*)a->pointer, globals);
                    TouchBody(shape);
                    break;
                case ShapeCommand.GetLocalPose:
                    a->position = shape->data.localPosition;
                    a->rotation = shape->data.localRotation;
                    break;
                case ShapeCommand.GetPose:
                    GetPose(shape, a);
                    break;
                case ShapeCommand.SetPose:
                    SetPose(shape, a);
                    break;
                case ShapeCommand.SetContactOffset:
                    shape->contactOffset = a->scalar;
                    break;
                case ShapeCommand.GetWorldBounds:
                    GetWorldBounds(shape, a);
                    break;
                case ShapeCommand.GetBoundsAtPose:
                    // The supplied pose replaces the shape's entire world pose, so ignore the shape's local pose.
                    ShapeMassProperties.WorldBounds(&shape->data, float3.zero, quaternion.identity, a->position, a->rotation, out float3 lower, out float3 upper);
                    WriteBounds(a, lower, upper);
                    break;
                case ShapeCommand.ShouldIgnoreCollision:
                    a->enabled = IgnoreTable.Contains(globals, a->value, a->value2) ? 1u : 0u;
                    break;
                case ShapeCommand.IgnoreCollision:
                    IgnoreTable.Set(globals, ref a->value, ref a->value2, a->enabled != 0);
                    break;
                case ShapeCommand.CleanupIgnoredColliders:
                    IgnoreTable.Forget(globals, a->value);
                    break;
                case ShapeCommand.GetGeometryType:
                    a->mode = shape->data.geometryType;
                    break;
                case ShapeCommand.SetGeometry:
                    StoreGeometry(shape, a, globals);
                    TouchBody(shape);
                    break;
                case ShapeCommand.GetGeometry:
                    a->mode = shape->data.geometryType;
                    a->vector = shape->data.extents;
                    a->pointer = shape->data.mesh;
                    break;
                case ShapeCommand.CreateHeightField:
                    a->pointer = HeightFields.Create(a, globals);
                    break;
                case ShapeCommand.UpdateHeightFieldRegion:
                    HeightFields.Update(a, globals);
                    break;
                case ShapeCommand.DestroyHeightField:
                    HeightFields.Release((HeightField*)a->pointer, globals);
                    break;
            }
        }

        /// <summary>
        /// Apply shape filter changes and wake the body to reevaluate touching pairs.
        /// Rebuild only when filter data baked into the collider changes.
        /// </summary>
        private static void SetFilter(Shape* shape, CommandArgs* a)
        {
            bool baked = shape->data.belongsTo != a->value || shape->data.collidesWith != a->value2 || shape->data.collisionDisabled != a->enabled;
            shape->data.belongsTo = a->value;
            shape->data.collidesWith = a->value2;
            shape->data.ignorePairId = a->mode;
            shape->data.collisionDisabled = a->enabled;
            shape->messages = (uint)a->size;
            if (baked)
            {
                TouchBody(shape);
                PairFilterVersion.Changed();
            }

            if (shape->body != null && (StoreRows.Hot(shape->body)->flags & BodyFlags.Sleeping) != 0)
            {
                BodyKinematics.Wake(StoreRows.Hot(shape->body));
            }
        }

        private static void TouchBody(Shape* shape)
        {
            if (shape->body != null)
            {
                CommandStore.MarkColliderDirty(shape->body);
            }
        }

        private static void SetMaterial(Shape* shape, MaterialData* material, StoreGlobals* globals)
        {
            MaterialData* next = material != null ? material : globals->defaultMaterial;
            StoreMemory.RetainMaterial(globals, next);
            StoreMemory.ReleaseMaterial(globals, shape->data.material);
            shape->data.material = next;
        }

        private static void StoreGeometry(Shape* shape, CommandArgs* a, StoreGlobals* globals)
        {
            using ProfilerMarker.AutoScope span = SetGeometryMarker.Auto();

            void* previousMesh = ShapeMassProperties.IsMeshGeometry(shape->data.geometryType) ? shape->data.mesh : null;
            if (shape->data.geometryType == StoreConstants.GeometryHeightField)
            {
                HeightFields.Release(shape, globals);
            }

            shape->data.geometryType = a->mode;
            shape->data.extents = a->vector;
            shape->data.mesh = a->pointer;
            if (ShapeMassProperties.IsMeshGeometry(shape->data.geometryType))
            {
                StoreMemory.RetainMesh(globals, shape->data.mesh);
            }
            else if (shape->data.geometryType == StoreConstants.GeometryHeightField)
            {
                HeightFields.Retain(shape);
            }

            StoreMemory.ReleaseMesh(globals, previousMesh);
        }

        private static Shape* Create(CommandArgs* a, StoreGlobals* globals)
        {
            using ProfilerMarker.AutoScope span = CreateShapeMarker.Auto();

            Shape* shape = StoreMemory.Allocate<Shape>(globals);
            shape->data.owner = shape;
            shape->data.localRotation = quaternion.identity;
            shape->data.flags = a->flags;
            shape->data.userData = a->pointer3;
            SetMaterial(shape, (MaterialData*)a->pointer2, globals);
            StoreGeometry(shape, a, globals);
            GroupTable.AddShape(globals, shape);
            return shape;
        }

        private static void Destroy(Shape* shape, StoreGlobals* globals)
        {
            using ProfilerMarker.AutoScope span = DestroyShapeMarker.Auto();

            GroupTable.RemoveShape(globals, shape);
            if (shape->body != null)
            {
                CommandStore.RemoveShape(shape->body, shape, globals);
            }

            if (ShapeMassProperties.IsMeshGeometry(shape->data.geometryType))
            {
                StoreMemory.ReleaseMesh(globals, shape->data.mesh);
            }
            else if (shape->data.geometryType == StoreConstants.GeometryHeightField)
            {
                HeightFields.Release(shape, globals);
            }

            StoreMemory.ReleaseMaterial(globals, shape->data.material);
            StoreMemory.Release(globals, shape);
        }

        /// <summary>
        /// Return the shape pose in SetPose coordinates: world space for static shapes, body-local space otherwise.
        /// </summary>
        private static void GetPose(Shape* shape, CommandArgs* a)
        {
            if (shape->body == null || (StoreRows.Hot(shape->body)->flags & BodyFlags.Static) == 0)
            {
                a->position = shape->data.localPosition;
                a->rotation = shape->data.localRotation;
                return;
            }

            BodyData* hot = StoreRows.Hot(shape->body);
            a->position = hot->pose->Position + math.rotate(hot->pose->Rotation, shape->data.localPosition);
            a->rotation = math.mul(hot->pose->Rotation, shape->data.localRotation);
        }

        /// <summary>
        /// Ignore small rounding differences and quaternion sign changes in body-local shape poses to avoid
        /// unnecessary collider rebuilds.
        /// </summary>
        private static bool IsRounding(float3 position, quaternion rotation, float3 newPosition, quaternion newRotation)
        {
            const float PositionTolerance = 1e-5f;
            const float RotationTolerance = 1e-5f;

            float tolerance = PositionTolerance * (math.cmax(math.abs(newPosition)) + 1.0f);
            if (math.any(math.abs(newPosition - position) > tolerance))
            {
                return false;
            }

            float4 change = math.dot(rotation.value, newRotation.value) >= 0.0f
                ? newRotation.value - rotation.value
                : newRotation.value + rotation.value;
            return math.all(math.abs(change) <= RotationTolerance);
        }

        private static void SetPose(Shape* shape, CommandArgs* a)
        {
            float3 position = shape->data.localPosition;
            quaternion rotation = shape->data.localRotation;
            if (shape->body != null && (StoreRows.Hot(shape->body)->flags & BodyFlags.Static) != 0)
            {
                BodyCommands.StoreStaticShapePose(&shape->data, StoreRows.Hot(shape->body), a->position, a->rotation);

                // Ignore rounding from converting static shape poses after a floating origin shift.
                float tolerance = 1e-6f * (math.cmax(math.abs(a->position)) + 1.0f);
                if (math.all(math.abs(shape->data.localPosition - position) <= tolerance)
                    && math.all(math.abs(shape->data.localRotation.value - rotation.value) <= 1e-6f))
                {
                    shape->data.localPosition = position;
                    shape->data.localRotation = rotation;
                }
            }
            else if (!IsRounding(position, rotation, a->position, a->rotation))
            {
                shape->data.localPosition = a->position;
                shape->data.localRotation = a->rotation;
            }

            // Unity sets the pose of every collider under a moved transform. Most keep the same pose relative to their
            // body.
            bool moved = math.any(shape->data.localPosition != position) || math.any(shape->data.localRotation.value != rotation.value);
            if (shape->body == null || !moved)
            {
                return;
            }

            // Only a static body can use PosedShape. Its built body uses the shape's pose as its own pose.
            if (ColliderBuilder.PosedShape(StoreRows.Hot(shape->body)) != null)
            {
                CommandStore.MarkStaticMoved(shape->body);
            }
            else
            {
                CommandStore.MarkCollidersMoved(shape->body);
            }
        }

        private static void GetWorldBounds(Shape* shape, CommandArgs* a)
        {
            float3 position = float3.zero;
            quaternion rotation = quaternion.identity;
            if (shape->body != null)
            {
                BodyData* hot = StoreRows.Hot(shape->body);
                position = hot->pose->Position;
                rotation = hot->pose->Rotation;
            }

            ShapeMassProperties.WorldBounds(&shape->data, shape->data.localPosition, shape->data.localRotation, position, rotation, out float3 lower, out float3 upper);
            WriteBounds(a, lower, upper);
        }

        private static void WriteBounds(CommandArgs* a, float3 lower, float3 upper)
        {
            a->vector = (lower + upper) * 0.5f;
            a->vector2 = (upper - lower) * 0.5f;
        }
    }
}
