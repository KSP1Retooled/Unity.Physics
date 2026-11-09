using System;
using AOT;
using Unity.Burst;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using UnityEngine;

namespace KSP.Retooled.Physics
{
    /// <summary>
    /// The plugin API's shape commands, by the id Shape.cpp passes. Attach checks the count against the plugin's. Only what
    /// runs in <see cref="ShapeKernel"/> has a name here that matters; the mesh commands stay in Mesh.cpp.
    /// </summary>
    internal enum ShapeCommand : uint
    {
        Invalid = 0,
        CreateShape,
        DestroyShape,
        GetActorDescriptor,
        GetUserData,
        GetFlags,
        SetFlags,
        GetFilterData,
        SetFilterData,
        SetModifiableContacts,
        SetSupportedMessages,
        SetMaterial,
        GetLocalPose,
        GetPose,
        SetPose,
        SetContactOffset,
        GetWorldBounds,
        GetBoundsAtPose,
        ShouldIgnoreCollision,
        IgnoreCollision,
        CleanupIgnoredColliders,
        CookCollisionMesh,
        CookCollisionMeshStream,
        CreateCollisionMeshFromStream,
        CreateHeightField,
        UpdateHeightFieldRegion,
        DestroyHeightField,
        DestroyCollisionMesh,
        ExtractCollisionMeshData,
        GetGeometryType,
        SetGeometry,
        GetGeometry,
        RemapTriangleIndexToSourceIndex,
        Count,
    }

    /// <summary>
    /// Everything a shape command does that the plugin doesn't answer itself. Shape commands arrive outside a step, mostly on
    /// the main thread: Unity sets poses from job workers too, which read rows as <see cref="BodyEntities"/> left them. As
    /// with the query functions, the plugin holds a Burst function pointer, and until the async compile delivers it is a
    /// stub running the same code in Mono.
    /// </summary>
    [BurstCompile]
    internal static unsafe class ShapeKernel
    {
        [BurstCompile(DisableSafetyChecks = true)]
        [MonoPInvokeCallback(typeof(NativeStore.CommandFunction))]
        internal static void Run(uint command, void* handle, NativeStore.CommandArgs* args, NativeStore.StoreGlobals* globals)
        {
            using Unity.Profiling.ProfilerMarker.AutoScope span = BackendMarkers.ShapeCommand.Auto();

            bool ranInMono = false;
            RunInMono(command, handle, args, globals, ref ranInMono);
            if (!ranInMono)
            {
                Execute((ShapeCommand)command, (NativeStore.Shape*)handle, args, globals);
            }
        }

        /// <summary>Burst drops this, so it only runs in the Mono form, which needs the exception ended here.</summary>
        [BurstDiscard]
        private static void RunInMono(uint command, void* handle, NativeStore.CommandArgs* args, NativeStore.StoreGlobals* globals, ref bool ranInMono)
        {
            ranInMono = true;
            BackendHost.NoteManagedCommand();
            try
            {
                Execute((ShapeCommand)command, (NativeStore.Shape*)handle, args, globals);
            }
            catch (Exception e)
            {
                Debug.LogException(e);
            }
        }

        private static void Execute(ShapeCommand command, NativeStore.Shape* shape, NativeStore.CommandArgs* a, NativeStore.StoreGlobals* globals)
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
                    shape->data.flags = a->flags;
                    TouchBody(shape);
                    break;
                case ShapeCommand.SetFilterData:
                    SetFilter(shape, a);
                    break;
                case ShapeCommand.SetMaterial:
                    SetMaterial(shape, (NativeStore.MaterialData*)a->pointer, globals);
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
                    // The pose replaces the shape's whole world pose, so the shape's own local pose takes no part.
                    ShapeMath.WorldBounds(&shape->data, float3.zero, quaternion.identity, a->position, a->rotation, out float3 lower, out float3 upper);
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
                    HeightFields.Release((NativeStore.HeightField*)a->pointer, globals);
                    break;
            }
        }

        /// <summary>
        /// Unity rewrites the filter of both colliders after every IgnoreCollision, to make PhysX pair them afresh, and contacts
        /// read the ignore id live. Only what is baked into a collider needs a rebuild. PhysX drops a shape's pairs on any new
        /// filter, so a sleeping body wakes to find out what it still touches.
        /// </summary>
        private static void SetFilter(NativeStore.Shape* shape, NativeStore.CommandArgs* a)
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
            }

            if (shape->body != null && (StoreOps.Hot(shape->body)->flags & NativeStore.BodyFlags.Sleeping) != 0)
            {
                JointKernel.WakeRow(StoreOps.Hot(shape->body));
            }
        }

        private static void TouchBody(NativeStore.Shape* shape)
        {
            if (shape->body != null)
            {
                StoreOps.MarkColliderDirty(shape->body);
            }
        }

        private static void SetMaterial(NativeStore.Shape* shape, NativeStore.MaterialData* material, NativeStore.StoreGlobals* globals)
        {
            NativeStore.MaterialData* next = material != null ? material : globals->defaultMaterial;
            StoreOps.RetainMaterial(globals, next);
            StoreOps.ReleaseMaterial(globals, shape->data.material);
            shape->data.material = next;
        }

        private static void StoreGeometry(NativeStore.Shape* shape, NativeStore.CommandArgs* a, NativeStore.StoreGlobals* globals)
        {
            using Unity.Profiling.ProfilerMarker.AutoScope span = BackendMarkers.SetGeometry.Auto();

            void* previousMesh = ShapeMath.IsMeshGeometry(shape->data.geometryType) ? shape->data.mesh : null;
            if (shape->data.geometryType == NativeStore.GeometryHeightField)
            {
                HeightFields.Release(shape, globals);
            }

            shape->data.geometryType = a->mode;
            shape->data.extents = a->vector;
            shape->data.mesh = a->pointer;
            if (ShapeMath.IsMeshGeometry(shape->data.geometryType))
            {
                StoreOps.RetainMesh(globals, shape->data.mesh);
            }
            else if (shape->data.geometryType == NativeStore.GeometryHeightField)
            {
                HeightFields.Retain(shape);
            }

            StoreOps.ReleaseMesh(globals, previousMesh);
        }

        private static NativeStore.Shape* Create(NativeStore.CommandArgs* a, NativeStore.StoreGlobals* globals)
        {
            using Unity.Profiling.ProfilerMarker.AutoScope span = BackendMarkers.CreateShape.Auto();

            NativeStore.Shape* shape = StoreOps.Allocate<NativeStore.Shape>(globals);
            shape->data.owner = shape;
            shape->data.localRotation = quaternion.identity;
            shape->data.flags = a->flags;
            shape->data.userData = a->pointer3;
            SetMaterial(shape, (NativeStore.MaterialData*)a->pointer2, globals);
            StoreGeometry(shape, a, globals);
            return shape;
        }

        private static void Destroy(NativeStore.Shape* shape, NativeStore.StoreGlobals* globals)
        {
            using Unity.Profiling.ProfilerMarker.AutoScope span = BackendMarkers.DestroyShape.Auto();

            if (shape->body != null)
            {
                StoreOps.RemoveShape(shape->body, shape, globals);
            }

            if (ShapeMath.IsMeshGeometry(shape->data.geometryType))
            {
                StoreOps.ReleaseMesh(globals, shape->data.mesh);
            }
            else if (shape->data.geometryType == NativeStore.GeometryHeightField)
            {
                HeightFields.Release(shape, globals);
            }

            StoreOps.ReleaseMaterial(globals, shape->data.material);
            StoreOps.Release(globals, shape);
        }

        /// <summary>
        /// Answers in SetPose's terms: relative to the body, but in world space for a static body's shape. Unity's transform
        /// sync reads this on its job workers and leaves out every collider whose new pose matches it, so an answer in
        /// other terms sends every collider under a moved transform through the main thread's pose writes.
        /// </summary>
        private static void GetPose(NativeStore.Shape* shape, NativeStore.CommandArgs* a)
        {
            if (shape->body == null || (StoreOps.Hot(shape->body)->flags & NativeStore.BodyFlags.Static) == 0)
            {
                a->position = shape->data.localPosition;
                a->rotation = shape->data.localRotation;
                return;
            }

            NativeStore.BodyData* hot = StoreOps.Hot(shape->body);
            a->position = hot->pose->Position + math.rotate(hot->pose->Rotation, shape->data.localPosition);
            a->rotation = math.mul(hot->pose->Rotation, shape->data.localRotation);
        }

        private static void SetPose(NativeStore.Shape* shape, NativeStore.CommandArgs* a)
        {
            float3 position = shape->data.localPosition;
            quaternion rotation = shape->data.localRotation;
            if (shape->body != null && (StoreOps.Hot(shape->body)->flags & NativeStore.BodyFlags.Static) != 0)
            {
                BodyKernel.StoreStaticShapePose(&shape->data, StoreOps.Hot(shape->body), a->position, a->rotation);

                // A floating origin shift moves a static and its shapes together, and taking the body's new pose out of
                // the shape's rounds differently each time. A change that small is rounding, not a move.
                float tolerance = 1e-6f * (math.cmax(math.abs(a->position)) + 1.0f);
                if (math.all(math.abs(shape->data.localPosition - position) <= tolerance)
                    && math.all(math.abs(shape->data.localRotation.value - rotation.value) <= 1e-6f))
                {
                    shape->data.localPosition = position;
                    shape->data.localRotation = rotation;
                }
            }
            else
            {
                shape->data.localPosition = a->position;
                shape->data.localRotation = a->rotation;
            }

            // Unity poses every collider under a transform that moved, and most of them keep their place in their body.
            bool moved = math.any(shape->data.localPosition != position) || math.any(shape->data.localRotation.value != rotation.value);
            if (shape->body == null || !moved)
            {
                return;
            }

            if (ColliderBuilder.PosedShape(StoreOps.Hot(shape->body)) != null)
            {
                StoreOps.MarkPoseChanged(shape->body, StoreOps.Hot(shape->body)->flags);
            }
            else
            {
                StoreOps.MarkCollidersMoved(shape->body);
            }
        }

        private static void GetWorldBounds(NativeStore.Shape* shape, NativeStore.CommandArgs* a)
        {
            float3 position = float3.zero;
            quaternion rotation = quaternion.identity;
            if (shape->body != null)
            {
                NativeStore.BodyData* hot = StoreOps.Hot(shape->body);
                position = hot->pose->Position;
                rotation = hot->pose->Rotation;
            }

            ShapeMath.WorldBounds(&shape->data, shape->data.localPosition, shape->data.localRotation, position, rotation, out float3 lower, out float3 upper);
            WriteBounds(a, lower, upper);
        }

        private static void WriteBounds(NativeStore.CommandArgs* a, float3 lower, float3 upper)
        {
            a->vector = (lower + upper) * 0.5f;
            a->vector2 = (upper - lower) * 0.5f;
        }
    }
}
