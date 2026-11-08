using System;
using System.Runtime.InteropServices;
using AOT;
using Unity.Burst;
using Unity.Mathematics;
using UnityEngine;

namespace KSP.Retooled.Physics
{
    /// <summary>
    /// The plugin API's world commands, by the id World.cpp passes. Attach checks the count against the plugin's. Those a world
    /// needs before attaching stay in World.cpp, so only some of them run in <see cref="WorldKernel"/>.
    /// </summary>
    internal enum WorldCommand : uint
    {
        Invalid = 0,
        CreateWorld,
        DestroyWorld,
        SetGravity,
        GetGravity,
        SetBounceThreshold,
        GetBounceThreshold,
        GetBasicStats,
        SetScratchBufferChunkCount,
        Simulate,
        GetLastSimulationStepResults,
        ReleaseLastSimulationStepData,
        GetLastSimulationStats,
        GetUserData,
        CreateBody,
        AddActorsToScene,
        DestroyBody,
        CreateArticulation,
        DestroyArticulation,
        CreateJoint,
        DestroyJoint,
        GetArticulationCount,
        GetArticulations,
        GetAllShapesByLayerMask,
        Count,
    }

    /// <summary>
    /// What a world's dense body array needs done: creating and destroying bodies, and walking them. World commands arrive on
    /// the main thread outside a step. As with the query functions, the plugin holds a Burst function pointer, and until the
    /// async compile delivers it is a stub running the same code in Mono.
    /// </summary>
    [BurstCompile]
    internal static unsafe class WorldKernel
    {
        // Rigidbody.maxLinearVelocity's default; Unity sends the rest of a body's limits after creating it.
        private const float DefaultMaxLinearVelocity = 1e16f;
        private const float DefaultMaxAngularVelocity = 7.0f;
        private const float DefaultMaxDepenetrationVelocity = 1e32f;
        private const float DefaultSleepThreshold = 0.005f;

        /// <summary>The head of Physics::SimulationStepStats. World.cpp checks the offsets.</summary>
        [StructLayout(LayoutKind.Sequential)]
        internal struct BodyCounts
        {
            public uint DynamicBodies;
            public uint ActiveDynamicBodies;
            public uint KinematicBodies;
            public uint ActiveKinematicBodies;
            public uint StaticBodies;
        }

        [BurstCompile(DisableSafetyChecks = true)]
        [MonoPInvokeCallback(typeof(NativeStore.CommandFunction))]
        internal static void Run(uint command, void* handle, NativeStore.CommandArgs* args, NativeStore.StoreGlobals* globals)
        {
            using Unity.Profiling.ProfilerMarker.AutoScope span = BackendMarkers.WorldCommand.Auto();

            bool ranInMono = false;
            RunInMono(command, handle, args, globals, ref ranInMono);
            if (!ranInMono)
            {
                Execute((WorldCommand)command, (NativeStore.WorldData*)handle, args, globals);
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
                Execute((WorldCommand)command, (NativeStore.WorldData*)handle, args, globals);
            }
            catch (Exception e)
            {
                Debug.LogException(e);
            }
        }

        private static void Execute(WorldCommand command, NativeStore.WorldData* world, NativeStore.CommandArgs* a, NativeStore.StoreGlobals* globals)
        {
            switch (command)
            {
                case WorldCommand.GetBasicStats:
                    a->value = world->bodyCount;
                    break;
                case WorldCommand.GetLastSimulationStats:
                    CountBodies(world, (BodyCounts*)a->pointer);
                    break;
                case WorldCommand.CreateBody:
                    a->pointer = CreateBody(world, a, globals);
                    break;
                case WorldCommand.DestroyBody:
                    DestroyBody(world, (NativeStore.Body*)a->pointer, globals);
                    break;
                case WorldCommand.CreateJoint:
                    a->pointer = JointKernel.Create(world, a, globals);
                    break;
                case WorldCommand.DestroyJoint:
                    JointKernel.Destroy(world, (NativeStore.JointData*)a->pointer, globals);
                    break;
                case WorldCommand.GetAllShapesByLayerMask:
                    CollectShapesByLayerMask(world, a, globals);
                    break;
            }
        }

        private static void CountBodies(NativeStore.WorldData* world, BodyCounts* counts)
        {
            using Unity.Profiling.ProfilerMarker.AutoScope span = BackendMarkers.CountBodies.Auto();

            for (uint i = 0; i < world->bodyCount; i++)
            {
                NativeStore.BodyFlags flags = world->bodies[i].flags;
                if ((flags & NativeStore.BodyFlags.Simulated) == 0)
                {
                    continue;
                }

                uint active = (flags & NativeStore.BodyFlags.Sleeping) == 0 ? 1u : 0u;
                if ((flags & NativeStore.BodyFlags.Static) != 0)
                {
                    counts->StaticBodies++;
                }
                else if ((flags & NativeStore.BodyFlags.Kinematic) != 0)
                {
                    counts->KinematicBodies++;
                    counts->ActiveKinematicBodies += active;
                }
                else
                {
                    counts->DynamicBodies++;
                    counts->ActiveDynamicBodies += active;
                }
            }
        }

        private static NativeStore.Body* CreateBody(NativeStore.WorldData* world, NativeStore.CommandArgs* a, NativeStore.StoreGlobals* globals)
        {
            using Unity.Profiling.ProfilerMarker.AutoScope span = BackendMarkers.CreateBody.Auto();

            if (world->bodyCount == world->bodyCapacity)
            {
                world->bodyCapacity = StoreOps.Grown(world->bodyCapacity);
                world->bodies = (NativeStore.BodyData*)StoreOps.Reallocate(globals, world->bodies, world->bodyCapacity * (ulong)sizeof(NativeStore.BodyData));
            }

            NativeStore.Body* body = StoreOps.Allocate<NativeStore.Body>(globals);
            body->world = world;
            body->index = world->bodyCount++;
            body->userData = a->pointer;

            NativeStore.BodyData* hot = world->bodies + body->index;
            *hot = default;
            hot->owner = body;
            StoreOps.PointAtHome(hot, body);
            hot->pose->Rotation = a->rotation;
            hot->pose->Position = a->position;
            hot->pose->Scale = 1.0f;
            hot->flags = (NativeStore.BodyFlags)a->flags | NativeStore.BodyFlags.ColliderDirty;
            if (a->enabled != 0)
            {
                hot->flags |= NativeStore.BodyFlags.Static;
            }

            hot->bodyMass->Mass = a->scalar;
            hot->mass->InverseMass = a->scalar > 0.0f ? 1.0f / a->scalar : 0.0f;
            hot->bodyMass->Inertia = new float3(1.0f);
            hot->mass->InverseInertia = new float3(1.0f);
            hot->mass->Transform.rot = quaternion.identity;
            hot->damping->Linear = a->scalar2;
            hot->damping->Angular = a->scalar3;
            hot->limits->MaxLinearVelocity = DefaultMaxLinearVelocity;
            hot->limits->MaxAngularVelocity = DefaultMaxAngularVelocity;
            hot->limits->MaxDepenetrationVelocity = DefaultMaxDepenetrationVelocity;
            hot->sleep->WakeCounter = StoreOps.WakeCounterReset;
            hot->sleep->Threshold = DefaultSleepThreshold;
            hot->target->Rotation = a->rotation;
            hot->target->Position = a->position;
            BodyEntities.Adopt(hot);

            world->staticsVersion++;
            StoreOps.MarkQueryDirty(world);
            return body;
        }

        private static void DestroyBody(NativeStore.WorldData* world, NativeStore.Body* body, NativeStore.StoreGlobals* globals)
        {
            if (body == null)
            {
                return;
            }

            using Unity.Profiling.ProfilerMarker.AutoScope span = BackendMarkers.DestroyBody.Auto();

            StoreOps.DetachAllShapes(body, globals);
            JointKernel.ForgetBody(world, body);

            NativeStore.BodyData* hot = world->bodies + body->index;
            BodyEntities.Release(hot);
            if (hot->collider->Value.IsCreated)
            {
                if (world->deadColliderCount == world->deadColliderCapacity)
                {
                    world->deadColliderCapacity = StoreOps.Grown(world->deadColliderCapacity);
                    world->deadColliders = (void**)StoreOps.Reallocate(globals, world->deadColliders, world->deadColliderCapacity * (ulong)sizeof(void*));
                }

                world->deadColliders[world->deadColliderCount++] = hot->collider->Value.GetUnsafePtr();
            }

            uint last = --world->bodyCount;
            if (body->index != last)
            {
                *hot = world->bodies[last];
                hot->owner->index = body->index;
            }

            StoreOps.Release(globals, body);
            world->staticsVersion++;
            StoreOps.MarkQueryDirty(world);
        }

        /// <summary>Unity destroys bodies before their world, so this only frees storage. A shape can outlive its body.</summary>
        internal static void FreeBodies(NativeStore.WorldData* world, NativeStore.StoreGlobals* globals)
        {
            for (uint i = 0; i < world->bodyCount; i++)
            {
                NativeStore.Body* body = world->bodies[i].owner;
                BodyEntities.Release(world->bodies + i);
                StoreOps.DetachAllShapes(body, globals);
                StoreOps.Release(globals, body);
            }

            world->bodyCount = 0;
        }

        /// <summary>The plugin hands the list to Unity's callback, then frees it.</summary>
        private static void CollectShapesByLayerMask(NativeStore.WorldData* world, NativeStore.CommandArgs* a, NativeStore.StoreGlobals* globals)
        {
            using Unity.Profiling.ProfilerMarker.AutoScope span = BackendMarkers.ShapesByLayerMask.Auto();

            uint capacity = 0;
            for (uint i = 0; i < world->bodyCount; i++)
            {
                capacity += world->bodies[i].shapeCount;
            }

            void** found = (void**)StoreOps.Reallocate(globals, null, math.max(capacity, 1u) * (ulong)sizeof(void*));
            ulong count = 0;
            for (uint i = 0; i < world->bodyCount; i++)
            {
                NativeStore.BodyData* hot = world->bodies + i;
                for (uint s = 0; s < hot->shapeCount; s++)
                {
                    if ((hot->shapes[s]->belongsTo & a->value) != 0)
                    {
                        found[count++] = hot->shapes[s]->userData;
                    }
                }
            }

            a->pointer = found;
            a->size = count;
        }
    }
}
