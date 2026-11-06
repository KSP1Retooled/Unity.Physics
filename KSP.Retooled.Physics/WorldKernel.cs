using System;
using System.Runtime.InteropServices;
using AOT;
using Unity.Burst;
using Unity.Mathematics;
using UnityEngine;

namespace KSP.Retooled.PhysicsBackend
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
        private const float DefaultWakeCounter = 0.4f;

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
            hot->rotation = a->rotation;
            hot->position = a->position;
            hot->flags = (NativeStore.BodyFlags)a->flags | NativeStore.BodyFlags.ColliderDirty;
            if (a->enabled != 0)
            {
                hot->flags |= NativeStore.BodyFlags.Static;
            }

            hot->mass = a->scalar;
            hot->inverseMass = a->scalar > 0.0f ? 1.0f / a->scalar : 0.0f;
            hot->inertia = new float3(1.0f);
            hot->inverseInertia = new float3(1.0f);
            hot->inertiaRotation = quaternion.identity;
            hot->linearDamping = a->scalar2;
            hot->angularDamping = a->scalar3;
            hot->maxLinearVelocity = DefaultMaxLinearVelocity;
            hot->maxAngularVelocity = DefaultMaxAngularVelocity;
            hot->maxDepenetrationVelocity = DefaultMaxDepenetrationVelocity;
            hot->wakeCounter = DefaultWakeCounter;
            hot->sleepThreshold = DefaultSleepThreshold;
            hot->targetRotation = hot->rotation;
            hot->targetPosition = hot->position;
            hot->owner = body;

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

            StoreOps.DetachAllShapes(body, globals);
            JointKernel.ForgetBody(world, body);

            NativeStore.BodyData* hot = world->bodies + body->index;
            if (hot->collider != null)
            {
                if (world->deadColliderCount == world->deadColliderCapacity)
                {
                    world->deadColliderCapacity = StoreOps.Grown(world->deadColliderCapacity);
                    world->deadColliders = (void**)StoreOps.Reallocate(globals, world->deadColliders, world->deadColliderCapacity * (ulong)sizeof(void*));
                }

                world->deadColliders[world->deadColliderCount++] = hot->collider;
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
                StoreOps.DetachAllShapes(body, globals);
                StoreOps.Release(globals, body);
            }

            world->bodyCount = 0;
        }

        /// <summary>The plugin hands the list to Unity's callback, then frees it.</summary>
        private static void CollectShapesByLayerMask(NativeStore.WorldData* world, NativeStore.CommandArgs* a, NativeStore.StoreGlobals* globals)
        {
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
