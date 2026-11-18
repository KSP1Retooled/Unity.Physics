using Unity.Profiling;
using PhysicsCollider = Unity.Physics.Collider;
using KSP.Retooled.Physics.Entities;
using KSP.Retooled.Physics.Host;
using KSP.Retooled.Physics.Interop;
using System;
using System.Runtime.InteropServices;
using AOT;
using Unity.Burst;
using Unity.Mathematics;
using UnityEngine;
using Unity.Physics.Store.Bodies;
using Unity.Physics.Store.Joints;
using Unity.Physics.Store.Memory;
using Unity.Physics.Store.Versions;
using Unity.Physics.Store.World;
using Unity.Physics.Stepping;

namespace KSP.Retooled.Physics.Commands
{
    /// <summary>
    /// Manage bodies in a world's dense array. World commands run on the main thread outside simulation.
    /// </summary>
    [BurstCompile]
    internal static unsafe class WorldCommands
    {
        private static readonly ProfilerMarker WorldCommandMarker = new ProfilerMarker(ProfilerCategory.Physics, "Physics.Commands.World");
        private static readonly ProfilerMarker CreateBodyMarker = new ProfilerMarker(ProfilerCategory.Physics, "Physics.Commands.CreateBody");
        private static readonly ProfilerMarker DestroyBodyMarker = new ProfilerMarker(ProfilerCategory.Physics, "Physics.Commands.DestroyBody");
        private static readonly ProfilerMarker CountBodiesMarker = new ProfilerMarker(ProfilerCategory.Physics, "Physics.Commands.CountBodies");
        private static readonly ProfilerMarker ShapesByLayerMaskMarker = new ProfilerMarker(ProfilerCategory.Physics, "Physics.Commands.ShapesByLayerMask");

        // The default value of Rigidbody.maxLinearVelocity. Unity sends the body's other limits after creating it.
        private const float DefaultMaxLinearVelocity = 1e16f;
        private const float DefaultMaxAngularVelocity = 7.0f;
        private const float DefaultMaxDepenetrationVelocity = 1e32f;
        private const float DefaultSleepThreshold = 0.005f;

        /// <summary>
        /// Matches the initial fields of Physics::SimulationStepStats. World.cpp checks the field offsets.
        /// </summary>
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
        [MonoPInvokeCallback(typeof(ManagedCallbacks.CommandFunction))]
        internal static void Run(uint command, void* handle, CommandArgs* args, StoreGlobals* globals)
        {
            // Reads run without the category's span, as in BodyCommands. The reads that count or collect have their own.
            if (IsRead((WorldCommand)command))
            {
                Dispatch(command, handle, args, globals);
                return;
            }

            using ProfilerMarker.AutoScope span = WorldCommandMarker.Auto();
            Dispatch(command, handle, args, globals);
        }

        /// <summary>Whether the command only reads the world. Reads run without the category's span.</summary>
        internal static bool IsRead(WorldCommand command)
        {
            switch (command)
            {
                case WorldCommand.GetBasicStats:
                case WorldCommand.GetLastSimulationStats:
                case WorldCommand.GetAllShapesByLayerMask:
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
                Execute((WorldCommand)command, (WorldData*)handle, args, globals);
            }
        }

        /// <summary>
        /// Burst omits this method. It runs only in Mono, where exceptions must be caught before returning to native
        /// code.
        /// </summary>
        [BurstDiscard]
        private static void RunInMono(uint command, void* handle, CommandArgs* args, StoreGlobals* globals, ref bool ranInMono)
        {
            ranInMono = true;
            BackendCounters.NoteManagedCommand();
            try
            {
                Execute((WorldCommand)command, (WorldData*)handle, args, globals);
            }
            catch (Exception e)
            {
                Debug.LogException(e);
            }
        }

        private static void Execute(WorldCommand command, WorldData* world, CommandArgs* a, StoreGlobals* globals)
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
                    DestroyBody(world, (Body*)a->pointer, globals);
                    break;
                case WorldCommand.CreateJoint:
                    a->pointer = JointCommands.Create(world, a, globals);
                    break;
                case WorldCommand.DestroyJoint:
                    JointCommands.Destroy(world, (JointData*)a->pointer, globals);
                    break;
                case WorldCommand.GetAllShapesByLayerMask:
                    CollectShapesByLayerMask(world, a, globals);
                    break;
            }
        }

        private static void CountBodies(WorldData* world, BodyCounts* counts)
        {
            using ProfilerMarker.AutoScope span = CountBodiesMarker.Auto();

            for (uint i = 0; i < world->bodyCount; i++)
            {
                BodyFlags flags = world->bodies[i].flags;
                if ((flags & BodyFlags.Simulated) == 0)
                {
                    continue;
                }

                uint active = (flags & BodyFlags.Sleeping) == 0 ? 1u : 0u;
                if ((flags & BodyFlags.Static) != 0)
                {
                    counts->StaticBodies++;
                }
                else if ((flags & BodyFlags.Kinematic) != 0)
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

        private static Body* CreateBody(WorldData* world, CommandArgs* a, StoreGlobals* globals)
        {
            using ProfilerMarker.AutoScope span = CreateBodyMarker.Auto();

            if (world->bodyCount == world->bodyCapacity)
            {
                world->bodyCapacity = StoreMemory.Grown(world->bodyCapacity);
                world->bodies = (BodyData*)StoreMemory.Reallocate(globals, world->bodies, world->bodyCapacity * (ulong)sizeof(BodyData));
            }

            Body* body = StoreMemory.Allocate<Body>(globals);
            body->world = world;
            body->index = world->bodyCount++;
            body->userData = a->pointer;

            BodyData* hot = world->bodies + body->index;
            *hot = default;
            hot->owner = body;
            StoreRows.PointAtHome(hot, body);
            hot->pose->Rotation = a->rotation;
            hot->pose->Position = a->position;
            hot->pose->Scale = 1.0f;
            hot->flags = (BodyFlags)a->flags | BodyFlags.ColliderDirty;
            if (a->enabled != 0)
            {
                hot->flags |= BodyFlags.Static;
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
            hot->sleep->WakeCounter = StoreMemory.WakeCounterReset;
            hot->sleep->Threshold = DefaultSleepThreshold;
            hot->target->Rotation = a->rotation;
            hot->target->Position = a->position;
            BodyEntities.Adopt(hot);

            // Every new body moves the topology version, so the new value is a number that no other body or joint
            // received.
            body->serial = Serials.Next();
            hot->serial = body->serial;

            // Append the new body to the array. The sequence of static bodies changes only if the new body is static.
            if ((hot->flags & BodyFlags.Static) != 0)
            {
                world->staticsVersion++;
            }

            QuerySnapshotVersion.Changed();
            StoreMemory.MarkQueryDirty(world);
            return body;
        }

        private static void DestroyBody(WorldData* world, Body* body, StoreGlobals* globals)
        {
            if (body == null)
            {
                return;
            }

            using ProfilerMarker.AutoScope span = DestroyBodyMarker.Auto();

            CommandStore.DetachAllShapes(body, globals);
            JointCommands.ForgetBody(world, body);
            VehicleCommands.ForgetBody(world, body);

            BodyData* hot = world->bodies + body->index;
            BodyEntities.Release(hot);
            if (hot->collider->Value.IsCreated)
            {
                if (world->deadColliderCount == world->deadColliderCapacity)
                {
                    world->deadColliderCapacity = StoreMemory.Grown(world->deadColliderCapacity);
                    world->deadColliders = (PhysicsCollider**)StoreMemory.Reallocate(globals, world->deadColliders, world->deadColliderCapacity * (ulong)sizeof(void*));
                }

                world->deadColliders[world->deadColliderCount++] = (PhysicsCollider*)hot->collider->Value.GetUnsafePtr();
            }

            // Replace the destroyed body with the last body in the array. The sequence of static bodies changes only if
            // either body is static.
            bool staticsMove = (hot->flags & BodyFlags.Static) != 0;
            uint last = --world->bodyCount;
            if (body->index != last)
            {
                staticsMove |= (world->bodies[last].flags & BodyFlags.Static) != 0;
                *hot = world->bodies[last];
                hot->owner->index = body->index;
            }

            StoreMemory.Release(globals, body);
            JointTopology.Changed();
            if (staticsMove)
            {
                world->staticsVersion++;
            }

            // The last body moved into the destroyed body's row, so the snapshot's rows no longer match the store.
            QuerySnapshotVersion.Changed();
            StoreMemory.MarkQueryDirty(world);
        }

        /// <summary>
        /// Release entities, detach shapes and free bodies remaining when the world is destroyed. Shapes may
        /// outlive their bodies.
        /// </summary>
        internal static void FreeBodies(WorldData* world, StoreGlobals* globals)
        {
            for (uint i = 0; i < world->bodyCount; i++)
            {
                Body* body = world->bodies[i].owner;
                BodyEntities.Release(world->bodies + i);
                CommandStore.DetachAllShapes(body, globals);
                StoreMemory.Release(globals, body);
            }

            world->bodyCount = 0;
        }

        /// <summary>Builds a list that the plugin passes to Unity's callback and then frees.</summary>
        private static void CollectShapesByLayerMask(WorldData* world, CommandArgs* a, StoreGlobals* globals)
        {
            using ProfilerMarker.AutoScope span = ShapesByLayerMaskMarker.Auto();

            uint capacity = 0;
            for (uint i = 0; i < world->bodyCount; i++)
            {
                capacity += world->bodies[i].shapeCount;
            }

            void** found = (void**)StoreMemory.Reallocate(globals, null, math.max(capacity, 1u) * (ulong)sizeof(void*));
            ulong count = 0;
            for (uint i = 0; i < world->bodyCount; i++)
            {
                BodyData* hot = world->bodies + i;
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
