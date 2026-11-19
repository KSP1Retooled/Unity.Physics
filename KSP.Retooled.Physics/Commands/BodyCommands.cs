using Unity.Profiling;
using KSP.Retooled.Physics.Entities;
using KSP.Retooled.Physics.Host;
using KSP.Retooled.Physics.Interop;
using System;
using AOT;
using Unity.Burst;
using Unity.Mathematics;
using UnityEngine;
using Unity.Physics.Dynamics.Bodies;
using Unity.Physics.Store.Bodies;
using Unity.Physics.Store.Memory;
using Unity.Physics.Store.Shapes;
using Unity.Physics.Store.Versions;
using Unity.Physics.Store.World;
using Unity.Physics.Stepping;
using Unity.Physics.Dynamics.Contacts;

namespace KSP.Retooled.Physics.Commands
{
    /// <summary>
    /// Implement body commands and accumulate forces and torques for the next step.
    /// When Unity reads poses from worker threads, the plugin handles those reads directly.
    /// </summary>
    [BurstCompile]
    internal static unsafe class BodyCommands
    {
        private static readonly ProfilerMarker BodyCommandMarker = new ProfilerMarker(ProfilerCategory.Physics, "Physics.Commands.Body");
        private static readonly ProfilerMarker AttachShapeMarker = new ProfilerMarker(ProfilerCategory.Physics, "Physics.Commands.AttachShape");
        private static readonly ProfilerMarker DetachShapeMarker = new ProfilerMarker(ProfilerCategory.Physics, "Physics.Commands.DetachShape");

        [BurstCompile(DisableSafetyChecks = true)]
        [MonoPInvokeCallback(typeof(ManagedCallbacks.CommandFunction))]
        internal static void Run(uint command, void* handle, CommandArgs* args, StoreGlobals* globals)
        {
            // Skip category markers for frequent reads; their overhead is comparable to the read itself.
            if (IsRead((BodyCommand)command))
            {
                Dispatch(command, handle, args, globals);
                return;
            }

            using ProfilerMarker.AutoScope span = BodyCommandMarker.Auto();
            Dispatch(command, handle, args, globals);
        }

        /// <summary>Whether the command only reads the body. Reads run without the category's span.</summary>
        internal static bool IsRead(BodyCommand command)
        {
            switch (command)
            {
                case BodyCommand.GetFlags:
                case BodyCommand.GetMass:
                case BodyCommand.GetDamping:
                case BodyCommand.GetPose:
                case BodyCommand.GetShapeCount:
                case BodyCommand.GetShapes:
                case BodyCommand.GetShapesUserData:
                case BodyCommand.GetUserData:
                case BodyCommand.GetLinearVelocity:
                case BodyCommand.GetAngularVelocity:
                case BodyCommand.GetMaxLinearVelocity:
                case BodyCommand.GetMaxAngularVelocity:
                case BodyCommand.GetMaxDepenetrationVelocity:
                case BodyCommand.GetInertiaTensor:
                case BodyCommand.GetInertiaTensorRotation:
                case BodyCommand.GetLocalCenterOfMass:
                case BodyCommand.GetWorldInertiaTensorMatrix:
                case BodyCommand.GetAccumulatedForce:
                case BodyCommand.GetAccumulatedTorque:
                case BodyCommand.GetLocalPointVelocity:
                case BodyCommand.GetWorldPointVelocity:
                case BodyCommand.GetClosestWorldPointOnBounds:
                case BodyCommand.GetKinematicTarget:
                case BodyCommand.GetIsSleeping:
                case BodyCommand.GetWakeCounter:
                case BodyCommand.GetSleepThresholdOverride:
                case BodyCommand.GetSolverIterationsOverride:
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
                Execute((BodyCommand)command, (Body*)handle, args, globals);
            }
        }

        /// <summary>
        /// Run the Mono fallback and catch exceptions before returning to native code. Omitted by Burst.
        /// </summary>
        [BurstDiscard]
        private static void RunInMono(uint command, void* handle, CommandArgs* args, StoreGlobals* globals, ref bool ranInMono)
        {
            ranInMono = true;
            BackendCounters.NoteManagedCommand();
            try
            {
                Execute((BodyCommand)command, (Body*)handle, args, globals);
            }
            catch (Exception e)
            {
                Debug.LogException(e);
            }
        }

        internal static void Execute(BodyCommand command, Body* body, CommandArgs* a, StoreGlobals* globals)
        {
            BodyData* hot = StoreRows.Hot(body);
            switch (command)
            {
                case BodyCommand.SetFlags:
                    SetFlags(body, hot, a);
                    break;
                case BodyCommand.SetMass:
                    hot->bodyMass->Mass = a->scalar;
                    hot->mass->InverseMass = MassProperties.InverseOrZero(a->scalar);
                    break;
                case BodyCommand.GetMass:
                    a->scalar = hot->bodyMass->Mass;
                    break;
                case BodyCommand.SetDamping:
                    hot->damping->Linear = a->scalar;
                    hot->damping->Angular = a->scalar2;
                    break;
                case BodyCommand.GetDamping:
                    a->scalar = hot->damping->Linear;
                    a->scalar2 = hot->damping->Angular;
                    break;
                case BodyCommand.SetPose:
                    SetPose(body, hot, a);
                    break;
                case BodyCommand.GetPose:
                    a->position = hot->pose->Position;
                    a->rotation = hot->pose->Rotation;
                    break;
                case BodyCommand.AttachShape:
                    AttachShape(body, a, globals);
                    break;
                case BodyCommand.DetachShape:
                    if (a->pointer != null)
                    {
                        using ProfilerMarker.AutoScope span = DetachShapeMarker.Auto();
                        CommandStore.RemoveShape(body, (Shape*)a->pointer, globals);
                    }

                    break;
                case BodyCommand.GetShapeCount:
                    a->size = body->shapeCount;
                    break;
                case BodyCommand.GetShapes:
                    a->size = math.min(a->size, body->shapeCount);
                    for (ulong i = 0; i < a->size; i++)
                    {
                        ((void**)a->pointer)[i] = body->shapes[i];
                    }

                    break;
                case BodyCommand.GetShapesUserData:
                    a->size = math.min(a->size, body->shapeCount);
                    for (ulong i = 0; i < a->size; i++)
                    {
                        ((void**)a->pointer)[i] = body->shapes[i]->userData;
                    }

                    break;
                case BodyCommand.GetUserData:
                    a->pointer = body->userData;
                    break;
                case BodyCommand.SetCollisionDetectionMode:
                    body->collisionDetectionMode = (int)a->mode;
                    break;
                case BodyCommand.AddForce:
                    ExternalForces.AddForce(hot, a);
                    break;
                case BodyCommand.AddTorque:
                    ExternalForces.AddTorque(hot, a);
                    break;
                case BodyCommand.ClearForce:
                    hot->forces->Force = float3.zero;
                    hot->forces->Acceleration = float3.zero;
                    hot->forces->Impulse = float3.zero;
                    hot->forces->VelocityChange = float3.zero;
                    break;
                case BodyCommand.ClearTorque:
                    hot->forces->Torque = float3.zero;
                    hot->forces->AngularAcceleration = float3.zero;
                    hot->forces->AngularImpulse = float3.zero;
                    hot->forces->AngularVelocityChange = float3.zero;
                    break;
                case BodyCommand.SetLinearVelocity:
                    hot->velocity->Linear = a->vector;
                    BodyKinematics.WakeIfAsked(hot, a);
                    break;
                case BodyCommand.GetLinearVelocity:
                    a->vector = hot->velocity->Linear;
                    break;
                case BodyCommand.SetAngularVelocity:
                    BodyKinematics.SetWorldAngularVelocity(*hot, a->vector);
                    BodyKinematics.WakeIfAsked(hot, a);
                    break;
                case BodyCommand.GetAngularVelocity:
                    a->vector = BodyKinematics.WorldAngularVelocity(*hot);
                    break;
                case BodyCommand.SetMaxLinearVelocity:
                    hot->limits->MaxLinearVelocity = a->scalar;
                    break;
                case BodyCommand.GetMaxLinearVelocity:
                    a->scalar = hot->limits->MaxLinearVelocity;
                    break;
                case BodyCommand.SetMaxAngularVelocity:
                    hot->limits->MaxAngularVelocity = a->scalar;
                    break;
                case BodyCommand.GetMaxAngularVelocity:
                    a->scalar = hot->limits->MaxAngularVelocity;
                    break;
                case BodyCommand.SetMaxDepenetrationVelocity:
                    hot->limits->MaxDepenetrationVelocity = a->scalar;
                    break;
                case BodyCommand.GetMaxDepenetrationVelocity:
                    a->scalar = hot->limits->MaxDepenetrationVelocity;
                    break;
                case BodyCommand.SetInertiaTensor:
                    MassProperties.SetInertia(hot, a->vector);
                    break;
                case BodyCommand.GetInertiaTensor:
                    a->vector = hot->bodyMass->Inertia;
                    break;
                case BodyCommand.SetInertiaTensorRotation:
                    BodyKinematics.SetInertiaRotation(*hot, a->rotation);
                    break;
                case BodyCommand.GetInertiaTensorRotation:
                    a->rotation = hot->mass->Transform.rot;
                    break;
                case BodyCommand.SetLocalCenterOfMass:
                    hot->mass->Transform.pos = a->vector;
                    break;
                case BodyCommand.GetLocalCenterOfMass:
                    a->vector = hot->mass->Transform.pos;
                    break;
                case BodyCommand.GetWorldInertiaTensorMatrix:
                    MassProperties.WorldInertiaTensor(hot, (float*)a->pointer);
                    break;
                case BodyCommand.RecomputeMassProperties:
                    MassProperties.RecomputeMassProperties(body, hot, a);
                    break;
                case BodyCommand.GetAccumulatedForce:
                    a->vector = ExternalForces.AccumulatedForce(hot, a->scalar);
                    break;
                case BodyCommand.GetAccumulatedTorque:
                    a->vector = ExternalForces.AccumulatedTorque(hot, a->scalar);
                    break;
                case BodyCommand.GetLocalPointVelocity:
                    a->vector = BodyKinematics.VelocityAt(hot, hot->pose->Position + math.rotate(hot->pose->Rotation, a->vector));
                    break;
                case BodyCommand.GetWorldPointVelocity:
                    a->vector = BodyKinematics.VelocityAt(hot, a->vector);
                    break;
                case BodyCommand.GetClosestWorldPointOnBounds:
                    MassProperties.ClosestPointOnBounds(body, hot, a);
                    break;
                case BodyCommand.SetKinematicTarget:
                    SetKinematicTarget(hot, a);
                    break;
                case BodyCommand.GetKinematicTarget:
                    bool hasTarget = (hot->flags & BodyFlags.HasKinematicTarget) != 0;
                    a->position = hasTarget ? hot->target->Position : hot->pose->Position;
                    a->rotation = hasTarget ? hot->target->Rotation : hot->pose->Rotation;
                    break;
                case BodyCommand.Sleep:
                    // Sleeping bodies are not stepped. Clear the target here so that it is not applied when the body
                    // next wakes.
                    hot->flags |= BodyFlags.Sleeping;
                    hot->flags &= ~BodyFlags.HasKinematicTarget;
                    hot->sleep->WakeCounter = 0.0f;
                    hot->velocity->Linear = float3.zero;
                    hot->velocity->Angular = float3.zero;
                    break;
                case BodyCommand.WakeUp:
                    BodyKinematics.Wake(hot);
                    break;
                case BodyCommand.SetWakeCounter:
                    hot->sleep->WakeCounter = a->scalar;
                    if (a->scalar > 0.0f)
                    {
                        hot->flags &= ~BodyFlags.Sleeping;
                    }

                    break;
                case BodyCommand.GetWakeCounter:
                    a->scalar = hot->sleep->WakeCounter;
                    break;
                case BodyCommand.SetSleepThresholdOverride:
                    hot->sleep->Threshold = a->scalar;
                    break;
                case BodyCommand.GetSleepThresholdOverride:
                    a->scalar = hot->sleep->Threshold;
                    break;
                case BodyCommand.SetSolverIterationsOverride:
                    body->positionIterations = a->value;
                    body->velocityIterations = a->value2;
                    break;
                case BodyCommand.GetSolverIterationsOverride:
                    a->value = body->positionIterations;
                    a->value2 = body->velocityIterations;
                    break;
            }
        }

        /// <summary>
        /// Set a body's pose and mark the static tree dirty when a static body moves.
        /// </summary>
        private static void SetPose(Body* body, BodyData* hot, CommandArgs* a)
        {
            float3 position = hot->pose->Position;
            quaternion rotation = hot->pose->Rotation;

            // Preserve the exact state when pose bits match, avoiding rounding from angular velocity
            // conversions.
            // Still honor wake requests. Bitwise comparison distinguishes signed zero and NaN payloads.
            if (math.all(math.asuint(a->position) == math.asuint(position)) && math.all(math.asuint(a->rotation.value) == math.asuint(rotation.value)))
            {
                BodyKinematics.WakeIfAsked(hot, a);
                return;
            }

            hot->pose->Position = a->position;
            BodyKinematics.SetRotation(*hot, a->rotation);
            BodyKinematics.WakeIfAsked(hot, a);
            if (math.any(hot->pose->Position != position) || math.any(hot->pose->Rotation.value != rotation.value))
            {
                if ((hot->flags & BodyFlags.Static) != 0)
                {
                    CommandStore.MarkStaticMoved(body);
                }
                else
                {
                    CommandStore.MarkPoseChanged(body, hot->flags);
                }
            }
        }

        private static void SetFlags(Body* body, BodyData* hot, CommandArgs* a)
        {
            BodyFlags bits = (BodyFlags)a->flags & BodyFlags.PublicMask;
            BodyFlags before = hot->flags;
            if (a->enabled != 0)
            {
                hot->flags |= bits;
            }
            else
            {
                hot->flags &= ~bits;
            }

            // Only simulated kinematic bodies move to their targets during a step. Clear the target for other bodies so
            // that it is not applied later.
            const BodyFlags followsTargets = BodyFlags.Kinematic | BodyFlags.Simulated;
            if ((hot->flags & followsTargets) != followsTargets)
            {
                hot->flags &= ~BodyFlags.HasKinematicTarget;
            }

            // Unity sets flags far more often than their values change. The static tree needs to be updated both when a
            // body becomes static and when it stops being static.
            if (hot->flags != before)
            {
                CommandStore.MarkPoseChanged(body, before | hot->flags);
            }

            // Whether a body moves determines the position of its joints in the vessel forest. Whether it is simulated
            // determines where the bodies for its joints are built.
            if (((hot->flags ^ before) & (BodyFlags.Kinematic | BodyFlags.Simulated)) != 0)
            {
                JointTopology.Changed();
            }

            // Match PhysX: changing isKinematic restarts touching pairs, even for sleeping bodies.
            // TouchingPairs.WakeRefilteredPairsJob reevaluates the pairs on the next step.
            if (((hot->flags ^ before) & BodyFlags.Kinematic) != 0)
            {
                PairFilterVersion.Changed();
            }
        }

        /// <summary>
        /// Convert a static shape's world-space pose to body-local storage.
        /// Unity supplies local shape poses for dynamic bodies, despite the plugin API comments.
        /// </summary>
        internal static void StoreStaticShapePose(ShapeData* shape, BodyData* body, float3 worldPosition, quaternion worldRotation)
        {
            quaternion inverse = math.conjugate(body->pose->Rotation);
            shape->localPosition = math.rotate(inverse, worldPosition - body->pose->Position);
            shape->localRotation = math.mul(inverse, worldRotation);
        }

        private static void AttachShape(Body* body, CommandArgs* a, StoreGlobals* globals)
        {
            using ProfilerMarker.AutoScope span = AttachShapeMarker.Auto();

            // A shape has one owner. If its entry remained with the old owner, that entry would point to freed memory
            // after the shape was destroyed.
            Shape* shape = (Shape*)a->pointer;
            if (shape == null)
            {
                return;
            }

            if (shape->body != null)
            {
                CommandStore.RemoveShape(shape->body, shape, globals);
            }

            BodyData* hot = StoreRows.Hot(body);
            if ((hot->flags & BodyFlags.Static) != 0 && a->enabled == 0)
            {
                StoreStaticShapePose(&shape->data, hot, a->position, a->rotation);
            }
            else
            {
                shape->data.localPosition = a->position;
                shape->data.localRotation = a->rotation;
            }

            CommandStore.AddShape(body, shape, globals);
        }

        private static void SetKinematicTarget(BodyData* hot, CommandArgs* a)
        {
            // Bodies that are not simulated are skipped during the step. Ignore their targets here so that a stored
            // target cannot cause a teleport later.
            if ((hot->flags & BodyFlags.Simulated) != 0)
            {
                hot->target->Position = a->position;
                hot->target->Rotation = a->rotation;
                hot->flags |= BodyFlags.HasKinematicTarget;
            }

            BodyKinematics.Wake(hot);
        }
    }
}
