using System;
using AOT;
using Unity.Burst;
using Unity.Mathematics;
using UnityEngine;

namespace KSP.Retooled.Physics
{
    /// <summary>The plugin API's joint commands, by the id Joint.cpp passes. Attach checks the count against the plugin's.</summary>
    internal enum JointCommand : uint
    {
        Invalid = 0,
        SetFlags,
        GetFlags,
        GetUserData,
        SetActorLocalPose,
        GetActorLocalPose,
        SetBreakForce,
        GetIsConstraintValid,
        GetForces,
        SetInvMassAndInertiaScale,
        SetActors,
        GetActors,
        WakeUpActors,
        Set6DofLimit,
        Get6DofLimit,
        Set6DofSpringLimit,
        Set6DofAxisLock,
        Set6DofAxisMotorConfiguration,
        Set6DofMotorTargetPosition,
        Set6DofMotorTargetRotation,
        Set6DofMotorTargetLinearVelocity,
        Set6DofMotorTargetAngularVelocity,
        Set6DofProjectionTolerance,
        SetDistanceLimit,
        SetDistanceSpringLimit,
        SetDistanceErrorTolerance,
        SetHingeMotorConfiguration,
        SetHingeMotorTargetVelocity,
        SetHingeLimit,
        SetHingeLimitEnabled,
        Count,
    }

    /// <summary>
    /// Everything a joint command does: keeping what Unity sends in the store. The step turns that into constraints in
    /// <see cref="BuildJointsJob"/>. Commands arrive on the main thread outside a step.
    /// </summary>
    [BurstCompile]
    internal static unsafe class JointKernel
    {
        [BurstCompile(DisableSafetyChecks = true)]
        [MonoPInvokeCallback(typeof(NativeStore.CommandFunction))]
        internal static void Run(uint command, void* handle, NativeStore.CommandArgs* args, NativeStore.StoreGlobals* globals)
        {
            using Unity.Profiling.ProfilerMarker.AutoScope span = BackendMarkers.JointCommand.Auto();

            bool ranInMono = false;
            RunInMono(command, handle, args, ref ranInMono);
            if (!ranInMono)
            {
                Execute((JointCommand)command, (NativeStore.JointData*)handle, args);
            }
        }

        /// <summary>Burst drops this, so it only runs in the Mono form, which needs the exception ended here.</summary>
        [BurstDiscard]
        private static void RunInMono(uint command, void* handle, NativeStore.CommandArgs* args, ref bool ranInMono)
        {
            ranInMono = true;
            BackendHost.NoteManagedCommand();
            try
            {
                Execute((JointCommand)command, (NativeStore.JointData*)handle, args);
            }
            catch (Exception e)
            {
                Debug.LogException(e);
            }
        }

        private static void Execute(JointCommand command, NativeStore.JointData* joint, NativeStore.CommandArgs* a)
        {
            NativeStore.JointAxis* axis = NativeStore.AxisOf(joint, (int)math.min(a->value, NativeStore.JointAxisCount - 1));
            switch (command)
            {
                case JointCommand.SetFlags:
                    joint->flags = a->enabled != 0 ? joint->flags | a->flags : joint->flags & ~a->flags;
                    break;
                case JointCommand.GetFlags:
                    a->flags = joint->flags;
                    break;
                case JointCommand.GetUserData:
                    a->pointer = joint->userData;
                    break;
                case JointCommand.SetActorLocalPose:
                    if (a->value == 0)
                    {
                        joint->rotation0 = a->rotation;
                        joint->position0 = a->position;
                    }
                    else
                    {
                        joint->rotation1 = a->rotation;
                        joint->position1 = a->position;
                    }

                    break;
                case JointCommand.GetActorLocalPose:
                    a->rotation = a->value == 0 ? joint->rotation0 : joint->rotation1;
                    a->position = a->value == 0 ? joint->position0 : joint->position1;
                    break;
                case JointCommand.SetBreakForce:
                    joint->breakForce = a->scalar;
                    joint->breakTorque = a->scalar2;
                    break;
                case JointCommand.GetIsConstraintValid:
                    a->enabled = (joint->state & (NativeStore.JointState.Broken | NativeStore.JointState.BodyDestroyed)) == 0 ? 1u : 0u;
                    break;
                case JointCommand.GetForces:
                    a->vector = joint->force;
                    a->vector2 = joint->torque;
                    break;
                case JointCommand.SetInvMassAndInertiaScale:
                    if (a->value == 0)
                    {
                        joint->inverseMassScale0 = a->scalar;
                    }
                    else
                    {
                        joint->inverseMassScale1 = a->scalar;
                    }

                    break;
                case JointCommand.SetActors:
                    CountEnd(joint->body0, -1);
                    CountEnd(joint->body1, -1);
                    joint->body0 = ((NativeStore.ActorDescriptor*)a->pointer)[0].actor;
                    joint->body1 = ((NativeStore.ActorDescriptor*)a->pointer)[1].actor;
                    CountEnd(joint->body0, 1);
                    CountEnd(joint->body1, 1);
                    joint->state &= ~NativeStore.JointState.BodyDestroyed;
                    break;
                case JointCommand.GetActors:
                    Describe(joint->body0, (NativeStore.ActorDescriptor*)a->pointer);
                    Describe(joint->body1, (NativeStore.ActorDescriptor*)a->pointer + 1);
                    break;
                case JointCommand.WakeUpActors:
                    Wake(joint->body0);
                    Wake(joint->body1);
                    break;
                case JointCommand.Set6DofLimit:
                    SetLimit(LimitAxis(joint, a->value), a);
                    break;
                case JointCommand.Get6DofLimit:
                    axis = LimitAxis(joint, a->value);
                    a->scalar = axis->min;
                    a->scalar2 = axis->max;
                    a->vector = new float3(axis->bounciness, axis->bounceMinVelocity, axis->contactDistance);
                    break;
                case JointCommand.Set6DofSpringLimit:
                    axis->limitSpring = a->scalar;
                    axis->limitDamping = a->scalar2;
                    break;
                case JointCommand.Set6DofAxisLock:
                    axis->dofLock = a->mode;
                    break;
                case JointCommand.Set6DofAxisMotorConfiguration:
                    axis->driveSpring = a->scalar;
                    axis->driveDamper = a->scalar2;
                    axis->driveMaxForce = a->scalar3;
                    axis->driveUsesAcceleration = a->enabled;
                    break;
                case JointCommand.Set6DofMotorTargetPosition:
                    joint->targetPosition = a->vector;
                    break;
                case JointCommand.Set6DofMotorTargetRotation:
                    joint->targetRotation = a->rotation;
                    break;
                case JointCommand.Set6DofMotorTargetLinearVelocity:
                    joint->targetLinearVelocity = a->vector;
                    break;
                case JointCommand.Set6DofMotorTargetAngularVelocity:
                    joint->targetAngularVelocity = a->vector;
                    break;
                case JointCommand.Set6DofProjectionTolerance:
                    joint->projectionLinearTolerance = a->scalar;
                    joint->projectionAngularTolerance = a->scalar2;
                    break;
                case JointCommand.SetDistanceLimit:
                    axis = NativeStore.AxisOf(joint, NativeStore.AxisDistance);
                    axis->min = a->scalar;
                    axis->max = a->scalar2;
                    break;
                case JointCommand.SetDistanceSpringLimit:
                    axis = NativeStore.AxisOf(joint, NativeStore.AxisDistance);
                    axis->limitSpring = a->scalar;
                    axis->limitDamping = a->scalar2;
                    break;
                case JointCommand.SetDistanceErrorTolerance:
                    joint->distanceErrorTolerance = a->scalar;
                    break;
                case JointCommand.SetHingeLimitEnabled:
                    SetState(joint, NativeStore.JointState.HingeLimitEnabled, a->enabled != 0);
                    break;
                case JointCommand.SetHingeMotorConfiguration:
                    joint->hingeMotorMaxForce = a->scalar;
                    SetState(joint, NativeStore.JointState.HingeMotorFreeSpin, a->enabled != 0);
                    break;
                case JointCommand.SetHingeMotorTargetVelocity:
                    joint->hingeMotorTargetVelocity = a->scalar;
                    break;
                case JointCommand.SetHingeLimit:
                    SetLimit(NativeStore.AxisOf(joint, NativeStore.AxisAngularX), a);
                    break;
            }
        }

        /// <summary>
        /// Unity sends the swing limit as one pair, (angle about y, angle about z), under any of the three swing axes, each
        /// repeating the whole pair, so they share a slot. A linear limit arrives as Distance.
        /// </summary>
        private static NativeStore.JointAxis* LimitAxis(NativeStore.JointData* joint, uint axis)
        {
            bool swing = axis == NativeStore.AxisAngularY || axis == NativeStore.AxisAngularZ;
            return NativeStore.AxisOf(joint, swing ? NativeStore.AxisAngularYZ : (int)math.min(axis, NativeStore.JointAxisCount - 1));
        }

        private static void SetLimit(NativeStore.JointAxis* axis, NativeStore.CommandArgs* a)
        {
            axis->min = a->scalar;
            axis->max = a->scalar2;
            axis->bounciness = a->vector.x;
            axis->bounceMinVelocity = a->vector.y;
            axis->contactDistance = a->vector.z;
        }

        private static void SetState(NativeStore.JointData* joint, NativeStore.JointState bit, bool set)
        {
            joint->state = set ? joint->state | bit : joint->state & ~bit;
        }

        private static void Describe(NativeStore.Body* body, NativeStore.ActorDescriptor* descriptor)
        {
            descriptor->actor = body;
            descriptor->userData = body != null ? body->userData : null;

            // Physics::ActorType: Invalid, Static, Dynamic.
            descriptor->type = body == null ? (ushort)0 : (StoreOps.Hot(body)->flags & NativeStore.BodyFlags.Static) != 0 ? (ushort)1 : (ushort)2;
        }

        private static void Wake(NativeStore.Body* body)
        {
            if (body != null)
            {
                WakeRow(StoreOps.Hot(body));
            }
        }

        internal static void WakeRow(NativeStore.BodyData* hot)
        {
            hot->flags &= ~NativeStore.BodyFlags.Sleeping;
            hot->sleep->WakeCounter = math.max(hot->sleep->WakeCounter, StoreOps.WakeCounterReset);
        }

        internal static NativeStore.JointData* Create(NativeStore.WorldData* world, NativeStore.CommandArgs* a, NativeStore.StoreGlobals* globals)
        {
            using Unity.Profiling.ProfilerMarker.AutoScope span = BackendMarkers.CreateJoint.Auto();

            if (world->jointCount == world->jointCapacity)
            {
                world->jointCapacity = StoreOps.Grown(world->jointCapacity);
                world->joints = (NativeStore.JointData**)StoreOps.Reallocate(globals, world->joints, world->jointCapacity * (ulong)sizeof(void*));
            }

            NativeStore.JointData* joint = StoreOps.Allocate<NativeStore.JointData>(globals);
            joint->rotation0 = quaternion.identity;
            joint->rotation1 = quaternion.identity;
            joint->targetRotation = quaternion.identity;
            joint->type = a->value;
            joint->flags = a->flags;
            joint->body0 = (NativeStore.Body*)a->pointer;
            joint->body1 = (NativeStore.Body*)a->pointer2;
            joint->userData = a->pointer3;
            CountEnd(joint->body0, 1);
            CountEnd(joint->body1, 1);
            joint->breakForce = float.PositiveInfinity;
            joint->breakTorque = float.PositiveInfinity;
            joint->inverseMassScale0 = 1.0f;
            joint->inverseMassScale1 = 1.0f;
            for (int i = 0; i < NativeStore.JointAxisCount; i++)
            {
                NativeStore.AxisOf(joint, i)->dofLock = NativeStore.DofLocked;
                NativeStore.AxisOf(joint, i)->driveMaxForce = float.PositiveInfinity;
            }

            joint->index = world->jointCount;
            world->joints[world->jointCount++] = joint;
            return joint;
        }

        internal static void Destroy(NativeStore.WorldData* world, NativeStore.JointData* joint, NativeStore.StoreGlobals* globals)
        {
            if (joint == null)
            {
                return;
            }

            using Unity.Profiling.ProfilerMarker.AutoScope span = BackendMarkers.DestroyJoint.Auto();

            CountEnd(joint->body0, -1);
            CountEnd(joint->body1, -1);
            uint last = --world->jointCount;
            if (joint->index != last)
            {
                world->joints[joint->index] = world->joints[last];
                world->joints[joint->index]->index = joint->index;
            }

            StoreOps.Release(globals, joint);
        }

        /// <summary>A joint can outlive a body it connects, and must not keep a pointer to the freed record.</summary>
        internal static void ForgetBody(NativeStore.WorldData* world, NativeStore.Body* body)
        {
            using Unity.Profiling.ProfilerMarker.AutoScope span = BackendMarkers.ForgetBodyJoints.Auto();

            NativeStore.BodyData* hot = world->bodies + body->index;
            for (uint i = 0; i < world->jointCount && hot->jointEnds > 0; i++)
            {
                NativeStore.JointData* joint = world->joints[i];
                if (joint->body0 == body)
                {
                    joint->body0 = null;
                    joint->state |= NativeStore.JointState.BodyDestroyed;
                    hot->jointEnds--;
                }

                if (joint->body1 == body)
                {
                    joint->body1 = null;
                    joint->state |= NativeStore.JointState.BodyDestroyed;
                    hot->jointEnds--;
                }
            }
        }

        private static void CountEnd(NativeStore.Body* body, int change)
        {
            if (body != null)
            {
                NativeStore.BodyData* hot = body->world->bodies + body->index;
                hot->jointEnds += change;
            }
        }
    }
}
