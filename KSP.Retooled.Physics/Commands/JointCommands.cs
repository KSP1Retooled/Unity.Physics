using Unity.Profiling;
using KSP.Retooled.Physics.Entities;
using KSP.Retooled.Physics.Host;
using KSP.Retooled.Physics.Interop;
using System;
using AOT;
using Unity.Burst;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using UnityEngine;
using Unity.Physics.Store.Bodies;
using Unity.Physics.Store.Joints;
using Unity.Physics.Store.Memory;
using Unity.Physics.Store.Versions;
using Unity.Physics.Store.World;
using Unity.Physics.Stepping;
using Unity.Physics.Dynamics.Joint.Translation;
using Unity.Physics.Dynamics.Joint.Loads;
using Unity.Physics.Dynamics.Bodies;

namespace KSP.Retooled.Physics.Commands
{
    /// <summary>
    /// Store main-thread joint commands for JointTranslation.BuildJointsJob. Commands arrive outside simulation.
    /// </summary>
    [BurstCompile]
    internal static unsafe class JointCommands
    {
        private static readonly ProfilerMarker JointCommandMarker = new ProfilerMarker(ProfilerCategory.Physics, "Physics.Commands.Joint");
        private static readonly ProfilerMarker CreateJointMarker = new ProfilerMarker(ProfilerCategory.Physics, "Physics.Commands.CreateJoint");
        private static readonly ProfilerMarker DestroyJointMarker = new ProfilerMarker(ProfilerCategory.Physics, "Physics.Commands.DestroyJoint");
        private static readonly ProfilerMarker ForgetBodyJointsMarker = new ProfilerMarker(ProfilerCategory.Physics, "Physics.Commands.ForgetBodyJoints");

        [BurstCompile(DisableSafetyChecks = true)]
        [MonoPInvokeCallback(typeof(ManagedCallbacks.CommandFunction))]
        internal static void Run(uint command, void* handle, CommandArgs* args, StoreGlobals* globals)
        {
            // A span costs about as much as a read, so reads run without the category's span, as in BodyCommands.
            if (IsRead((JointCommand)command))
            {
                Dispatch(command, handle, args);
                return;
            }

            using ProfilerMarker.AutoScope span = JointCommandMarker.Auto();
            Dispatch(command, handle, args);
        }

        /// <summary>Whether the command only reads the joint. Reads run without the category's span.</summary>
        internal static bool IsRead(JointCommand command)
        {
            switch (command)
            {
                case JointCommand.GetFlags:
                case JointCommand.GetUserData:
                case JointCommand.GetActorLocalPose:
                case JointCommand.GetIsConstraintValid:
                case JointCommand.GetForces:
                case JointCommand.GetActors:
                case JointCommand.Get6DofLimit:
                    return true;
                default:
                    return false;
            }
        }

        private static void Dispatch(uint command, void* handle, CommandArgs* args)
        {
            bool ranInMono = false;
            RunInMono(command, handle, args, ref ranInMono);
            if (!ranInMono)
            {
                Execute((JointCommand)command, (JointData*)handle, args);
            }
        }

        /// <summary>
        /// Burst omits this method. It runs only in Mono, where exceptions must be caught and logged before returning
        /// to native code.
        /// </summary>
        [BurstDiscard]
        private static void RunInMono(uint command, void* handle, CommandArgs* args, ref bool ranInMono)
        {
            ranInMono = true;
            BackendCounters.NoteManagedCommand();
            try
            {
                Execute((JointCommand)command, (JointData*)handle, args);
            }
            catch (Exception e)
            {
                Debug.LogException(e);
            }
        }

        private static void Execute(JointCommand command, JointData* joint, CommandArgs* a)
        {
            JointAxis* axis = JointData.AxisOf(joint, (int)math.min(a->value, StoreConstants.JointAxisCount - 1));
            if (!Changes(command))
            {
                Run(command, joint, a, axis);
                return;
            }

            // The game often sends a setting again with the value that the joint already has. Keep a copy of the record,
            // so that only a command that really changes a setting moves the joint's settings stamp.
            bool setsTarget = SetsTarget(command);
            JointData before = default;
            if (!setsTarget)
            {
                before = *joint;
            }

            // Reuse groups and forest roles when the joints still have compatible settings.
            bool wasMergeable = JointTranslation.BuildJointsJob.Mergeable(joint);
            Body* body0 = joint->body0, body1 = joint->body1;
            quaternion rotation0 = joint->rotation0, rotation1 = joint->rotation1;
            var flags = joint->flags;
            Run(command, joint, a, axis);
            bool mergeable = JointTranslation.BuildJointsJob.Mergeable(joint);
            bool moved = joint->body0 != body0 || joint->body1 != body1 || joint->flags != flags
                || !math.all(joint->rotation0.value == rotation0.value) || !math.all(joint->rotation1.value == rotation1.value);
            bool canChangeLimit = command == JointCommand.Set6DofAxisLock || command == JointCommand.Set6DofLimit || command == JointCommand.Set6DofSpringLimit;
            bool limitsChanged = mergeable && wasMergeable && canChangeLimit && !JointTranslation.BuildJointsJob.SameAngularLimits(&before, joint);
            if (mergeable != wasMergeable || (mergeable && (moved || limitsChanged)))
            {
                JointTopology.Changed();
            }

            if (!setsTarget && UnsafeUtility.MemCmp(&before, joint, sizeof(JointData)) != 0)
            {
                joint->settingsStamp = JointSettings.Changed();
            }
        }

        private static void Run(JointCommand command, JointData* joint, CommandArgs* a, JointAxis* axis)
        {
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
                    JointLoadStates.Refresh(joint);
                    break;
                case JointCommand.GetIsConstraintValid:
                    a->enabled = (joint->state & (JointState.Broken | JointState.BodyDestroyed)) == 0 ? 1u : 0u;
                    break;
                case JointCommand.GetForces:
                    // While the joint's world has a step state, the loads are in its dense array, not in the record.
                    JointLoadState* state = JointLoadStates.Find(joint);
                    a->vector = state != null ? state->Force : joint->force;
                    a->vector2 = state != null ? state->Torque : joint->torque;
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
                    joint->body0 = ((ActorDescriptor*)a->pointer)[0].actor;
                    joint->body1 = ((ActorDescriptor*)a->pointer)[1].actor;
                    CountEnd(joint->body0, 1);
                    CountEnd(joint->body1, 1);
                    joint->state &= ~JointState.BodyDestroyed;
                    JointLoadStates.Refresh(joint);
                    break;
                case JointCommand.GetActors:
                    Describe(joint->body0, (ActorDescriptor*)a->pointer);
                    Describe(joint->body1, (ActorDescriptor*)a->pointer + 1);
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
                    axis = JointData.AxisOf(joint, StoreConstants.AxisDistance);
                    axis->min = a->scalar;
                    axis->max = a->scalar2;
                    break;
                case JointCommand.SetDistanceSpringLimit:
                    axis = JointData.AxisOf(joint, StoreConstants.AxisDistance);
                    axis->limitSpring = a->scalar;
                    axis->limitDamping = a->scalar2;
                    break;
                case JointCommand.SetDistanceErrorTolerance:
                    joint->distanceErrorTolerance = a->scalar;
                    break;
                case JointCommand.SetHingeLimitEnabled:
                    SetState(joint, JointState.HingeLimitEnabled, a->enabled != 0);
                    break;
                case JointCommand.SetHingeMotorConfiguration:
                    joint->hingeMotorMaxForce = a->scalar;
                    SetState(joint, JointState.HingeMotorFreeSpin, a->enabled != 0);
                    break;
                case JointCommand.SetHingeMotorTargetVelocity:
                    joint->hingeMotorTargetVelocity = a->scalar;
                    break;
                case JointCommand.SetHingeLimit:
                    SetLimit(JointData.AxisOf(joint, StoreConstants.AxisAngularX), a);
                    break;
            }
        }

        /// <summary>
        /// Returns whether a command may modify the joint. Commands that only read the joint or wake its bodies return
        /// false.
        /// </summary>
        private static bool Changes(JointCommand command)
        {
            switch (command)
            {
                case JointCommand.GetFlags:
                case JointCommand.GetUserData:
                case JointCommand.GetActorLocalPose:
                case JointCommand.GetIsConstraintValid:
                case JointCommand.GetForces:
                case JointCommand.GetActors:
                case JointCommand.WakeUpActors:
                case JointCommand.Get6DofLimit:
                    return false;
                default:
                    return true;
            }
        }

        /// <summary>Returns whether a command only sets a drive target, which the vessel edges never read.</summary>
        private static bool SetsTarget(JointCommand command)
        {
            switch (command)
            {
                case JointCommand.Set6DofMotorTargetPosition:
                case JointCommand.Set6DofMotorTargetRotation:
                case JointCommand.Set6DofMotorTargetLinearVelocity:
                case JointCommand.Set6DofMotorTargetAngularVelocity:
                case JointCommand.SetHingeMotorTargetVelocity:
                    return true;
                default:
                    return false;
            }
        }

        // Swing commands repeat the y/z limit pair, so share one slot. Linear limits use the Distance slot.
        private static JointAxis* LimitAxis(JointData* joint, uint axis)
        {
            bool swing = axis == StoreConstants.AxisAngularY || axis == StoreConstants.AxisAngularZ;
            return JointData.AxisOf(joint, swing ? StoreConstants.AxisAngularYZ : (int)math.min(axis, StoreConstants.JointAxisCount - 1));
        }

        private static void SetLimit(JointAxis* axis, CommandArgs* a)
        {
            axis->min = a->scalar;
            axis->max = a->scalar2;
            axis->bounciness = a->vector.x;
            axis->bounceMinVelocity = a->vector.y;
            axis->contactDistance = a->vector.z;
        }

        private static void SetState(JointData* joint, JointState bit, bool set)
        {
            joint->state = set ? joint->state | bit : joint->state & ~bit;
        }

        private static void Describe(Body* body, ActorDescriptor* descriptor)
        {
            descriptor->actor = body;
            descriptor->userData = body != null ? body->userData : null;

            // Physics::ActorType: Invalid, Static, Dynamic.
            descriptor->type = body == null ? (ushort)0 : (StoreRows.Hot(body)->flags & BodyFlags.Static) != 0 ? (ushort)1 : (ushort)2;
        }

        private static void Wake(Body* body)
        {
            if (body != null)
            {
                BodyKinematics.Wake(StoreRows.Hot(body));
            }
        }

        internal static JointData* Create(WorldData* world, CommandArgs* a, StoreGlobals* globals)
        {
            using ProfilerMarker.AutoScope span = CreateJointMarker.Auto();

            if (world->jointCount == world->jointCapacity)
            {
                world->jointCapacity = StoreMemory.Grown(world->jointCapacity);
                world->joints = (JointData**)StoreMemory.Reallocate(globals, world->joints, world->jointCapacity * (ulong)sizeof(void*));
            }

            JointData* joint = StoreMemory.Allocate<JointData>(globals);
            joint->rotation0 = quaternion.identity;
            joint->rotation1 = quaternion.identity;
            joint->targetRotation = quaternion.identity;
            joint->type = a->value;
            joint->flags = a->flags;
            joint->body0 = (Body*)a->pointer;
            joint->body1 = (Body*)a->pointer2;
            joint->userData = a->pointer3;
            CountEnd(joint->body0, 1);
            CountEnd(joint->body1, 1);
            joint->breakForce = float.PositiveInfinity;
            joint->breakTorque = float.PositiveInfinity;
            joint->inverseMassScale0 = 1.0f;
            joint->inverseMassScale1 = 1.0f;
            for (int i = 0; i < StoreConstants.JointAxisCount; i++)
            {
                JointData.AxisOf(joint, i)->dofLock = StoreConstants.DofLocked;
                JointData.AxisOf(joint, i)->driveMaxForce = float.PositiveInfinity;
            }

            joint->index = world->jointCount;
            world->joints[world->jointCount++] = joint;

            // Every new joint moves the topology version, so the new value is a number that no other joint received.
            joint->serial = Serials.Next();
            JointLoadStates.Created(world, joint);
            return joint;
        }

        internal static void Destroy(WorldData* world, JointData* joint, StoreGlobals* globals)
        {
            if (joint == null)
            {
                return;
            }

            using ProfilerMarker.AutoScope span = DestroyJointMarker.Auto();

            JointTopology.Changed();
            CountEnd(joint->body0, -1);
            CountEnd(joint->body1, -1);
            JointLoadStates.Destroying(world, joint);
            uint last = --world->jointCount;
            if (joint->index != last)
            {
                world->joints[joint->index] = world->joints[last];
                world->joints[joint->index]->index = joint->index;
            }

            StoreMemory.Release(globals, joint);
        }

        /// <summary>
        /// A joint can outlive a body that it connects. Clear the pointer to that body before its record is freed.
        /// </summary>
        internal static void ForgetBody(WorldData* world, Body* body)
        {
            using ProfilerMarker.AutoScope span = ForgetBodyJointsMarker.Auto();

            BodyData* hot = world->bodies + body->index;
            if (hot->jointEnds > 0)
            {
                JointTopology.Changed();
            }

            UnsafeList<JointLoadState>* states = JointLoadStates.Of(world);
            for (uint i = 0; i < world->jointCount && hot->jointEnds > 0; i++)
            {
                JointData* joint = world->joints[i];
                if (joint->body0 == body)
                {
                    joint->body0 = null;
                    joint->state |= JointState.BodyDestroyed;
                    hot->jointEnds--;
                }

                if (joint->body1 == body)
                {
                    joint->body1 = null;
                    joint->state |= JointState.BodyDestroyed;
                    hot->jointEnds--;
                }

                if (states != null && (joint->state & JointState.BodyDestroyed) != 0)
                {
                    JointLoadStates.Refresh(states->Ptr + i, joint);
                }
            }
        }

        private static void CountEnd(Body* body, int change)
        {
            if (body != null)
            {
                BodyData* hot = body->world->bodies + body->index;
                hot->jointEnds += change;
            }
        }
    }
}
