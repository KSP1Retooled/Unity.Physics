using System;
using AOT;
using Unity.Burst;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using UnityEngine;

namespace KSP.Retooled.Physics
{
    /// <summary>
    /// The plugin API's joint commands, identified by the IDs that Joint.cpp passes. Attach checks that the command
    /// count matches the plugin's count.
    /// </summary>
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
    /// A version counter for changes that can affect how the step groups joints into units and assigns their roles in
    /// the vessel forest ( <see cref="GroupJointsJob"/> ). These changes include creating, destroying, breaking or
    /// reconfiguring a joint; creating or destroying a body; and changing whether a body is kinematic or simulated. If
    /// the version stays the same, a world reuses the groups from the previous step. The plugin mirrors the records in
    /// the store, so there is no space to add this counter to those records. Instead, all worlds share one counter. A
    /// change in any world causes every world to rebuild its groups.
    /// </summary>
    internal static class JointTopology
    {
        private readonly struct Key { }

        private static readonly SharedStatic<int> Shared = SharedStatic<int>.GetOrCreate<Key>();

        internal static uint Version => (uint)Shared.Data;

        /// <summary>Moves the version on and returns the new value.</summary>
        internal static unsafe uint Changed()
        {
            return (uint)System.Threading.Interlocked.Increment(ref UnsafeUtility.AsRef<int>(Shared.UnsafeDataPointer));
        }
    }

    /// <summary>
    /// Counts the commands that changed a joint's settings: its anchors, frames, break limits, axis locks and drives.
    /// A command that only sets a drive target, or that sets a value that the joint already has, leaves this version the
    /// same. The joint that a command changed records the new version in its settings stamp. While this version stays
    /// the same, the vessel fill jobs reuse every edge and closure that they computed in an earlier step. When it moves,
    /// they compute again only the edges and closures whose joints have a stamp that differs from the one they saw.
    /// All worlds share this counter, as they share <see cref="JointTopology"/>.
    /// </summary>
    internal static class JointSettings
    {
        private readonly struct Key { }

        private static readonly SharedStatic<int> Shared = SharedStatic<int>.GetOrCreate<Key>();

        internal static uint Version => (uint)Shared.Data;

        /// <summary>Moves the version on and returns the new value.</summary>
        internal static unsafe uint Changed()
        {
            return (uint)System.Threading.Interlocked.Increment(ref UnsafeUtility.AsRef<int>(Shared.UnsafeDataPointer));
        }
    }

    /// <summary>
    /// Handles joint commands by storing the values that Unity sends. <see cref="BuildJointsJob"/> uses those values to
    /// build constraints for the step. Commands arrive on the main thread, outside a step.
    /// </summary>
    [BurstCompile]
    internal static unsafe class JointKernel
    {
        [BurstCompile(DisableSafetyChecks = true)]
        [MonoPInvokeCallback(typeof(NativeStore.CommandFunction))]
        internal static void Run(uint command, void* handle, NativeStore.CommandArgs* args, NativeStore.StoreGlobals* globals)
        {
            // A span costs about as much as a read, so reads run without the category's span, as in BodyKernel.
            if (IsRead((JointCommand)command))
            {
                Dispatch(command, handle, args);
                return;
            }

            using Unity.Profiling.ProfilerMarker.AutoScope span = BackendMarkers.JointCommand.Auto();
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

        private static void Dispatch(uint command, void* handle, NativeStore.CommandArgs* args)
        {
            bool ranInMono = false;
            RunInMono(command, handle, args, ref ranInMono);
            if (!ranInMono)
            {
                Execute((JointCommand)command, (NativeStore.JointData*)handle, args);
            }
        }

        /// <summary>
        /// Burst omits this method. It runs only in Mono, where exceptions must be caught and logged before returning
        /// to native code.
        /// </summary>
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
            if (!Changes(command))
            {
                Run(command, joint, a, axis);
                return;
            }

            // The game often sends a setting again with the value that the joint already has. Keep a copy of the record,
            // so that only a command that really changes a setting moves the joint's settings stamp.
            bool setsTarget = SetsTarget(command);
            NativeStore.JointData before = default;
            if (!setsTarget)
            {
                before = *joint;
            }

            // The groups and forest roles depend on which joints are mergeable. For mergeable joints, they also depend
            // on the bodies, frame orientations and flags. The existing groups and roles can be reused if a command
            // changes none of these. Examples include changing a servo target without changing mergeability, or tuning
            // a drive on a joint that remains unmergeable.
            bool wasMergeable = BuildJointsJob.Mergeable(joint);
            NativeStore.Body* body0 = joint->body0, body1 = joint->body1;
            quaternion rotation0 = joint->rotation0, rotation1 = joint->rotation1;
            var flags = joint->flags;
            Run(command, joint, a, axis);
            bool mergeable = BuildJointsJob.Mergeable(joint);
            bool moved = joint->body0 != body0 || joint->body1 != body1 || joint->flags != flags
                || !math.all(joint->rotation0.value == rotation0.value) || !math.all(joint->rotation1.value == rotation1.value);
            if (mergeable != wasMergeable || (mergeable && moved))
            {
                JointTopology.Changed();
            }

            if (!setsTarget && UnsafeUtility.MemCmp(&before, joint, sizeof(NativeStore.JointData)) != 0)
            {
                joint->settingsStamp = JointSettings.Changed();
            }
        }

        private static void Run(JointCommand command, NativeStore.JointData* joint, NativeStore.CommandArgs* a, NativeStore.JointAxis* axis)
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
                    a->enabled = (joint->state & (NativeStore.JointState.Broken | NativeStore.JointState.BodyDestroyed)) == 0 ? 1u : 0u;
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
                    joint->body0 = ((NativeStore.ActorDescriptor*)a->pointer)[0].actor;
                    joint->body1 = ((NativeStore.ActorDescriptor*)a->pointer)[1].actor;
                    CountEnd(joint->body0, 1);
                    CountEnd(joint->body1, 1);
                    joint->state &= ~NativeStore.JointState.BodyDestroyed;
                    JointLoadStates.Refresh(joint);
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

        /// <summary>
        /// Unity sends the swing limits as a pair: the angle about y and the angle about z. Commands for any of the
        /// three swing axes repeat the entire pair, so the limits share one storage slot. A linear limit uses the
        /// Distance slot.
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

            // Every new joint moves the topology version, so the new value is a number that no other joint received.
            joint->serial = JointTopology.Changed();
            JointLoadStates.Created(world, joint);
            return joint;
        }

        internal static void Destroy(NativeStore.WorldData* world, NativeStore.JointData* joint, NativeStore.StoreGlobals* globals)
        {
            if (joint == null)
            {
                return;
            }

            using Unity.Profiling.ProfilerMarker.AutoScope span = BackendMarkers.DestroyJoint.Auto();

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

            StoreOps.Release(globals, joint);
        }

        /// <summary>
        /// A joint can outlive a body that it connects. Clear the pointer to that body before its record is freed.
        /// </summary>
        internal static void ForgetBody(NativeStore.WorldData* world, NativeStore.Body* body)
        {
            using Unity.Profiling.ProfilerMarker.AutoScope span = BackendMarkers.ForgetBodyJoints.Auto();

            NativeStore.BodyData* hot = world->bodies + body->index;
            if (hot->jointEnds > 0)
            {
                JointTopology.Changed();
            }

            UnsafeList<JointLoadState>* states = JointLoadStates.Of(world);
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

                if (states != null && (joint->state & NativeStore.JointState.BodyDestroyed) != 0)
                {
                    JointLoadStates.Refresh(states->Ptr + i, joint);
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
