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

namespace KSP.Retooled.Physics
{
    /// <summary>
    /// Body command IDs shared with Body.cpp. Attach verifies the command count.
    /// </summary>
    internal enum BodyCommand : uint
    {
        Invalid = 0,
        SetFlags,
        GetFlags,
        SetMass,
        GetMass,
        SetDamping,
        GetDamping,
        SetPose,
        GetPose,
        AttachShape,
        DetachShape,
        GetShapeCount,
        GetShapes,
        GetShapesUserData,
        GetUserData,
        SetCollisionDetectionMode,
        AddForce,
        AddTorque,
        ClearForce,
        ClearTorque,
        SetLinearVelocity,
        GetLinearVelocity,
        SetAngularVelocity,
        GetAngularVelocity,
        SetMaxLinearVelocity,
        GetMaxLinearVelocity,
        SetMaxAngularVelocity,
        GetMaxAngularVelocity,
        SetMaxDepenetrationVelocity,
        GetMaxDepenetrationVelocity,
        SetInertiaTensor,
        GetInertiaTensor,
        SetInertiaTensorRotation,
        GetInertiaTensorRotation,
        SetLocalCenterOfMass,
        GetLocalCenterOfMass,
        GetWorldInertiaTensorMatrix,
        RecomputeMassProperties,
        GetAccumulatedForce,
        GetAccumulatedTorque,
        GetLocalPointVelocity,
        GetWorldPointVelocity,
        GetClosestWorldPointOnBounds,
        SetKinematicTarget,
        GetKinematicTarget,
        GetIsSleeping,
        Sleep,
        WakeUp,
        SetWakeCounter,
        GetWakeCounter,
        SetSleepThresholdOverride,
        GetSleepThresholdOverride,
        SetSolverIterationsOverride,
        GetSolverIterationsOverride,
        Count,
    }

    /// <summary>
    /// Implement body commands and accumulate forces and torques for the next step.
    /// Unity may read poses from workers; the plugin handles those reads directly without entering this kernel.
    /// </summary>
    [BurstCompile]
    internal static unsafe class BodyKernel
    {
        private const uint OverrideCenterOfMass = 1;
        private const uint OverrideInertiaTensor = 2;

        // Physics::ForceMode.
        private const uint ForceModeForce = 0;
        private const uint ForceModeImpulse = 1;
        private const uint ForceModeVelocityChange = 2;
        private const uint ForceModeAcceleration = 5;

        private const BodyFlags Immobile = BodyFlags.Kinematic | BodyFlags.Static;

        [BurstCompile(DisableSafetyChecks = true)]
        [MonoPInvokeCallback(typeof(NativeStore.CommandFunction))]
        internal static void Run(uint command, void* handle, CommandArgs* args, StoreGlobals* globals)
        {
            // Skip category markers for frequent reads; their overhead is comparable to the read itself.
            if (IsRead((BodyCommand)command))
            {
                Dispatch(command, handle, args, globals);
                return;
            }

            using Unity.Profiling.ProfilerMarker.AutoScope span = BackendMarkers.BodyCommand.Auto();
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
            BackendHost.NoteManagedCommand();
            try
            {
                Execute((BodyCommand)command, (Body*)handle, args, globals);
            }
            catch (Exception e)
            {
                Debug.LogException(e);
            }
        }

        private static void Execute(BodyCommand command, Body* body, CommandArgs* a, StoreGlobals* globals)
        {
            BodyData* hot = StoreOps.Hot(body);
            switch (command)
            {
                case BodyCommand.SetFlags:
                    SetFlags(body, hot, a);
                    break;
                case BodyCommand.SetMass:
                    (*hot).BodyMass()->Mass = a->scalar;
                    (*hot).Mass()->InverseMass = InverseOrZero(a->scalar);
                    break;
                case BodyCommand.GetMass:
                    a->scalar = (*hot).BodyMass()->Mass;
                    break;
                case BodyCommand.SetDamping:
                    (*hot).Damping()->Linear = a->scalar;
                    (*hot).Damping()->Angular = a->scalar2;
                    break;
                case BodyCommand.GetDamping:
                    a->scalar = (*hot).Damping()->Linear;
                    a->scalar2 = (*hot).Damping()->Angular;
                    break;
                case BodyCommand.SetPose:
                    SetPose(body, hot, a);
                    break;
                case BodyCommand.GetPose:
                    a->position = (*hot).Pose()->Position;
                    a->rotation = (*hot).Pose()->Rotation;
                    break;
                case BodyCommand.AttachShape:
                    AttachShape(body, a, globals);
                    break;
                case BodyCommand.DetachShape:
                    if (a->pointer != null)
                    {
                        using Unity.Profiling.ProfilerMarker.AutoScope span = BackendMarkers.DetachShape.Auto();
                        StoreOps.RemoveShape(body, (Shape*)a->pointer, globals);
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
                    AddForce(hot, a);
                    break;
                case BodyCommand.AddTorque:
                    AddTorque(hot, a);
                    break;
                case BodyCommand.ClearForce:
                    (*hot).Forces()->Force = float3.zero;
                    (*hot).Forces()->Acceleration = float3.zero;
                    (*hot).Forces()->Impulse = float3.zero;
                    (*hot).Forces()->VelocityChange = float3.zero;
                    break;
                case BodyCommand.ClearTorque:
                    (*hot).Forces()->Torque = float3.zero;
                    (*hot).Forces()->AngularAcceleration = float3.zero;
                    (*hot).Forces()->AngularImpulse = float3.zero;
                    (*hot).Forces()->AngularVelocityChange = float3.zero;
                    break;
                case BodyCommand.SetLinearVelocity:
                    (*hot).Velocity()->Linear = a->vector;
                    WakeIfAsked(hot, a);
                    break;
                case BodyCommand.GetLinearVelocity:
                    a->vector = (*hot).Velocity()->Linear;
                    break;
                case BodyCommand.SetAngularVelocity:
                    BodyKinematics.SetWorldAngularVelocity(*hot, a->vector);
                    WakeIfAsked(hot, a);
                    break;
                case BodyCommand.GetAngularVelocity:
                    a->vector = BodyKinematics.WorldAngularVelocity(*hot);
                    break;
                case BodyCommand.SetMaxLinearVelocity:
                    (*hot).Limits()->MaxLinearVelocity = a->scalar;
                    break;
                case BodyCommand.GetMaxLinearVelocity:
                    a->scalar = (*hot).Limits()->MaxLinearVelocity;
                    break;
                case BodyCommand.SetMaxAngularVelocity:
                    (*hot).Limits()->MaxAngularVelocity = a->scalar;
                    break;
                case BodyCommand.GetMaxAngularVelocity:
                    a->scalar = (*hot).Limits()->MaxAngularVelocity;
                    break;
                case BodyCommand.SetMaxDepenetrationVelocity:
                    (*hot).Limits()->MaxDepenetrationVelocity = a->scalar;
                    break;
                case BodyCommand.GetMaxDepenetrationVelocity:
                    a->scalar = (*hot).Limits()->MaxDepenetrationVelocity;
                    break;
                case BodyCommand.SetInertiaTensor:
                    SetInertia(hot, a->vector);
                    break;
                case BodyCommand.GetInertiaTensor:
                    a->vector = (*hot).BodyMass()->Inertia;
                    break;
                case BodyCommand.SetInertiaTensorRotation:
                    BodyKinematics.SetInertiaRotation(*hot, a->rotation);
                    break;
                case BodyCommand.GetInertiaTensorRotation:
                    a->rotation = (*hot).Mass()->Transform.rot;
                    break;
                case BodyCommand.SetLocalCenterOfMass:
                    (*hot).Mass()->Transform.pos = a->vector;
                    break;
                case BodyCommand.GetLocalCenterOfMass:
                    a->vector = (*hot).Mass()->Transform.pos;
                    break;
                case BodyCommand.GetWorldInertiaTensorMatrix:
                    WorldInertiaTensor(hot, (float*)a->pointer);
                    break;
                case BodyCommand.RecomputeMassProperties:
                    RecomputeMassProperties(body, hot, a);
                    break;
                case BodyCommand.GetAccumulatedForce:
                    a->vector = AccumulatedForce(hot, a->scalar);
                    break;
                case BodyCommand.GetAccumulatedTorque:
                    a->vector = AccumulatedTorque(hot, a->scalar);
                    break;
                case BodyCommand.GetLocalPointVelocity:
                    a->vector = VelocityAt(hot, (*hot).Pose()->Position + math.rotate((*hot).Pose()->Rotation, a->vector));
                    break;
                case BodyCommand.GetWorldPointVelocity:
                    a->vector = VelocityAt(hot, a->vector);
                    break;
                case BodyCommand.GetClosestWorldPointOnBounds:
                    ClosestPointOnBounds(body, hot, a);
                    break;
                case BodyCommand.SetKinematicTarget:
                    SetKinematicTarget(hot, a);
                    break;
                case BodyCommand.GetKinematicTarget:
                    bool hasTarget = (hot->flags & BodyFlags.HasKinematicTarget) != 0;
                    a->position = hasTarget ? (*hot).Target()->Position : (*hot).Pose()->Position;
                    a->rotation = hasTarget ? (*hot).Target()->Rotation : (*hot).Pose()->Rotation;
                    break;
                case BodyCommand.Sleep:
                    // Sleeping bodies are not stepped. Clear the target here so that it is not applied when the body
                    // next wakes.
                    hot->flags |= BodyFlags.Sleeping;
                    hot->flags &= ~BodyFlags.HasKinematicTarget;
                    (*hot).Sleep()->WakeCounter = 0.0f;
                    (*hot).Velocity()->Linear = float3.zero;
                    (*hot).Velocity()->Angular = float3.zero;
                    break;
                case BodyCommand.WakeUp:
                    Wake(hot);
                    break;
                case BodyCommand.SetWakeCounter:
                    (*hot).Sleep()->WakeCounter = a->scalar;
                    if (a->scalar > 0.0f)
                    {
                        hot->flags &= ~BodyFlags.Sleeping;
                    }

                    break;
                case BodyCommand.GetWakeCounter:
                    a->scalar = (*hot).Sleep()->WakeCounter;
                    break;
                case BodyCommand.SetSleepThresholdOverride:
                    (*hot).Sleep()->Threshold = a->scalar;
                    break;
                case BodyCommand.GetSleepThresholdOverride:
                    a->scalar = (*hot).Sleep()->Threshold;
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

        private static float InverseOrZero(float value)
        {
            return value > 0.0f ? 1.0f / value : 0.0f;
        }

        private static void Wake(BodyData* hot)
        {
            hot->flags &= ~BodyFlags.Sleeping;
            (*hot).Sleep()->WakeCounter = math.max((*hot).Sleep()->WakeCounter, StoreMemory.WakeCounterReset);
        }

        private static void WakeIfAsked(BodyData* hot, CommandArgs* a)
        {
            if (a->enabled != 0)
            {
                Wake(hot);
            }
        }

        private static void SetInertia(BodyData* hot, float3 inertia)
        {
            (*hot).BodyMass()->Inertia = inertia;
            (*hot).Mass()->InverseInertia = new float3(InverseOrZero(inertia.x), InverseOrZero(inertia.y), InverseOrZero(inertia.z));
        }

        /// <summary>
        /// Set a body's pose and mark the static tree dirty when a static body moves.
        /// </summary>
        private static void SetPose(Body* body, BodyData* hot, CommandArgs* a)
        {
            float3 position = (*hot).Pose()->Position;
            quaternion rotation = (*hot).Pose()->Rotation;

            // Preserve the exact state when pose bits match, avoiding rounding from angular velocity
            // conversions.
            // Still honor wake requests. Bitwise comparison distinguishes signed zero and NaN payloads.
            if (math.all(math.asuint(a->position) == math.asuint(position)) && math.all(math.asuint(a->rotation.value) == math.asuint(rotation.value)))
            {
                WakeIfAsked(hot, a);
                return;
            }

            (*hot).Pose()->Position = a->position;
            BodyKinematics.SetRotation(*hot, a->rotation);
            WakeIfAsked(hot, a);
            if (math.any((*hot).Pose()->Position != position) || math.any((*hot).Pose()->Rotation.value != rotation.value))
            {
                if ((hot->flags & BodyFlags.Static) != 0)
                {
                    StoreOps.MarkStaticMoved(body);
                }
                else
                {
                    StoreOps.MarkPoseChanged(body, hot->flags);
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
                StoreOps.MarkPoseChanged(body, before | hot->flags);
            }

            // Whether a body moves determines the position of its joints in the vessel forest. Whether it is simulated
            // determines where the bodies for its joints are built.
            if (((hot->flags ^ before) & (BodyFlags.Kinematic | BodyFlags.Simulated)) != 0)
            {
                JointTopology.Changed();
            }

            // Match PhysX: changing isKinematic restarts touching pairs, even for sleeping bodies.
            // WakeRefilteredPairsJob reevaluates the pairs on the next step.
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
            quaternion inverse = math.conjugate((*body).Pose()->Rotation);
            shape->localPosition = math.rotate(inverse, worldPosition - (*body).Pose()->Position);
            shape->localRotation = math.mul(inverse, worldRotation);
        }

        private static void AttachShape(Body* body, CommandArgs* a, StoreGlobals* globals)
        {
            using Unity.Profiling.ProfilerMarker.AutoScope span = BackendMarkers.AttachShape.Auto();

            // A shape has one owner. If its entry remained with the old owner, that entry would point to freed memory
            // after the shape was destroyed.
            Shape* shape = (Shape*)a->pointer;
            if (shape == null)
            {
                return;
            }

            if (shape->body != null)
            {
                StoreOps.RemoveShape(shape->body, shape, globals);
            }

            BodyData* hot = StoreOps.Hot(body);
            if ((hot->flags & BodyFlags.Static) != 0 && a->enabled == 0)
            {
                StoreStaticShapePose(&shape->data, hot, a->position, a->rotation);
            }
            else
            {
                shape->data.localPosition = a->position;
                shape->data.localRotation = a->rotation;
            }

            StoreOps.AddShape(body, shape, globals);
        }

        private static void AddForce(BodyData* hot, CommandArgs* a)
        {
            if ((hot->flags & Immobile) != 0)
            {
                return;
            }

            switch (a->mode)
            {
                case ForceModeForce:
                    (*hot).Forces()->Force += a->vector;
                    break;
                case ForceModeImpulse:
                    (*hot).Forces()->Impulse += a->vector;
                    break;
                case ForceModeVelocityChange:
                    (*hot).Forces()->VelocityChange += a->vector;
                    break;
                case ForceModeAcceleration:
                    (*hot).Forces()->Acceleration += a->vector;
                    break;
            }

            WakeIfAsked(hot, a);
        }

        private static void AddTorque(BodyData* hot, CommandArgs* a)
        {
            if ((hot->flags & Immobile) != 0)
            {
                return;
            }

            switch (a->mode)
            {
                case ForceModeForce:
                    (*hot).Forces()->Torque += a->vector;
                    break;
                case ForceModeImpulse:
                    (*hot).Forces()->AngularImpulse += a->vector;
                    break;
                case ForceModeVelocityChange:
                    (*hot).Forces()->AngularVelocityChange += a->vector;
                    break;
                case ForceModeAcceleration:
                    (*hot).Forces()->AngularAcceleration += a->vector;
                    break;
            }

            WakeIfAsked(hot, a);
        }

        private static void WorldInertiaTensor(BodyData* hot, float* matrix)
        {
            quaternion rotation = math.mul((*hot).Pose()->Rotation, (*hot).Mass()->Transform.rot);
            float3x3 axes = new float3x3(rotation);
            float3 moments = (*hot).BodyMass()->Inertia;
            for (int row = 0; row < 3; row++)
            {
                for (int column = 0; column < 3; column++)
                {
                    float sum = 0.0f;
                    for (int axis = 0; axis < 3; axis++)
                    {
                        sum += axes[axis][row] * moments[axis] * axes[axis][column];
                    }

                    matrix[row * 3 + column] = sum;
                }
            }
        }

        /// <summary>
        /// Diagonalize a symmetric 3 by 3 matrix using cyclic Jacobi rotations.
        /// Store eigenvectors in the columns of axes. Both arrays use row-major order.
        /// </summary>
        private static void Diagonalize(double* a, double* axes)
        {
            for (int i = 0; i < 9; i++)
            {
                axes[i] = i % 4 == 0 ? 1.0 : 0.0;
            }

            for (int sweep = 0; sweep < 32; sweep++)
            {
                double offDiagonal = math.abs(a[1]) + math.abs(a[2]) + math.abs(a[5]);
                if (offDiagonal < 1e-14 * (math.abs(a[0]) + math.abs(a[4]) + math.abs(a[8])))
                {
                    return;
                }

                for (int p = 0; p < 2; p++)
                {
                    for (int q = p + 1; q < 3; q++)
                    {
                        double pq = a[p * 3 + q];
                        if (pq == 0.0)
                        {
                            continue;
                        }

                        double theta = (a[q * 3 + q] - a[p * 3 + p]) / (2.0 * pq);
                        double t = (theta >= 0.0 ? 1.0 : -1.0) / (math.abs(theta) + math.sqrt(theta * theta + 1.0));
                        double c = 1.0 / math.sqrt(t * t + 1.0);
                        double s = t * c;

                        for (int k = 0; k < 3; k++)
                        {
                            double akp = a[k * 3 + p];
                            double akq = a[k * 3 + q];
                            a[k * 3 + p] = c * akp - s * akq;
                            a[k * 3 + q] = s * akp + c * akq;
                        }

                        for (int k = 0; k < 3; k++)
                        {
                            double apk = a[p * 3 + k];
                            double aqk = a[q * 3 + k];
                            a[p * 3 + k] = c * apk - s * aqk;
                            a[q * 3 + k] = s * apk + c * aqk;
                        }

                        for (int k = 0; k < 3; k++)
                        {
                            double vkp = axes[k * 3 + p];
                            double vkq = axes[k * 3 + q];
                            axes[k * 3 + p] = c * vkp - s * vkq;
                            axes[k * 3 + q] = s * vkp + c * vkq;
                        }
                    }
                }
            }
        }

        private static quaternion QuaternionFromAxes(double* m)
        {
            double trace = m[0] + m[4] + m[8];
            double x, y, z, w;
            if (trace > 0.0)
            {
                double s = math.sqrt(trace + 1.0) * 2.0;
                w = 0.25 * s;
                x = (m[7] - m[5]) / s;
                y = (m[2] - m[6]) / s;
                z = (m[3] - m[1]) / s;
            }
            else if (m[0] > m[4] && m[0] > m[8])
            {
                double s = math.sqrt(1.0 + m[0] - m[4] - m[8]) * 2.0;
                w = (m[7] - m[5]) / s;
                x = 0.25 * s;
                y = (m[1] + m[3]) / s;
                z = (m[2] + m[6]) / s;
            }
            else if (m[4] > m[8])
            {
                double s = math.sqrt(1.0 + m[4] - m[0] - m[8]) * 2.0;
                w = (m[2] - m[6]) / s;
                x = (m[1] + m[3]) / s;
                y = 0.25 * s;
                z = (m[5] + m[7]) / s;
            }
            else
            {
                double s = math.sqrt(1.0 + m[8] - m[0] - m[4]) * 2.0;
                w = (m[3] - m[1]) / s;
                x = (m[2] + m[6]) / s;
                y = (m[5] + m[7]) / s;
                z = 0.25 * s;
            }

            return new quaternion((float)x, (float)y, (float)z, (float)w);
        }

        private static float3 ShapeCenterInBody(ShapeData* shape, float3 localCenter)
        {
            return shape->localPosition + math.rotate(shape->localRotation, localCenter);
        }

        /// <summary>
        /// Distribute mass among non-trigger shapes by volume, following PhysX updateMassAndInertia.
        /// </summary>
        private static void RecomputeMassProperties(Body* body, BodyData* hot, CommandArgs* a)
        {
            using Unity.Profiling.ProfilerMarker.AutoScope span = BackendMarkers.RecomputeMassProperties.Auto();

            bool overrideCenter = (a->flags & OverrideCenterOfMass) != 0;
            bool overrideInertia = (a->flags & OverrideInertiaTensor) != 0;

            float totalVolume = 0.0f;
            float3 weightedCenter = float3.zero;
            for (uint i = 0; i < body->shapeCount; i++)
            {
                ShapeData* shape = body->shapes[i];

                // PhysX cannot calculate mass properties for a triangle mesh and rejects the calculation for the whole
                // body. A moving body with a triangle mesh must be kinematic.
                if (shape->geometryType == StoreConstants.GeometryTriangleMesh)
                {
                    totalVolume = 0.0f;
                    break;
                }

                if (TakesMass(shape, out float volume, out float3 localCenter, out _))
                {
                    totalVolume += volume;
                    weightedCenter += ShapeCenterInBody(shape, localCenter) * volume;
                }
            }

            if (totalVolume <= 0.0f)
            {
                (*hot).Mass()->Transform.pos = overrideCenter ? a->vector : float3.zero;
                SetInertia(hot, overrideInertia ? a->vector2 : new float3(1.0f));
                BodyKinematics.SetInertiaRotation(*hot, overrideInertia ? a->rotation : quaternion.identity);
                return;
            }

            float3 center = overrideCenter ? a->vector : weightedCenter / totalVolume;
            (*hot).Mass()->Transform.pos = center;
            if (overrideInertia)
            {
                SetInertia(hot, a->vector2);
                BodyKinematics.SetInertiaRotation(*hot, a->rotation);
    return;
            }

            double* tensor = stackalloc double[9];
            double* principalAxes = stackalloc double[9];
            for (int i = 0; i < 9; i++)
            {
                tensor[i] = 0.0;
            }

            for (uint i = 0; i < body->shapeCount; i++)
            {
                ShapeData* shape = body->shapes[i];
                if (TakesMass(shape, out float volume, out float3 localCenter, out float3x3 inertiaPerUnitMass))
                {
                    AddToTensor(tensor, shape, (*hot).BodyMass()->Mass * (double)volume / totalVolume, ShapeCenterInBody(shape, localCenter) - center, inertiaPerUnitMass);
                }
            }

            Diagonalize(tensor, principalAxes);

            // The calculated eigenvectors may form a left-handed basis. A quaternion can represent only a proper
            // rotation.
            double determinant = principalAxes[0] * (principalAxes[4] * principalAxes[8] - principalAxes[5] * principalAxes[7])
                - principalAxes[1] * (principalAxes[3] * principalAxes[8] - principalAxes[5] * principalAxes[6])
                + principalAxes[2] * (principalAxes[3] * principalAxes[7] - principalAxes[4] * principalAxes[6]);
            if (determinant < 0.0)
            {
                principalAxes[2] = -principalAxes[2];
                principalAxes[5] = -principalAxes[5];
                principalAxes[8] = -principalAxes[8];
            }

            SetInertia(hot, new float3((float)tensor[0], (float)tensor[4], (float)tensor[8]));
            BodyKinematics.SetInertiaRotation(*hot, QuaternionFromAxes(principalAxes));
        }

        private static bool TakesMass(ShapeData* shape, out float volume, out float3 center, out float3x3 inertiaPerUnitMass)
        {
            bool isTrigger = (shape->flags & StoreConstants.ShapeFlagTrigger) != 0;
            bool hasMass = ShapeMath.MassProperties(shape, out volume, out center, out inertiaPerUnitMass);
            return !isTrigger && hasMass;
        }

        /// <summary>
        /// Add a shape's rotated inertia and parallel axis contribution to the body's inertia tensor.
        /// </summary>
        private static void AddToTensor(double* tensor, ShapeData* shape, double mass, float3 offset, float3x3 inertiaPerUnitMass)
        {
            float3x3 axes = new float3x3(shape->localRotation);
            double offsetSquared = math.dot(offset, offset);
            for (int row = 0; row < 3; row++)
            {
                for (int column = 0; column < 3; column++)
                {
                    double value = (row == column ? offsetSquared : 0.0) - (double)offset[row] * offset[column];
                    for (int first = 0; first < 3; first++)
                    {
                        for (int second = 0; second < 3; second++)
                        {
                            value += (double)axes[first][row] * inertiaPerUnitMass[first][second] * axes[second][column];
                        }
                    }

                    tensor[row * 3 + column] += mass * value;
                }
            }
        }

        private static float3 AccumulatedForce(BodyData* hot, float deltaTime)
        {
            float perStep = deltaTime > 0.0f ? 1.0f / deltaTime : 0.0f;
            return (*hot).Forces()->Force + (*hot).Forces()->Acceleration * (*hot).BodyMass()->Mass + (*hot).Forces()->Impulse * perStep + (*hot).Forces()->VelocityChange * (*hot).BodyMass()->Mass * perStep;
        }

        /// <summary>
        /// Apply the principal moments and axes of inertia to an angular acceleration or velocity change.
        /// </summary>
        private static float3 ApplyInertia(BodyData* hot, float3 angular)
        {
            quaternion rotation = math.mul((*hot).Pose()->Rotation, (*hot).Mass()->Transform.rot);
            return math.rotate(rotation, math.rotate(math.conjugate(rotation), angular) * (*hot).BodyMass()->Inertia);
        }

        private static float3 AccumulatedTorque(BodyData* hot, float deltaTime)
        {
            float perStep = deltaTime > 0.0f ? 1.0f / deltaTime : 0.0f;
            return (*hot).Forces()->Torque
                + ApplyInertia(hot, (*hot).Forces()->AngularAcceleration)
                + (*hot).Forces()->AngularImpulse * perStep
                + ApplyInertia(hot, (*hot).Forces()->AngularVelocityChange) * perStep;
        }

        private static float3 VelocityAt(BodyData* hot, float3 worldPoint)
        {
            float3 worldCenter = (*hot).Pose()->Position + math.rotate((*hot).Pose()->Rotation, (*hot).Mass()->Transform.pos);
            return (*hot).Velocity()->Linear + math.cross(BodyKinematics.WorldAngularVelocity(*hot), worldPoint - worldCenter);
        }

        private static void ClosestPointOnBounds(Body* body, BodyData* hot, CommandArgs* a)
        {
            using Unity.Profiling.ProfilerMarker.AutoScope span = BackendMarkers.ClosestPointOnBounds.Auto();

            float3 position = a->vector;
            float3 closest = position;
            if (body->shapeCount != 0)
            {
                float3 lower = new float3(float.PositiveInfinity);
                float3 upper = new float3(float.NegativeInfinity);
                for (uint i = 0; i < body->shapeCount; i++)
                {
                    ShapeData* shape = body->shapes[i];
                    ShapeMath.WorldBounds(shape, shape->localPosition, shape->localRotation, (*hot).Pose()->Position, (*hot).Pose()->Rotation, out float3 shapeLower, out float3 shapeUpper);
                    lower = math.min(lower, shapeLower);
                    upper = math.max(upper, shapeUpper);
                }

                closest = math.clamp(position, lower, upper);
            }

            a->vector = closest;
            a->scalar = math.lengthsq(closest - position);
        }

        private static void SetKinematicTarget(BodyData* hot, CommandArgs* a)
        {
            // Bodies that are not simulated are skipped during the step. Ignore their targets here so that a stored
            // target cannot cause a teleport later.
            if ((hot->flags & BodyFlags.Simulated) != 0)
            {
                (*hot).Target()->Position = a->position;
                (*hot).Target()->Rotation = a->rotation;
                hot->flags |= BodyFlags.HasKinematicTarget;
            }

            Wake(hot);
        }
    }
}
