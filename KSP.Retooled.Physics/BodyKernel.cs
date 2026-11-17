using System;
using AOT;
using Unity.Burst;
using Unity.Mathematics;
using UnityEngine;

namespace KSP.Retooled.Physics
{
    /// <summary>
    /// Defines the plugin API's body commands using the ids passed by Body.cpp. Attach checks that the number of
    /// commands matches the plugin's count.
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
    /// Implements body commands. Forces and torques accumulate here until <see cref="ApplyForcesJob"/> applies them
    /// during the next step.
    /// </summary>
    /// <remarks>
    /// The plugin stores a Burst function pointer, as it does for query functions. Until asynchronous compilation
    /// finishes, that pointer refers to a stub that runs the same code in Mono. Body commands arrive outside a step,
    /// mostly on the main thread. Unity also reads poses from worker threads. The plugin answers those reads itself,
    /// through the rows in the state that <see cref="BodyEntities"/> left them, and so it does for the main thread's
    /// reads right after a step (see Body.cpp). GetPose reaches this kernel only on the main thread at other times.
    /// </remarks>
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

        private const NativeStore.BodyFlags Immobile = NativeStore.BodyFlags.Kinematic | NativeStore.BodyFlags.Static;

        [BurstCompile(DisableSafetyChecks = true)]
        [MonoPInvokeCallback(typeof(NativeStore.CommandFunction))]
        internal static void Run(uint command, void* handle, NativeStore.CommandArgs* args, NativeStore.StoreGlobals* globals)
        {
            // Unity reads every awake body's pose after each step, and the poses of the moved bodies before each step. A
            // span costs about as much as such a read, so reads run without the category's span. A read that does more
            // work, such as GetClosestWorldPointOnBounds, still has its own span.
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

        private static void Dispatch(uint command, void* handle, NativeStore.CommandArgs* args, NativeStore.StoreGlobals* globals)
        {
            bool ranInMono = false;
            RunInMono(command, handle, args, globals, ref ranInMono);
            if (!ranInMono)
            {
                Execute((BodyCommand)command, (NativeStore.Body*)handle, args, globals);
            }
        }

        /// <summary>
        /// Burst omits this method, so it runs only in Mono. The Mono version must catch the exception here.
        /// </summary>
        [BurstDiscard]
        private static void RunInMono(uint command, void* handle, NativeStore.CommandArgs* args, NativeStore.StoreGlobals* globals, ref bool ranInMono)
        {
            ranInMono = true;
            BackendHost.NoteManagedCommand();
            try
            {
                Execute((BodyCommand)command, (NativeStore.Body*)handle, args, globals);
            }
            catch (Exception e)
            {
                Debug.LogException(e);
            }
        }

        private static void Execute(BodyCommand command, NativeStore.Body* body, NativeStore.CommandArgs* a, NativeStore.StoreGlobals* globals)
        {
            NativeStore.BodyData* hot = StoreOps.Hot(body);
            switch (command)
            {
                case BodyCommand.SetFlags:
                    SetFlags(body, hot, a);
                    break;
                case BodyCommand.SetMass:
                    hot->bodyMass->Mass = a->scalar;
                    hot->mass->InverseMass = InverseOrZero(a->scalar);
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
                        using Unity.Profiling.ProfilerMarker.AutoScope span = BackendMarkers.DetachShape.Auto();
                        StoreOps.RemoveShape(body, (NativeStore.Shape*)a->pointer, globals);
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
                    WakeIfAsked(hot, a);
                    break;
                case BodyCommand.GetLinearVelocity:
                    a->vector = hot->velocity->Linear;
                    break;
                case BodyCommand.SetAngularVelocity:
                    StoreOps.SetWorldAngularVelocity(*hot, a->vector);
                    WakeIfAsked(hot, a);
                    break;
                case BodyCommand.GetAngularVelocity:
                    a->vector = StoreOps.WorldAngularVelocity(*hot);
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
                    SetInertia(hot, a->vector);
                    break;
                case BodyCommand.GetInertiaTensor:
                    a->vector = hot->bodyMass->Inertia;
                    break;
                case BodyCommand.SetInertiaTensorRotation:
                    StoreOps.SetInertiaRotation(*hot, a->rotation);
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
                    a->vector = VelocityAt(hot, hot->pose->Position + math.rotate(hot->pose->Rotation, a->vector));
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
                    bool hasTarget = (hot->flags & NativeStore.BodyFlags.HasKinematicTarget) != 0;
                    a->position = hasTarget ? hot->target->Position : hot->pose->Position;
                    a->rotation = hasTarget ? hot->target->Rotation : hot->pose->Rotation;
                    break;
                case BodyCommand.Sleep:
                    // Sleeping bodies are not stepped. Clear the target here so that it is not applied when the body
                    // next wakes.
                    hot->flags |= NativeStore.BodyFlags.Sleeping;
                    hot->flags &= ~NativeStore.BodyFlags.HasKinematicTarget;
                    hot->sleep->WakeCounter = 0.0f;
                    hot->velocity->Linear = float3.zero;
                    hot->velocity->Angular = float3.zero;
                    break;
                case BodyCommand.WakeUp:
                    Wake(hot);
                    break;
                case BodyCommand.SetWakeCounter:
                    hot->sleep->WakeCounter = a->scalar;
                    if (a->scalar > 0.0f)
                    {
                        hot->flags &= ~NativeStore.BodyFlags.Sleeping;
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

        private static float InverseOrZero(float value)
        {
            return value > 0.0f ? 1.0f / value : 0.0f;
        }

        private static void Wake(NativeStore.BodyData* hot)
        {
            hot->flags &= ~NativeStore.BodyFlags.Sleeping;
            hot->sleep->WakeCounter = math.max(hot->sleep->WakeCounter, StoreOps.WakeCounterReset);
        }

        private static void WakeIfAsked(NativeStore.BodyData* hot, NativeStore.CommandArgs* a)
        {
            if (a->enabled != 0)
            {
                Wake(hot);
            }
        }

        private static void SetInertia(NativeStore.BodyData* hot, float3 inertia)
        {
            hot->bodyMass->Inertia = inertia;
            hot->mass->InverseInertia = new float3(InverseOrZero(inertia.x), InverseOrZero(inertia.y), InverseOrZero(inertia.z));
        }

        /// <summary>
        /// Unity sets the pose of every body beneath a transform that moved. Most of those bodies are already at the
        /// supplied pose. Moving a static body requires rebuilding the static tree.
        /// </summary>
        private static void SetPose(NativeStore.Body* body, NativeStore.BodyData* hot, NativeStore.CommandArgs* a)
        {
            float3 position = hot->pose->Position;
            quaternion rotation = hot->pose->Rotation;

            // A pose that equals the stored one to the last bit changes nothing. Setting the rotation again would convert
            // the angular velocity to world space and back, which can change its last bits, so the body stays as it is.
            // The body still wakes if Unity asks, as a PhysX body does. The bits are compared rather than the floats, so
            // that -0 against 0, or a NaN, still counts as a change.
            if (math.all(math.asuint(a->position) == math.asuint(position)) && math.all(math.asuint(a->rotation.value) == math.asuint(rotation.value)))
            {
                WakeIfAsked(hot, a);
                return;
            }

            hot->pose->Position = a->position;
            StoreOps.SetRotation(*hot, a->rotation);
            WakeIfAsked(hot, a);
            if (math.any(hot->pose->Position != position) || math.any(hot->pose->Rotation.value != rotation.value))
            {
                if ((hot->flags & NativeStore.BodyFlags.Static) != 0)
                {
                    StoreOps.MarkStaticMoved(body);
                }
                else
                {
                    StoreOps.MarkPoseChanged(body, hot->flags);
                }
            }
        }

        private static void SetFlags(NativeStore.Body* body, NativeStore.BodyData* hot, NativeStore.CommandArgs* a)
        {
            NativeStore.BodyFlags bits = (NativeStore.BodyFlags)a->flags & NativeStore.BodyFlags.PublicMask;
            NativeStore.BodyFlags before = hot->flags;
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
            const NativeStore.BodyFlags followsTargets = NativeStore.BodyFlags.Kinematic | NativeStore.BodyFlags.Simulated;
            if ((hot->flags & followsTargets) != followsTargets)
            {
                hot->flags &= ~NativeStore.BodyFlags.HasKinematicTarget;
            }

            // Unity sets flags far more often than their values change. The static tree needs to be updated both when a
            // body becomes static and when it stops being static.
            if (hot->flags != before)
            {
                StoreOps.MarkPoseChanged(body, before | hot->flags);
            }

            // Whether a body moves determines the position of its joints in the vessel forest. Whether it is simulated
            // determines where the bodies for its joints are built.
            if (((hot->flags ^ before) & (NativeStore.BodyFlags.Kinematic | NativeStore.BodyFlags.Simulated)) != 0)
            {
                JointTopology.Changed();
            }
        }

        /// <summary>
        /// Unity supplies world-space poses for static bodies' shapes here and in SetPose. It supplies local poses for
        /// dynamic bodies' shapes, despite the comments in the plugin API. The store keeps every shape's pose relative
        /// to its body.
        /// </summary>
        internal static void StoreStaticShapePose(NativeStore.ShapeData* shape, NativeStore.BodyData* body, float3 worldPosition, quaternion worldRotation)
        {
            quaternion inverse = math.conjugate(body->pose->Rotation);
            shape->localPosition = math.rotate(inverse, worldPosition - body->pose->Position);
            shape->localRotation = math.mul(inverse, worldRotation);
        }

        private static void AttachShape(NativeStore.Body* body, NativeStore.CommandArgs* a, NativeStore.StoreGlobals* globals)
        {
            using Unity.Profiling.ProfilerMarker.AutoScope span = BackendMarkers.AttachShape.Auto();

            // A shape has one owner. If its entry remained with the old owner, that entry would point to freed memory
            // after the shape was destroyed.
            NativeStore.Shape* shape = (NativeStore.Shape*)a->pointer;
            if (shape == null)
            {
                return;
            }

            if (shape->body != null)
            {
                StoreOps.RemoveShape(shape->body, shape, globals);
            }

            NativeStore.BodyData* hot = StoreOps.Hot(body);
            if ((hot->flags & NativeStore.BodyFlags.Static) != 0 && a->enabled == 0)
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

        private static void AddForce(NativeStore.BodyData* hot, NativeStore.CommandArgs* a)
        {
            if ((hot->flags & Immobile) != 0)
            {
                return;
            }

            switch (a->mode)
            {
                case ForceModeForce:
                    hot->forces->Force += a->vector;
                    break;
                case ForceModeImpulse:
                    hot->forces->Impulse += a->vector;
                    break;
                case ForceModeVelocityChange:
                    hot->forces->VelocityChange += a->vector;
                    break;
                case ForceModeAcceleration:
                    hot->forces->Acceleration += a->vector;
                    break;
            }

            WakeIfAsked(hot, a);
        }

        private static void AddTorque(NativeStore.BodyData* hot, NativeStore.CommandArgs* a)
        {
            if ((hot->flags & Immobile) != 0)
            {
                return;
            }

            switch (a->mode)
            {
                case ForceModeForce:
                    hot->forces->Torque += a->vector;
                    break;
                case ForceModeImpulse:
                    hot->forces->AngularImpulse += a->vector;
                    break;
                case ForceModeVelocityChange:
                    hot->forces->AngularVelocityChange += a->vector;
                    break;
                case ForceModeAcceleration:
                    hot->forces->AngularAcceleration += a->vector;
                    break;
            }

            WakeIfAsked(hot, a);
        }

        private static void WorldInertiaTensor(NativeStore.BodyData* hot, float* matrix)
        {
            quaternion rotation = math.mul(hot->pose->Rotation, hot->mass->Transform.rot);
            float3x3 axes = new float3x3(rotation);
            float3 moments = hot->bodyMass->Inertia;
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
        /// Uses the cyclic Jacobi method to rotate a symmetric matrix until it is diagonal. The accumulated rotation is
        /// stored in <paramref name="axes"/>, whose columns become the eigenvectors. Both arrays use the index [row * 3
        /// + column].
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

        private static float3 ShapeCenterInBody(NativeStore.ShapeData* shape, float3 localCenter)
        {
            return shape->localPosition + math.rotate(shape->localRotation, localCenter);
        }

        /// <summary>
        /// Distributes the body's mass among its shapes in proportion to their volumes, excluding triggers. This
        /// follows PhysX's updateMassAndInertia.
        /// </summary>
        private static void RecomputeMassProperties(NativeStore.Body* body, NativeStore.BodyData* hot, NativeStore.CommandArgs* a)
        {
            using Unity.Profiling.ProfilerMarker.AutoScope span = BackendMarkers.RecomputeMassProperties.Auto();

            bool overrideCenter = (a->flags & OverrideCenterOfMass) != 0;
            bool overrideInertia = (a->flags & OverrideInertiaTensor) != 0;

            float totalVolume = 0.0f;
            float3 weightedCenter = float3.zero;
            for (uint i = 0; i < body->shapeCount; i++)
            {
                NativeStore.ShapeData* shape = body->shapes[i];

                // PhysX cannot calculate mass properties for a triangle mesh and rejects the calculation for the whole
                // body. A moving body with a triangle mesh must be kinematic.
                if (shape->geometryType == NativeStore.GeometryTriangleMesh)
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
                hot->mass->Transform.pos = overrideCenter ? a->vector : float3.zero;
                SetInertia(hot, overrideInertia ? a->vector2 : new float3(1.0f));
                StoreOps.SetInertiaRotation(*hot, overrideInertia ? a->rotation : quaternion.identity);
                return;
            }

            float3 center = overrideCenter ? a->vector : weightedCenter / totalVolume;
            hot->mass->Transform.pos = center;
            if (overrideInertia)
            {
                SetInertia(hot, a->vector2);
                StoreOps.SetInertiaRotation(*hot, a->rotation);
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
                NativeStore.ShapeData* shape = body->shapes[i];
                if (TakesMass(shape, out float volume, out float3 localCenter, out float3x3 inertiaPerUnitMass))
                {
                    AddToTensor(tensor, shape, hot->bodyMass->Mass * (double)volume / totalVolume, ShapeCenterInBody(shape, localCenter) - center, inertiaPerUnitMass);
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
            StoreOps.SetInertiaRotation(*hot, QuaternionFromAxes(principalAxes));
        }

        private static bool TakesMass(NativeStore.ShapeData* shape, out float volume, out float3 center, out float3x3 inertiaPerUnitMass)
        {
            bool isTrigger = (shape->flags & NativeStore.ShapeFlagTrigger) != 0;
            bool hasMass = ShapeMath.MassProperties(shape, out volume, out center, out inertiaPerUnitMass);
            return !isTrigger && hasMass;
        }

        /// <summary>
        /// Adds the contribution of a shape with the given mass to the inertia tensor about the body's centre of mass.
        /// This includes the shape's inertia rotated into the body's frame and the parallel axis term.
        /// </summary>
        private static void AddToTensor(double* tensor, NativeStore.ShapeData* shape, double mass, float3 offset, float3x3 inertiaPerUnitMass)
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

        private static float3 AccumulatedForce(NativeStore.BodyData* hot, float deltaTime)
        {
            float perStep = deltaTime > 0.0f ? 1.0f / deltaTime : 0.0f;
            return hot->forces->Force + hot->forces->Acceleration * hot->bodyMass->Mass + hot->forces->Impulse * perStep + hot->forces->VelocityChange * hot->bodyMass->Mass * perStep;
        }

        /// <summary>
        /// Converts angular acceleration or a change in angular velocity into torque using the principal axes and
        /// moments of inertia. These inputs do not include the effect of inertia.
        /// </summary>
        private static float3 ApplyInertia(NativeStore.BodyData* hot, float3 angular)
        {
            quaternion rotation = math.mul(hot->pose->Rotation, hot->mass->Transform.rot);
            return math.rotate(rotation, math.rotate(math.conjugate(rotation), angular) * hot->bodyMass->Inertia);
        }

        private static float3 AccumulatedTorque(NativeStore.BodyData* hot, float deltaTime)
        {
            float perStep = deltaTime > 0.0f ? 1.0f / deltaTime : 0.0f;
            return hot->forces->Torque
                + ApplyInertia(hot, hot->forces->AngularAcceleration)
                + hot->forces->AngularImpulse * perStep
                + ApplyInertia(hot, hot->forces->AngularVelocityChange) * perStep;
        }

        private static float3 VelocityAt(NativeStore.BodyData* hot, float3 worldPoint)
        {
            float3 worldCenter = hot->pose->Position + math.rotate(hot->pose->Rotation, hot->mass->Transform.pos);
            return hot->velocity->Linear + math.cross(StoreOps.WorldAngularVelocity(*hot), worldPoint - worldCenter);
        }

        private static void ClosestPointOnBounds(NativeStore.Body* body, NativeStore.BodyData* hot, NativeStore.CommandArgs* a)
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
                    NativeStore.ShapeData* shape = body->shapes[i];
                    ShapeMath.WorldBounds(shape, shape->localPosition, shape->localRotation, hot->pose->Position, hot->pose->Rotation, out float3 shapeLower, out float3 shapeUpper);
                    lower = math.min(lower, shapeLower);
                    upper = math.max(upper, shapeUpper);
                }

                closest = math.clamp(position, lower, upper);
            }

            a->vector = closest;
            a->scalar = math.lengthsq(closest - position);
        }

        private static void SetKinematicTarget(NativeStore.BodyData* hot, NativeStore.CommandArgs* a)
        {
            // Bodies that are not simulated are skipped during the step. Ignore their targets here so that a stored
            // target cannot cause a teleport later.
            if ((hot->flags & NativeStore.BodyFlags.Simulated) != 0)
            {
                hot->target->Position = a->position;
                hot->target->Rotation = a->rotation;
                hot->flags |= NativeStore.BodyFlags.HasKinematicTarget;
            }

            Wake(hot);
        }
    }
}
