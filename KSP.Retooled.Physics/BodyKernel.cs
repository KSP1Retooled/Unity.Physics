using System;
using AOT;
using Unity.Burst;
using Unity.Mathematics;
using UnityEngine;

namespace KSP.Retooled.Physics
{
    /// <summary>The plugin API's body commands, by the id Body.cpp passes. Attach checks the count against the plugin's.</summary>
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
    /// Everything a body command does. The accumulated forces and torques it collects are consumed by the step in
    /// <see cref="ApplyForcesJob"/>.
    /// </summary>
    /// <remarks>
    /// As with the query functions, the plugin holds a Burst function pointer, and until the async compile delivers it is a
    /// stub running the same code in Mono. Body commands arrive on the main thread outside a step.
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
            using Unity.Profiling.ProfilerMarker.AutoScope span = BackendMarkers.BodyCommand.Auto();

            bool ranInMono = false;
            RunInMono(command, handle, args, globals, ref ranInMono);
            if (!ranInMono)
            {
                Execute((BodyCommand)command, (NativeStore.Body*)handle, args, globals);
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
                case BodyCommand.GetFlags:
                    a->flags = (uint)(hot->flags & NativeStore.BodyFlags.PublicMask);
                    break;
                case BodyCommand.SetMass:
                    hot->mass->Mass = a->scalar;
                    hot->mass->InverseMass = InverseOrZero(a->scalar);
                    break;
                case BodyCommand.GetMass:
                    a->scalar = hot->mass->Mass;
                    break;
                case BodyCommand.SetDamping:
                    hot->limits->LinearDamping = a->scalar;
                    hot->limits->AngularDamping = a->scalar2;
                    break;
                case BodyCommand.GetDamping:
                    a->scalar = hot->limits->LinearDamping;
                    a->scalar2 = hot->limits->AngularDamping;
                    break;
                case BodyCommand.SetPose:
                    hot->pose->Position = a->position;
                    hot->pose->Rotation = a->rotation;
                    WakeIfAsked(hot, a);
                    StoreOps.MarkPoseChanged(body, hot->flags);
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
                    hot->velocity->Angular = a->vector;
                    WakeIfAsked(hot, a);
                    break;
                case BodyCommand.GetAngularVelocity:
                    a->vector = hot->velocity->Angular;
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
                    a->vector = hot->mass->Inertia;
                    break;
                case BodyCommand.SetInertiaTensorRotation:
                    hot->mass->InertiaRotation = a->rotation;
                    break;
                case BodyCommand.GetInertiaTensorRotation:
                    a->rotation = hot->mass->InertiaRotation;
                    break;
                case BodyCommand.SetLocalCenterOfMass:
                    hot->mass->LocalCenterOfMass = a->vector;
                    break;
                case BodyCommand.GetLocalCenterOfMass:
                    a->vector = hot->mass->LocalCenterOfMass;
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
                case BodyCommand.GetIsSleeping:
                    a->enabled = (hot->flags & NativeStore.BodyFlags.Sleeping) != 0 ? 1u : 0u;
                    break;
                case BodyCommand.Sleep:
                    // A sleeping body doesn't step, so a target kept here would land when it next wakes.
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
            hot->mass->Inertia = inertia;
            hot->mass->InverseInertia = new float3(InverseOrZero(inertia.x), InverseOrZero(inertia.y), InverseOrZero(inertia.z));
        }

        private static void SetFlags(NativeStore.Body* body, NativeStore.BodyData* hot, NativeStore.CommandArgs* a)
        {
            NativeStore.BodyFlags bits = (NativeStore.BodyFlags)a->flags & NativeStore.BodyFlags.PublicMask;
            if (a->enabled != 0)
            {
                hot->flags |= bits;
            }
            else
            {
                hot->flags &= ~bits;
            }

            // Only a simulated kinematic body reaches a target, and the step skips the rest, so one left here would land later.
            const NativeStore.BodyFlags followsTargets = NativeStore.BodyFlags.Kinematic | NativeStore.BodyFlags.Simulated;
            if ((hot->flags & followsTargets) != followsTargets)
            {
                hot->flags &= ~NativeStore.BodyFlags.HasKinematicTarget;
            }

            StoreOps.MarkPoseChanged(body, hot->flags);
        }

        /// <summary>
        /// Unity gives a static body's shape a world-space pose here and in SetPose, unlike a dynamic body's, whatever the plugin
        /// API's comments say. The store keeps every shape relative to its body.
        /// </summary>
        internal static void StoreStaticShapePose(NativeStore.ShapeData* shape, NativeStore.BodyData* body, float3 worldPosition, quaternion worldRotation)
        {
            quaternion inverse = math.conjugate(body->pose->Rotation);
            shape->localPosition = math.rotate(inverse, worldPosition - body->pose->Position);
            shape->localRotation = math.mul(inverse, worldRotation);
        }

        private static void AttachShape(NativeStore.Body* body, NativeStore.CommandArgs* a, NativeStore.StoreGlobals* globals)
        {
            // A shape has one owner, and the old owner's entry would dangle once the shape is destroyed.
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
            quaternion rotation = math.mul(hot->pose->Rotation, hot->mass->InertiaRotation);
            float3x3 axes = new float3x3(rotation);
            float3 moments = hot->mass->Inertia;
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
        /// Rotates a symmetric matrix until it is diagonal, by cyclic Jacobi, collecting the rotation in
        /// <paramref name="axes"/>, whose columns end up as the eigenvectors. Both index as [row * 3 + column].
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

        /// <summary>Distributes the body's mass over its non-trigger shapes by volume, as PhysX's updateMassAndInertia does.</summary>
        private static void RecomputeMassProperties(NativeStore.Body* body, NativeStore.BodyData* hot, NativeStore.CommandArgs* a)
        {
            bool overrideCenter = (a->flags & OverrideCenterOfMass) != 0;
            bool overrideInertia = (a->flags & OverrideInertiaTensor) != 0;

            float totalVolume = 0.0f;
            float3 weightedCenter = float3.zero;
            for (uint i = 0; i < body->shapeCount; i++)
            {
                NativeStore.ShapeData* shape = body->shapes[i];

                // PhysX can't integrate a triangle mesh, and gives up on the whole body. Only a kinematic body can carry one.
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
                hot->mass->LocalCenterOfMass = overrideCenter ? a->vector : float3.zero;
                SetInertia(hot, overrideInertia ? a->vector2 : new float3(1.0f));
                hot->mass->InertiaRotation = overrideInertia ? a->rotation : quaternion.identity;
                return;
            }

            float3 center = overrideCenter ? a->vector : weightedCenter / totalVolume;
            hot->mass->LocalCenterOfMass = center;
            if (overrideInertia)
            {
                SetInertia(hot, a->vector2);
                hot->mass->InertiaRotation = a->rotation;
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
                    AddToTensor(tensor, shape, hot->mass->Mass * (double)volume / totalVolume, ShapeCenterInBody(shape, localCenter) - center, inertiaPerUnitMass);
                }
            }

            Diagonalize(tensor, principalAxes);

            // Eigenvectors may come out left-handed, and a quaternion can only express a proper rotation.
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
            hot->mass->InertiaRotation = QuaternionFromAxes(principalAxes);
        }

        private static bool TakesMass(NativeStore.ShapeData* shape, out float volume, out float3 center, out float3x3 inertiaPerUnitMass)
        {
            bool isTrigger = (shape->flags & NativeStore.ShapeFlagTrigger) != 0;
            bool hasMass = ShapeMath.MassProperties(shape, out volume, out center, out inertiaPerUnitMass);
            return !isTrigger && hasMass;
        }

        /// <summary>Adds a shape of the given mass to a tensor about the body's centre of mass: its own inertia rotated into the body's frame, and the parallel axis term.</summary>
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
            return hot->forces->Force + hot->forces->Acceleration * hot->mass->Mass + hot->forces->Impulse * perStep + hot->forces->VelocityChange * hot->mass->Mass * perStep;
        }

        /// <summary>Acceleration and velocity-change torques go through the principal axes, because they carry no inertia of their own.</summary>
        private static float3 ApplyInertia(NativeStore.BodyData* hot, float3 angular)
        {
            quaternion rotation = math.mul(hot->pose->Rotation, hot->mass->InertiaRotation);
            return math.rotate(rotation, math.rotate(math.conjugate(rotation), angular) * hot->mass->Inertia);
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
            float3 worldCenter = hot->pose->Position + math.rotate(hot->pose->Rotation, hot->mass->LocalCenterOfMass);
            return hot->velocity->Linear + math.cross(hot->velocity->Angular, worldPoint - worldCenter);
        }

        private static void ClosestPointOnBounds(NativeStore.Body* body, NativeStore.BodyData* hot, NativeStore.CommandArgs* a)
        {
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
            // The step skips a body that isn't simulated, so it could only ever reach this target by a later teleport.
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
