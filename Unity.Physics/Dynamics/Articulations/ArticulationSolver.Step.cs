using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Numerics.Kernels;
using Unity.Physics.Dynamics.Bodies;
using static Unity.Physics.Dynamics.Contacts.ManifoldRecords;
using Unity.Physics.Store.Articulations;
using Unity.Physics.Store.Bodies;
using Unity.Physics.Store.Memory;
using Unity.Physics.Store.World;
using static Unity.Physics.Dynamics.Articulations.ArticulationKinematics;

namespace Unity.Physics.Dynamics.Articulations
{
    internal static unsafe partial class ArticulationSolver
    {
        internal static void Begin(WorldData* world, float deltaTime)
        {
            if (deltaTime <= 0f)
            {
                return;
            }

            for (ArticulationData* tree = world->articulations; tree != null; tree = tree->Next)
            {
                if (tree->Count == 0 || (tree->Flags & ArticulationData.Simulated) == 0)
                {
                    continue;
                }

                if (tree->Asleep)
                {
                    WakeFromContact(tree);
                    if (tree->Asleep)
                    {
                        continue;
                    }
                }

                Refresh(tree, true);
                ClampLinkVelocities(tree);
                ArticulationFrame* frames = BuildFrames(tree);
                bool friction = ConfigureDrives(tree, frames, deltaTime);
                SolveForces(tree, frames, deltaTime, friction);

                BodyData* root = StoreMemory.Row(tree->Links[0]->Body);
                SetVelocity(root, Velocity(root) + deltaTime * frames[0].Motion);
                for (int i = 1; i < tree->Count; i++)
                {
                    ArticulationLink* link = tree->Links[i];
                    link->StepStartVelocity = link->Velocity;
                    link->Velocity += deltaTime * (float3)frames[i].Result;
                }

                Refresh(tree, false);
                for (int i = 0; i < tree->Count; i++)
                {
                    ArticulationLink* link = tree->Links[i];
                    BodyData* body = StoreMemory.Row(link->Body);
                    link->FreeLinearVelocity = body->velocity->Linear;
                    link->FreeAngularVelocity = body->velocity->Angular;
                    *body->forces = default;
                }

                FreeFrames(frames);
            }
        }

        private static void WakeFromContact(ArticulationData* tree)
        {
            for (int i = 0; i < tree->Count; i++)
            {
                if ((StoreMemory.Row(tree->Links[i]->Body)->flags & BodyFlags.Sleeping) == 0)
                {
                    ArticulationSleep.Wake(tree);
                    return;
                }
            }
        }

        private static void ClampLinkVelocities(ArticulationData* tree)
        {
            for (int i = tree->FixedBase ? 1 : 0; i < tree->Count; i++)
            {
                ArticulationLink* link = tree->Links[i];
                BodyData* body = StoreMemory.Row(link->Body);
                float3 linear = body->velocity->Linear;
                float3 limited = ExternalForces.ApplyForcesJob.ClampLength(linear, body->limits->MaxLinearVelocity);
                if (math.any(limited != linear))
                {
                    if (i == 0)
                    {
                        body->velocity->Linear = limited;
                    }
                    else
                    {
                        SetLinkVelocity(link, limited, false);
                    }

                    Refresh(tree, false);
                }

                float3 angular = BodyKinematics.WorldAngularVelocity(*body);
                limited = ExternalForces.ApplyForcesJob.ClampLength(angular, body->limits->MaxAngularVelocity);
                if (math.any(limited != angular))
                {
                    if (i == 0)
                    {
                        BodyKinematics.SetWorldAngularVelocity(*body, limited);
                    }
                    else
                    {
                        SetLinkVelocity(link, limited, true);
                    }

                    Refresh(tree, false);
                }
            }
        }

        private static bool ConfigureDrives(ArticulationData* tree, ArticulationFrame* frames, float dt)
        {
            bool factored = false;
            bool friction = false;
            for (int i = 1; i < tree->Count; i++)
            {
                ArticulationLink* link = tree->Links[i];
                ArticulationFrame* frame = frames + i;
                for (int axis = 0; axis < 3; axis++)
                {
                    int bit = 1 << axis;
                    if ((link->Axes & bit) == 0)
                    {
                        continue;
                    }

                    ArticulationAxis* drive = ArticulationLink.Axis(link, axis);
                    double stiffness = drive->Stiffness;
                    double damping = drive->Damping;
                    bool needsFriction = link->Friction > 0f;
                    friction |= needsFriction;
                    if (drive->DriveType != 0 || needsFriction)
                    {
                        if (!factored)
                        {
                            Factor(tree, frames, false);
                            factored = true;
                        }

                        ClearLoads(tree, frames);
                        frame->RightHandSide[axis] = 1.0;
                        Solve(tree, frames, false);
                        frame->UnitMass[axis] = 1.0 / math.max(frame->Result[axis], 1e-12);
                    }

                    double scale = drive->DriveType != 0 ? frame->UnitMass[axis] : 1.0;
                    double velocity = link->Velocity[axis];
                    double speedTarget = drive->TargetVelocity;
                    double error = (double)drive->Target - link->Position[axis];
                    if (drive->DriveType == 2 || drive->DriveType == 3)
                    {
                        // A target drive asks to reach the target position in one step. A velocity drive
                        // leaves position free. Use high damping and still enforce the drive's force limit.
                        damping = 10000.0 / dt;
                        stiffness = 0.0;
                        speedTarget = drive->DriveType == 2 ? error / dt : speedTarget;
                    }

                    frame->Drive[axis] = scale * (stiffness * (error - dt * velocity) + damping * (speedTarget - velocity));
                    frame->Compliance[axis] = scale * dt * (stiffness * dt + damping);
                    frame->DriveLimit[axis] = (tree->Flags & ArticulationData.DriveLimitsAsForces) != 0
                        ? drive->ForceLimit : (double)drive->ForceLimit / dt;
                    if (stiffness != 0.0 || damping != 0.0)
                    {
                        frame->DrivenAxes |= bit;
                    }
                }
            }

            return friction;
        }

        private static void SolveForces(ArticulationData* tree, ArticulationFrame* frames, float dt, bool friction)
        {
            bool changed;
            uint pass = 0;
            uint positionPasses = math.clamp(tree->PositionIterations, 1u, 255u);
            uint velocityPasses = math.clamp(tree->VelocityIterations, 1u, 255u);
            do
            {
                Factor(tree, frames, true);
                for (int i = 0; i < tree->Count; i++)
                {
                    ArticulationLink* link = tree->Links[i];
                    BodyData* body = StoreMemory.Row(link->Body);
                    ArticulationFrame* frame = frames + i;
                    double6 velocity = Velocity(body);
                    double3 force = ExternalForces.AccumulatedForce(body, dt);
                    double3 torque = ExternalForces.AccumulatedTorque(body, dt);
                    if ((body->flags & BodyFlags.Gravity) != 0)
                    {
                        force += frame->Mass * (double3)tree->World->gravity;
                    }

                    double3 angular = Angular(velocity);
                    force -= (frame->Mass * body->damping->Linear) * Linear(velocity);
                    torque -= math.cross(angular, math.mul(frame->Inertia, angular))
                        + body->damping->Angular * math.mul(frame->Inertia, angular);
                    frame->AppliedLoad = frame->Load = Spatial(-force, -torque);
                    frame->RightHandSide = (double3)link->Force + frame->Drive + frame->Friction;
                }

                Solve(tree, frames, true);
                changed = EnforceBounds(tree, frames, dt, pass < positionPasses, pass < velocityPasses);
                if (!changed && friction)
                {
                    changed = UpdateFriction(tree, frames, dt);
                    friction = false;
                }

                pass++;
            }
            while (changed);

            for (int i = 1; i < tree->Count; i++)
            {
                ArticulationLink* link = tree->Links[i];
                ArticulationFrame* frame = frames + i;
                link->DriveForce = (float3)math.clamp(frame->Drive - frame->Compliance * frame->Result,
                    -frame->DriveLimit, frame->DriveLimit);
            }
        }

        private static bool UpdateFriction(ArticulationData* tree, ArticulationFrame* frames, float dt)
        {
            for (int i = 0; i < tree->Count; i++)
            {
                ArticulationFrame* frame = frames + i;
                frame->Load = frame->AppliedLoad + frame->ApplyInertia(frame->Motion);
            }

            bool changed = false;
            for (int i = tree->Count - 1; i > 0; i--)
            {
                ArticulationLink* link = tree->Links[i];
                ArticulationFrame* frame = frames + i;
                if (link->Friction > 0f)
                {
                    double3 anchorToCenter = (double3)Center(StoreMemory.Row(link->Body)) - Anchor(link).pos;
                    double6 reaction = ShiftWrench(frame->Load, anchorToCenter);
                    double limit = link->Friction * (math.length(Linear(reaction)) + math.length(Angular(reaction)));
                    double3 velocity = (double3)link->Velocity + dt * frame->Result;
                    frame->Friction = math.clamp((-1.0 / dt) * frame->UnitMass * velocity, -limit, limit);
                    changed |= math.any(frame->Friction != double3.zero);
                }

                frames[link->Parent->Index].Load += ShiftWrench(frame->Load, frame->Offset);
            }

            return changed;
        }

        private static bool EnforceBounds(ArticulationData* tree, ArticulationFrame* frames, float dt,
            bool positionPass, bool velocityPass)
        {
            bool changed = false;
            for (int i = 1; i < tree->Count; i++)
            {
                ArticulationLink* link = tree->Links[i];
                ArticulationFrame* frame = frames + i;
                for (int axis = 0; axis < 3; axis++)
                {
                    int bit = 1 << axis;
                    if ((link->Axes & bit) == 0)
                    {
                        continue;
                    }

                    double speed = link->Velocity[axis] + dt * frame->Result[axis];
                    double maximum = link->MaxVelocity;
                    double minimum = -maximum;
                    ArticulationAxis* setting = ArticulationLink.Axis(link, axis);
                    if (setting->Limited && setting->Lower <= setting->Upper)
                    {
                        minimum = math.max(minimum, (setting->Lower - (double)link->Position[axis]) / dt);
                        maximum = math.min(maximum, (setting->Upper - (double)link->Position[axis]) / dt);
                        maximum = math.max(minimum, maximum);
                    }

                    if (positionPass && (frame->BoundAxes & bit) == 0 && (speed > maximum || speed < minimum))
                    {
                        frame->BoundAxes |= bit;
                        frame->LimitSpeed[axis] = (math.clamp(speed, minimum, maximum) - link->Velocity[axis]) / dt;
                        changed = true;
                    }

                    if (velocityPass && (frame->DrivenAxes & bit) != 0)
                    {
                        double driveForce = frame->Drive[axis] - frame->Compliance[axis] * frame->Result[axis];
                        double clamped = math.clamp(driveForce, -frame->DriveLimit[axis], frame->DriveLimit[axis]);
                        if (driveForce != clamped)
                        {
                            frame->DrivenAxes &= ~bit;
                            frame->Drive[axis] = clamped;
                            frame->Compliance[axis] = 0.0;
                            changed = true;
                        }
                    }
                }
            }

            return changed;
        }

        [BurstCompile]
        internal struct FinishJob : IJob
        {
            [NativeDisableUnsafePtrRestriction]
            public WorldData* World;
            [ReadOnly] public NativeArray<int> Indices;
            public NativeArray<MotionData> MotionData;
            public NativeArray<MotionVelocity> MotionVelocity;
            public float DeltaTime;
            public NativeList<ManifoldRecord> Contacts;
            [ReadOnly] public NativeList<ManifoldPoint> Points;

            public void Execute()
            {
                if (DeltaTime <= 0f)
                {
                    return;
                }

                for (ArticulationData* tree = World->articulations; tree != null; tree = tree->Next)
                {
                    if (tree->Count == 0 || (tree->Flags & ArticulationData.Simulated) == 0)
                    {
                        continue;
                    }

                    if (tree->Asleep)
                    {
                        WakeFromContact(tree);
                    }

                    if (!tree->Asleep)
                    {
                        ApplyContactChanges(tree);
                        Refresh(tree, false);
                    }
                }

                if (Contacts.Length != 0)
                {
                    new ArticulationContactSolver
                    {
                        World = World,
                        Indices = Indices,
                        MotionVelocities = MotionVelocity,
                        Records = Contacts,
                        Points = Points,
                        DeltaTime = DeltaTime,
                    }.Solve();
                }

                for (ArticulationData* tree = World->articulations; tree != null; tree = tree->Next)
                {
                    if (tree->Count == 0 || (tree->Flags & ArticulationData.Simulated) == 0)
                    {
                        continue;
                    }

                    if (!tree->Asleep)
                    {
                        Integrate(tree, DeltaTime);
                        ArticulationSleep.Test(tree, DeltaTime);
                    }

                    CopySolvedMotions(tree);
                }
            }

            private void ApplyContactChanges(ArticulationData* tree)
            {
                ArticulationFrame* frames = null;
                for (int i = tree->FixedBase ? 1 : 0; i < tree->Count; i++)
                {
                    ArticulationLink* link = tree->Links[i];
                    int index = Indices[(int)link->Body->index];
                    if ((uint)index >= MotionVelocity.Length)
                    {
                        continue;
                    }

                    MotionVelocity solved = MotionVelocity[index];
                    float3 linear = solved.LinearVelocity - link->FreeLinearVelocity;
                    float3 angular = solved.AngularVelocity - link->FreeAngularVelocity;
                    if (math.lengthsq(linear) + math.lengthsq(angular) <= 1e-14f || !math.all(math.isfinite(linear)) || !math.all(math.isfinite(angular)))
                    {
                        continue;
                    }

                    if (frames == null)
                    {
                        frames = BuildFrames(tree);
                    }

                    quaternion rotation = MotionData[index].WorldFromMotion.rot;
                    // Convert the velocity change from the contact solve back to its spatial impulse.
                    double6 change = Spatial(linear, math.rotate(rotation, angular));
                    frames[i].Load = -1.0 * frames[i].ApplyInertia(change);
                }

                if (frames != null)
                {
                    Factor(tree, frames, false);
                    Solve(tree, frames, false);
                    AddVelocityChange(tree, frames);
                    FreeFrames(frames);
                }
            }

            private void CopySolvedMotions(ArticulationData* tree)
            {
                for (int i = 0; i < tree->Count; i++)
                {
                    Body* link = tree->Links[i]->Body;
                    int index = Indices[(int)link->index];
                    if ((uint)index >= MotionVelocity.Length)
                    {
                        continue;
                    }

                    BodyData* body = StoreMemory.Row(link);
                    MotionData motion = MotionData[index];
                    motion.WorldFromMotion = math.mul(Pose(body), motion.BodyFromMotion);
                    MotionData[index] = motion;
                    MotionVelocity velocity = MotionVelocity[index];
                    velocity.LinearVelocity = body->velocity->Linear;
                    velocity.AngularVelocity = body->velocity->Angular;
                    MotionVelocity[index] = velocity;
                }
            }
        }

        private static void Integrate(ArticulationData* tree, float dt)
        {
            BodyData* root = StoreMemory.Row(tree->Links[0]->Body);
            if (!tree->FixedBase)
            {
                double6 velocity = Velocity(root);
                float3 linear = (float3)Linear(velocity);
                float3 angular = (float3)Angular(velocity);
                float3 center = Center(root) + dt * linear;
                root->pose->Rotation = math.normalize(math.mul(Rotation(dt * angular), root->pose->Rotation));
                root->pose->Position = center - math.rotate(root->pose->Rotation, root->mass->Transform.pos);
                SetVelocity(root, Spatial(linear, angular));
            }

            for (int i = 1; i < tree->Count; i++)
            {
                ArticulationLink* link = tree->Links[i];
                link->Velocity = math.clamp(link->Velocity, -link->MaxVelocity, link->MaxVelocity);
                link->Position += dt * link->Velocity;
                for (int axis = 0; axis < 3; axis++)
                {
                    ArticulationAxis* axisData = ArticulationLink.Axis(link, axis);
                    if ((link->Axes & (1 << axis)) != 0 && axisData->Limited && axisData->Lower <= axisData->Upper)
                    {
                        float position = link->Position[axis];
                        link->Position[axis] = math.clamp(position, axisData->Lower, axisData->Upper);
                        if (link->Position[axis] != position)
                        {
                            link->Velocity[axis] = 0f;
                        }
                    }
                }

                link->Acceleration = (link->Velocity - link->StepStartVelocity) / dt;
                if (link->Type == 3)
                {
                    WrapSphericalRotation(link);
                }
            }

            Refresh(tree, true);
        }

        private static void WrapSphericalRotation(ArticulationLink* link)
        {
            float angle = math.length(link->Position);
            if (angle <= math.PI)
            {
                return;
            }

            float turns = 2f * math.PI * math.floor((angle + math.PI) / (2f * math.PI));
            float scale = 1f - turns / angle;
            float3 direction = link->Position / angle;
            float3 radial = direction * math.dot(direction, link->Velocity);
            link->Position *= scale;
            link->Velocity = radial + scale * (link->Velocity - radial);
        }
    }
}
