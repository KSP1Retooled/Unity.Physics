using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using Unity.Physics.Dynamics.Articulations;
using Unity.Physics.Store.Articulations;
using Unity.Physics.Store.Memory;
using Unity.Physics.Store.World;

namespace KSP.Retooled.Physics.Commands
{
    internal static unsafe class ArticulationJointCommands
    {
        internal static void Execute(ArticulationCommand command, ArticulationLink* link, CommandArgs* a, StoreGlobals* globals)
        {
            ArticulationData* tree = link->Tree;
            switch (command)
            {
                case ArticulationCommand.SetLinkJointType:
                    if (link->Type != a->value)
                    {
                        link->Type = a->value;
                        link->Position = link->Velocity = float3.zero;
                        link->UpdateAxes();
                        ArticulationSleep.Wake(tree);
                    }

                    break;
                case ArticulationCommand.GetLinkJointType:
                    a->value = link->Type;
                    break;
                case ArticulationCommand.SetLinkJointAxisLock:
                    ArticulationAxis* axis = ArticulationLink.CommandAxis(link, a->mode);
                    if (axis != null && axis->Lock != a->value)
                    {
                        axis->Lock = a->value;
                        link->UpdateAxes();
                        ArticulationSleep.Wake(tree);
                    }

                    break;
                case ArticulationCommand.SetLinkJointLimit:
                    axis = ArticulationLink.CommandAxis(link, a->mode);
                    if (axis != null)
                    {
                        SetLimits(link, axis, a->scalar, a->scalar2);
                        ArticulationSleep.Wake(tree);
                    }

                    break;
                case ArticulationCommand.SetLinkJointActorLocalPose:
                    if (a->mode == 0)
                    {
                        link->ParentAnchor = new RigidTransform(a->rotation, a->position);
                    }
                    else
                    {
                        link->ChildAnchor = new RigidTransform(a->rotation, a->position);
                    }

                    ArticulationSleep.Wake(tree);
                    break;
                case ArticulationCommand.SetLinkJointAxisMotorConfiguration:
                    axis = ArticulationLink.CommandAxis(link, a->mode);
                    if (axis != null)
                    {
                        SetLimits(link, axis, a->vector.x, a->vector.y);
                        axis->Stiffness = math.max(0f, a->vector.z);
                        axis->Damping = math.max(0f, a->vector2.x);
                        axis->ForceLimit = math.max(0f, a->vector2.y);
                        axis->Target = a->vector2.z;
                        axis->TargetVelocity = a->scalar;
                        axis->DriveType = a->value;
                        ArticulationSleep.Wake(tree);
                    }

                    break;
                case ArticulationCommand.SetLinkJointFriction:
                    link->Friction = math.max(0f, a->scalar);
                    break;
                case ArticulationCommand.SetLinkJointMaxVelocity:
                    link->MaxVelocity = math.max(0f, a->scalar);
                    break;
                case ArticulationCommand.GetLinkJointMaxVelocity:
                    a->scalar = link->MaxVelocity;
                    break;
                case ArticulationCommand.GetLinkJointDofCount:
                    a->size = (uint)link->DofCount;
                    break;
                case ArticulationCommand.GetLinkJointForcesForAcceleration:
                    float3 acceleration = Unpack(link, a->vector, a->value);
                    ArticulationCommands.PrepareRows(tree);
                    a->vector = Pack(link, (float3)ArticulationQueries.ForcesForAcceleration(link, acceleration));
                    a->value = (uint)link->DofCount;
                    break;
                default:
                    Reduced(command, link, a);
                    break;
            }
        }

        private static void SetLimits(ArticulationLink* link, ArticulationAxis* axis, float lower, float upper)
        {
            // Unity 6000.7 sends revolute drive limits in degrees. It sends spherical limits in radians.
            axis->Lower = link->Type == 2 ? math.radians(lower) : lower;
            axis->Upper = link->Type == 2 ? math.radians(upper) : upper;
        }

        private static void Reduced(ArticulationCommand command, ArticulationLink* link, CommandArgs* a)
        {
            switch (command)
            {
                case ArticulationCommand.SetLinkJointPosition:
                    link->Position = Unpack(link, a->vector, a->value);
                    ArticulationSleep.Wake(link->Tree);
                    ArticulationCommands.Refresh(link->Tree, true);
                    return;
                case ArticulationCommand.SetLinkJointVelocity:
                    link->Velocity = Unpack(link, a->vector, a->value);
                    ArticulationSleep.Wake(link->Tree);
                    ArticulationCommands.Refresh(link->Tree, false);
                    return;
                case ArticulationCommand.SetLinkJointAcceleration:
                    link->Acceleration = Unpack(link, a->vector, a->value);
                    return;
                case ArticulationCommand.SetLinkJointForce:
                    link->Force = Unpack(link, a->vector, a->value);
                    ArticulationSleep.Wake(link->Tree);
                    return;
            }

            a->vector = Pack(link, command switch
            {
                ArticulationCommand.GetLinkJointPosition => link->Position,
                ArticulationCommand.GetLinkJointVelocity => link->Velocity,
                ArticulationCommand.GetLinkJointAcceleration => link->Acceleration,
                ArticulationCommand.GetLinkJointForce => link->Force,
                ArticulationCommand.GetLinkJointMotorForce => link->DriveForce,
                _ => float3.zero,
            });
            a->value = (uint)link->DofCount;
        }

        private static float3 Pack(ArticulationLink* link, float3 value)
        {
            float3 packed = default;
            int at = 0;
            for (int axis = 0; axis < 3; axis++)
            {
                if ((link->Axes & (1 << axis)) != 0)
                {
                    packed[at++] = value[axis];
                }
            }

            return packed;
        }

        private static float3 Unpack(ArticulationLink* link, float3 value, uint count)
        {
            float3 unpacked = default;
            int at = 0;
            for (int axis = 0; axis < 3 && at < count; axis++)
            {
                if ((link->Axes & (1 << axis)) != 0)
                {
                    unpacked[axis] = value[at++];
                }
            }

            return unpacked;
        }

        internal static void Bulk(ArticulationCommand command, ArticulationData* tree, CommandArgs* a, StoreGlobals* globals)
        {
            // Unity includes six reserved base entries in floating-base joint buffers. The root pose and
            // velocity use their own commands. GetDofStartIndices also skips these entries for the first joint.
            int firstJoint = tree->Count != 0 && !tree->FixedBase ? 6 : 0;
            int dofs = 0;
            for (int i = 0; i < tree->Count; i++)
            {
                dofs += tree->Links[i]->DofCount;
            }

            switch (command)
            {
                case ArticulationCommand.SetJointsPositions:
                case ArticulationCommand.SetJointsVelocities:
                case ArticulationCommand.SetJointsAccelerations:
                case ArticulationCommand.SetJointsForces:
                    if (a->size == (ulong)(firstJoint + dofs) && dofs != 0 && a->pointer != null)
                    {
                        ReadState(tree, command, (float*)a->pointer + firstJoint);
                        if (command != ArticulationCommand.SetJointsAccelerations)
                        {
                            ArticulationSleep.Wake(tree);
                        }

                        if (command == ArticulationCommand.SetJointsPositions || command == ArticulationCommand.SetJointsVelocities)
                        {
                            ArticulationCommands.Refresh(tree, command == ArticulationCommand.SetJointsPositions);
                        }
                    }

                    return;
                case ArticulationCommand.GetDenseJacobianAsFloatBuffer:
                    a->value = (uint)math.max(0, tree->Count - (tree->FixedBase ? 1 : 0)) * 6;
                    a->value2 = (uint)(dofs + firstJoint);
                    a->size = (ulong)a->value * a->value2;
                    a->pointer = AllocateFloats(globals, a->size);
                    if (a->pointer != null)
                    {
                        UnsafeUtility.MemClear(a->pointer, (long)(a->size * sizeof(float)));
                        ArticulationCommands.PrepareRows(tree);
                        ArticulationQueries.Jacobian(tree, (int)a->value2, (float*)a->pointer);
                    }

                    return;
            }

            a->size = (ulong)(dofs + firstJoint);
            float* buffer = AllocateFloats(globals, a->size);
            a->pointer = buffer;
            if (buffer == null)
            {
                return;
            }

            if (firstJoint != 0)
            {
                UnsafeUtility.MemClear(buffer, firstJoint * sizeof(float));
                buffer += firstJoint;
            }

            if (dofs == 0)
            {
                return;
            }

            switch (command)
            {
                case ArticulationCommand.GetJointsGravityForces:
                case ArticulationCommand.GetJointsCoriolisCentrifugalForces:
                case ArticulationCommand.GetJointsExternalForces:
                    ArticulationCommands.PrepareRows(tree);
                    ArticulationQueries.ForceKind kind = command == ArticulationCommand.GetJointsGravityForces
                        ? ArticulationQueries.ForceKind.Gravity
                        : command == ArticulationCommand.GetJointsExternalForces
                            ? ArticulationQueries.ForceKind.External
                            : ArticulationQueries.ForceKind.Coriolis;
                    ArticulationQueries.Forces(tree, kind, a->scalar, buffer);
                    break;
                default:
                    WriteState(tree, command, buffer);
                    break;
            }
        }

        private static float* AllocateFloats(StoreGlobals* globals, ulong count)
        {
            return count == 0 ? null : (float*)StoreMemory.Reallocate(globals, null, count * sizeof(float));
        }

        private static void ReadState(ArticulationData* tree, ArticulationCommand command, float* buffer)
        {
            for (int i = 1; i < tree->Count; i++)
            {
                ArticulationLink* link = tree->Links[i];
                float3 value = default;
                for (int axis = 0; axis < 3; axis++)
                {
                    if ((link->Axes & (1 << axis)) != 0)
                    {
                        value[axis] = *buffer++;
                    }
                }

                switch (command)
                {
                    case ArticulationCommand.SetJointsPositions:
                        link->Position = value;
                        break;
                    case ArticulationCommand.SetJointsVelocities:
                        link->Velocity = value;
                        break;
                    case ArticulationCommand.SetJointsAccelerations:
                        link->Acceleration = value;
                        break;
                    case ArticulationCommand.SetJointsForces:
                        link->Force = value;
                        break;
                }
            }
        }

        private static void WriteState(ArticulationData* tree, ArticulationCommand command, float* buffer)
        {
            for (int i = 1; i < tree->Count; i++)
            {
                ArticulationLink* link = tree->Links[i];
                float3 value = command switch
                {
                    ArticulationCommand.GetJointsPositions => link->Position,
                    ArticulationCommand.GetJointsVelocities => link->Velocity,
                    ArticulationCommand.GetJointsAccelerations => link->Acceleration,
                    ArticulationCommand.GetJointsForces => link->Force,
                    _ => link->DriveForce,
                };
                for (int axis = 0; axis < 3; axis++)
                {
                    if ((link->Axes & (1 << axis)) != 0)
                    {
                        *buffer++ = value[axis];
                    }
                }
            }
        }
    }
}
