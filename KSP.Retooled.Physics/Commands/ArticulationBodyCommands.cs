using KSP.Retooled.Physics.Queries;
using Unity.Mathematics;
using Unity.Physics.Dynamics.Articulations;
using Unity.Physics.Dynamics.Bodies;
using Unity.Physics.Store.Articulations;
using Unity.Physics.Store.Bodies;
using Unity.Physics.Store.Memory;
using Unity.Physics.Store.World;

namespace KSP.Retooled.Physics.Commands
{
    internal static unsafe class ArticulationBodyCommands
    {
        internal static void Execute(ArticulationCommand command, ArticulationLink* link, CommandArgs* a, StoreGlobals* globals)
        {
            Body* body = link->Body;
            switch (command)
            {
                case ArticulationCommand.GetLinkIndex:
                    a->size = (uint)link->Index;
                    return;
                case ArticulationCommand.GetLinkJoint:
                    a->pointer = link->Parent != null ? body : null;
                    return;
                case ArticulationCommand.SetLinkFlags:
                    a->flags &= (uint)BodyFlags.Gravity;
                    BodyCommands.Execute(BodyCommand.SetFlags, body, a, globals);
                    return;
                case ArticulationCommand.GetLinkFlags:
                    a->flags = (uint)(StoreMemory.Row(body)->flags & BodyFlags.Gravity);
                    return;
                case ArticulationCommand.GetLinkClosestWorldPoint:
                    a->scalar = QueryEntry.ArticulationClosestPoint(body, new RigidTransform(a->rotation, a->position),
                        a->vector, out a->vector);
                    return;
                case ArticulationCommand.SetLinkPose:
                    SetPose(link, a, globals);
                    return;
                case ArticulationCommand.SetLinkLinearVelocity:
                case ArticulationCommand.SetLinkAngularVelocity:
                    if (link->Parent == null && !link->Tree->FixedBase)
                    {
                        BodyCommands.Execute(command == ArticulationCommand.SetLinkLinearVelocity
                            ? BodyCommand.SetLinearVelocity : BodyCommand.SetAngularVelocity, body, a, globals);
                        ArticulationCommands.Refresh(link->Tree, false);
                    }
                    else if (link->Parent != null)
                    {
                        ArticulationCommands.PrepareRows(link->Tree);
                        ArticulationSolver.SetLinkVelocity(link, a->vector,
                            command == ArticulationCommand.SetLinkAngularVelocity);
                        ArticulationKinematics.Refresh(link->Tree, false);
                    }

                    if (a->enabled != 0)
                    {
                        ArticulationSleep.Wake(link->Tree);
                    }

                    return;
                case ArticulationCommand.AddForceToLink:
                case ArticulationCommand.AddTorqueToLink:
                    if (a->enabled != 0 && math.any(a->vector != float3.zero) && (link->Parent != null || !link->Tree->FixedBase))
                    {
                        ArticulationSleep.Wake(link->Tree);
                    }

                    BodyCommands.Execute(command == ArticulationCommand.AddForceToLink
                        ? BodyCommand.AddForce : BodyCommand.AddTorque, body, a, globals);
                    return;
            }

            BodyCommand bodyCommand = command switch
            {
                ArticulationCommand.SetLinkMass => BodyCommand.SetMass,
                ArticulationCommand.GetLinkMass => BodyCommand.GetMass,
                ArticulationCommand.SetLinkDamping => BodyCommand.SetDamping,
                ArticulationCommand.GetLinkDamping => BodyCommand.GetDamping,
                ArticulationCommand.GetLinkPose => BodyCommand.GetPose,
                ArticulationCommand.AttachShapeToLink => BodyCommand.AttachShape,
                ArticulationCommand.DetachShapeFromLink => BodyCommand.DetachShape,
                ArticulationCommand.GetLinkShapeCount => BodyCommand.GetShapeCount,
                ArticulationCommand.GetLinkShapesUserData => BodyCommand.GetShapesUserData,
                ArticulationCommand.GetLinkUserData => BodyCommand.GetUserData,
                ArticulationCommand.SetLinkCollisionDetectionMode => BodyCommand.SetCollisionDetectionMode,
                ArticulationCommand.ClearForceFromLink => BodyCommand.ClearForce,
                ArticulationCommand.ClearTorqueFromLink => BodyCommand.ClearTorque,
                ArticulationCommand.GetLinkLinearVelocity => BodyCommand.GetLinearVelocity,
                ArticulationCommand.GetLinkAngularVelocity => BodyCommand.GetAngularVelocity,
                ArticulationCommand.SetLinkMaxLinearVelocity => BodyCommand.SetMaxLinearVelocity,
                ArticulationCommand.GetLinkMaxLinearVelocity => BodyCommand.GetMaxLinearVelocity,
                ArticulationCommand.SetLinkMaxAngularVelocity => BodyCommand.SetMaxAngularVelocity,
                ArticulationCommand.GetLinkMaxAngularVelocity => BodyCommand.GetMaxAngularVelocity,
                ArticulationCommand.SetLinkMaxDepenetrationVelocity => BodyCommand.SetMaxDepenetrationVelocity,
                ArticulationCommand.GetLinkMaxDepenetrationVelocity => BodyCommand.GetMaxDepenetrationVelocity,
                ArticulationCommand.SetLinkInertiaTensor => BodyCommand.SetInertiaTensor,
                ArticulationCommand.GetLinkInertiaTensor => BodyCommand.GetInertiaTensor,
                ArticulationCommand.SetLinkInertiaTensorRotation => BodyCommand.SetInertiaTensorRotation,
                ArticulationCommand.GetLinkInertiaTensorRotation => BodyCommand.GetInertiaTensorRotation,
                ArticulationCommand.SetLinkLocalCenterOfMass => BodyCommand.SetLocalCenterOfMass,
                ArticulationCommand.GetLinkLocalCenterOfMass => BodyCommand.GetLocalCenterOfMass,
                ArticulationCommand.GetLinkWorldInertiaTensorMatrix => BodyCommand.GetWorldInertiaTensorMatrix,
                ArticulationCommand.RecomputeLinkMassProperties => BodyCommand.RecomputeMassProperties,
                ArticulationCommand.GetLinkAccumulatedForce => BodyCommand.GetAccumulatedForce,
                ArticulationCommand.GetLinkAccumulatedTorque => BodyCommand.GetAccumulatedTorque,
                ArticulationCommand.GetLinkLocalPointVelocity => BodyCommand.GetLocalPointVelocity,
                ArticulationCommand.GetLinkWorldPointVelocity => BodyCommand.GetWorldPointVelocity,
                _ => BodyCommand.Invalid,
            };
            BodyCommands.Execute(bodyCommand, body, a, globals);
        }

        private static void SetPose(ArticulationLink* link, CommandArgs* a, StoreGlobals* globals)
        {
            BodyCommands.Execute(BodyCommand.SetPose, link->Body, a, globals);
            if (a->enabled != 0)
            {
                ArticulationSleep.Wake(link->Tree);
            }

            if (link->Parent == null)
            {
                ArticulationCommands.Refresh(link->Tree, true);
            }
        }
    }
}
