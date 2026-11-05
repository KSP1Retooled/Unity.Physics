// Joint commands. A joint exists as a record, so that Unity holds a valid handle, but it constrains nothing yet: the
// getters answer zero and the setters are dropped.

#include "Commands.h"

using namespace PhysicsCommands;

namespace Retooled
{
    Joint* CreateJointRecord(const WorldCmd::CreateJoint& command)
    {
        LogUnimplementedOnce(kWorldCategory, static_cast<uint32_t>(PhysicsCommands::World::CreateJoint));

        Joint* joint = new Joint();
        joint->userData = command.userData;
        joint->body0 = static_cast<Body*>(command.body0);
        joint->body1 = static_cast<Body*>(command.body1);
        joint->type = static_cast<int32_t>(command.type);
        joint->flags = static_cast<int32_t>(command.flags);
        return joint;
    }

    void DestroyJointRecord(Joint* joint)
    {
        delete joint;
    }

    template <typename T>
    static void BindStub(FunctionBinding* table)
    {
        table[static_cast<size_t>(T::command)] = &ZeroedStub<T, kJointCategory>;
    }

    void BindJointCommands(FunctionBinding* table)
    {
        namespace Cmd = PhysicsCommands::JointData;
        BindStub<Cmd::GetFlags>(table);
        BindStub<Cmd::GetUserData>(table);
        BindStub<Cmd::GetActorLocalPose>(table);
        BindStub<Cmd::GetIsConstraintValid>(table);
        BindStub<Cmd::GetForces>(table);
        BindStub<Cmd::GetActors>(table);
        BindStub<Cmd::Get6DofLimit>(table);
    }
}
