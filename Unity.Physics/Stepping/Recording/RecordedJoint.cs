using Unity.Burst;
using Unity.Mathematics;
using Unity.Physics.Store.Joints;
using Unity.Physics.Store.World;

namespace Unity.Physics.Stepping.Recording
{
    [BurstCompile]
    internal unsafe struct RecordedJoint
    {
        internal JointData Data;
        internal ulong EngineId;
        internal ulong Address;
        internal int Body0;
        internal int Body1;

        // Five target setters intentionally leave JointSettings unchanged because vessel edges ignore targets.
        [BurstCompile]
        internal static int MatchesTargetsAndEnds(WorldData* world, uint idOffset, RecordedJoint* saved)
        {
            for (uint i = 0; i < world->jointCount; i++)
            {
                JointData* joint = world->joints[i];
                ref RecordedJoint old = ref saved[i];
                if ((ulong)joint != old.Address || joint->serial != old.Data.serial
                    || joint->state != old.Data.state || joint->settingsStamp != old.Data.settingsStamp
                    || (joint->body0 == null ? -1 : (int)joint->body0->index) != old.Body0
                    || (joint->body1 == null ? -1 : (int)joint->body1->index) != old.Body1
                    || FlightRecorder.EngineId(joint->userData, idOffset) != old.EngineId
                    || math.any(math.asuint(joint->targetRotation.value) != math.asuint(old.Data.targetRotation.value))
                    || math.any(math.asuint(joint->targetPosition) != math.asuint(old.Data.targetPosition))
                    || math.any(math.asuint(joint->targetLinearVelocity) != math.asuint(old.Data.targetLinearVelocity))
                    || math.any(math.asuint(joint->targetAngularVelocity) != math.asuint(old.Data.targetAngularVelocity))
                    || math.asuint(joint->hingeMotorTargetVelocity) != math.asuint(old.Data.hingeMotorTargetVelocity))
                {
                    return 0;
                }
            }

            return 1;
        }

        [BurstCompile]
        internal static void Capture(WorldData* world, uint idOffset, RecordedJoint* output)
        {
            for (uint i = 0; i < world->jointCount; i++)
            {
                JointData* joint = world->joints[i];
                ref RecordedJoint record = ref output[i];
                record.Data = *joint;
                record.Data.body0 = null;
                record.Data.body1 = null;
                record.Data.userData = null;
                record.EngineId = FlightRecorder.EngineId(joint->userData, idOffset);
                record.Address = (ulong)joint;
                record.Body0 = joint->body0 == null ? -1 : (int)joint->body0->index;
                record.Body1 = joint->body1 == null ? -1 : (int)joint->body1->index;
            }
        }
    }
}
