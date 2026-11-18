using Unity.Burst;
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
