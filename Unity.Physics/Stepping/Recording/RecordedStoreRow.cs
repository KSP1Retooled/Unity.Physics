using Unity.Burst;
using Unity.Physics.Store.Bodies;
using Unity.Physics.Store.World;

namespace Unity.Physics.Stepping.Recording
{
    [BurstCompile]
    internal unsafe struct RecordedStoreRow
    {
        internal BodyHome Home;
        internal ulong Address;
        internal ulong EngineId;
        internal BodyFlags Flags;
        internal uint Serial;
        internal int JointEnds;
        internal uint ShapeCount;
        internal uint OwnerSerial;
        internal int CollisionDetectionMode;
        internal uint PositionIterations;
        internal uint VelocityIterations;
        internal int MovingCompound;

        [BurstCompile]
        internal static void Capture(WorldData* world, uint engineIdOffset, RecordedStoreRow* rows)
        {
            for (uint i = 0; i < world->bodyCount; i++)
            {
                BodyData* source = world->bodies + i;
                var row = new RecordedStoreRow
                {
                    Home = new BodyHome
                    {
                        pose = *source->pose,
                        velocity = *source->velocity,
                        mass = *source->mass,
                        bodyMass = *source->bodyMass,
                        damping = *source->damping,
                        limits = *source->limits,
                        sleep = *source->sleep,
                        forces = *source->forces,
                        target = *source->target,
                    },
                    Flags = source->flags,
                    Serial = source->serial,
                    JointEnds = source->jointEnds,
                    ShapeCount = source->shapeCount,
                };

                Body* body = source->owner;
                if (body != null)
                {
                    row.Address = (ulong)body;
                    row.EngineId = FlightRecorder.EngineId(body->userData, engineIdOffset);
                    row.OwnerSerial = body->serial;
                    row.CollisionDetectionMode = body->collisionDetectionMode;
                    row.PositionIterations = body->positionIterations;
                    row.VelocityIterations = body->velocityIterations;
                }

                rows[i] = row;
            }
        }
    }
}
