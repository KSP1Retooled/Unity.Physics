using Unity.Burst;
using Unity.Mathematics;
using Unity.Physics.Store.Bodies;
using Unity.Physics.Dynamics.Bodies;
using Unity.Physics.Store.World;

namespace Unity.Physics.Stepping.Recording
{
    [BurstCompile]
    internal unsafe struct RecordedBody
    {
        internal uint Serial;
        internal BodyFlags Flags;
        internal ulong EngineId;
        internal float3 Position;
        internal quaternion Rotation;
        internal float3 Linear;
        internal float3 Angular;
        internal float3 WorldAngular;
        internal int Island;
        internal int IslandSize;
        internal int SolverPath;

        [BurstCompile]
        internal static void Capture(WorldData* world, uint engineIdOffset, RecordedIsland* islands, RecordedBody* output)
        {
            for (uint i = 0; i < world->bodyCount; i++)
            {
                RecordedBody body = Read(world->bodies + i, engineIdOffset);
                if (islands != null && islands[i].Size > 0)
                {
                    body.Island = islands[i].Index;
                    body.IslandSize = islands[i].Size;
                    body.SolverPath = (int)islands[i].Path;
                }

                output[i] = body;
            }
        }

        internal static RecordedBody Read(BodyData* body, uint engineIdOffset)
        {
            return new RecordedBody
            {
                Serial = body->serial,
                EngineId = body->owner != null ? FlightRecorder.EngineId(body->owner->userData, engineIdOffset) : 0,
                Flags = body->flags,
                Position = body->pose->Position,
                Rotation = body->pose->Rotation,
                Linear = body->velocity->Linear,
                Angular = body->velocity->Angular,
                WorldAngular = BodyKinematics.WorldAngularVelocity(*body),
                Island = -1,
            };
        }

        internal readonly bool Dynamic => (Flags & (BodyFlags.Static | BodyFlags.Kinematic)) == 0;

        internal readonly bool Finite => math.all(math.isfinite(Position)) && math.all(math.isfinite(Rotation.value))
            && math.all(math.isfinite(Linear)) && math.all(math.isfinite(WorldAngular));
    }
}
