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
        internal static void Capture(WorldData* world, uint engineIdOffset, RecordedIsland* islands, byte* restoredMotion, RecordedBody* output)
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

                if (restoredMotion != null && restoredMotion[i] != 0)
                {
                    body.SolverPath |= (int)RecordedSolverPath.RestoredMotion;
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

        [BurstCompile]
        internal static int FindTrigger(RecordedBody* before, RecordedBody* after, int count,
            RecordedBody* previous, int previousCount, float linear, float angular,
            FlightTriggerReason* reason, int* previousRow)
        {
            for (int i = 0; i < count; i++)
            {
                *reason = CheckStep(before[i], after[i], linear, angular);
                if (*reason != FlightTriggerReason.None)
                {
                    return i;
                }

                uint serial = before[i].Serial;
                if (serial == 0 || previousCount == 0)
                {
                    continue;
                }

                if (i >= previousCount || serial != previous[i].Serial)
                {
                    *reason = FlightTriggerReason.ReorderedBodies;
                    return i;
                }

                *reason = Check(previous[i], before[i], linear, angular);
                if (*reason != FlightTriggerReason.None)
                {
                    *previousRow = i;
                    return i;
                }
            }

            return -1;
        }

        internal static FlightTriggerReason CheckStep(in RecordedBody before, in RecordedBody after, float linear, float angular)
        {
            return (after.SolverPath & (int)RecordedSolverPath.RestoredMotion) != 0
                ? FlightTriggerReason.RestoredMotion
                : Check(before, after, linear, angular);
        }

        internal static FlightTriggerReason Check(in RecordedBody before, in RecordedBody after, float linear, float angular)
        {
            if (!before.Finite || !after.Finite)
            {
                return FlightTriggerReason.NonFiniteMotion;
            }

            if (after.Dynamic)
            {
                if (linear >= 0 && math.lengthsq(after.Linear - before.Linear) > linear * linear)
                {
                    return FlightTriggerReason.LinearVelocity;
                }

                if (angular >= 0 && math.lengthsq(after.WorldAngular - before.WorldAngular) > angular * angular)
                {
                    return FlightTriggerReason.AngularVelocity;
                }
            }

            return FlightTriggerReason.None;
        }

        internal readonly bool Dynamic => (Flags & (BodyFlags.Static | BodyFlags.Kinematic)) == 0;

        internal readonly bool Finite => math.all(math.isfinite(Position)) && math.all(math.isfinite(Rotation.value))
            && math.all(math.isfinite(Linear)) && math.all(math.isfinite(WorldAngular));
    }
}
