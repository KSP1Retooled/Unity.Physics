using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Entities;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Physics;
using static Unity.Physics.Math;
using PhysicsJoint = Unity.Physics.Joint;
using Unity.Physics.Dynamics.Bodies;
using Unity.Physics.Store.Bodies;
using Unity.Physics.Store.Joints;
using Unity.Physics.Store.World;
using Unity.Physics.Stepping;
using Unity.Physics.Dynamics.Joint.Translation;

namespace KSP.Retooled.Physics
{
    /// <summary>
    /// Wake sleeping dynamic bodies connected to awake bodies by active joints. Repeat until no more bodies wake.
    /// </summary>
    [BurstCompile]
    internal unsafe struct WakeJointedBodiesJob : IJob
    {
        [NativeDisableUnsafePtrRestriction]
        public WorldData* World;

        public void Execute()
        {
            // Only a sleeping body can be woken.
            bool anyAsleep = false;
            for (uint i = 0; i < World->bodyCount && !anyAsleep; i++)
            {
                anyAsleep = (World->bodies[i].flags & BodyFlags.Sleeping) != 0;
            }

            while (anyAsleep && WakeThroughJoints(World))
            {
            }
        }

        /// <summary>
        /// Makes one pass over the joints to wake connected bodies. The caller repeats the pass until no more bodies
        /// wake.
        /// </summary>
        internal static bool WakeThroughJoints(WorldData* world)
        {
            bool woke = false;
            for (uint i = 0; i < world->jointCount; i++)
            {
                JointData* joint = world->joints[i];
                if (joint->body0 == null || joint->body1 == null || (joint->state & JointConstraints.Inactive) != 0)
                {
                    continue;
                }

                BodyData* first = StoreOps.Hot(joint->body0);
                BodyData* second = StoreOps.Hot(joint->body1);
                woke |= WakeFrom(first, second);
                woke |= WakeFrom(second, first);
            }

            return woke;
        }

        private static bool WakeFrom(BodyData* awake, BodyData* sleeping)
        {
            const BodyFlags immobile = BodyFlags.Static | BodyFlags.Kinematic;
            bool otherMoves = (awake->flags & (immobile | BodyFlags.Sleeping)) == 0;
            if (!otherMoves || (sleeping->flags & BodyFlags.Sleeping) == 0 || (sleeping->flags & immobile) != 0)
            {
                return false;
            }

            JointKernel.WakeRow(sleeping);
            return true;
        }
    }

}
