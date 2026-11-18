using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Physics.Store.Bodies;
using Unity.Physics.Store.Joints;
using Unity.Physics.Store.Memory;
using Unity.Physics.Store.World;
using Unity.Physics.Stepping;
using Unity.Physics.Dynamics.Joint.Translation;
using Unity.Physics.Dynamics.Bodies;

namespace Unity.Physics.Dynamics.Sleep
{
    internal static partial class FallAsleep
    {
        /// <summary>
        /// Puts the bodies listed by <see cref="SleepJob"/> to sleep and stops them at their current positions.
        /// </summary>
        [BurstCompile]
        internal unsafe struct FallAsleepJob : IJob
        {
            [NativeDisableUnsafePtrRestriction]
            public WorldData* World;

            [ReadOnly]
            public NativeList<int> Falling;

            public void Execute()
            {
                for (int i = 0; i < Falling.Length; i++)
                {
                    BodyData* body = World->bodies + Falling[i];
                    body->flags |= BodyFlags.Sleeping;
                    body->sleep->WakeCounter = 0.0f;
                    body->velocity->Linear = float3.zero;
                    body->velocity->Angular = float3.zero;
                    body->sleep->LinearSum = float3.zero;
                    body->sleep->AngularSum = float3.zero;
                }
            }
        }
    }
}
