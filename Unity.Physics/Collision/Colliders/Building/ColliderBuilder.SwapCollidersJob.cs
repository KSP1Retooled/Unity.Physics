using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Entities;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Physics.Store.Bodies;
using Unity.Physics.Store.Shapes;
using StoreMesh = Unity.Physics.Store.Shapes.Mesh;
using Unity.Physics.Store.World;
using Unity.Physics.Stepping;
using Unity.Profiling;

namespace Unity.Physics.Collision.Colliders.Building
{
    internal static unsafe partial class ColliderBuilder
    {
        /// <summary>
        /// Assigns each body's newly built collider and releases its previous collider.
        /// </summary>
        [BurstCompile]
        private struct SwapCollidersJob : IJob
        {
            [NativeDisableUnsafePtrRestriction]
            public BodyData* Bodies;

            [ReadOnly]
            public NativeArray<Work> Work;

            [ReadOnly]
            public NativeArray<long> Results;

            public void Execute()
            {
                for (int i = 0; i < Results.Length; i++)
                {
                    if (Results[i] == Failed)
                    {
                        continue;
                    }

                    BodyData* body = Bodies + Work[i].Body;
                    Release(body->collider->Value.GetUnsafePtr());
                    SetCollider(body, FromPointer((void*)Results[i]));
                }
            }
        }
    }
}
