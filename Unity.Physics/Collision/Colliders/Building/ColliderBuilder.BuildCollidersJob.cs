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
        [BurstCompile]
        private struct BuildCollidersJob : IJobParallelForDefer
        {
            [NativeDisableUnsafePtrRestriction]
            public BodyData* Bodies;

            [ReadOnly]
            public NativeArray<Work> Work;

            public NativeArray<long> Results;

            public void Execute(int index)
            {
                BodyData* body = Bodies + Work[index].Body;
                void* collider = Work[index].Whole ? null : Repose(body);
                Results[index] = (long)(collider != null ? collider : Build(body));
            }
        }
    }
}
