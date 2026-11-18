using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;

namespace Unity.Physics
{
    public static partial class Integrator
    {
        // Schedule integration for bodies outside islands. Bodies within islands are integrated during their islands'
        // substeps.
        internal static JobHandle ScheduleIntegrateBodiesOutsideIslandsJob(in DynamicsWorld world, NativeArray<bool> bodiesInIslands,
            float timeStep, JobHandle inputDeps)
        {
            return new IntegrateBodiesOutsideIslandsJob
            {
                MotionDatas = world.MotionDatas,
                MotionVelocities = world.MotionVelocities,
                BodiesInIslands = bodiesInIslands,
                TimeStep = timeStep
            }.Schedule(world.NumMotions, 64, inputDeps);
        }

        [BurstCompile]
        private struct IntegrateBodiesOutsideIslandsJob : IJobParallelFor
        {
            public NativeArray<MotionData> MotionDatas;
            public NativeArray<MotionVelocity> MotionVelocities;

            [ReadOnly]
            public NativeArray<bool> BodiesInIslands;

            public float TimeStep;

            public void Execute(int i)
            {
                if (BodiesInIslands.Length == 0 || !BodiesInIslands[i])
                {
                    IntegrateMotion(i, MotionDatas, MotionVelocities, TimeStep);
                }
            }
        }
    }
}
