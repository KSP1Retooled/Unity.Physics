using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Entities;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Physics.Dynamics.Bodies;
using Unity.Physics.Store.Bodies;
using Unity.Physics.Store.Shapes;
using Unity.Physics.Store.World;
using Unity.Physics.Stepping;
using Unity.Physics.Collision.Colliders.Building;
using Unity.Physics.Dynamics.Joint.Translation;

namespace Unity.Physics.Stepping
{
    internal static partial class StoreBodies
    {
        /// <summary>
        /// Partition simulated bodies into dynamic and static lists and assign built indices to store rows.
        /// Run on the main thread because scheduling needs the counts.
        /// </summary>
        [BurstCompile]
        internal unsafe struct PartitionBodiesJob : IJob
        {
            [NativeDisableUnsafePtrRestriction]
            public WorldData* World;

            public NativeList<int> DynamicIndices;
            public NativeList<int> StaticIndices;
            public NativeList<int> BuiltIndices;

            /// <summary>The flags of a dynamic body that receives its pose during write-back instead of from its motion.</summary>
            public BodyFlags PosedByWriteBack;

            /// <summary>Set to 1 if no dynamic body has a flag in <see cref="PosedByWriteBack"/>, and to 0 otherwise.</summary>
            public NativeReference<int> MotionsMatchStore;

            public void Execute()
            {
                int count = (int)World->bodyCount;
                DynamicIndices.Clear();
                StaticIndices.Clear();
                BuiltIndices.ResizeUninitialized(count);
                int matches = 1;
                for (int i = 0; i < count; i++)
                {
                    BodyFlags flags = World->bodies[i].flags;
                    BuiltIndices[i] = -1;
                    if ((flags & BodyFlags.Simulated) == 0)
                    {
                        continue;
                    }

                    if ((flags & BodyFlags.Static) != 0)
                    {
                        BuiltIndices[i] = StaticIndices.Length;
                        StaticIndices.Add(i);
                    }
                    else
                    {
                        BuiltIndices[i] = DynamicIndices.Length;
                        DynamicIndices.Add(i);
                        if ((flags & PosedByWriteBack) != 0)
                        {
                            matches = 0;
                        }
                    }
                }

                int dynamicCount = DynamicIndices.Length;
                for (int s = 0; s < StaticIndices.Length; s++)
                {
                    BuiltIndices[StaticIndices[s]] += dynamicCount;
                }

                MotionsMatchStore.Value = matches;
            }
        }
    }
}
