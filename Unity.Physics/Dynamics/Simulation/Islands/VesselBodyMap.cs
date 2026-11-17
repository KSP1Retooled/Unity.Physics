using System;
using System.Threading;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Physics.Dynamics.Vessels.Model;

namespace Unity.Physics.Dynamics.Simulation.Islands
{
    /// <summary>
    /// Body-to-vessel mapping, with each vessel's dynamic part count and lowest dynamic body.
    /// </summary>
    internal unsafe struct VesselBodyMap
    {
        /// <summary>For each dynamic body, the index of its vessel in the step's vessel set, or -1.</summary>
        public int* BodyVessel;

        /// <summary>For each vessel, the number of its parts whose bodies are dynamic.</summary>
        public int* DynamicParts;

        /// <summary>For each vessel, its lowest dynamic body, or int.MaxValue if it has none.</summary>
        public int* KeyBody;

        /// <summary>Builds the map in temporary memory, for a step without a <see cref="SolverIslandCache"/>.</summary>
        public static VesselBodyMap Build(in VesselSet vessels, int dynamicBodies, Allocator allocator)
        {
            var map = new VesselBodyMap
            {
                BodyVessel = (int*)UnsafeUtility.Malloc((long)math.max(1, dynamicBodies) * sizeof(int), UnsafeUtility.AlignOf<int>(), allocator),
                DynamicParts = (int*)UnsafeUtility.Malloc((long)math.max(1, vessels.VesselCount) * sizeof(int), UnsafeUtility.AlignOf<int>(), allocator),
                KeyBody = (int*)UnsafeUtility.Malloc((long)math.max(1, vessels.VesselCount) * sizeof(int), UnsafeUtility.AlignOf<int>(), allocator),
            };
            map.Fill(vessels, dynamicBodies);
            return map;
        }

        public void Fill(in VesselSet vessels, int dynamicBodies)
        {
            UnsafeUtility.MemSet(BodyVessel, 0xFF, (long)dynamicBodies * sizeof(int));
            for (int v = 0; v < vessels.VesselCount; v++)
            {
                VesselRecord vessel = vessels.Vessels[v];
                int count = 0;
                int key = int.MaxValue;
                for (int part = vessel.FirstPart; part < vessel.FirstPart + vessel.PartCount; part++)
                {
                    int body = vessels.PartBodies[part];
                    if (body >= 0 && body < dynamicBodies)
                    {
                        BodyVessel[body] = v;
                        count++;
                        key = math.min(key, body);
                    }
                }

                DynamicParts[v] = count;
                KeyBody[v] = key;
            }
        }
    }
}
