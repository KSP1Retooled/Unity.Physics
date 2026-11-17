using System;
using System.Runtime.InteropServices;
using AOT;
using Unity.Burst;
using UnityEngine;
using Unity.Physics.Store.Shapes;
using Unity.Physics.Collision.Colliders.Building;

namespace KSP.Retooled.Physics
{
    /// <summary>Plugin entry points for cooking and validating baked mesh data.</summary>
    [BurstCompile]
    internal static unsafe class MeshCook
    {
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        internal delegate void CookFunction(CookInput* input, CookOutput* output);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        internal delegate int ValidateFunction(byte* stream, long size, int convex);

        /// <summary>
        /// Writes the cooked mesh to output memory owned by the caller. If no output bytes are supplied, sets size to
        /// the required capacity. Otherwise, sets size to the number of bytes written, which may be smaller.
        /// </summary>
        [BurstCompile(DisableSafetyChecks = true, FloatMode = FloatMode.Strict)]
        [MonoPInvokeCallback(typeof(CookFunction))]
        internal static void Cook(CookInput* input, CookOutput* output)
        {
            using Unity.Profiling.ProfilerMarker.AutoScope span = BackendMarkers.CookMesh.Auto();

            bool ranInMono = false;
            CookInMono(input, output, ref ranInMono);
            if (!ranInMono)
            {
                MeshCooker.Run(input, output);
            }
        }

        /// <summary>
        /// Burst excludes this method, so it runs only in Mono. Catch exceptions here so that they do not leave the
        /// managed callback.
        /// </summary>
        [BurstDiscard]
        private static void CookInMono(CookInput* input, CookOutput* output, ref bool ranInMono)
        {
            ranInMono = true;
            BackendHost.NoteManagedCook();
            try
            {
                MeshCooker.Run(input, output);
            }
            catch (Exception e)
            {
                output->status = (int)MeshCookStatus.BadVertices;
                Debug.LogException(e);
            }
        }

        /// <summary>
        /// Checks whether the bytes describe a cooked mesh of the requested kind and can be read safely.
        /// </summary>
        [BurstCompile(DisableSafetyChecks = true)]
        [MonoPInvokeCallback(typeof(ValidateFunction))]
        internal static int Validate(byte* stream, long size, int convex)
        {
            using Unity.Profiling.ProfilerMarker.AutoScope span = BackendMarkers.ValidateMesh.Auto();

            return MeshCooker.Validate(stream, size, convex);
        }
    }
}
