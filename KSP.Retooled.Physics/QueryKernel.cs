using System;
using System.Runtime.InteropServices;
using AOT;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Physics;
using UnityEngine;

namespace KSP.Retooled.PhysicsBackend
{
    /// <summary>
    /// What a query function reads: a copy of the built collision world and a snapshot of the store's bodies and shapes. It
    /// lives in unmanaged memory and holds no managed reference, so a Burst function on any thread can use it. The plugin
    /// keeps it behind the world's query lock, and <see cref="WorldState"/> republishes it after every build. A worker
    /// reads the last one published even when the store has moved on, so it must not lead back into the store.
    /// </summary>
    internal unsafe struct QueryWorld
    {
        public CollisionWorld Collision;

        // One per body of Collision.
        [NativeDisableUnsafePtrRestriction]
        public QueryBody* Bodies;

        // Each body's shapes in store order, from QueryBody.FirstShape.
        [NativeDisableUnsafePtrRestriction]
        public QueryShape* Shapes;
    }

    /// <summary>A built body as queries see it.</summary>
    internal struct QueryBody
    {
        public int FirstShape;
        public int SingleShape;     // the shape a collider that isn't a compound stands for, relative to FirstShape
        public int IsStatic;
    }

    /// <summary>A shape as queries see it.</summary>
    internal struct QueryShape
    {
        public ulong EngineObjectId;
        public uint Flags;
    }

    /// <summary>
    /// The queries the plugin calls without going through Mono. Each is a Burst function pointer, and until the async compile
    /// delivers, that pointer is a stub running the same method in Mono, so a query always works.
    /// </summary>
    /// <remarks>
    /// Safety checks stay off because a Burst exception can't cross the native caller; the Mono form catches its own.
    /// </remarks>
    [BurstCompile]
    internal static unsafe class QueryKernel
    {
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        internal delegate int RayCastFunction(QueryWorld* world, NativeStore.RayCastInput* input, NativeStore.RayCastHit* hit);

        [BurstCompile(DisableSafetyChecks = true)]
        [MonoPInvokeCallback(typeof(RayCastFunction))]
        internal static int RayCast(QueryWorld* world, NativeStore.RayCastInput* input, NativeStore.RayCastHit* hit)
        {
            int result = 0;
            bool ranInMono = false;
            RayCastInMono(world, input, hit, ref result, ref ranInMono);
            return ranInMono ? result : CastRay(world, input, hit);
        }

        /// <summary>Burst drops this, so it only runs in the Mono form, which needs the exception ended here.</summary>
        [BurstDiscard]
        private static void RayCastInMono(
            QueryWorld* world,
            NativeStore.RayCastInput* input,
            NativeStore.RayCastHit* hit,
            ref int result,
            ref bool ranInMono)
        {
            ranInMono = true;
            BackendHost.NoteManagedQuery();
            try
            {
                // Inside a job, which is where most of these calls run, the safety system only lets a native container
                // be touched through a job's own copy of it. Burst has no such checks, so only this form needs one.
                int cast = 0;
                new RayCastJob { World = *world, Input = input, Hit = hit, Result = &cast }.Run();
                result = cast;
            }
            catch (Exception e)
            {
                Debug.LogException(e);
            }
        }

        private struct RayCastJob : IJob
        {
            [ReadOnly]
            public QueryWorld World;

            [NativeDisableUnsafePtrRestriction]
            public NativeStore.RayCastInput* Input;

            [NativeDisableUnsafePtrRestriction]
            public NativeStore.RayCastHit* Hit;

            [NativeDisableUnsafePtrRestriction]
            public int* Result;

            public void Execute()
            {
                *Result = CastRay((QueryWorld*)UnsafeUtility.AddressOf(ref World), Input, Hit);
            }
        }

        private static int CastRay(QueryWorld* world, NativeStore.RayCastInput* input, NativeStore.RayCastHit* hit)
        {
            float3 direction = math.normalizesafe(input->direction);
            float distance = math.min(input->maxDistance, QueryCollector.MaxRayLength);
            RaycastInput ray = new RaycastInput
            {
                Start = input->origin,
                End = input->origin + direction * distance,
                Filter = new CollisionFilter
                {
                    BelongsTo = uint.MaxValue,
                    CollidesWith = (uint)input->layerMask,
                },
            };

            QueryCollector collector = new QueryCollector(world, input->filterOptions, direction);
            if (!world->Collision.CastRay(ray, ref collector))
            {
                return 0;
            }

            hit->point = collector.Hit.Position;
            hit->normal = collector.Hit.SurfaceNormal;
            // A hit names a primitive of the blob, which merges and reorders triangles. Nothing in the game reads the source
            // triangle or the barycentric coordinates, so mapping back isn't worth the memory, and these stay zero.
            hit->faceId = 0;
            hit->distance = collector.Hit.Fraction * distance;
            hit->uv = float2.zero;
            hit->engineObjectId = collector.HitEngineObjectId;
            return 1;
        }
    }
}
