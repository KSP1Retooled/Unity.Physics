using System;
using System.Collections.Generic;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Entities;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Physics.Dynamics.Bodies;
using Unity.Physics.Store.Bodies;
using Unity.Physics.Store.Memory;
using Unity.Physics.Store.Reports;
using StoreContactPoint = Unity.Physics.Store.Reports.ContactPoint;
using TriggerEvent = Unity.Physics.Store.Reports.TriggerEvent;
using Unity.Physics.Store.Shapes;
using Unity.Physics.Store.World;
using Unity.Physics.Stepping;
using Unity.Physics.Collision.Colliders.Building;
using Unity.Physics.Collision.Filter.Groups;
using Unity.Physics.Collision.Queries.Collectors;
using Unity.Physics.Dynamics.Joint.Grouping;
using Unity.Physics.Dynamics.Vessels.Detection;

namespace Unity.Physics.Dynamics.Contacts
{
    internal static partial class WarmStart
    {
        /// <summary>
        /// Warm start nearby contact points with an equal share of the previous manifold's normal impulse.
        /// </summary>
        [BurstCompile]
        internal unsafe struct WarmStartJob : IJacobiansJob
        {
            // Use less than the full impulse. A manifold that needs no additional impulse produces no event and would start
            // cold next step.
            internal const float Fraction = 0.95f;

            // Maximum separation, in metres, for warm starting. Must not exceed Solver.ContactRowReach:
            // each warmed point needs a row that can remove excess impulse.
            private const float FarthestPoint = 0.1f;

            [ReadOnly]
            public NativeHashMap<JacobianKey, float2> WarmStarts;

            [NativeDisableUnsafePtrRestriction]
            public MotionVelocity* Motions;

            public int MotionCount;

            public void Execute(ref ModifiableJacobianHeader header, ref ModifiableContactJacobian jacobian)
            {
                JacobianKey key = new JacobianKey { BodyA = header.BodyIndexA, BodyB = header.BodyIndexB, Normal = jacobian.Normal };
                if ((header.Flags & JacobianFlags.Disabled) != 0 || !WarmStarts.TryGetValue(key, out float2 shared))
                {
                    return;
                }

                float impulse = Fraction * shared.x / shared.y;
                for (int i = 0; i < jacobian.NumContacts; i++)
                {
                    ContactJacAngAndVelToReachCp point = header.GetAngularJacobian(i);
                    if (point.ContactDistance > FarthestPoint)
                    {
                        continue;
                    }

                    point.Jac.Impulse = impulse;
                    header.SetAngularJacobian(i, point);

                    Push(header.BodyIndexA, impulse * jacobian.Normal, impulse * point.Jac.AngularA);
                    Push(header.BodyIndexB, -impulse * jacobian.Normal, impulse * point.Jac.AngularB);
                }
            }

            public void Execute(ref ModifiableJacobianHeader header, ref ModifiableTriggerJacobian jacobian)
            {
            }

            private void Push(int body, float3 linearImpulse, float3 angularImpulse)
            {
                if (body < MotionCount)
                {
                    ref MotionVelocity motion = ref Motions[body];
                    motion.LinearVelocity += linearImpulse * motion.InverseMass;
                    motion.AngularVelocity += angularImpulse * motion.InverseInertia;
                }
            }
        }
    }
}
