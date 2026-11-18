using System;
using Unity.Burst;
using Unity.Burst.Intrinsics;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Physics.Dynamics.Vessels.Model;
using Unity.Physics.Dynamics.Vessels.Sections;
using Unity.Physics.Store.Bodies;
using Unity.Physics.Store.Joints;
using Unity.Physics.Store.World;
using Unity.Physics.Stepping;
using Unity.Physics.Store.Memory;
using Unity.Physics.Dynamics.Joint.Translation;
using Unity.Physics.Dynamics.Joint.Grouping;

namespace Unity.Physics.Dynamics.Vessels.Detection
{
    internal static partial class VesselClosureFill
    {
        /// <summary>
        /// Fill closures from leading body0 to body1, with rigid linear rows and per-axis angular drives.
        /// </summary>
        [BurstCompile]
        internal unsafe struct FillVesselClosuresJob : IJobParallelFor
        {
            [NativeDisableUnsafePtrRestriction]
            public WorldData* World;

            [ReadOnly]
            public NativeArray<JointUnit> Units;

            // See the comment on VesselEdgeFill.FillVesselEdgesJob.Groups.
            [NativeDisableParallelForRestriction]
            [NativeDisableContainerSafetyRestriction]
            public NativeArray<JointGroup> Groups;

            [NativeDisableParallelForRestriction]
            [NativeDisableContainerSafetyRestriction]
            public NativeArray<JointGroupMember> Members;

            [ReadOnly]
            public NativeArray<int> UnitVessels;

            [ReadOnly]
            public NativeArray<byte> Active;

            [ReadOnly]
            public NativeArray<int> ClosureUnits;

            [ReadOnly]
            public NativeArray<int> ClosurePartA;

            [ReadOnly]
            public NativeArray<int> ClosurePartB;

            /// <summary>
            /// The compiled vessels and the store body of each part. ClosurePartA and ClosurePartB count from the first
            /// part of their vessel.
            /// </summary>
            [ReadOnly]
            public NativeArray<VesselRecord> Vessels;

            [ReadOnly]
            public NativeArray<int> PartStoreBodies;

            public NativeArray<VesselClosure> Closures;

            /// <summary>See <see cref="VesselEdgeFill.FillVesselEdgesJob.RefillAll"/>.</summary>
            public bool RefillAll;

            /// <summary>See <see cref="VesselEdgeFill.FillVesselEdgesJob.SettingsChanged"/>.</summary>
            public bool SettingsChanged;

            public NativeArray<VesselClosure> ClosureCache;

            [NativeDisableParallelForRestriction]
            public NativeArray<RigidTransform> ClosureCacheFrames;

            public NativeArray<uint> ClosureCacheStamps;

            public void Execute(int closure)
            {
                if (ClosureUnits[closure] < 0)
                {
                    // A closure whose joint went away is a tombstone. With no stiffness and no damping, it has no effect.
                    Closures[closure] = new VesselClosure
                    {
                        PartA = ClosurePartA[closure],
                        PartB = ClosurePartB[closure],
                        Frame = double3x3.identity,
                        Rest = new double4(0.0, 0.0, 0.0, 1.0),
                    };
                    return;
                }

                JointUnit unit = Units[ClosureUnits[closure]];
                if (Active[UnitVessels[ClosureUnits[closure]]] == 0)
                {
                    // The vessel takes no part in this step. See the same case in VesselEdgeFill.FillVesselEdgesJob.
                    Closures[closure] = new VesselClosure
                    {
                        PartA = ClosurePartA[closure],
                        PartB = ClosurePartB[closure],
                        Frame = double3x3.identity,
                        Rest = new double4(0.0, 0.0, 0.0, 1.0),
                    };
                    ClosureCacheFrames[2 * closure] = default;
                    ClosureCacheFrames[2 * closure + 1] = default;
                    return;
                }

                // Find bodies through vessel parts so that cache reuse avoids reading the leading joint record.
                JointTranslation.BuildJointsJob.PrepareVesselGroup(World, unit, Groups, Members);
                int firstPart = Vessels[UnitVessels[ClosureUnits[closure]]].FirstPart;
                BodyData* bodyA = World->bodies + PartStoreBodies[firstPart + ClosurePartA[closure]];
                BodyData* bodyB = World->bodies + PartStoreBodies[firstPart + ClosurePartB[closure]];
                RigidTransform motionA = bodyA->mass->Transform;
                RigidTransform motionB = bodyB->mass->Transform;
                uint stamp = RefillAll || SettingsChanged ? UnitParameters.Stamp(World, unit, Groups, Members) : ClosureCacheStamps[closure];
                if (!RefillAll && stamp == ClosureCacheStamps[closure] && UnitParameters.Same(ClosureCacheFrames[2 * closure], motionA)
                    && UnitParameters.Same(ClosureCacheFrames[2 * closure + 1], motionB))
                {
                    Closures[closure] = ClosureCache[closure];
                    return;
                }

                UnitParameters parameters = UnitParameters.Of(World, unit, Groups, Members);
                float nearest = math.cmin(JointTranslation.BuildJointsJob.NearAngularLimits(World->joints[unit.Joint]));
                double unlockRotation = nearest < float.PositiveInfinity ? math.sin(0.375 * math.max(0.001, nearest)) : 0.0;
                var filled = new VesselClosure
                {
                    PartA = ClosurePartA[closure],
                    PartB = ClosurePartB[closure],
                    AnchorA = UnitParameters.InMotionFrame(bodyA, parameters.Anchor0),
                    AnchorB = UnitParameters.InMotionFrame(bodyB, parameters.Anchor1),
                    Frame = UnitParameters.Frame(bodyA, parameters.Rotation0),
                    Rest = UnitParameters.Rest(bodyA, parameters.Rotation0, bodyB, parameters.Rotation1),
                    UnlockRotationSquared = unlockRotation * unlockRotation,
                    LinearStiffness = double.PositiveInfinity,
                    AngularStiffness = parameters.Stiffness,
                    AngularDamping = parameters.Damping,
                };
                Closures[closure] = filled;
                ClosureCache[closure] = filled;
                ClosureCacheFrames[2 * closure] = motionA;
                ClosureCacheFrames[2 * closure + 1] = motionB;
                ClosureCacheStamps[closure] = stamp;
            }
        }
    }
}
