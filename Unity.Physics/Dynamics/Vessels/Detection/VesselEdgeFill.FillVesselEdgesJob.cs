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
    internal static partial class VesselEdgeFill
    {
        /// <summary>
        /// Fill part body indices and edges from stored joints, converting frames and anchors to motion coordinates.
        /// </summary>
        [BurstCompile]
        internal unsafe struct FillVesselEdgesJob : IJobParallelFor
        {
            [NativeDisableUnsafePtrRestriction]
            public WorldData* World;

            [ReadOnly]
            public NativeArray<JointUnit> Units;

            // Concurrent fill jobs write disjoint groups: active edges here, active closures in
            // VesselClosureFill.FillVesselClosuresJob,
            // and inactive units in JointTranslation.BuildJointsJob. Do not read groups owned by another job.
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
            public NativeArray<int> PartStoreBodies;

            [ReadOnly]
            public NativeArray<int> PartUnits;

            [ReadOnly]
            public NativeArray<int> PartParents;

            [ReadOnly]
            public NativeArray<byte> PartIsBody0;

            public NativeArray<VesselEdge> Edges;

            /// <summary>
            /// For each part, the index of its vessel's first part. Add this to a parent's relative index to find its index
            /// in the full list.
            /// </summary>
            [ReadOnly]
            public NativeArray<int> PartVesselFirst;

            /// <summary>
            /// True when the cached edges can't be reused, because the units or the compiled vessels changed since the
            /// cache was filled.
            /// </summary>
            public bool RefillAll;

            /// <summary>
            /// True when some joint's settings changed since the cache was filled. Each entry then compares the settings
            /// stamp of its unit with the stamp that it recorded.
            /// </summary>
            public bool SettingsChanged;

            public NativeArray<VesselEdge> EdgeCache;

            [NativeDisableParallelForRestriction]
            public NativeArray<RigidTransform> EdgeCacheFrames;

            public NativeArray<uint> EdgeCacheStamps;

            public void Execute(int part)
            {
                int storeBody = PartStoreBodies[part];
                int unit = PartUnits[part];
                if (unit < 0)
                {
                    // A root, or an edge whose joint went away. Such an edge is a tombstone: the solver treats it as broken
                    // and never locks it.
                    bool root = PartParents[part] < 0;
                    Edges[part] = new VesselEdge
                    {
                        Parent = PartParents[part],
                        State = root ? VesselEdgeState.Yielding : VesselEdgeState.Broken,
                        MayLock = root,
                        Frame = double3x3.identity,
                        Rest = new double4(0.0, 0.0, 0.0, 1.0),
                    };
                    return;
                }

                if (Active[UnitVessels[unit]] == 0)
                {
                    // Inactive vessels are not solved. Invalidate cached frames so that reactivation recomputes
                    // changed settings.
                    Edges[part] = new VesselEdge
                    {
                        Parent = PartParents[part],
                        State = VesselEdgeState.Broken,
                        Frame = double3x3.identity,
                        Rest = new double4(0.0, 0.0, 0.0, 1.0),
                    };
                    EdgeCacheFrames[2 * part] = default;
                    EdgeCacheFrames[2 * part + 1] = default;
                    return;
                }

                JointTranslation.BuildJointsJob.PrepareVesselGroup(World, Units[unit], Groups, Members);
                BodyData* child = World->bodies + storeBody;
                BodyData* parent = World->bodies + PartStoreBodies[PartVesselFirst[part] + PartParents[part]];
                RigidTransform childMotion = child->mass->Transform;
                RigidTransform parentMotion = parent->mass->Transform;
                uint stamp = RefillAll || SettingsChanged ? UnitParameters.Stamp(World, Units[unit], Groups, Members) : EdgeCacheStamps[part];
                if (!RefillAll && stamp == EdgeCacheStamps[part] && UnitParameters.Same(EdgeCacheFrames[2 * part], childMotion)
                    && UnitParameters.Same(EdgeCacheFrames[2 * part + 1], parentMotion))
                {
                    Edges[part] = EdgeCache[part];
                    return;
                }

                UnitParameters parameters = UnitParameters.Of(World, Units[unit], Groups, Members);
                bool childIsBody0 = PartIsBody0[part] != 0;
                float3 childAnchor = childIsBody0 ? parameters.Anchor0 : parameters.Anchor1;
                float3 parentAnchor = childIsBody0 ? parameters.Anchor1 : parameters.Anchor0;
                quaternion childFrame = childIsBody0 ? parameters.Rotation0 : parameters.Rotation1;
                quaternion parentFrame = childIsBody0 ? parameters.Rotation1 : parameters.Rotation0;

                // Use the softer swing axis because the edge uses one spring for both swing axes. Use the damping from that
                // same axis.
                int swing = parameters.Stiffness.y <= parameters.Stiffness.z ? 1 : 2;
                var angles = JointTranslation.BuildJointsJob.NearAngularLimits(World->joints[Units[unit].Joint]);
                var stiffness = new double3(parameters.Stiffness.x, parameters.Stiffness[swing], parameters.Stiffness[swing]);
                bool3 narrow = angles < float.PositiveInfinity & stiffness < double.PositiveInfinity;
                double flexTorque = math.cmin(math.select(double.PositiveInfinity, math.max(1e-12, (double3)angles * stiffness), narrow));
                float nearest = math.cmin(angles);
                double unlockRotation = nearest < float.PositiveInfinity ? math.sin(0.375 * math.max(0.001, nearest)) : 0.0;
                var edge = new VesselEdge
                {
                    Parent = PartParents[part],
                    State = VesselEdgeState.Yielding,
                    MayLock = true,
                    ParentAnchor = UnitParameters.InMotionFrame(parent, parentAnchor),
                    ChildAnchor = UnitParameters.InMotionFrame(child, childAnchor),
                    Frame = UnitParameters.Frame(parent, parentFrame),
                    Rest = UnitParameters.Rest(parent, parentFrame, child, childFrame),
                    TwistStiffness = parameters.Stiffness.x,
                    SwingStiffness = parameters.Stiffness[swing],
                    TwistDamping = parameters.Damping.x,
                    SwingDamping = parameters.Damping[swing],
                    BreakForce = parameters.BreakForce,
                    BreakTorque = parameters.BreakTorque,
                    FlexYieldTorque = flexTorque,
                    UnlockRotationSquared = unlockRotation * unlockRotation,
                };
                Edges[part] = edge;
                EdgeCache[part] = edge;
                EdgeCacheFrames[2 * part] = childMotion;
                EdgeCacheFrames[2 * part + 1] = parentMotion;
                EdgeCacheStamps[part] = stamp;
            }
        }
    }
}
