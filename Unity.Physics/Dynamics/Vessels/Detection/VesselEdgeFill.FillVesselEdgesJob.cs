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

            [ReadOnly]
            public NativeArray<JointForestRole> Roles;

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
                JointUnit jointUnit = Units[unit];
                uint oldStamp = EdgeCacheStamps[part];
                uint stamp = RefillAll || SettingsChanged ? UnitParameters.Stamp(World, jointUnit, Groups, Members) : oldStamp;
                if (!RefillAll && (stamp == oldStamp || Roles[unit].IsServoTree())
                    && UnitParameters.Same(EdgeCacheFrames[2 * part], childMotion)
                    && UnitParameters.Same(EdgeCacheFrames[2 * part + 1], parentMotion))
                {
                    VesselEdge cached = EdgeCache[part];
                    if (stamp == oldStamp)
                    {
                        Edges[part] = cached;
                        return;
                    }

                    // Frame changes recompile the vessel. Of the remaining servo settings, only the break limits affect the edge.
                    JointData* changed = World->joints[jointUnit.Joint];
                    if (cached.BreakForce == changed->breakForce && cached.BreakTorque == changed->breakTorque)
                    {
                        Edges[part] = cached;
                        EdgeCacheStamps[part] = stamp;
                        return;
                    }
                }

                JointData* joint = World->joints[jointUnit.Joint];
                ServoAxis servo = Roles[unit] switch
                {
                    JointForestRole.ServoTree => ServoAxis.AngularX,
                    JointForestRole.PistonTree => ServoAxis.LinearX,
                    _ => ServoAxis.None,
                };
                UnitParameters parameters = UnitParameters.Of(World, jointUnit, Groups, Members, servo);
                bool childIsBody0 = PartIsBody0[part] != 0;
                float3 childAnchor = childIsBody0 ? parameters.Anchor0 : parameters.Anchor1;
                float3 parentAnchor = childIsBody0 ? parameters.Anchor1 : parameters.Anchor0;
                quaternion childFrame = childIsBody0 ? parameters.Rotation0 : parameters.Rotation1;
                quaternion parentFrame = childIsBody0 ? parameters.Rotation1 : parameters.Rotation0;
                bool frameInChild = servo == ServoAxis.LinearX && childIsBody0;

                double flexTorque = double.PositiveInfinity;
                double unlockRotation = 0.0;
                if (servo == ServoAxis.None)
                {
                    var angles = JointTranslation.BuildJointsJob.NearAngularLimits(joint);
                    bool3 narrow = angles < float.PositiveInfinity & parameters.Stiffness < double.PositiveInfinity;
                    flexTorque = math.cmin(math.select(double.PositiveInfinity, math.max(1e-12, (double3)angles * parameters.Stiffness), narrow));
                    float nearest = math.cmin(angles);
                    unlockRotation = nearest < float.PositiveInfinity ? math.sin(0.375 * math.max(0.001, nearest)) : 0.0;
                }
                var edge = new VesselEdge
                {
                    Parent = PartParents[part],
                    State = servo != ServoAxis.None ? VesselEdgeState.FreeAxis : VesselEdgeState.Yielding,
                    MayLock = servo == ServoAxis.None,
                    ServoAxis = servo,
                    FrameInChild = frameInChild,
                    ParentAnchor = UnitParameters.InMotionFrame(parent, parentAnchor),
                    ChildAnchor = UnitParameters.InMotionFrame(child, childAnchor),
                    Frame = frameInChild ? UnitParameters.Frame(child, childFrame) : UnitParameters.Frame(parent, parentFrame),
                    Rest = UnitParameters.Rest(parent, parentFrame, child, childFrame),
                    AngularStiffness = parameters.Stiffness,
                    AngularDamping = parameters.Damping,
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
