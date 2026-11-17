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
using Unity.Physics.Dynamics.Contacts;

namespace Unity.Physics.Collision.Filter.Groups
{
    internal static partial class PairFilter
    {
        /// <summary>
        /// Remove overlapping body pairs when all collider pairs are ignored by Physics.IgnoreCollision or collision
        /// groups.
        /// Cache results until filters or collider builds change. ContactMaterials.ReadContactsJob filters individual collider pairs.
        /// </summary>
        [BurstCompile]
        internal unsafe struct DropIgnoredPairsJob : IJobParallelForDefer
        {
            [NativeDisableParallelForRestriction]
            public NativeArray<DispatchPairSequencer.DispatchPair> Pairs;

            [NativeDisableUnsafePtrRestriction]
            public BodyData* Store;

            [NativeDisableUnsafePtrRestriction]
            public StoreGlobals* Globals;

            [NativeDisableUnsafePtrRestriction]
            public RigidBody* Bodies;

            [ReadOnly]
            public NativeHashMap<ulong, KnownBodyPairs.KnownBodyPair> Known;

            [ReadOnly]
            public NativeReference<uint> WholeBuilds;

            /// <summary>
            /// Stores results calculated during this step for <see cref="RememberBodyPairsJob"/> to cache.
            /// </summary>
            public NativeQueue<KeyValue>.ParallelWriter Learned;

            [NativeDisableUnsafePtrRestriction]
            public WorldData* World;

            /// <summary>The joint units, to find the leading joint of a vessel's edge or closure.</summary>
            [ReadOnly]
            public NativeArray<JointUnit> Units;

            /// <summary>See <see cref="VesselBuild.BodyVessels"/>. Empty when vessels are off.</summary>
            [ReadOnly]
            public NativeArray<int> BodyVessels;

            /// <summary>For each vessel, 1 when it takes part in this step.</summary>
            [ReadOnly]
            public NativeArray<byte> VesselActive;

            /// <summary>See <see cref="VesselBuild.PairUnits"/>.</summary>
            [ReadOnly]
            public NativeParallelMultiHashMap<ulong, int> VesselPairUnits;

            internal struct KeyValue
            {
                public ulong Key;
                public KnownBodyPairs.KnownBodyPair Value;
            }

            public void Execute(int index)
            {
                DispatchPairSequencer.DispatchPair pair = Pairs[index];
                if (pair.IsValid && pair.IsContact && pair.IsCollisionEnabled && (IsVesselJointed(pair, index) || IsIgnored(pair)))
                {
                    pair.DisableCollision();
                    Pairs[index] = pair;
                }
            }

            /// <summary>
            /// Return whether an active vessel joint disables collision for this body pair.
            /// Preserve the pair when another joint between the bodies allows collision.
            /// </summary>
            private bool IsVesselJointed(DispatchPairSequencer.DispatchPair pair, int index)
            {
                if (BodyVessels.Length == 0)
                {
                    return false;
                }

                Entity entityA = Bodies[pair.BodyIndexA].Entity;
                Entity entityB = Bodies[pair.BodyIndexB].Entity;
                int rowA = entityA.Index;
                int rowB = entityB.Index;
                if (entityA.Version == 0 || entityB.Version == 0 || rowA >= BodyVessels.Length || rowB >= BodyVessels.Length)
                {
                    return false;
                }

                int vessel = BodyVessels[rowA];
                if (vessel < 0 || vessel != BodyVessels[rowB] || VesselActive[vessel] == 0)
                {
                    return false;
                }

                if (!VesselPairUnits.TryGetFirstValue(VesselBuild.PairKey(rowA, rowB), out int unit, out NativeParallelMultiHashMapIterator<ulong> iterator))
                {
                    return false;
                }

                bool disabled = false;
                do
                {
                    disabled |= (World->joints[Units[unit].Joint]->flags & StoreConstants.JointFlagCollisionEnabled) == 0;
                }
                while (!disabled && VesselPairUnits.TryGetNextValue(out unit, ref iterator));

                if (!disabled || index == 0)
                {
                    return disabled;
                }

                DispatchPairSequencer.DispatchPair previous = Pairs[index - 1];
                bool permittedByJoint = previous.IsJoint && previous.BodyIndexA == pair.BodyIndexA && previous.BodyIndexB == pair.BodyIndexB;
                return !permittedByJoint;
            }

            private bool IsIgnored(DispatchPairSequencer.DispatchPair pair)
            {
                uint rowA = (uint)Bodies[pair.BodyIndexA].Entity.Index;
                uint rowB = (uint)Bodies[pair.BodyIndexB].Entity.Index;

                // When one of the bodies has no collider with an ignore id, only the collision groups can stop the pair.
                // Reading the groups costs less than the cache lookup, so the result isn't cached.
                if (!HasIgnoreIds(Store + rowA) || !HasIgnoreIds(Store + rowB))
                {
                    return IsIgnored(Store + rowA, Store + rowB);
                }

                uint low = math.min(rowA, rowB);
                uint high = math.max(rowA, rowB);
                ulong key = ((ulong)low << 32) | high;
                uint ignoreVersion = Globals->ignoreVersion;
                uint wholeBuilds = WholeBuilds.Value;

                // When a body is destroyed, the store moves its last body into the destroyed body's row. That changes neither
                // version, so the serial numbers tell whether the rows still hold the bodies that the entry was made for.
                uint serialLow = Store[low].owner->serial;
                uint serialHigh = Store[high].owner->serial;
                if (Known.TryGetValue(key, out KnownBodyPairs.KnownBodyPair known) && known.IgnoreVersion == ignoreVersion && known.WholeBuilds == wholeBuilds
                    && known.SerialLow == serialLow && known.SerialHigh == serialHigh)
                {
                    return known.Ignored;
                }

                bool ignored = IsIgnored(Store + rowA, Store + rowB);
                Learned.Enqueue(new KeyValue
                {
                    Key = key,
                    Value = new KnownBodyPairs.KnownBodyPair
                    {
                        IgnoreVersion = ignoreVersion,
                        WholeBuilds = wholeBuilds,
                        SerialLow = serialLow,
                        SerialHigh = serialHigh,
                        Ignored = ignored,
                    },
                });
                return ignored;
            }

            private static bool HasIgnoreIds(BodyData* row)
            {
                for (int i = 0; i < row->shapeCount; i++)
                {
                    if (row->shapes[i]->ignorePairId != 0)
                    {
                        return true;
                    }
                }

                return false;
            }

            private bool IsIgnored(BodyData* rowA, BodyData* rowB)
            {
                for (int a = 0; a < rowA->shapeCount; a++)
                {
                    ShapeData* shapeA = rowA->shapes[a];
                    if (!ColliderBuilder.IsSupported(shapeA))
                    {
                        continue;
                    }

                    for (int b = 0; b < rowB->shapeCount; b++)
                    {
                        ShapeData* shapeB = rowB->shapes[b];
                        if (ColliderBuilder.IsSupported(shapeB) && !IgnoreTable.Ignores(Globals, shapeA, shapeB))
                        {
                            return false;
                        }
                    }
                }

                return true;
            }
        }
    }
}
