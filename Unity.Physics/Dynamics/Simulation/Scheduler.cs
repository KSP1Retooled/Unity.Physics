using Unity.Physics.Configuration;
//#define SERIAL_RADIX_SORT

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Jobs.LowLevel.Unsafe;
using Unity.Mathematics;
using UnityEngine.Assertions;

namespace Unity.Physics
{
    /// <summary>
    /// Builds phased pairs of interacting bodies, used to parallelize work items during the
    /// simulation step.
    /// </summary>
    partial struct DispatchPairSequencer
    {
        /// <summary>
        /// A pair of interacting bodies (either potentially colliding, or constrained together using a
        /// Joint). The joints and flags are compressed into a single 32 bit value as follows:
        ///
        ///      30        20        10
        ///    21098765432109876543210987654321
        /// 0b_11111111111111111111111111111111
        ///    ED [     Joint:29 bits         ]
        ///
        ///
        /// This gives a limit of
        ///    2^32-1 / 2 Rigid bodies (/2 to remove negative values), with a collision pair supported for each body pair.
        ///    2^29-1 / 2 Joints (1 bit used for [E]nable Collision flag, 1 unused bit, 1 bit used for [D]irect Solver flag)
        ///
        /// Note: The [E]nable Collision flag must appear before any other bits, to ensure collisions will always have a higher
        /// hash value than joints, making them appear after joints in the sorted dispatch pair array, which is important for the
        /// joints' "Enable Collision" feature and for solving collisions last.
        ///
        /// We additionally choose indices so that BodyIndexA < BodyIndexB. This has subtle side-effects:
        /// * If one body in the pair is static, it will be body B.
        /// * Indices used for jointed pairs are not necessarily the same as selected in the joint
        /// * For some body A, all its static collisions will be contiguous.
        /// </summary>
        [DebuggerDisplay("{IsValid ? \"Valid\" : \"Invalid\"}, {IsContact ? \"Contact\" : \"Joint\"}, {SolverType}, Enable Collision: {IsCollisionEnabled}, [{BodyIndexA}, {BodyIndexB}]")]
        internal struct DispatchPair
            : IEquatable<DispatchPair>
        {
            int m_BodyIndexA;
            int m_BodyIndexB;
            uint m_Data;

            static readonly uint k_InvalidJointIndex = 0b_0001_1111_1111_1111_1111_1111_1111_1111; // 29 bits
            static readonly uint k_JointMask = ~k_InvalidJointIndex;
            static readonly uint k_DirectSolverBit = 1u << 30;
            static readonly uint k_EnableCollisionBit = k_DirectSolverBit << 1;

            /// <summary>   Indicates whether this pair is valid. </summary>
            ///
            /// <value> True if this pair is valid, false if not. </value>
            public bool IsValid => m_BodyIndexA != DispatchPair.Invalid.BodyIndexA || BodyIndexASubArraySortKey != DispatchPair.Invalid.BodyIndexASubArraySortKey;

            /// <summary>   Indicates whether this pair is a joint. </summary>
            ///
            /// <value> True if this pair is a joint, false if not. </value>
            public bool IsJoint => JointIndex != k_InvalidJointIndex;

            /// <summary>   Indicates whether this pair is a contact. </summary>
            ///
            /// <value> True if this pair is a contact, false if not. </value>
            public bool IsContact => !IsJoint;

            /// <summary>   Gets or sets the Solver type used for this pair. </summary>
            ///
            /// <value> The Solver type used for this pair. </value>
            public SolverType SolverType
            {
                get => (m_Data & k_DirectSolverBit) != 0 ? SolverType.Direct : SolverType.Iterative;
                set => m_Data = (m_Data & ~k_DirectSolverBit) | (value == SolverType.Direct ? k_DirectSolverBit : 0);
            }

            /// <summary>   Gets the invalid dispatch pair. </summary>
            ///
            /// <value> The invalid dispatch pair. </value>
            public static DispatchPair Invalid => new DispatchPair { m_BodyIndexA = ~0, m_BodyIndexB = ~0, m_Data = ~(uint)0 };

            /// <summary>   Gets or sets the body index a. </summary>
            ///
            /// <value> The body index a. </value>
            public int BodyIndexA
            {
                set => m_BodyIndexA = value;
                get => m_BodyIndexA;
            }

            /// <summary>   Gets the body index b. </summary>
            ///
            /// <value> The body index b. </value>
            public int BodyIndexB
            {
                set => m_BodyIndexB = value;
                get => m_BodyIndexB;
            }

            /// <summary>   Gets the zero-based index of the joint. </summary>
            ///
            /// <value> The joint index. </value>
            public int JointIndex
            {
                get => (int)(m_Data & k_InvalidJointIndex);
                internal set
                {
                    Assert.IsTrue(value < k_InvalidJointIndex);
                    m_Data = (m_Data & k_JointMask) | (uint)value;
                }
            }

            /// <summary>
            /// Gets a value indicating whether collision is enabled for this pair.
            /// If the pair is a joint, this indicates whether the joint allows collision between the bodies.
            /// If the pair is a collision, this indicates whether the collision is enabled.
            /// </summary>
            internal bool IsCollisionEnabled => (m_Data & k_EnableCollisionBit) != 0;

            /// <summary>
            /// Disables collision for this pair.
            /// </summary>
            internal void DisableCollision()
            {
                SafetyChecks.CheckAreEqualAndThrow(IsContact, true);
                m_Data &= ~k_EnableCollisionBit;
            }

            /// <summary>
            /// Sort key for sorting of DispatchPair sub-arrays with identical BodyIndexA.
            /// Packed version of BodyIndexB (leading) and rest of data.
            /// </summary>
            internal ulong BodyIndexASubArraySortKey => (ulong)m_BodyIndexB << 32 | m_Data;

            /// <summary>
            /// Return whether pairs can share one phase sequentially: the same two bodies, or one dynamic body with
            /// any statics.
            /// </summary>
            internal bool IsSameLink(DispatchPair other, int numDynamicBodies) => BodyIndexA == other.BodyIndexA
                && (BodyIndexB == other.BodyIndexB || (BodyIndexB >= numDynamicBodies && other.BodyIndexB >= numDynamicBodies));

            /// <summary>   Creates collision pair from a broad phase overlap result. </summary>
            ///
            /// <param name="overlapResult"> The overlap result. </param>
            ///
            /// <returns>   The new dispatch pair representing collision pair. </returns>
            public static DispatchPair CreateCollisionPair(Broadphase.OverlapResult overlapResult)
            {
                // Note: The "Enable Collision" flag is set to true deliberately, since initially, all contacts
                // are enabled. They might get disabled later during the pair sorting if none of the joints between the
                // corresponding body pair allows collision.
                // Furthermore, setting the "Enable Collision" and all bits related to the Joint (k_InvalidJointIndex)
                // ensures that joints always appear before contacts in the sorted dispatch pair array.
                // This serves two purposes:
                // 1) It guarantees that joints are always solved before the higher priority contacts in the iterative
                //    solver, yielding lower constraint error for contacts than for joints after every solver iteration.
                // 2) It is a requirement in the joints' "Enable Collision" feature, in which collisions between a given
                //    body pair are enabled if any of the joints between them allows it.
                return Create(overlapResult.BodyPair, (int)k_InvalidJointIndex, enableCollision: true, overlapResult.SolverType);
            }

            /// <summary>   Creates a joint. </summary>
            ///
            /// <param name="pair">             The body pair. </param>
            /// <param name="jointIndex">       Zero-based index of the joint. </param>
            /// <param name="enableCollision">  This joint enables collision between the given body pair. </param>
            /// <param name="solverType">       Solver type used for this joint. </param>
            ///
            /// <returns>   The new dispatch pair representing the joint. </returns>
            public static DispatchPair CreateJoint(BodyIndexPair pair, int jointIndex, bool enableCollision, SolverType solverType = SolverType.Iterative)
            {
                Assert.IsTrue(jointIndex < k_InvalidJointIndex);
                return Create(pair, jointIndex, enableCollision, solverType);
            }

            static DispatchPair Create(BodyIndexPair pair, int jointIndex, bool enableCollision, SolverType solverType)
            {
                int selectedA = math.min(pair.BodyIndexA, pair.BodyIndexB);
                int selectedB = math.max(pair.BodyIndexA, pair.BodyIndexB);
                return new DispatchPair
                {
                    BodyIndexA = selectedA,
                    BodyIndexB = selectedB,
                    m_Data =
                        (solverType == SolverType.Direct ? k_DirectSolverBit : 0) |
                        (enableCollision ? k_EnableCollisionBit : 0) |
                        (uint)(jointIndex & k_InvalidJointIndex)
                };
            }

            public bool Equals(DispatchPair other)
            {
                return m_BodyIndexA == other.m_BodyIndexA && BodyIndexASubArraySortKey == other.BodyIndexASubArraySortKey;
            }

            public override int GetHashCode()
            {
                return HashCode.Combine(m_BodyIndexA.GetHashCode(), BodyIndexASubArraySortKey.GetHashCode());
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static void SortDispatchPairs(NativeList<DispatchPair> dispatchPairs) => dispatchPairs.Sort(new DispatchPairComparer());

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static void SortDispatchPairs(NativeArray<DispatchPair> dispatchPairs) => dispatchPairs.Sort(new DispatchPairComparer());

        [NoAlias]
        internal struct SolverSchedulerInfo : IDisposable
        {
            // The serial step uses at most one work item per island bucket.
            internal const int k_MaxSingleThreadedWorkItems = NumIslandBuckets;

            [NoAlias]
            public IterativeSolverSchedulerInfo IterativePairsIterativeScheduling;
            [NoAlias]
            public IterativeSolverSchedulerInfo DirectPairsIterativeScheduling;
            [NoAlias]
            public DirectSolverSchedulerInfo DirectPairsDirectScheduling;

            [NoAlias]
            public NativeReference<int> NumIterativeWorkItems;

            public SolverSchedulerInfo(int numPhases, Allocator allocator = Allocator.TempJob)
            {
                IterativePairsIterativeScheduling = new IterativeSolverSchedulerInfo(numPhases, allocator);
                DirectPairsIterativeScheduling = new IterativeSolverSchedulerInfo(numPhases, allocator);
                DirectPairsDirectScheduling = DirectSolverSchedulerInfo.Create(allocator);
                NumIterativeWorkItems = new NativeReference<int>(0, allocator);
            }

            /// <summary>
            ///     Finalizes the iterative solver scheduler info.
            ///     <para/>
            ///     Sets <see cref="NumIterativeWorkItems">total iterative work item count</see> and
            ///     the <see cref="IterativeSolverSchedulerInfo.FirstWorkItemIndex">work item start
            ///     indices</see> for all iterative solvers.
            /// </summary>
            public void FinalizeIterativeWorkItemInfo()
            {
                var numWorkItems = 0;
                IterativePairsIterativeScheduling.FirstWorkItemIndex.Value = 0;
                numWorkItems += IterativePairsIterativeScheduling.NumWorkItems[0];
                DirectPairsIterativeScheduling.FirstWorkItemIndex.Value = numWorkItems;

                NumIterativeWorkItems.Value = IterativePairsIterativeScheduling.NumWorkItems[0] +
                    DirectPairsIterativeScheduling.NumWorkItems[0];
            }

            public void Dispose()
            {
                IterativePairsIterativeScheduling.Dispose();
                DirectPairsIterativeScheduling.Dispose();
                DirectPairsDirectScheduling.Dispose();
                NumIterativeWorkItems.Dispose();
            }

            public JobHandle ScheduleDisposeJob(JobHandle inputDeps)
            {
                return JobHandle.CombineDependencies(
                    IterativePairsIterativeScheduling.ScheduleDisposeJob(inputDeps),
                    DirectPairsIterativeScheduling.ScheduleDisposeJob(inputDeps),
                    JobHandle.CombineDependencies(DirectPairsDirectScheduling.ScheduleDisposeJob(inputDeps),
                        NumIterativeWorkItems.Dispose(inputDeps)));
            }
        }

        /// <summary>
        /// Island information for a dispatch pair in the direct solver
        /// </summary>
        [NoAlias]
        [DebuggerDisplay("DispatchPair Index: {UnphasedDispatchPairIndex}, Island Index {IslandIndex}")]
        internal struct DispatchPairIslandInfo
        {
            /// <summary> Index of the dispatch pair in the unphased (lexicographically sorted) array </summary>
            public int UnphasedDispatchPairIndex;
            /// <summary> Index of the island this dispatch pair belongs to </summary>
            public int IslandIndex;
        }

        /// <summary>
        /// The vessels of one direct island: the range of their indices in
        /// <see cref="DirectSolverSchedulerInfo.IslandVessels"/>, the number of their parts, and the number of the
        /// island's bodies that are not parts of its vessels.
        /// </summary>
        internal struct IslandVesselInfo
        {
            public int FirstVessel;
            public int VesselCount;
            public int VesselParts;
            public int OtherBodies;
        }

        /// <summary>
        /// Jacobian mapping information, mapping a dispatch pair in the direct solver to the
        /// location of its associated data in the Jacobian stream buffers.
        /// </summary>
        internal struct DispatchPairJacobianMapping
        {
            public UnsafeStream.ReaderState ReaderState;
            /// <summary> Work item index (buffer) where this dispatch pair's Jacobian data is located in the Jacobian stream </summary>
            public int WorkItemIndex;

            /// <summary>
            /// The number of consecutive constraint blocks. A contact pair has one block per manifold.
            /// </summary>
            public int NumBlocks;

            /// <summary> Indicates whether this mapping is valid </summary>
            public bool IsValid => WorkItemIndex >= 0;

            /// <summary> Invalid mapping, used for unmapped entries </summary>
            public static DispatchPairJacobianMapping Invalid => new DispatchPairJacobianMapping { WorkItemIndex = -1 };
        }

        [NoAlias]
        internal struct DirectSolverSchedulerInfo : IDisposable
        {
            /// <summary>
            /// Map from unphased direct pair index (used to identify pairs in the direct pair islands)
            /// to phased direct pair index (used to identify direct pairs in the phases of the iterative solver scheduling of
            /// the direct pairs. See <see cref="SolverSchedulerInfo.DirectPairsIterativeScheduling"/>).
            /// <para/> Note that the phased and unphased indices are both local to the direct pairs sub-array of the
            /// full dispatch pair array.
            /// <para/>
            /// With this map we can jump from the unphased dispatch pair indices with which the islands are
            /// organized (see <see cref="DispatchPairIslandInfo.UnphasedDispatchPairIndex"/>) to the (local) phased
            /// dispatch pair indices with which the direct pairs are organized in the global dispatch pair
            /// array (see <see cref="SolverSchedulerInfo.DirectPairsIterativeScheduling"/>).
            /// This map is among others used to allow locating the Jacobian data in the Jacobian stream buffers for
            /// a given unphased dispatch pair that is part of an island. The mapping from (local) phased dispatch pair
            /// index to said Jacobian data location is stored in <see cref="PhasedDispatchPairJacobianMappings"/> below
            /// and is produced in the <see cref="Solver.BuildDirectSolverJacobiansMapJob"/>.
            /// </summary>
            [NoAlias]
            public NativeList<int> UnphasedToPhasedDispatchPairMap;

            /// <summary>
            /// Mapping from the phased direct dispatch pair index (organized in phases
            /// in <see cref="SolverSchedulerInfo.DirectPairsIterativeScheduling"/>) to the Jacobian data location
            /// in the Jacobian stream buffers.
            /// </summary>
            [NoAlias]
            public NativeList<DispatchPairJacobianMapping> PhasedDispatchPairJacobianMappings;

            /// <summary>
            /// Island information for the direct pairs, grouped sequentially by islands, with
            /// "DispatchPairIslandInfoCounts[islandIndex]" many entries each, where islandIndex ranges from
            /// 0 to "DispatchPairIslandInfoCounts.Length - 1".
            /// </summary>
            [NoAlias]
            public NativeList<DispatchPairIslandInfo> DispatchPairIslandInfos;

            /// <summary> Indices into the DispatchPairIslandInfos array, indicating the first entry of each island. </summary>
            [NoAlias]
            public NativeList<int> FirstDispatchPairIslandInfoIndices;

            /// <summary> Number of dispatch pairs in each island, indexed by the island index. </summary>
            [NoAlias]
            public NativeList<int> DispatchPairIslandInfoCounts;

            /// <summary>
            /// The dynamic bodies grouped by island. <see cref="FirstIslandBodyIndices"/> identifies the start of each
            /// group. When an island integrates its own substeps, it integrates the bodies in this group.
            /// </summary>
            [NoAlias]
            public NativeList<int> IslandBodies;

            /// <summary> Indices into <see cref="IslandBodies"/>, indicating the first entry of each island. </summary>
            [NoAlias]
            public NativeList<int> FirstIslandBodyIndices;

            /// <summary>
            /// Indicates whether each dynamic body belongs to an island. The list is empty if there are no islands.
            /// </summary>
            [NoAlias]
            public NativeList<bool> BodiesInIslands;

            /// <summary>
            /// Contact count per island. Contacts follow joints in DispatchPairIslandInfos but are excluded from
            /// joint-info counts.
            /// </summary>
            [NoAlias]
            public NativeList<int> IslandContactCounts;

            /// <summary>
            /// The vessels of each island, indexed by the island index. The list is empty when the islands were found
            /// by a job that does not know the vessels.
            /// </summary>
            [NoAlias]
            public NativeList<IslandVesselInfo> IslandVesselInfos;

            /// <summary>
            /// The indices of each island's vessels in the step's vessel set, in ascending order within each island. See
            /// <see cref="IslandVesselInfos"/>.
            /// </summary>
            [NoAlias]
            public NativeList<int> IslandVessels;

            /// <summary> Number of islands to be solved with the direct solver. </summary>
            public int NumIslands => DispatchPairIslandInfoCounts.Length;

            /// <summary>
            /// First direct dispatch pair index in the global dispatch pair array indicating the beginning of the
            /// direct dispatch pairs sub-array.
            /// </summary>
            [NoAlias]
            public NativeReference<int> FirstDispatchPairIndex;

            public static DirectSolverSchedulerInfo Create(Allocator allocator = Allocator.TempJob)
            {
                return new DirectSolverSchedulerInfo
                {
                    UnphasedToPhasedDispatchPairMap = new NativeList<int>(allocator),
                    PhasedDispatchPairJacobianMappings = new NativeList<DispatchPairJacobianMapping>(allocator),
                    DispatchPairIslandInfos = new NativeList<DispatchPairIslandInfo>(allocator),
                    FirstDispatchPairIslandInfoIndices = new NativeList<int>(128, allocator),
                    DispatchPairIslandInfoCounts = new NativeList<int>(128, allocator),
                    IslandBodies = new NativeList<int>(allocator),
                    FirstIslandBodyIndices = new NativeList<int>(128, allocator),
                    BodiesInIslands = new NativeList<bool>(allocator),
                    IslandContactCounts = new NativeList<int>(128, allocator),
                    IslandVesselInfos = new NativeList<IslandVesselInfo>(allocator),
                    IslandVessels = new NativeList<int>(allocator),
                    FirstDispatchPairIndex = new NativeReference<int>(0, allocator)
                };
            }

            public void ResizeMaps(int numDispatchPairs)
            {
                PhasedDispatchPairJacobianMappings.Clear();
                PhasedDispatchPairJacobianMappings.ResizeUninitialized(numDispatchPairs);

                UnphasedToPhasedDispatchPairMap.Clear();
                UnphasedToPhasedDispatchPairMap.ResizeUninitialized(numDispatchPairs);
            }

            public void Dispose()
            {
                UnphasedToPhasedDispatchPairMap.Dispose();
                PhasedDispatchPairJacobianMappings.Dispose();
                DispatchPairIslandInfos.Dispose();
                FirstDispatchPairIslandInfoIndices.Dispose();
                DispatchPairIslandInfoCounts.Dispose();
                IslandBodies.Dispose();
                FirstIslandBodyIndices.Dispose();
                BodiesInIslands.Dispose();
                IslandContactCounts.Dispose();
                IslandVesselInfos.Dispose();
                IslandVessels.Dispose();
                FirstDispatchPairIndex.Dispose();
            }

            public JobHandle ScheduleDisposeJob(JobHandle inputDeps)
            {
                JobHandle islandBodies = JobHandle.CombineDependencies(IslandBodies.Dispose(inputDeps),
                    JobHandle.CombineDependencies(FirstIslandBodyIndices.Dispose(inputDeps), IslandContactCounts.Dispose(inputDeps)),
                    BodiesInIslands.Dispose(inputDeps));
                islandBodies = JobHandle.CombineDependencies(islandBodies, IslandVesselInfos.Dispose(inputDeps), IslandVessels.Dispose(inputDeps));

                return JobHandle.CombineDependencies(UnphasedToPhasedDispatchPairMap.Dispose(inputDeps), PhasedDispatchPairJacobianMappings.Dispose(inputDeps),
                    JobHandle.CombineDependencies(DispatchPairIslandInfos.Dispose(inputDeps), islandBodies,
                        JobHandle.CombineDependencies(FirstDispatchPairIslandInfoIndices.Dispose(inputDeps),
                            DispatchPairIslandInfoCounts.Dispose(inputDeps), FirstDispatchPairIndex.Dispose(inputDeps))));
            }
        }

        /// <summary>
        /// A phased set of dispatch pairs used to schedule solver jobs and distribute work between them.
        /// </summary>
        [NoAlias]
        internal struct IterativeSolverSchedulerInfo : IDisposable
        {
            // A structure which describes the number of items in a single phase
            internal struct SolvePhaseInfo
            {
                internal int DispatchPairCount; // The total number of pairs in this phase
                internal int BatchSize; // The number of items per thread work item; at most, the number of pairs
                internal int NumWorkItems; // The amount of "subtasks" of size BatchSize in this phase
                internal int FirstWorkItemIndexOffset;  // Offset to first work item index of this phase. Calculated as
                                                        // the sum of NumWorkItems in all previous phases.
                                                        // Index for this work item in the global work item array is
                                                        // IterativeSolverSchedulerInfo.FirstWorkItemIndex + SolvePhaseInfo.FirstWorkItemIndexOffset.
                internal int FirstDispatchPairOffset;   // Offset to first dispatch pair in this phase within the sub-array
                                                        // of dispatch pairs for this scheduler info.
                                                        // Index for this pair in the global dispatch pair array is
                                                        // IterativeSolverSchedulerInfo.FirstDispatchPairIndex + SolvePhaseInfo.FirstDispatchPairOffset.
                internal int LinkedNumDynamicBodies;    // Positive when all pairs of a link must stay in one work item.
                                                        // See DispatchPair.IsSameLink.
            }

            [NoAlias]
            internal NativeArray<SolvePhaseInfo> PhaseInfo;
            [NoAlias]
            internal NativeArray<int> NumActivePhases;
            [NoAlias]
            internal NativeReference<int> FirstWorkItemIndex;   // Index of first work item used for this scheduler info. Indicates where the
                                                                // solver data can be found in the corresponding streams (jacobian and contact).
                                                                // All work item indices specified in the SolvePhaseInfo of this scheduler info
                                                                // are relative to this index.
            [NoAlias]
            // Total number of work items in all phases for this scheduler info.
            internal NativeArray<int> NumWorkItems;

            [NoAlias]
            // Index of first dispatch pair for this scheduler info in the global dispatch pair array.
            internal NativeReference<int> FirstDispatchPairIndex;

            [NoAlias]
            // Total number of dispatch pairs in all phases for this scheduler info.
            internal NativeReference<int> NumDispatchPairs;

            /// <summary>
            /// The first pair and pair count for each work item in phases that group pairs by link.
            /// </summary>
            [NoAlias]
            internal NativeList<int2> LinkedWorkItemRanges;

            // For a given work item index offset returns phase id.
            internal int FindPhaseId(int workItemIndexOffset)
            {
                int phaseId = 0;
                for (int i = NumActivePhases[0] - 1; i >= 0; i--)
                {
                    if (workItemIndexOffset >= PhaseInfo[i].FirstWorkItemIndexOffset)
                    {
                        phaseId = i;
                        break;
                    }
                }

                return phaseId;
            }

            // For a given work item index offset returns index into PhasedDispatchPairs and number of pairs to read.
            internal int GetWorkItemReadOffset(int workItemIndexOffset, out int pairReadCount)
            {
                SolvePhaseInfo phaseInfo = PhaseInfo[FindPhaseId(workItemIndexOffset)];

                int numItemsToRead = phaseInfo.BatchSize;
                int readStartOffset = phaseInfo.FirstDispatchPairOffset + (workItemIndexOffset - phaseInfo.FirstWorkItemIndexOffset) * phaseInfo.BatchSize;

                int lastWorkItemIndexOffset = phaseInfo.FirstWorkItemIndexOffset + phaseInfo.NumWorkItems - 1;
                bool isLastWorkItemInPhase = workItemIndexOffset == lastWorkItemIndexOffset;
                if (isLastWorkItemInPhase)
                {
                    int numPairsBeforeLastWorkItem = (phaseInfo.NumWorkItems - 1) * phaseInfo.BatchSize;
                    numItemsToRead = phaseInfo.DispatchPairCount - numPairsBeforeLastWorkItem;
                    readStartOffset = phaseInfo.FirstDispatchPairOffset + numPairsBeforeLastWorkItem;
                }

                pairReadCount = numItemsToRead;

                return readStartOffset;
            }

            /// <summary>
            /// Return a work item's pair range, using precomputed link ranges in linked phases.
            /// </summary>
            internal int GetLinkedWorkItemReadOffset(int workItemIndexOffset, out int pairReadCount)
            {
                if (PhaseInfo[FindPhaseId(workItemIndexOffset)].LinkedNumDynamicBodies == 0)
                {
                    return GetWorkItemReadOffset(workItemIndexOffset, out pairReadCount);
                }

                int2 range = LinkedWorkItemRanges[workItemIndexOffset];
                pairReadCount = range.y;
                return range.x;
            }

            /// <summary>
            /// Compute linked work-item ranges before pair-reading jobs run.
            /// Do not discover ranges while Jacobian construction may invalidate pairs.
            /// </summary>
            internal void FindLinkedWorkItemRanges(NativeArray<DispatchPair> dispatchPairs)
            {
                LinkedWorkItemRanges.Clear();
                for (int workItem = 0; workItem < NumWorkItems[0]; workItem++)
                {
                    int start = FindLinkedReadOffset(workItem, dispatchPairs, out int pairReadCount);
                    LinkedWorkItemRanges.Add(new int2(start, pairReadCount));
                }
            }

            /// <summary>
            /// Assign an entire link to the work item where it starts, preventing concurrent workers from splitting
            /// a link.
            /// </summary>
            int FindLinkedReadOffset(int workItemIndexOffset, NativeArray<DispatchPair> dispatchPairs, out int pairReadCount)
            {
                int start = GetWorkItemReadOffset(workItemIndexOffset, out pairReadCount);
                SolvePhaseInfo phaseInfo = PhaseInfo[FindPhaseId(workItemIndexOffset)];
                int numDynamicBodies = phaseInfo.LinkedNumDynamicBodies;
                if (numDynamicBodies == 0)
                {
                    return start;
                }

                int firstPair = FirstDispatchPairIndex.Value;
                int phaseStart = phaseInfo.FirstDispatchPairOffset;
                int phaseEnd = phaseStart + phaseInfo.DispatchPairCount;
                int end = start + pairReadCount;
                while (start < end && start > phaseStart && dispatchPairs[firstPair + start].IsSameLink(dispatchPairs[firstPair + start - 1], numDynamicBodies))
                {
                    start++;
                }

                if (start == end)
                {
                    pairReadCount = 0;
                    return start;
                }

                while (end < phaseEnd && dispatchPairs[firstPair + end].IsSameLink(dispatchPairs[firstPair + end - 1], numDynamicBodies))
                {
                    end++;
                }

                pairReadCount = end - start;
                return start;
            }

            public IterativeSolverSchedulerInfo(int numPhases, Allocator allocator = Allocator.TempJob)
            {
                PhaseInfo = new NativeArray<SolvePhaseInfo>(numPhases, allocator);
                NumActivePhases = new NativeArray<int>(1, allocator);
                FirstWorkItemIndex = new NativeReference<int>(0, allocator);
                NumWorkItems = new NativeArray<int>(1, allocator);
                FirstDispatchPairIndex = new NativeReference<int>(0, allocator);
                NumDispatchPairs = new NativeReference<int>(0, allocator);
                LinkedWorkItemRanges = new NativeList<int2>(allocator);
            }

            public void Dispose()
            {
                if (PhaseInfo.IsCreated)
                {
                    PhaseInfo.Dispose();
                }

                if (NumActivePhases.IsCreated)
                {
                    NumActivePhases.Dispose();
                }

                if (FirstWorkItemIndex.IsCreated)
                {
                    FirstWorkItemIndex.Dispose();
                }

                if (NumWorkItems.IsCreated)
                {
                    NumWorkItems.Dispose();
                }

                if (FirstDispatchPairIndex.IsCreated)
                {
                    FirstDispatchPairIndex.Dispose();
                }

                if (NumDispatchPairs.IsCreated)
                {
                    NumDispatchPairs.Dispose();
                }

                if (LinkedWorkItemRanges.IsCreated)
                {
                    LinkedWorkItemRanges.Dispose();
                }
            }

            public JobHandle ScheduleDisposeJob(JobHandle inputDeps)
            {
                return JobHandle.CombineDependencies(new DisposeJob { PhaseInfo = PhaseInfo, NumActivePhases = NumActivePhases, NumWorkItems = NumWorkItems }.Schedule(inputDeps),
                    JobHandle.CombineDependencies(FirstWorkItemIndex.Dispose(inputDeps), NumDispatchPairs.Dispose(inputDeps), FirstDispatchPairIndex.Dispose(inputDeps)),
                    LinkedWorkItemRanges.Dispose(inputDeps));
            }

            // A job to dispose the phase information
            [BurstCompile]
            struct DisposeJob : IJob
            {
                [DeallocateOnJobCompletion]
                public NativeArray<SolvePhaseInfo> PhaseInfo;

                [DeallocateOnJobCompletion]
                public NativeArray<int> NumActivePhases;

                [DeallocateOnJobCompletion]
                public NativeArray<int> NumWorkItems;

                public void Execute() {}
            }
        }

        struct DispatchPairComparer : IComparer<DispatchPair>
        {
            public int Compare(DispatchPair x, DispatchPair y)
            {
                return math.select(x.BodyIndexA.CompareTo(y.BodyIndexA),
                    x.BodyIndexASubArraySortKey.CompareTo(y.BodyIndexASubArraySortKey),
                    x.BodyIndexA == y.BodyIndexA);
            }
        }

        /// <summary>
        /// Schedule a set of jobs to merge streams of body pairs and joints into a sorted array of
        /// dispatch pairs.
        /// </summary>
        internal static SimulationJobHandles ScheduleCreatePhasedDispatchPairsJob(in SimulationStepInput stepInput,
            ref NativeStream dynamicVsDynamicBroadphasePairsStream, ref NativeStream staticVsDynamicBroadphasePairsStream,
            JobHandle inputDeps, ref NativeList<DispatchPair> dispatchPairs, out SolverSchedulerInfo solverSchedulerInfo,
            out NativeList<DispatchPair> pairsToSortByIsland, bool multiThreaded = true, NativeReference<int> islandPairCount = default,
            IslandPairSortCache islandPairSorts = default)
        {
            SimulationJobHandles returnHandles = default;
            pairsToSortByIsland = default;

            // Use one sorting job for small island workloads to avoid the parallel chain's scheduling overhead.
            if (!multiThreaded || (islandPairCount.IsCreated && islandPairCount.Value <= SolverSwitches.SerialIslandPairLimit))
            {
                dispatchPairs = new NativeList<DispatchPair>(Allocator.TempJob);
                pairsToSortByIsland = new NativeList<DispatchPair>(Allocator.TempJob);
                solverSchedulerInfo = new SolverSchedulerInfo(NumIslandPhases);
                JobHandle sorted = new SortPairsForIslandsJob
                {
                    Joints = stepInput.World.DynamicsWorld.InternalJointsList.AsDeferredJobArray(),
                    DynamicVsDynamicPairs = dynamicVsDynamicBroadphasePairsStream.AsReader(),
                    StaticVsDynamicPairs = staticVsDynamicBroadphasePairsStream.AsReader(),
                    NumBodies = stepInput.World.NumBodies,
                    UnsortedPairs = dispatchPairs,
                    SortedPairs = pairsToSortByIsland,
                    PairCount = islandPairCount,
                    Cache = islandPairSorts,
                }.Schedule(inputDeps);

                returnHandles.FinalExecutionHandle = sorted;
                returnHandles.FinalDisposeHandle = JobHandle.CombineDependencies(
                    dynamicVsDynamicBroadphasePairsStream.Dispose(sorted), staticVsDynamicBroadphasePairsStream.Dispose(sorted));
                return returnHandles;
            }

            var numWorkers = math.max(1, JobsUtility.JobWorkerCount);

            var unsortedPairs = new NativeList<DispatchPair>(Allocator.TempJob);
            var sortedPairs = new NativeList<DispatchPair>(Allocator.TempJob);
            var jointStreamPrefixSum = new NativeList<int>(numWorkers + 1, Allocator.TempJob);
            var dynamicVsDynamicStreamPrefixSum = new NativeList<int>(Allocator.TempJob);
            var staticVsDynamicStreamPrefixSum = new NativeList<int>(Allocator.TempJob);
            var jointDispatchPairsStream = new NativeStream(numWorkers, Allocator.TempJob);

            using var disposeHandles = new NativeList<JobHandle>(32, Allocator.Temp);

            // Merge broadphase pairs and joint pairs into the unsorted array in parallel:

            JobHandle dynamicVsDynamicStreamPrefixSumHandle = new CreateStreamPrefixSumJob
            {
                Reader = dynamicVsDynamicBroadphasePairsStream.AsReader(),
                StreamPrefixSum = dynamicVsDynamicStreamPrefixSum
            }.Schedule(inputDeps);

            JobHandle staticVsDynamicStreamPrefixSumHandle = new CreateStreamPrefixSumJob
            {
                Reader = staticVsDynamicBroadphasePairsStream.AsReader(),
                StreamPrefixSum = staticVsDynamicStreamPrefixSum
            }.Schedule(inputDeps);

            JobHandle prefixSumHandle = JobHandle.CombineDependencies(dynamicVsDynamicStreamPrefixSumHandle, staticVsDynamicStreamPrefixSumHandle);

            // Allocate unsorted and sorted dispatch pair arrays
            JobHandle allocateDispatchPairs = new AllocateDispatchPairsJob
            {
                Joints = stepInput.World.DynamicsWorld.InternalJointsList.AsDeferredJobArray(),
                DynamicVsDynamicStreamPrefixSum = dynamicVsDynamicStreamPrefixSum,
                StaticVsDynamicStreamPrefixSum = staticVsDynamicStreamPrefixSum,
                UnsortedDispatchPairs = unsortedPairs,
                SortedDispatchPairs = sortedPairs
            }.Schedule(prefixSumHandle);

            // Here we launch three parallel chains of jobs for creation of the unsorted dispatch pair array,
            // organized as [dynamics vs. dynamic collision pairs][static vs. dynamic collision pairs][joint pairs].
            // These are:
            // 1) Creating dynamic vs. dynamic collision dispatch pairs: CreateCollisionDispatchPairsJob
            // 2) Creating static vs. dynamic collision dispatch pairs: CreateCollisionDispatchPairsJob
            // 3) Creating joint dispatch pairs: CreateJointDispatchPairsJob -> CreateStreamPrefixSumJob -> MergeJointDispatchPairsStreamJob
            // Finally, we shrink the dispatch pair array using the ShrinkDispatchPairListJob, allocated above assuming
            // all joints are valid, in case there are any invalid joints.

            JobHandle dynamicVsDynamicPairsHandle;
            JobHandle staticVsDynamicPairsHandle;
            JobHandle jointPairsHandle;
            // 1) Creating dynamic vs. dynamic collision dispatch pairs: CreateCollisionDispatchPairsJob
            dynamicVsDynamicPairsHandle = new CreateCollisionDispatchPairsJob
            {
                DispatchPairs = unsortedPairs.AsDeferredJobArray(),
                PrecedingStreamHistogram = default, // no preceding stream for dynamic vs. dynamic pairs in the dispatch pair array
                OverlapResultsStreamHistogram = dynamicVsDynamicStreamPrefixSum.AsDeferredJobArray(),
                OverlapResultsReader = dynamicVsDynamicBroadphasePairsStream.AsReader(),
            }.ScheduleUnsafe(dynamicVsDynamicBroadphasePairsStream, 1, allocateDispatchPairs);

            // 2) Creating static vs. dynamic collision dispatch pairs: CreateCollisionDispatchPairsJob
            staticVsDynamicPairsHandle = new CreateCollisionDispatchPairsJob
            {
                DispatchPairs = unsortedPairs.AsDeferredJobArray(),
                PrecedingStreamHistogram = dynamicVsDynamicStreamPrefixSum.AsDeferredJobArray(),
                OverlapResultsStreamHistogram = staticVsDynamicStreamPrefixSum.AsDeferredJobArray(),
                OverlapResultsReader = staticVsDynamicBroadphasePairsStream.AsReader(),
            }.ScheduleUnsafe(staticVsDynamicBroadphasePairsStream, 1, allocateDispatchPairs);

            // 3) Creating joint dispatch pairs:
            jointPairsHandle = new CreateJointDispatchPairsJob
            {
                Joints = stepInput.World.DynamicsWorld.InternalJointsList.AsDeferredJobArray(),
                JointDispatchPairWriter = jointDispatchPairsStream.AsWriter(),
                NumWorkers = numWorkers,
            }.ScheduleUnsafe(jointDispatchPairsStream, 1, inputDeps);

            jointPairsHandle = new CreateStreamPrefixSumJob
            {
                Reader = jointDispatchPairsStream.AsReader(),
                StreamPrefixSum = jointStreamPrefixSum
            }.Schedule(jointPairsHandle);

            jointPairsHandle = new MergeJointDispatchPairsStreamJob
            {
                DispatchPairs = unsortedPairs,
                JointDispatchPairReader = jointDispatchPairsStream.AsReader(),
                JointDispatchPairStreamHistogram = jointStreamPrefixSum.AsDeferredJobArray(),
                DynamicVsDynamicStreamHistogram = dynamicVsDynamicStreamPrefixSum.AsDeferredJobArray(),
                StaticVsDynamicStreamHistogram = staticVsDynamicStreamPrefixSum.AsDeferredJobArray()
            }.ScheduleUnsafe(jointDispatchPairsStream, 1, JobHandle.CombineDependencies(jointPairsHandle, allocateDispatchPairs));

            JobHandle dispatchPairsHandle = JobHandle.CombineDependencies(jointPairsHandle,  dynamicVsDynamicPairsHandle, staticVsDynamicPairsHandle);

            dispatchPairsHandle = new ShrinkDispatchPairListJob
            {
                UnsortedDispatchPairs = unsortedPairs,
                SortedDispatchPairs = sortedPairs,
                DynamicVsDynamicStreamPrefixSum = dynamicVsDynamicStreamPrefixSum.AsDeferredJobArray(),
                StaticVsDynamicStreamPrefixSum = staticVsDynamicStreamPrefixSum.AsDeferredJobArray(),
                JointStreamPrefixSum = jointStreamPrefixSum.AsDeferredJobArray()
            }.Schedule(dispatchPairsHandle);

            // dispose dependent resources once done with creation of unsorted pair array
            disposeHandles.Add(jointStreamPrefixSum.Dispose(dispatchPairsHandle));
            disposeHandles.Add(dynamicVsDynamicStreamPrefixSum.Dispose(dispatchPairsHandle));
            disposeHandles.Add(staticVsDynamicStreamPrefixSum.Dispose(dispatchPairsHandle));
            disposeHandles.Add(jointDispatchPairsStream.Dispose(dispatchPairsHandle));

            // Now, dispose the broad phase pairs
            disposeHandles.Add(dynamicVsDynamicBroadphasePairsStream.Dispose(dispatchPairsHandle));
            disposeHandles.Add(staticVsDynamicBroadphasePairsStream.Dispose(dispatchPairsHandle));


            // Sort the pairs so that collision pairs can be disabled by their joints.
            JobHandle sortHandle = ScheduleSortJob(stepInput.World.NumBodies, unsortedPairs, sortedPairs, dispatchPairsHandle);

            sortHandle = new SetDisableCollisionFlagsJob
            {
                DispatchPairs = sortedPairs.AsDeferredJobArray(),
                NumWorkers = numWorkers
            }.Schedule(numWorkers, 1, sortHandle);

            dispatchPairs = unsortedPairs;
            solverSchedulerInfo = new SolverSchedulerInfo(NumIslandPhases);
            pairsToSortByIsland = sortedPairs;
            if (islandPairCount.IsCreated)
            {
                sortHandle = new CountIslandPairsJob { Pairs = sortedPairs, PairCount = islandPairCount }.Schedule(sortHandle);
            }

            returnHandles.FinalExecutionHandle = sortHandle;
            returnHandles.FinalDisposeHandle = JobHandle.CombineDependencies(disposeHandles.AsArray());
            return returnHandles;
        }

        #region Helpers

        /// <summary>
        /// Helper function to schedule jobs to sort an array of dispatch pairs. The first single
        /// threaded job is a single pass Radix sort on the bits associated with DispatchPair.BodyIndexA,
        /// resulting in sub arrays with the same bodyA index. The second parallel job dispatches default
        /// sorts on each sub array.
        /// </summary>
        static JobHandle ScheduleSortJob(
            int numBodies,
            NativeList<DispatchPair> unsortedPairsIn,
            NativeList<DispatchPair> sortedPairsOut,
            JobHandle inputDependencies)
        {
#if SERIAL_RADIX_SORT   // Serial Radix Sort with Parallel Bucket Sort:
            var histogram = new NativeArray<int>(numBodies + 1, Allocator.TempJob);

            // Calculate number digits needed to encode all body indices,
            // which corresponds to the position of the most significant bit.
            int maxBodyIndex = numBodies - 1;

            // Perform single pass of single threaded radix sort.
            inputDependencies = new RadixSortPerBodyAJob
            {
                InputArray = unsortedPairsIn.AsDeferredJobArray(),
                OutputArray = sortedPairsOut.AsDeferredJobArray(),
                MaxIndex = maxBodyIndex,
                BodyIndexHistogram = histogram
            }.Schedule(inputDependencies);

            // Sort sub arrays with default sort.
            int numPerBatch = math.max(1, maxBodyIndex / 32);

            inputDependencies = new SortSubArraysJob
            {
                InOutArray = sortedPairsOut.AsDeferredJobArray(),
                NextElementIndex = histogram
            }.Schedule(histogram.Length, numPerBatch, inputDependencies);

            return inputDependencies;

#else       // Parallel Radix Sort with Parallel Bucket Sort:

            // In the first phase, we keep some workers available for other tasks that are still happening
            // from the previous stages in the pipeline, such as container disposal.
            int numAvailableWorkers = (int)(JobsUtility.JobWorkerCount * 0.2f);
            var numWorkersPhase1 = math.max(1, JobsUtility.JobWorkerCount - numAvailableWorkers);

            var histogramLength = numBodies + 1;
            var sortKeyHistogram = new NativeArray<int>(histogramLength, Allocator.TempJob);

            // Calculate number digits needed to encode all body indices,
            // which corresponds to the position of the most significant bit.
            int maxBodyIndex = numBodies - 1;

            // Perform radix sort in parallel using bodyA index as sort key:

            var sortHandle = new RadixSortHistogramJob
            {
                DispatchPairs = unsortedPairsIn.AsDeferredJobArray(),
                SortKeyHistogram = sortKeyHistogram,
                NumWorkers = numWorkersPhase1
            }.Schedule(numWorkersPhase1, 1, inputDependencies);

            var sortKeyPrefixSum = sortKeyHistogram;
            sortHandle = new RadixSortPrefixSumJob
            {
                SortKeyPrefixSum = sortKeyPrefixSum,
            }.Schedule(sortHandle);

            var numWorkersPhase2 = math.max(1, JobsUtility.JobWorkerCount);
            sortHandle = new RadixSortGatherJob
            {
                InputDispatchPairs = unsortedPairsIn.AsDeferredJobArray(),
                OutputDispatchPairs = sortedPairsOut.AsDeferredJobArray(),
                SortKeyPrefixSum = sortKeyPrefixSum,
                NumWorkers = numWorkersPhase2
            }.Schedule(numWorkersPhase2, 1, sortHandle);

            // Sort sub arrays by bodyB index and remaining dispatch pair values with default sort:

            int numPerBatch = math.max(1, maxBodyIndex / 32);

            // next element index array corresponds to the prefix sum of the last worker
            var nextElementIndex = sortKeyPrefixSum;
            sortHandle = new SortSubArraysJob
            {
                InOutArray = sortedPairsOut.AsDeferredJobArray(),
                NextElementIndex = nextElementIndex
            }.Schedule(nextElementIndex.Length, numPerBatch, sortHandle);

            return sortHandle;
#endif
        }

        #endregion

        #region Jobs

        [BurstCompile]
        struct SetDisableCollisionFlagsJob : IJobParallelFor
        {
            [NativeDisableParallelForRestriction]
            public NativeArray<DispatchPair> DispatchPairs;

            public int NumWorkers;

            public void Execute(int workerIndex)
            {
                if (DispatchPairs.Length == 0)
                {
                    return;
                }

                // We are parallelizing the joints' "Enable Collision" feature using a parallel-for job with a given number
                // of workers (e.g., one per job system worker) and starting each worker at different equally distributed
                // spots in the dispatch pair array. Starting at the beginning of their section, each worker first moves
                // forward in the array until the body pair indices change in order to find the beginning of the section it
                // is responsible for. Then, it starts the collision disabling operation.
                // The worker stops processing when it moves to the point beyond its associated section where again the
                // body pair index changes. This approach will make sure that each worker only processes sections of
                // dispatch pairs which begin and end with a new body pair, which is a core component in the way
                // the "Enable Collision" feature is implemented.

                var indicesPerWorker = (int)math.ceil(DispatchPairs.Length / (float)NumWorkers);
                var startIndex = workerIndex * indicesPerWorker;
                if (startIndex > DispatchPairs.Length - 1)
                {
                    // no work for this worker
                    return;
                }

                var endIndex = math.min(startIndex + indicesPerWorker, DispatchPairs.Length);

                // move to beginning of next valid body pair section in the sub-array for this worker
                if (startIndex != 0)
                {
                    var previousPair = DispatchPairs[startIndex - 1];
                    var startPair = DispatchPairs[startIndex];
                    while (startIndex < endIndex - 1 && startPair.BodyIndexA == previousPair.BodyIndexA && startPair.BodyIndexB == previousPair.BodyIndexB)
                    {
                        startPair = DispatchPairs[++startIndex];
                    }
                }

                // now, start processing the section for this worker
                int lastPairA = -1;
                int lastPairB = -1;

                bool contactsPermitted = true; // Will avoid creating a contact (=non-joint) pair if this is false

                for (int i = startIndex; i < DispatchPairs.Length; ++i)
                {
                    var pair = DispatchPairs[i];
                    int bodyIndexA = pair.BodyIndexA;
                    int bodyIndexB = pair.BodyIndexB;

                    bool indicesChanged = !(lastPairA == bodyIndexA && lastPairB == bodyIndexB);

                    if (i >= endIndex && indicesChanged)
                    {
                        // We reached the end of the section for this worker and an index change occurred.
                        // Stop!
                        break;
                    }

                    bool isJoint = pair.IsJoint;

                    if (indicesChanged || contactsPermitted || isJoint)
                    {
                        // If _any_ Joint between each Pair has Enable Collision set to true, then contacts will be permitted.
                        // Warning: For this to work, in a processed, continuous sequence of dispatch pairs with a given
                        // fixed body pair, joints must always appear before contacts for this implementation to work correctly.
                        // The DispatchPairs in the incoming pre-sorted dispatch pair array (dispatchPairs) is organized and sorted
                        // in the scheduler accordingly before this function is called.
                        bool thisPermitsContacts = isJoint && pair.IsCollisionEnabled;
                        contactsPermitted = (contactsPermitted && !indicesChanged) || thisPermitsContacts;

                        lastPairA = bodyIndexA;
                        lastPairB = bodyIndexB;
                    }
                    else
                    {
                        // Joints don't allow collision for this pair. Disable.
                        pair.DisableCollision();
                        DispatchPairs[i] = pair;
                    }
                }
            }
        }

        [BurstCompile]
        struct AllocateDispatchPairsJob : IJob
        {
            [ReadOnly]
            public NativeArray<Joint> Joints;

            [ReadOnly]
            public NativeList<int> DynamicVsDynamicStreamPrefixSum;

            [ReadOnly]
            public NativeList<int> StaticVsDynamicStreamPrefixSum;

            [WriteOnly]
            public NativeList<DispatchPair> UnsortedDispatchPairs;

            [WriteOnly]
            public NativeList<DispatchPair> SortedDispatchPairs;

            public void Execute()
            {
                var numDynamicVsDynamicOverlapResults = DynamicVsDynamicStreamPrefixSum[^ 1];
                var numStaticVsDynamicOverlapResults = StaticVsDynamicStreamPrefixSum[^ 1];

                var numDispatchPairsConservative =
                    numDynamicVsDynamicOverlapResults +
                    numStaticVsDynamicOverlapResults +
                    Joints.Length;

                if (numDispatchPairsConservative == 0)
                {
                    return;
                }

                UnsortedDispatchPairs.ResizeUninitialized(numDispatchPairsConservative);
                SortedDispatchPairs.ResizeUninitialized(numDispatchPairsConservative);
            }
        }

        [BurstCompile]
        struct ShrinkDispatchPairListJob : IJob
        {
            [WriteOnly]
            public NativeList<DispatchPair> UnsortedDispatchPairs;

            [WriteOnly]
            public NativeList<DispatchPair> SortedDispatchPairs;

            [ReadOnly]
            public NativeArray<int> JointStreamPrefixSum;

            [ReadOnly]
            public NativeArray<int> DynamicVsDynamicStreamPrefixSum;

            [ReadOnly]
            public NativeArray<int> StaticVsDynamicStreamPrefixSum;

            public void Execute()
            {
                var numDispatchPairs = JointStreamPrefixSum[^ 1]
                    + DynamicVsDynamicStreamPrefixSum[^ 1]
                    + StaticVsDynamicStreamPrefixSum[^ 1];

                UnsortedDispatchPairs.ResizeUninitialized(numDispatchPairs);
                SortedDispatchPairs.ResizeUninitialized(numDispatchPairs);
            }
        }


        [BurstCompile]
        struct CreateJointDispatchPairsJob : IJobParallelForDefer
        {
            [ReadOnly]
            public NativeArray<Joint> Joints;

            public NativeStream.Writer JointDispatchPairWriter;

            public int NumWorkers;

            public void Execute(int workerIndex)
            {
                if (Joints.Length == 0)
                {
                    return;
                }

                var indicesPerWorker = (int)math.ceil(Joints.Length / (float)NumWorkers);
                var startIndex = workerIndex * indicesPerWorker;
                if (startIndex > Joints.Length - 1)
                {
                    // no work for this worker
                    return;
                }

                var endIndex = math.min(startIndex + indicesPerWorker, Joints.Length);

                JointDispatchPairWriter.BeginForEachIndex(workerIndex);

                for (int i = startIndex; i < endIndex; i++)
                {
                    var joint = Joints[i];
                    if (joint.BodyPair.IsValid)
                    {
                        // Note: Since only valid joints are commited to the dispatch pair stream the joint indices will
                        // have some gaps, but that is alright, since the joint index is only used as a key for sorting
                        // of the dispatch pairs.
                        JointDispatchPairWriter.Write(DispatchPair.CreateJoint(joint.BodyPair, i, joint.EnableCollision, joint.SolverType));
                    }
                }

                JointDispatchPairWriter.EndForEachIndex();
            }
        }

        [BurstCompile]
        struct CreateStreamPrefixSumJob : IJob
        {
            public NativeStream.Reader Reader;

            public NativeList<int> StreamPrefixSum; // Note: can not be marked [WriteOnly] due to use of NativeList<T>.Length (via end expression "^ 1")

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public static void CreateStreamPrefixSum(NativeStream.Reader reader, NativeList<int> streamPrefixSum)
            {
                streamPrefixSum.ResizeUninitialized(reader.ForEachCount + 1); // +1 for total count entry at the end

                int totalCount = 0;
                for (int i = 0; i < reader.ForEachCount; i++)
                {
                    streamPrefixSum[i] = totalCount;
                    totalCount += reader.BeginForEachIndex(i);
                }

                // store total stream element count at the end of the histogram
                streamPrefixSum[^ 1] = totalCount;
            }

            public void Execute()
            {
                CreateStreamPrefixSum(Reader, StreamPrefixSum);
            }
        }

        [BurstCompile]
        struct MergeJointDispatchPairsStreamJob : IJobParallelForDefer
        {
            [WriteOnly, NativeDisableContainerSafetyRestriction]
            public NativeList<DispatchPair> DispatchPairs;

            public NativeStream.Reader JointDispatchPairReader;

            [ReadOnly]
            public NativeArray<int> JointDispatchPairStreamHistogram;
            [ReadOnly]
            public NativeArray<int> DynamicVsDynamicStreamHistogram;
            [ReadOnly]
            public NativeArray<int> StaticVsDynamicStreamHistogram;

            public void Execute(int index)
            {
                // calculate index for first dispatch pair in this worker, considering that the joint pairs are
                // located after the collision pairs in the dispatch pair array.
                var currentPairIndex = JointDispatchPairStreamHistogram[index]
                    + DynamicVsDynamicStreamHistogram[^ 1] + StaticVsDynamicStreamHistogram[^ 1];

                var jointPairs = JointDispatchPairReader.BeginForEachIndex(index);
                for (int j = 0; j < jointPairs; ++j)
                {
                    var jointPair = JointDispatchPairReader.Read<DispatchPair>();
                    DispatchPairs[currentPairIndex++] = jointPair;
                }
                JointDispatchPairReader.EndForEachIndex();
            }
        }

        [BurstCompile]
        struct CreateCollisionDispatchPairsJob : IJobParallelForDefer
        {
            [WriteOnly, NativeDisableContainerSafetyRestriction]
            public NativeArray<DispatchPair> DispatchPairs;

            [NativeDisableContainerSafetyRestriction] // to allow default value here
            public NativeArray<int> PrecedingStreamHistogram;

            [ReadOnly]
            public NativeArray<int> OverlapResultsStreamHistogram;

            public NativeStream.Reader OverlapResultsReader;

            public void Execute(int index)
            {
                // calculate index for first dispatch pair in this worker, considering that there might be a stream of
                // dispatch pairs in front with the number of pairs provided by the corresponding stream's
                // histogram (see PrecedingStreamHistogram).
                var currentPairIndex = OverlapResultsStreamHistogram[index];
                if (PrecedingStreamHistogram.IsCreated)
                {
                    currentPairIndex += PrecedingStreamHistogram[^ 1];
                }

                var overlapResults = OverlapResultsReader.BeginForEachIndex(index);
                for (int j = 0; j < overlapResults; ++j)
                {
                    var overlapResult = OverlapResultsReader.Read<Broadphase.OverlapResult>();

                    DispatchPairs[currentPairIndex++] = DispatchPair.CreateCollisionPair(overlapResult);
                }
                OverlapResultsReader.EndForEachIndex();
            }
        }

        /// <summary>
        /// Goes through the provided lexicographically sorted dispatch pair array and disables collisions
        /// for all collision pairs for which all joints between the same body pair have the "Enable Collision"
        /// flag set to false.
        /// </summary>
        internal static void SetDisableCollisionFlags(NativeList<DispatchPair> pairs)
        {
            int lastPairA = -1;
            int lastPairB = -1;

            bool contactsPermitted = true; // Will avoid creating a contact (=non-joint) pair if this is false

            for (int i = 0; i < pairs.Length; i++)
            {
                var pair = pairs[i];
                int bodyIndexA = pair.BodyIndexA;
                int bodyIndexB = pair.BodyIndexB;

                bool indicesChanged = !(lastPairA == bodyIndexA && lastPairB == bodyIndexB);
                bool isJoint = pair.IsJoint;

                if (indicesChanged || contactsPermitted || isJoint)
                {
                    // If _any_ Joint between a given pair has Enable Collision set to true, then contacts will be permitted
                    // for this pair.
                    // Warning: For this to work, in a processed, continuous sequence of dispatch pairs with a given fixed
                    // body pair (sub-sequence of pairs with the same two bodies), joints must always appear before
                    // contacts for this implementation to work correctly.
                    // The DispatchPairs in the incoming pre-sorted dispatch pair array (dispatchPairs) is organized and
                    // sorted in the scheduler accordingly before this function is called (lexicographically sorted by
                    // first body A, then body B, then joints and then contacts).
                    bool thisPermitsContacts = isJoint && pair.IsCollisionEnabled;
                    contactsPermitted = (contactsPermitted && !indicesChanged) || thisPermitsContacts;

                    lastPairA = bodyIndexA;
                    lastPairB = bodyIndexB;
                }
                else // collision pair for which contacts are not permitted based on the "IsCollisionEnabled" flag in the preceding joint pairs.
                {
                    // Joints don't allow collision for this collision pair. Disable.
                    pair.DisableCollision();
                    pairs[i] = pair;
                }
            }
        }

        /// <summary>   Sorts an array of dispatch pairs by Body A index. </summary>
        [BurstCompile]
        internal struct RadixSortPerBodyAJob : IJob
        {
            [ReadOnly]
            public NativeArray<DispatchPair> InputArray;
            [NativeDisableParallelForRestriction]
            public NativeArray<DispatchPair> OutputArray;
            [NativeDisableParallelForRestriction]
            public NativeArray<int> BodyIndexHistogram;

            public int MaxIndex;

            public void Execute()
            {
                RadixSortPerBodyA(InputArray, OutputArray, BodyIndexHistogram, MaxIndex);
            }

            // Performs single pass of Radix sort on NativeArray<DispatchPair> based on BodyIndexA.
            public static void RadixSortPerBodyA(NativeArray<DispatchPair> inputArray, NativeArray<DispatchPair> outputArray,
                NativeArray<int> bodyIndexHistogram, int maxBodyIndex)
            {
                SafetyChecks.CheckAreEqualAndThrow(inputArray.Length, outputArray.Length);

                var length = inputArray.Length;

                // Count body indices
                for (int i = 0; i < length; i++)
                {
                    var bodyIndex = inputArray[i].BodyIndexA;
                    bodyIndexHistogram[bodyIndex]++;
                }

                // Calculate start index for each body index (prefix sum calculation)
                int prev = bodyIndexHistogram[0];
                bodyIndexHistogram[0] = 0;
                for (int i = 1; i <= maxBodyIndex; i++)
                {
                    int current = bodyIndexHistogram[i];
                    bodyIndexHistogram[i] = bodyIndexHistogram[i - 1] + prev;
                    prev = current;
                }

                // Copy elements into buckets based on body A index, that is, into contiguous target array sections
                // with the same body A index
                for (int i = 0; i < length; i++)
                {
                    var pair = inputArray[i];
                    var bodyIndex = pair.BodyIndexA;
                    int index = bodyIndexHistogram[bodyIndex]++;
                    outputArray[index] = pair;
                }
            }
        }

        [BurstCompile]
        struct RadixSortHistogramJob : IJobParallelFor
        {
            [ReadOnly, NativeDisableParallelForRestriction]
            public NativeArray<DispatchPair> DispatchPairs;

            [NativeDisableParallelForRestriction]
            public NativeArray<int> SortKeyHistogram;

            public int NumWorkers;

            public void Execute(int workerIndex)
            {
                var indicesPerWorker = (int)math.ceil(DispatchPairs.Length / (float)NumWorkers);
                var startIndex = workerIndex * indicesPerWorker;
                if (startIndex > DispatchPairs.Length - 1)
                {
                    // no work for this worker
                    return;
                }

                var endIndex = math.min(startIndex + indicesPerWorker, DispatchPairs.Length);
                var numIndices = endIndex - startIndex;
                var workerSubArray = DispatchPairs.GetSubArray(startIndex, numIndices);

                unsafe
                {
                    var histogramPtr = (int*)SortKeyHistogram.GetUnsafePtr();

                    // count keys in sub-array
                    for (int i = 0; i < numIndices; i++)
                    {
                        var bodyIndex = workerSubArray[i].BodyIndexA;
                        // @todo: in the future, to reduce the number of atomic operations and false sharing, and the
                        // resultant thread contention we can fill a per-job hash map with the count per body A index
                        // while going over the pair sub-array of this worker, and then do a single atomic add to the
                        // shared histogram per body A index at the end of this job's Execute function.
                        var counter = new UnsafeAtomicCounter32(histogramPtr + bodyIndex);
                        counter.Add(1);
                    }
                }
            }
        }

        [BurstCompile]
        struct RadixSortPrefixSumJob : IJob
        {
            [NativeDisableParallelForRestriction]
            public NativeArray<int> SortKeyPrefixSum;

            public void Execute()
            {
                var last = SortKeyPrefixSum[0];
                SortKeyPrefixSum[0] = 0;
                for (var i = 1; i < SortKeyPrefixSum.Length; ++i)
                {
                    var current = SortKeyPrefixSum[i];
                    SortKeyPrefixSum[i] = SortKeyPrefixSum[i - 1] + last;
                    last = current;
                }
            }
        }

        [BurstCompile]
        struct RadixSortGatherJob : IJobParallelFor
        {
            [ReadOnly, NativeDisableParallelForRestriction]
            public NativeArray<DispatchPair> InputDispatchPairs;

            [WriteOnly, NativeDisableParallelForRestriction]
            public NativeArray<DispatchPair> OutputDispatchPairs;

            [NativeDisableParallelForRestriction]
            public NativeArray<int> SortKeyPrefixSum;

            public int NumWorkers;

            public void Execute(int workerIndex)
            {
                var indicesPerWorker = (int)math.ceil(InputDispatchPairs.Length / (float)NumWorkers);
                var startIndex = workerIndex * indicesPerWorker;
                if (startIndex > InputDispatchPairs.Length - 1)
                {
                    // no work for this worker
                    return;
                }
                var endIndex = math.min(startIndex + indicesPerWorker, InputDispatchPairs.Length);
                var numIndices = endIndex - startIndex;
                var workerSubArray = InputDispatchPairs.GetSubArray(startIndex, numIndices);

                unsafe
                {
                    var prefixSumPtr = (int*)SortKeyPrefixSum.GetUnsafePtr();

                    // Copy elements into buckets based on bodyA index
                    for (int i = 0; i < numIndices; i++)
                    {
                        var pair = workerSubArray[i];
                        var bodyIndex = pair.BodyIndexA;
                        var counter = new UnsafeAtomicCounter32(prefixSumPtr + bodyIndex);
                        var index = counter.Add(1);
                        OutputDispatchPairs[index] = pair;
                    }
                }
            }
        }

        /// <summary>   Sorts slices of an array in parallel. </summary>
        [BurstCompile]
        internal struct SortSubArraysJob : IJobParallelFor
        {
            [NativeDisableParallelForRestriction]
            public NativeArray<DispatchPair> InOutArray;

            // Array used to locate the beginning of the next pair sub-array section with identical body A index.
            // Note: After completion of RadixSortPerBodyAJob, RadixSortPerBodyAJob.BodyIndexHistogram is the prefix sum
            // shifted by one element to the left, that is, RadixSortPerBodyAJob.BodyIndexHistogram[i] = array index of
            // the first pair with body A index == i + 1.
            [ReadOnly, NativeDisableParallelForRestriction, DeallocateOnJobCompletion]
            public NativeArray<int> NextElementIndex;

            public void Execute(int workItemIndex)
            {
                int startIndex = 0;
                if (workItemIndex > 0)
                {
                    startIndex = NextElementIndex[workItemIndex - 1];
                }

                if (startIndex < InOutArray.Length)
                {
                    int length = NextElementIndex[workItemIndex] - startIndex;
                    DefaultSortOfSubArrays(InOutArray, startIndex, length);
                }
            }

            struct BodyIndexASubArrayComparer : IComparer<DispatchPair>
            {
                public int Compare(DispatchPair x, DispatchPair y)
                {
                    return x.BodyIndexASubArraySortKey.CompareTo(y.BodyIndexASubArraySortKey);
                }
            }

            // Sorts sub array by BodyIndexB and remaining data (joint etc.) using default sort
            static void DefaultSortOfSubArrays(NativeArray<DispatchPair> inOutArray, int startIndex, int length)
            {
                // inOutArray[startIndex] to inOutArray[startIndex + length - 1] have the same bodyA index
                // so we can do a simple sorting.
                if (length > 2)
                {
                    var subArray = new NativeSlice<DispatchPair>(inOutArray, startIndex, length);
                    NativeSortExtension.Sort(subArray, new BodyIndexASubArrayComparer());
                }
                else if (length == 2)
                {
                    if (inOutArray[startIndex].BodyIndexASubArraySortKey > inOutArray[startIndex + 1].BodyIndexASubArraySortKey)
                    {
                        var temp = inOutArray[startIndex + 1];
                        inOutArray[startIndex + 1] = inOutArray[startIndex];
                        inOutArray[startIndex] = temp;
                    }
                }
            }
        }

        #endregion
    }
}
