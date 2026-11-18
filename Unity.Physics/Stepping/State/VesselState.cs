using Unity.Collections;
using Unity.Jobs;
using Unity.Physics.Configuration;
using Unity.Physics.Dynamics.Joint.Loads;
using Unity.Physics.Dynamics.Vessels.Detection;
using Unity.Physics.Dynamics.Vessels.Model;
using Unity.Physics.Store.Versions;
using Unity.Physics.Store.World;
using System;

namespace Unity.Physics.Stepping.State
{
    /// <summary>
    /// Owns the compiled vessels and the caches for their edges and closures.
    /// Compiles on first use, joint regrouping, or tombstone expiry. Reuses constraints while the joint settings
    /// and motion frames match.
    /// </summary>
    internal unsafe partial struct VesselState
    {
        // The vessels compiled from the joints, and whether compilation has been completed for the current groups.
        internal VesselBuild vesselBuild;
        internal VesselSet vesselSet;
        private bool compiled;

        // Whether the vessel build holds cached edges and closures, and the joint settings version that they were
        // computed from. See VesselEdgeFill.FillVesselEdgesJob.RefillAll.
        private bool cacheFilled;
        private uint cachedSettings;

        // The jobs that fill the vessels' edges and closures for this step. Only the solve waits for them.
        internal JobHandle vesselsFilled;

        // The step at which the oldest tombstone in the vessels expires, or -1. See VesselCompiler.CompileVesselsJob.
        private int tombstoneExpiry;

        internal void Initialize()
        {
            vesselBuild = VesselBuild.Create();
        }

        internal void InitializeTombstones()
        {
            tombstoneExpiry = -1;
        }

        internal void Dispose()
        {
            vesselBuild.Dispose();
        }

        internal bool CompileAndChoose(WorldData* store, ref JointBuildState joints, ref BodyPartition partition,
            bool vessels, bool regrouped, ref uint stepCount, JobHandle groupsFound)
        {
            // Recompile vessels when topology or forest settings change, support is enabled, or a tombstone
            // expires.
            // Run synchronously because island scheduling needs the vessel count.
            bool tombstoneExpired = tombstoneExpiry >= 0 && (int)stepCount >= tombstoneExpiry;
            bool compiledThisStep = false;
            if (vessels && (!compiled || regrouped || tombstoneExpired))
            {
                compiledThisStep = true;
                groupsFound.Complete();
                new VesselCompiler.CompileVesselsJob
                {
                    World = store,
                    Units = joints.jointUnits.AsArray(),
                    Roles = joints.jointRoles.AsArray(),
                    Groups = joints.jointGroups.AsArray(),
                    Members = joints.jointGroupMembers.AsArray(),
                    Build = vesselBuild,
                    Step = (int)stepCount,
                    TombstoneSteps = VesselSwitches.TombstoneSteps,
                    MinimumParts = VesselSwitches.MinimumParts,
                }.Run();
                compiled = true;
                CompilationVersion++;
                tombstoneExpiry = vesselBuild.TombstoneExpiry(VesselSwitches.TombstoneSteps);
            }
            else if (!vessels && compiled)
            {
                vesselBuild.Clear(0);
                compiled = false;
                CompilationVersion++;
                tombstoneExpiry = -1;
            }

            vesselSet = default;
            if (vessels)
            {
                if (StepCacheSwitches.MemoryChecks)
                {
                    ValidatePartRows(store);
                }
                vesselBuild.ChooseStepVessels(store, partition.builtIndices.AsArray(), partition.dynamicIndices.Length);
                vesselSet = vesselBuild.Set();
            }
            return compiledThisStep;
        }

        internal void FillConstraints(WorldData* store, ref JointBuildState joints, bool compiledThisStep,
            JobHandle groupsFound, JobHandle emitted, JobHandle dependency)
        {
            const int ParallelBatchSize = 64;

            // The solve needs the vessel constraints. Fill them alongside joint and broadphase work.
            vesselsFilled = default;
            if (vesselSet.IsCreated)
            {
                // Reuse vessel constraints unless motion frames or joint settings changed.
                // Check per-joint settings stamps only when the shared version changes.
                uint settings = JointSettings.Version;
                bool resized = vesselBuild.SizeCaches();
                bool refillAll = resized || compiledThisStep || !cacheFilled || StepCacheSwitches.RefillVesselEdges;
                bool settingsChanged = settings != cachedSettings;
                cacheFilled = true;
                cachedSettings = settings;
                JobHandle edgesFilled = new VesselEdgeFill.FillVesselEdgesJob
                {
                    World = store,
                    Units = joints.jointUnits.AsDeferredJobArray(),
                    Groups = joints.jointGroups.AsDeferredJobArray(),
                    Members = joints.jointGroupMembers.AsDeferredJobArray(),
                    UnitVessels = vesselBuild.UnitVessels.AsArray(),
                    Active = vesselBuild.Active.AsArray(),
                    PartStoreBodies = vesselBuild.PartStoreBodies.AsArray(),
                    PartUnits = vesselBuild.PartUnits.AsArray(),
                    PartParents = vesselBuild.PartParents.AsArray(),
                    PartIsBody0 = vesselBuild.PartIsBody0.AsArray(),
                    PartVesselFirst = vesselBuild.PartVesselFirst.AsArray(),
                    Edges = vesselBuild.Edges.AsArray(),
                    RefillAll = refillAll,
                    SettingsChanged = settingsChanged,
                    EdgeCache = vesselBuild.EdgeCache.AsArray(),
                    EdgeCacheFrames = vesselBuild.EdgeCacheFrames.AsArray(),
                    EdgeCacheStamps = vesselBuild.EdgeCacheStamps.AsArray(),
                }.Schedule(vesselBuild.PartUnits.Length, ParallelBatchSize, groupsFound);

                JobHandle closuresFilled = new VesselClosureFill.FillVesselClosuresJob
                {
                    World = store,
                    Units = joints.jointUnits.AsDeferredJobArray(),
                    Groups = joints.jointGroups.AsDeferredJobArray(),
                    Members = joints.jointGroupMembers.AsDeferredJobArray(),
                    UnitVessels = vesselBuild.UnitVessels.AsArray(),
                    Active = vesselBuild.Active.AsArray(),
                    ClosureUnits = vesselBuild.ClosureUnits.AsArray(),
                    ClosurePartA = vesselBuild.ClosurePartA.AsArray(),
                    ClosurePartB = vesselBuild.ClosurePartB.AsArray(),
                    Vessels = vesselBuild.Vessels.AsArray(),
                    PartStoreBodies = vesselBuild.PartStoreBodies.AsArray(),
                    Closures = vesselBuild.Closures.AsArray(),
                    RefillAll = refillAll,
                    SettingsChanged = settingsChanged,
                    ClosureCache = vesselBuild.ClosureCache.AsArray(),
                    ClosureCacheFrames = vesselBuild.ClosureCacheFrames.AsArray(),
                    ClosureCacheStamps = vesselBuild.ClosureCacheStamps.AsArray(),
                }.Schedule(vesselBuild.ClosureUnits.Length, ParallelBatchSize, groupsFound);
                // The joint loads at the end of the step need this list. Waiting for JointTranslation.BuildJointsJob keeps the job
                // from reading a group while JointTranslation.BuildJointsJob writes it.
                JobHandle listed = new VesselLoads.ListOtherJointsJob
                {
                    Units = joints.jointUnits.AsDeferredJobArray(),
                    Groups = joints.jointGroups.AsDeferredJobArray(),
                    Members = joints.jointGroupMembers.AsDeferredJobArray(),
                    UnitVessels = vesselBuild.UnitVessels.AsArray(),
                    Active = vesselBuild.Active.AsArray(),
                    OtherJoints = joints.otherJoints,
                    OtherGroups = joints.otherGroups,
                }.Schedule(emitted);
                JobHandle loadsCleared = vesselBuild.ScheduleClearLoads(dependency);
                vesselsFilled = JobHandle.CombineDependencies(JobHandle.CombineDependencies(edgesFilled, closuresFilled, listed), loadsCleared);
            }
        }

        private void ValidatePartRows(WorldData* store)
        {
            for (int part = 0; part < vesselBuild.PartStoreBodies.Length; part++)
            {
                int row = vesselBuild.PartStoreBodies[part];
                if ((uint)row >= store->bodyCount || store->bodies[row].serial != vesselBuild.PartBodySerials[part]
                    || (IntPtr)store->bodies[row].owner != vesselBuild.PartBodyRecords[part])
                {
                    throw new InvalidOperationException($"Vessel part {part} refers to a different body at store row {row}.");
                }
            }
        }
    }
}
