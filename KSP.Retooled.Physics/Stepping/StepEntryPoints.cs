using KSP.Retooled.Physics.Entities;
using KSP.Retooled.Physics.Host;
using KSP.Retooled.Physics.Reports;
using KSP.Retooled.Physics.Interop;
using Unity.Burst;
using Unity.Jobs;
using Unity.Physics;
using Unity.Physics.Collision.World.Trees;
using Unity.Physics.Stepping;
using Unity.Profiling;

namespace KSP.Retooled.Physics.Stepping
{
    /// <summary>
    /// Burst entry points for StoreStepper. Current selects the scene being stepped by the shared physics system
    /// group.
    /// </summary>
    [BurstCompile]
    internal static unsafe class StepEntryPoints
    {
        private static readonly ProfilerMarker StepWaitForSolveMarker = new ProfilerMarker(ProfilerCategory.Physics, "Physics.Step.WaitForSolve");

        private readonly struct CurrentKey { }

        private static readonly SharedStatic<nint> CurrentState = SharedStatic<nint>.GetOrCreate<CurrentKey>();

        internal static StoreStepper* Current
        {
            get => (StoreStepper*)CurrentState.Data;
            set => CurrentState.Data = (nint)value;
        }

        [BurstCompile]
        internal static void ApplyForces(float deltaTime)
        {
            NoteRunInMono();
            StoreStepper* state = Current;
            state->BeginForces();
            BodyEntities.PrepareForJobs(state->Store);
            state->ApplyForces(deltaTime);
        }

        [BurstCompile]
        internal static void BuildWorld(float deltaTime, JobHandle* dependency)
        {
            *dependency = Current->BuildWorld(deltaTime, *dependency);
        }

        [BurstCompile]
        internal static void SortIslands(SimulationSingleton* singleton, SimulationStepInput* input, JobHandle* dependency)
        {
            *dependency = Current->SortIslands(*singleton, ref *input, *dependency);
        }

        [BurstCompile]
        internal static void ReadContacts(SimulationSingleton* singleton, float deltaTime, JobHandle* dependency)
        {
            *dependency = Current->ReadContacts(*singleton, deltaTime, *dependency);
        }

        [BurstCompile]
        internal static void PrepareSolve(SimulationSingleton* singleton, SimulationStepInput* input, JobHandle* dependency)
        {
            *dependency = Current->PrepareSolve(*singleton, ref *input, *dependency);
        }

        [BurstCompile]
        internal static void FinishStep(Simulation* simulation, float deltaTime)
        {
            Current->preparedSimulation = null;
            FinishStep(Current, *simulation, deltaTime, default);
        }

        /// <summary>
        /// Finish a prepared step, possibly while solving is still running. Return one, or zero if no step is
        /// prepared.
        /// </summary>
        [BurstCompile]
        internal static int FinishPreparedStep(float deltaTime, JobHandle* systems)
        {
            Simulation* simulation = Current->preparedSimulation;
            if (simulation == null)
            {
                return 0;
            }

            Current->preparedSimulation = null;
            FinishStep(Current, *simulation, deltaTime, *systems);
            return 1;
        }

        [BurstCompile]
        internal static void BuildForQueries(StoreStepper* state)
        {
            NoteRunInMono();
            if (state->BeginQueryBuild())
            {
                BodyEntities.PrepareForJobs(state->Store);
                state->BuildForQueries();
            }
        }

        private static void FinishStep(StoreStepper* state, Simulation simulation, float deltaTime, JobHandle systems)
        {
            JobHandle pending = systems;
            bool finished = false;
            try
            {
                JobHandle contactsReady = state->ScheduleBeforeContacts(ref simulation, deltaTime, ref pending,
                    out JobHandle snapshot, out JobHandle sleep);
                JobHandle published = ContactReport.Schedule(state, contactsReady);
                pending = JobHandle.CombineDependencies(pending, published);
                JobHandle fallen = state->ScheduleSleep(published, snapshot, sleep);
                pending = JobHandle.CombineDependencies(pending, fallen);

                using (StepWaitForSolveMarker.Auto())
                {
                    pending.Complete();
                }

                finished = true;
            }
            finally
            {
                if (!finished)
                {
                    pending.Complete();
                }
            }

            state->PublishQueries();
        }

        /// <summary>
        /// Help remaining dynamic refit work from a query on any thread; see SharedDynamicTreeRefit.Help.
        /// </summary>
        [BurstCompile(DisableSafetyChecks = true)]
        [AOT.MonoPInvokeCallback(typeof(ManagedCallbacks.RefitFunction))]
        internal static void HelpRefit(void* refit)
        {
            bool ranInMono = false;
            HelpRefitInMono(refit, ref ranInMono);
            if (!ranInMono)
            {
                ((SharedDynamicTreeRefit*)refit)->Help();
            }
        }

        /// <summary>Burst omits this method, so it runs only in Mono, which must not let an exception reach the plugin.</summary>
        [BurstDiscard]
        private static void HelpRefitInMono(void* refit, ref bool ranInMono)
        {
            ranInMono = true;
            try
            {
                ((SharedDynamicTreeRefit*)refit)->Help();
            }
            catch (System.Exception e)
            {
                UnityEngine.Debug.LogException(e);
            }
        }

        /// <summary>Burst omits this method, so it runs only in Mono.</summary>
        [BurstDiscard]
        private static void NoteRunInMono()
        {
            BackendCounters.NoteManagedStep();
        }
    }
}
