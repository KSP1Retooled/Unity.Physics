using System;
using System.Collections.Generic;
using System.Threading;
using AOT;
using Unity.Burst;
using Unity.Entities;
using Unity.Physics.Systems;
using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
#endif

namespace KSP.Retooled.Physics
{
    /// <summary>
    /// Connects the managed core to the plugin. The plugin's store survives a domain reload, and all managed state can
    /// be rebuilt from it. A reload therefore detaches the core to free managed state, then attaches it again. While
    /// the core is detached, the plugin performs no simulation and reports no hits. Commands that require the core do
    /// nothing and return zero. Detaching also moves body components back from entities, because the entities do not
    /// survive the domain reload.
    /// </summary>
    /// <remarks>
    /// The player runs on Mono, so callbacks use delegates that are kept alive in static fields and marked for reverse
    /// P/Invoke. Exceptions must not cross the native boundary, so each entry point catches them here.
    /// </remarks>
    public static unsafe class BackendHost
    {
        private const string Tag = "[RetooledPhysics] ";

        private static readonly NativeStore.SimulateCallback SimulateDelegate = OnSimulate;
        private static readonly NativeStore.BuildQueryWorldCallback BuildQueryWorldDelegate = OnBuildQueryWorld;
        private static readonly NativeStore.DestroyWorldCallback DestroyWorldDelegate = OnDestroyWorld;

        private static readonly Dictionary<IntPtr, WorldState> Worlds = new Dictionary<IntPtr, WorldState>();

        private static IntPtr queryFunction;
        private static IntPtr cookMeshFunction;
        private static IntPtr validateMeshFunction;
        private static IntPtr bodyCommandFunction;
        private static IntPtr shapeCommandFunction;
        private static IntPtr worldCommandFunction;
        private static IntPtr jointCommandFunction;
        private static IntPtr vehicleCommandFunction;
        private static bool attached;
        private static World entityWorld;
        private static long managedJobRuns;
        private static long managedQueries;
        private static long managedCooks;
        private static long managedCommands;
        private static long managedSteps;

        /// <summary>
        /// Counts bodies processed by the Mono version of the force job. The job runs in Mono until asynchronous Burst
        /// compilation finishes.
        /// </summary>
        public static long ManagedJobRuns => Interlocked.Read(ref managedJobRuns);

        /// <summary>
        /// Counts queries that ran in Mono while the query functions were waiting for asynchronous Burst compilation.
        /// </summary>
        public static long ManagedQueries => Interlocked.Read(ref managedQueries);

        /// <summary>
        /// Counts mesh cooks that ran in Mono while the cook function was waiting for asynchronous Burst compilation.
        /// </summary>
        public static long ManagedCooks => Interlocked.Read(ref managedCooks);

        /// <summary>
        /// Counts body, shape, world, joint and vehicle commands that ran in Mono while their kernels were waiting for
        /// asynchronous Burst compilation.
        /// </summary>
        public static long ManagedCommands => Interlocked.Read(ref managedCommands);

        /// <summary>
        /// Counts steps and query world builds that ran in Mono while the step functions were waiting for asynchronous
        /// Burst compilation.
        /// </summary>
        public static long ManagedSteps => Interlocked.Read(ref managedSteps);

        /// <summary>
        /// Records the managed ID of the thread that most recently ran a step. The value is zero before the first step.
        /// </summary>
        public static int LastStepThreadId { get; private set; }

        /// <summary>
        /// Counts steps that have run in this domain. Detaching and attaching again within the same domain does not
        /// reset the count.
        /// </summary>
        public static long StepCount { get; private set; }

        /// <summary>
        /// Records the thread that attached the core. All paths that attach the core run on the main thread.
        /// </summary>
        public static int MainThreadId { get; private set; }

#if UNITY_EDITOR
        [InitializeOnLoadMethod]
        private static void AttachInEditor()
        {
            if (!BackendInfo.IsRetooled)
            {
                return;
            }

            AssemblyReloadEvents.beforeAssemblyReload += Detach;
            EditorApplication.quitting += Detach;
            Attach();
        }
#endif

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.SubsystemRegistration)]
        private static void AttachInPlayer()
        {
            if (BackendInfo.IsRetooled)
            {
#if !UNITY_EDITOR
                // Application.quitting also fires when leaving play mode in the editor. Do not detach then, because
                // edit mode still needs the core.
                Application.quitting += Detach;
#endif
                Attach();
            }
        }

        public static void Attach()
        {
            if (attached)
            {
                return;
            }

            string mismatch = NativeStore.FirstSizeMismatch();
            if (mismatch != null)
            {
                Debug.LogError($"{Tag}not attaching: {mismatch}.");
                return;
            }

            NativeStore.ManagedCallbacks callbacks = new NativeStore.ManagedCallbacks
            {
                simulate = System.Runtime.InteropServices.Marshal.GetFunctionPointerForDelegate(SimulateDelegate),
                buildQueryWorld = System.Runtime.InteropServices.Marshal.GetFunctionPointerForDelegate(BuildQueryWorldDelegate),
                query = QueryFunction(),
                destroyWorld = System.Runtime.InteropServices.Marshal.GetFunctionPointerForDelegate(DestroyWorldDelegate),
                cookMesh = CookMeshFunction(),
                validateMesh = ValidateMeshFunction(),
                bodyCommand = BodyCommandFunction(),
                shapeCommand = ShapeCommandFunction(),
                worldCommand = WorldCommandFunction(),
                jointCommand = JointCommandFunction(),
                vehicleCommand = VehicleCommandFunction(),
            };

            NativeStore.RetooledPhysics_Attach(&callbacks);
            NativeStore.InstallAddressWait();
            IgnoreTable.Reslot(NativeStore.RetooledPhysics_Globals());
            attached = true;
            MainThreadId = Thread.CurrentThread.ManagedThreadId;

            if (entityWorld != null)
            {
                BodyEntities.Bind(entityWorld.EntityManager);
            }
        }

        public static void Detach()
        {
            if (!attached)
            {
                return;
            }

            if (BodyEntities.Bound)
            {
                BodyEntities.Unbind();
            }

            Unity.Physics.AddressWait.Install(IntPtr.Zero, IntPtr.Zero);
            NativeStore.RetooledPhysics_Detach();
            attached = false;

            int worldCount = NativeStore.RetooledPhysics_WorldCount();
            for (int i = 0; i < worldCount; i++)
            {
                ReleaseWorld(NativeStore.RetooledPhysics_GetWorld(i));
            }
        }

        /// <summary>
        /// Moves body components to entities in <paramref name="world"/>, unless an Entities world is already bound.
        /// </summary>
        internal static void BindWorld(World world)
        {
            if (entityWorld != null || !BackendInfo.IsRetooled)
            {
                return;
            }

            entityWorld = world;
            if (attached)
            {
                BodyEntities.Bind(world.EntityManager);
            }
        }

        /// <summary>
        /// Moves body components back from <paramref name="world"/> before it is disposed, if that world is bound.
        /// </summary>
        internal static void UnbindWorld(World world)
        {
            if (entityWorld != world)
            {
                return;
            }

            entityWorld = null;
            if (BodyEntities.Bound)
            {
                BodyEntities.Unbind();
            }
        }

        internal static void NoteManagedJobRun()
        {
            Interlocked.Increment(ref managedJobRuns);
        }

        internal static void NoteManagedCook()
        {
            Interlocked.Increment(ref managedCooks);
        }

        internal static void NoteManagedCommand()
        {
            Interlocked.Increment(ref managedCommands);
        }

        internal static void NoteManagedQuery()
        {
            Interlocked.Increment(ref managedQueries);
        }

        internal static void NoteManagedStep()
        {
            Interlocked.Increment(ref managedSteps);
        }

        /// <summary>
        /// Requests the function pointer from Burst once per domain. Burst compiles the method in the background. The
        /// returned pointer initially refers to a stub that runs the method in Mono until compilation finishes. If
        /// Burst is disabled, the stub continues running in Mono.
        /// </summary>
        private static IntPtr QueryFunction()
        {
            if (queryFunction == IntPtr.Zero)
            {
                queryFunction = BurstCompiler.CompileFunctionPointer<QueryKernel.QueryFunction>(QueryKernel.Run).Value;
            }

            return queryFunction;
        }

        private static IntPtr CookMeshFunction()
        {
            if (cookMeshFunction == IntPtr.Zero)
            {
                cookMeshFunction = BurstCompiler.CompileFunctionPointer<MeshCook.CookFunction>(MeshCook.Cook).Value;
            }

            return cookMeshFunction;
        }

        private static IntPtr ValidateMeshFunction()
        {
            if (validateMeshFunction == IntPtr.Zero)
            {
                validateMeshFunction = BurstCompiler.CompileFunctionPointer<MeshCook.ValidateFunction>(MeshCook.Validate).Value;
            }

            return validateMeshFunction;
        }

        private static IntPtr BodyCommandFunction()
        {
            if (bodyCommandFunction == IntPtr.Zero)
            {
                bodyCommandFunction = BurstCompiler.CompileFunctionPointer<NativeStore.CommandFunction>(BodyKernel.Run).Value;
            }

            return bodyCommandFunction;
        }

        private static IntPtr ShapeCommandFunction()
        {
            if (shapeCommandFunction == IntPtr.Zero)
            {
                shapeCommandFunction = BurstCompiler.CompileFunctionPointer<NativeStore.CommandFunction>(ShapeKernel.Run).Value;
            }

            return shapeCommandFunction;
        }

        private static IntPtr WorldCommandFunction()
        {
            if (worldCommandFunction == IntPtr.Zero)
            {
                worldCommandFunction = BurstCompiler.CompileFunctionPointer<NativeStore.CommandFunction>(WorldKernel.Run).Value;
            }

            return worldCommandFunction;
        }

        private static IntPtr JointCommandFunction()
        {
            if (jointCommandFunction == IntPtr.Zero)
            {
                jointCommandFunction = BurstCompiler.CompileFunctionPointer<NativeStore.CommandFunction>(JointKernel.Run).Value;
            }

            return jointCommandFunction;
        }

        private static IntPtr VehicleCommandFunction()
        {
            if (vehicleCommandFunction == IntPtr.Zero)
            {
                vehicleCommandFunction = BurstCompiler.CompileFunctionPointer<NativeStore.CommandFunction>(VehicleKernel.Run).Value;
            }

            return vehicleCommandFunction;
        }

        /// <summary>
        /// Updates the query worlds to include all changes made on the main thread. Workers cannot rebuild a query
        /// world, so they read the last published version. Main-thread code calls this after moving objects and before
        /// scheduling query jobs, following any Physics.SyncTransforms call. Simulation steps and main-thread queries
        /// already perform this update. Call this method on the main thread only.
        /// </summary>
        public static void SyncQueryWorlds()
        {
            if (attached)
            {
                NativeStore.RetooledPhysics_SyncQueryWorlds();
            }
        }

        private static void NoteStep()
        {
            LastStepThreadId = Thread.CurrentThread.ManagedThreadId;
            if (StepCount++ == 0)
            {
                Debug.Log($"{Tag}first step on thread {LastStepThreadId}, main thread {MainThreadId}");
            }
        }

        private static WorldState StateOf(NativeStore.WorldData* world)
        {
            // The caller holds the lock for this world. Queries on different worlds can still run concurrently.
            lock (Worlds)
            {
                IntPtr key = (IntPtr)world;
                if (!Worlds.TryGetValue(key, out WorldState state))
                {
                    state = new WorldState(world);
                    Worlds.Add(key, state);
                }

                return state;
            }
        }

        private static void ReleaseWorld(NativeStore.WorldData* world)
        {
            lock (Worlds)
            {
                IntPtr key = (IntPtr)world;
                if (Worlds.TryGetValue(key, out WorldState state))
                {
                    state.Dispose();
                    Worlds.Remove(key);
                }
            }
        }

        [MonoPInvokeCallback(typeof(NativeStore.SimulateCallback))]
        private static void OnSimulate(NativeStore.WorldData* world, float deltaTime)
        {
            using Unity.Profiling.ProfilerMarker.AutoScope span = BackendMarkers.Step.Auto();

            try
            {
                NoteStep();
                StateOf(world).Step(entityWorld?.GetExistingSystemManaged<PhysicsSystemGroup>(), deltaTime);
            }
            catch (Exception e)
            {
                world->queryWorld = null;
                world->queryDirty = 1;
                Debug.LogException(e);
            }
        }

        [MonoPInvokeCallback(typeof(NativeStore.BuildQueryWorldCallback))]
        private static int OnBuildQueryWorld(NativeStore.WorldData* world)
        {
            using Unity.Profiling.ProfilerMarker.AutoScope span = BackendMarkers.BuildQueryWorld.Auto();

            try
            {
                return StateOf(world).BuildForQueries();
            }
            catch (Exception e)
            {
                world->queryWorld = null;
                Debug.LogException(e);
                return NativeStore.QueryWorldFailed;
            }
        }

        [MonoPInvokeCallback(typeof(NativeStore.DestroyWorldCallback))]
        private static void OnDestroyWorld(NativeStore.WorldData* world)
        {
            using Unity.Profiling.ProfilerMarker.AutoScope span = BackendMarkers.DestroyWorld.Auto();

            try
            {
                ReleaseWorld(world);
                WorldKernel.FreeBodies(world, NativeStore.RetooledPhysics_Globals());
            }
            catch (Exception e)
            {
                Debug.LogException(e);
            }
        }
    }
}
