using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
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

        // The Mono versions of the kernels. In the editor, the plugin calls these until Burst has compiled the kernels.
        // The static fields keep the delegates alive while the plugin holds pointers to them.
        private static readonly QueryKernel.QueryFunction MonoQuery = QueryKernel.Run;
        private static readonly MeshCook.CookFunction MonoCookMesh = MeshCook.Cook;
        private static readonly MeshCook.ValidateFunction MonoValidateMesh = MeshCook.Validate;
        private static readonly NativeStore.CommandFunction MonoBodyCommand = BodyKernel.Run;
        private static readonly NativeStore.CommandFunction MonoShapeCommand = ShapeKernel.Run;
        private static readonly NativeStore.CommandFunction MonoWorldCommand = WorldKernel.Run;
        private static readonly NativeStore.CommandFunction MonoJointCommand = JointKernel.Run;
        private static readonly NativeStore.CommandFunction MonoVehicleCommand = VehicleKernel.Run;

        private static Kernels compiledKernels;
        private static volatile bool kernelsCompiled;
        private static bool compiledKernelsInPlugin;
#if UNITY_EDITOR
        private static Thread compileThread;
#endif
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

        /// <summary>Whether the managed core is attached to the plugin.</summary>
        public static bool IsAttached => attached;

#if UNITY_EDITOR
        [InitializeOnLoadMethod]
        private static void AttachInEditor()
        {
            if (!BackendInfo.IsRetooled)
            {
                return;
            }

            // The compile thread must not still be inside Burst when the domain unloads, so these handlers wait for it.
            AssemblyReloadEvents.beforeAssemblyReload += FinishKernelCompile;
            EditorApplication.quitting += FinishKernelCompile;
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

            AttachToPlugin();
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
            compiledKernelsInPlugin = false;

            int worldCount = NativeStore.RetooledPhysics_WorldCount();
            for (int i = 0; i < worldCount; i++)
            {
                NativeStore.WorldData* world = NativeStore.RetooledPhysics_GetWorld(i);
                lock (Worlds)
                {
                    // Unity keeps the trigger pairs that it holds, so the world's next state needs the touching pairs.
                    if (Worlds.TryGetValue((IntPtr)world, out WorldState state))
                    {
                        state.SaveTouching();
                    }
                }

                ReleaseWorld(world);
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
        /// Waits on the main thread until Burst has compiled the query, mesh and command kernels, then gives the
        /// compiled kernels to the plugin. Returns false if the compile did not finish within <paramref name="seconds"/>.
        /// Tests and benchmarks that check or measure compiled kernels call this first, because in the editor the
        /// kernels run in Mono until their compile finishes. Outside the editor, the kernels are compiled from the start.
        /// </summary>
        public static bool WaitForCompiledKernels(double seconds)
        {
#if UNITY_EDITOR
            Thread thread = compileThread;
            if (thread != null && !thread.Join(TimeSpan.FromSeconds(seconds)))
            {
                return false;
            }

            InstallCompiledKernels();
#endif
            return kernelsCompiled;
        }

        /// <summary>
        /// Gives the plugin the current kernels with the other callbacks. The plugin can be given new kernels while it is
        /// attached. A thread that is running a kernel at that moment finishes with the version it started.
        /// </summary>
        private static void AttachToPlugin()
        {
            Kernels kernels = CurrentKernels();
            NativeStore.ManagedCallbacks callbacks = new NativeStore.ManagedCallbacks
            {
                simulate = Marshal.GetFunctionPointerForDelegate(SimulateDelegate),
                buildQueryWorld = Marshal.GetFunctionPointerForDelegate(BuildQueryWorldDelegate),
                query = kernels.query,
                destroyWorld = Marshal.GetFunctionPointerForDelegate(DestroyWorldDelegate),
                cookMesh = kernels.cookMesh,
                validateMesh = kernels.validateMesh,
                bodyCommand = kernels.bodyCommand,
                shapeCommand = kernels.shapeCommand,
                worldCommand = kernels.worldCommand,
                jointCommand = kernels.jointCommand,
                vehicleCommand = kernels.vehicleCommand,
            };

            NativeStore.RetooledPhysics_Attach(&callbacks);
            compiledKernelsInPlugin = kernels.compiled;
        }

        /// <summary>
        /// Returns the compiled kernels once they exist, and the Mono kernels until then. In the editor, this starts the
        /// compile in the background. In a player, Burst only looks up the code that the build compiled. That lookup is
        /// quick, so this asks for the compiled kernels right away.
        /// </summary>
        private static Kernels CurrentKernels()
        {
            if (!kernelsCompiled)
            {
#if UNITY_EDITOR
                StartKernelCompile();
#else
                compiledKernels = CompileKernels();
                kernelsCompiled = true;
#endif
            }

            return kernelsCompiled ? compiledKernels : MonoKernels();
        }

        /// <summary>
        /// Requests a function pointer from Burst for each kernel. In the editor, the first request after a code change
        /// can block for a minute or more while Burst catches up. A returned pointer can still refer to a stub that runs
        /// the kernel in Mono until Burst finishes its code. If Burst is disabled, the stub keeps running in Mono.
        /// </summary>
        private static Kernels CompileKernels()
        {
            return new Kernels
            {
                query = BurstCompiler.CompileFunctionPointer<QueryKernel.QueryFunction>(QueryKernel.Run).Value,
                cookMesh = BurstCompiler.CompileFunctionPointer<MeshCook.CookFunction>(MeshCook.Cook).Value,
                validateMesh = BurstCompiler.CompileFunctionPointer<MeshCook.ValidateFunction>(MeshCook.Validate).Value,
                bodyCommand = BurstCompiler.CompileFunctionPointer<NativeStore.CommandFunction>(BodyKernel.Run).Value,
                shapeCommand = BurstCompiler.CompileFunctionPointer<NativeStore.CommandFunction>(ShapeKernel.Run).Value,
                worldCommand = BurstCompiler.CompileFunctionPointer<NativeStore.CommandFunction>(WorldKernel.Run).Value,
                jointCommand = BurstCompiler.CompileFunctionPointer<NativeStore.CommandFunction>(JointKernel.Run).Value,
                vehicleCommand = BurstCompiler.CompileFunctionPointer<NativeStore.CommandFunction>(VehicleKernel.Run).Value,
                compiled = true,
            };
        }

        private static Kernels MonoKernels()
        {
            return new Kernels
            {
                query = Marshal.GetFunctionPointerForDelegate(MonoQuery),
                cookMesh = Marshal.GetFunctionPointerForDelegate(MonoCookMesh),
                validateMesh = Marshal.GetFunctionPointerForDelegate(MonoValidateMesh),
                bodyCommand = Marshal.GetFunctionPointerForDelegate(MonoBodyCommand),
                shapeCommand = Marshal.GetFunctionPointerForDelegate(MonoShapeCommand),
                worldCommand = Marshal.GetFunctionPointerForDelegate(MonoWorldCommand),
                jointCommand = Marshal.GetFunctionPointerForDelegate(MonoJointCommand),
                vehicleCommand = Marshal.GetFunctionPointerForDelegate(MonoVehicleCommand),
                compiled = false,
            };
        }

#if UNITY_EDITOR
        /// <summary>
        /// Starts the compile of the kernels on a background thread, once per domain. If the main thread made the first
        /// request to Burst after a code change, the domain reload would wait for it. The plugin calls the Mono kernels
        /// in the meantime, and an editor update gives it the compiled kernels once the thread has finished.
        /// </summary>
        private static void StartKernelCompile()
        {
            if (compileThread != null)
            {
                return;
            }

            compileThread = new Thread(CompileKernelsInBackground)
            {
                IsBackground = true,
                Name = "Retooled kernel compile",
            };
            compileThread.Start();
            EditorApplication.update += InstallCompiledKernels;
        }

        private static void CompileKernelsInBackground()
        {
            System.Diagnostics.Stopwatch watch = System.Diagnostics.Stopwatch.StartNew();
            try
            {
                compiledKernels = CompileKernels();
                kernelsCompiled = true;
                Debug.Log($"{Tag}the kernel compile took {watch.Elapsed.TotalSeconds:F1} s on a background thread");
            }
            catch (Exception e)
            {
                Debug.LogException(e);
            }
        }

        /// <summary>
        /// Gives the compiled kernels to the plugin once the compile thread has finished. If the compile failed, the
        /// plugin keeps the Mono kernels. Runs on the main thread, because attaching records the main thread.
        /// </summary>
        private static void InstallCompiledKernels()
        {
            if (compileThread == null || compileThread.IsAlive)
            {
                return;
            }

            EditorApplication.update -= InstallCompiledKernels;
            if (attached && kernelsCompiled && !compiledKernelsInPlugin)
            {
                AttachToPlugin();
            }
        }

        /// <summary>
        /// Waits for the compile thread before the domain unloads, so that no thread is inside Burst at that point.
        /// </summary>
        private static void FinishKernelCompile()
        {
            compileThread?.Join();
        }
#endif

        /// <summary>The kernels that the plugin calls: all compiled by Burst, or all in Mono.</summary>
        private struct Kernels
        {
            public IntPtr query;
            public IntPtr cookMesh;
            public IntPtr validateMesh;
            public IntPtr bodyCommand;
            public IntPtr shapeCommand;
            public IntPtr worldCommand;
            public IntPtr jointCommand;
            public IntPtr vehicleCommand;
            public bool compiled;
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
