using KSP.Retooled.Physics.Commands;
using KSP.Retooled.Physics.Entities;
using KSP.Retooled.Physics.Host;
using KSP.Retooled.Physics.Queries;
using KSP.Retooled.Physics.Stepping;
using KSP.Retooled.Physics.Interop;
using System;
using Unity.Profiling;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Threading;
using AOT;
using Unity.Burst;
using Unity.Entities;
using Unity.Physics.Base.Threading;
using Unity.Physics.Systems;
using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
using Unity.Physics.Store.World;
#endif
using Unity.Physics.Collision.Filter.Groups;
using Unity.Physics.Dynamics.Bodies;

namespace KSP.Retooled.Physics
{
    /// <summary>
    /// Connect the managed core to the native plugin. Detaching frees managed state; the native store survives
    /// domain reloads.
    /// While detached, simulation and queries are disabled, and commands that require the core return zero.
    /// Callbacks must catch exceptions before returning across the native boundary.
    /// </summary>
    public static unsafe partial class BackendHost
    {
        private const string Tag = "[RetooledPhysics] ";

        private static readonly ManagedCallbacks.SimulateCallback SimulateDelegate = OnSimulate;
        private static readonly ManagedCallbacks.BuildQueryWorldCallback BuildQueryWorldDelegate = OnBuildQueryWorld;
        private static readonly ManagedCallbacks.DestroyWorldCallback DestroyWorldDelegate = OnDestroyWorld;

        private static readonly Dictionary<IntPtr, WorldState> Worlds = new Dictionary<IntPtr, WorldState>();

        // Keep Mono delegates alive while the plugin uses their function pointers during Burst compilation.
        private static readonly QueryEntry.QueryFunction MonoQuery = QueryEntry.Run;
        private static readonly MeshCook.CookFunction MonoCookMesh = MeshCook.Cook;
        private static readonly MeshCook.ValidateFunction MonoValidateMesh = MeshCook.Validate;
        private static readonly ManagedCallbacks.CommandFunction MonoBodyCommand = BodyCommands.Run;
        private static readonly ManagedCallbacks.CommandFunction MonoShapeCommand = ShapeCommands.Run;
        private static readonly ManagedCallbacks.CommandFunction MonoWorldCommand = WorldCommands.Run;
        private static readonly ManagedCallbacks.CommandFunction MonoJointCommand = JointCommands.Run;
        private static readonly ManagedCallbacks.CommandFunction MonoVehicleCommand = VehicleCommands.Run;
        private static readonly ManagedCallbacks.RefitFunction MonoHelpRefit = StepEntryPoints.HelpRefit;

        private static Kernels compiledKernels;
        private static volatile bool kernelsCompiled;
        private static bool compiledKernelsInPlugin;
#if UNITY_EDITOR
        private static Thread compileThread;
#endif
        private static bool attached;
        private static World entityWorld;

        /// <summary>
        /// Number of bodies processed by the Mono force job while Burst compilation is pending.
        /// </summary>
        public static long ManagedJobRuns => ExternalForces.ManagedJobRuns;

        /// <summary>
        /// Number of queries executed in Mono while Burst compilation is pending.
        /// </summary>
        public static long ManagedQueries => BackendCounters.ManagedQueries;

        /// <summary>
        /// Number of meshes cooked in Mono while Burst compilation is pending.
        /// </summary>
        public static long ManagedCooks => BackendCounters.ManagedCooks;

        /// <summary>
        /// Number of body, shape, world, joint and vehicle commands executed in Mono while Burst compilation is
        /// pending.
        /// </summary>
        public static long ManagedCommands => BackendCounters.ManagedCommands;

        /// <summary>
        /// Number of steps and query world builds executed in Mono while Burst compilation is pending.
        /// </summary>
        public static long ManagedSteps => BackendCounters.ManagedSteps;

        /// <summary>
        /// Managed thread ID of the most recent step, or zero before the first step.
        /// </summary>
        public static int LastStepThreadId { get; private set; }

        /// <summary>
        /// Number of steps in this domain. Detaching and reattaching does not reset the count.
        /// </summary>
        public static long StepCount { get; private set; }

        /// <summary>
        /// ID of the main thread that attached the core.
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

            string mismatch = LayoutChecks.FirstSizeMismatch();
            if (mismatch != null)
            {
                Debug.LogError($"{Tag}not attaching: {mismatch}.");
                return;
            }

            AttachToPlugin();
            AddressWaitInstall.InstallAddressWait();
            IgnoreTable.Reslot(NativeLibrary.RetooledPhysics_Globals());
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

            // Finish refits before clearing helpRefit. Otherwise a worker holding a query lock can wait for a
            // refit
            // that it can no longer help, while detach waits for that lock.
            lock (Worlds)
            {
                foreach (WorldState state in Worlds.Values)
                {
                    state.FinishPendingRefit();
                }
            }

            if (BodyEntities.Bound)
            {
                BodyEntities.Unbind();
            }

            AddressWait.Install(IntPtr.Zero, IntPtr.Zero);
            NativeLibrary.RetooledPhysics_Detach();
            attached = false;
            compiledKernelsInPlugin = false;

            int worldCount = NativeLibrary.RetooledPhysics_WorldCount();
            for (int i = 0; i < worldCount; i++)
            {
                WorldData* world = NativeLibrary.RetooledPhysics_GetWorld(i);
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
        /// Move body components to entities in world, unless an Entities world is already bound.
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
        /// Move body components out of the bound world before it is disposed.
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

        /// <summary>
        /// Wait on the main thread for Burst kernel compilation and install the compiled kernels.
        /// Return false if compilation does not finish within seconds. In players, the kernels are already
        /// compiled.
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
        /// Install the current kernels and callbacks. Calls already in progress finish with the previous
        /// kernels.
        /// </summary>
        private static void AttachToPlugin()
        {
            Kernels kernels = CurrentKernels();
            ManagedCallbacks callbacks = new ManagedCallbacks
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
                helpRefit = kernels.helpRefit,
            };

            NativeLibrary.RetooledPhysics_Attach(&callbacks);
            compiledKernelsInPlugin = kernels.compiled;
        }

        /// <summary>
        /// Return the compiled kernels if available, otherwise the Mono kernels.
        /// Start compilation in the background in the editor; look up precompiled code in players.
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
        /// Request Burst function pointers for all kernels. The pointers may use Mono stubs until compilation
        /// finishes.
        /// </summary>
        private static Kernels CompileKernels()
        {
            return new Kernels
            {
                query = BurstCompiler.CompileFunctionPointer<QueryEntry.QueryFunction>(QueryEntry.Run).Value,
                cookMesh = BurstCompiler.CompileFunctionPointer<MeshCook.CookFunction>(MeshCook.Cook).Value,
                validateMesh = BurstCompiler.CompileFunctionPointer<MeshCook.ValidateFunction>(MeshCook.Validate).Value,
                bodyCommand = BurstCompiler.CompileFunctionPointer<ManagedCallbacks.CommandFunction>(BodyCommands.Run).Value,
                shapeCommand = BurstCompiler.CompileFunctionPointer<ManagedCallbacks.CommandFunction>(ShapeCommands.Run).Value,
                worldCommand = BurstCompiler.CompileFunctionPointer<ManagedCallbacks.CommandFunction>(WorldCommands.Run).Value,
                jointCommand = BurstCompiler.CompileFunctionPointer<ManagedCallbacks.CommandFunction>(JointCommands.Run).Value,
                vehicleCommand = BurstCompiler.CompileFunctionPointer<ManagedCallbacks.CommandFunction>(VehicleCommands.Run).Value,
                helpRefit = BurstCompiler.CompileFunctionPointer<ManagedCallbacks.RefitFunction>(StepEntryPoints.HelpRefit).Value,
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
                helpRefit = Marshal.GetFunctionPointerForDelegate(MonoHelpRefit),
                compiled = false,
            };
        }

#if UNITY_EDITOR
        /// <summary>
        /// Start kernel compilation once per domain on a background thread to avoid blocking the main thread.
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
        /// Install compiled kernels on the main thread when compilation finishes. Keep the Mono kernels if
        /// compilation fails.
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
        /// Wait for kernel compilation before unloading the domain.
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
            public IntPtr helpRefit;
            public bool compiled;
        }

        /// <summary>
        /// Publish main-thread changes to the query worlds before scheduling queries on workers.
        /// Call on the main thread after Physics.SyncTransforms. Steps and main-thread queries already perform
        /// this update.
        /// </summary>
        public static void SyncQueryWorlds()
        {
            if (attached)
            {
                NativeLibrary.RetooledPhysics_SyncQueryWorlds();
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

        private static WorldState StateOf(WorldData* world)
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

        private static void ReleaseWorld(WorldData* world)
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

        [MonoPInvokeCallback(typeof(ManagedCallbacks.SimulateCallback))]
        private static void OnSimulate(WorldData* world, float deltaTime)
        {
            using ProfilerMarker.AutoScope span = StepMarker.Auto();

            WorldState state = null;
            try
            {
                NoteStep();
                state = StateOf(world);
                state.Step(entityWorld?.GetExistingSystemManaged<PhysicsSystemGroup>(), deltaTime);
            }
            catch (Exception e)
            {
                world->queryWorld = null;
                world->queryDirty = 1;
                state?.AfterFailedStep();
                Debug.LogException(e);
            }
        }

        [MonoPInvokeCallback(typeof(ManagedCallbacks.BuildQueryWorldCallback))]
        private static int OnBuildQueryWorld(WorldData* world)
        {
            using ProfilerMarker.AutoScope span = BuildQueryWorldMarker.Auto();

            try
            {
                return StateOf(world).BuildForQueries();
            }
            catch (Exception e)
            {
                world->queryWorld = null;
                Debug.LogException(e);
                return StoreConstants.QueryWorldFailed;
            }
        }

        [MonoPInvokeCallback(typeof(ManagedCallbacks.DestroyWorldCallback))]
        private static void OnDestroyWorld(WorldData* world)
        {
            using ProfilerMarker.AutoScope span = DestroyWorldMarker.Auto();

            try
            {
                ReleaseWorld(world);
                WorldCommands.FreeBodies(world, NativeLibrary.RetooledPhysics_Globals());
            }
            catch (Exception e)
            {
                Debug.LogException(e);
            }
        }
    }
}
