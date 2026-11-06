using System;
using System.Collections.Generic;
using System.Threading;
using AOT;
using Unity.Burst;
using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
#endif

namespace KSP.Retooled.PhysicsBackend
{
    /// <summary>
    /// Connects the managed core to the plugin. The plugin's store outlives a domain reload and everything managed holds
    /// can be rebuilt from it, so a reload is a detach, which frees managed state, followed by an attach. Until then the
    /// plugin answers with no simulation and no hits, and every command that needs the core is a no-op answering zero.
    /// </summary>
    /// <remarks>
    /// The player runs on Mono, so the callbacks are delegates kept alive in statics and marked for reverse P/Invoke.
    /// Nothing may throw across the native boundary, which is why each entry point ends the exception here.
    /// </remarks>
    public static unsafe class BackendHost
    {
        private const string Tag = "[RetooledPhysics] ";

        private static readonly NativeStore.SimulateCallback SimulateDelegate = OnSimulate;
        private static readonly NativeStore.BuildQueryWorldCallback BuildQueryWorldDelegate = OnBuildQueryWorld;
        private static readonly NativeStore.DestroyWorldCallback DestroyWorldDelegate = OnDestroyWorld;

        private static readonly Dictionary<IntPtr, WorldState> Worlds = new Dictionary<IntPtr, WorldState>();

        private static IntPtr rayCastFunction;
        private static IntPtr cookMeshFunction;
        private static IntPtr validateMeshFunction;
        private static IntPtr bodyCommandFunction;
        private static IntPtr shapeCommandFunction;
        private static IntPtr worldCommandFunction;
        private static IntPtr jointCommandFunction;
        private static bool attached;
        private static long managedJobRuns;
        private static long managedQueries;
        private static long managedCooks;
        private static long managedCommands;

        /// <summary>
        /// Bodies a force job has handled in its Mono form, which it runs until the async Burst compile has delivered it.
        /// </summary>
        public static long ManagedJobRuns => Interlocked.Read(ref managedJobRuns);

        /// <summary>Queries that ran in Mono, which they do until the async Burst compile has delivered the query functions.</summary>
        public static long ManagedQueries => Interlocked.Read(ref managedQueries);

        /// <summary>Mesh cooks that ran in Mono, which they do until the async Burst compile has delivered the cook function.</summary>
        public static long ManagedCooks => Interlocked.Read(ref managedCooks);

        /// <summary>Body, shape, world and joint commands that ran in Mono, which they do until the async Burst compile has delivered their kernels.</summary>
        public static long ManagedCommands => Interlocked.Read(ref managedCommands);

        /// <summary>The managed id of the thread that last ran a step, 0 before the first one.</summary>
        public static int LastStepThreadId { get; private set; }

        /// <summary>Steps run since this domain attached.</summary>
        public static long StepCount { get; private set; }

        /// <summary>The thread that attached, which is the main thread in every path that attaches.</summary>
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
                // In the editor this also fires on leaving play mode, when edit mode still needs the core.
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
                rayCast = RayCastFunction(),
                destroyWorld = System.Runtime.InteropServices.Marshal.GetFunctionPointerForDelegate(DestroyWorldDelegate),
                cookMesh = CookMeshFunction(),
                validateMesh = ValidateMeshFunction(),
                bodyCommand = BodyCommandFunction(),
                shapeCommand = ShapeCommandFunction(),
                worldCommand = WorldCommandFunction(),
                jointCommand = JointCommandFunction(),
            };

            NativeStore.RetooledPhysics_Attach(&callbacks);
            attached = true;
            MainThreadId = Thread.CurrentThread.ManagedThreadId;
        }

        public static void Detach()
        {
            if (!attached)
            {
                return;
            }

            NativeStore.RetooledPhysics_Detach();
            attached = false;

            int worldCount = NativeStore.RetooledPhysics_WorldCount();
            for (int i = 0; i < worldCount; i++)
            {
                ReleaseWorld(NativeStore.RetooledPhysics_GetWorld(i));
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

        /// <summary>
        /// Asks Burst for the function once per domain. Burst compiles in the background and hands back a stub that runs
        /// the method in Mono until the compile lands, or for good if Burst is off.
        /// </summary>
        private static IntPtr RayCastFunction()
        {
            if (rayCastFunction == IntPtr.Zero)
            {
                rayCastFunction = BurstCompiler.CompileFunctionPointer<QueryKernel.RayCastFunction>(QueryKernel.RayCast).Value;
            }

            return rayCastFunction;
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

        /// <summary>
        /// Brings the query world up to date with everything the main thread has done. Workers can't rebuild it and read
        /// the last one published, so main-thread code calls this between moving things and scheduling query jobs, after
        /// any Physics.SyncTransforms. A step and a main-thread query already do the same. Main thread only.
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
            // Each world's own lock is held here, but queries on different worlds can run side by side.
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
            try
            {
                NoteStep();
                StateOf(world).Step(deltaTime);
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
