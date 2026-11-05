using System;
using System.Collections.Generic;
using AOT;
using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
#endif

namespace KSP.Retooled.PhysicsBackend
{
    /// <summary>
    /// Connects the managed core to the plugin. The plugin's store outlives a domain reload and everything managed holds
    /// can be rebuilt from it, so a reload is a detach, which frees managed state, followed by an attach. Until then the
    /// plugin answers with no simulation and no hits.
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
        private static readonly NativeStore.RayCastCallback RayCastDelegate = OnRayCast;
        private static readonly NativeStore.DestroyWorldCallback DestroyWorldDelegate = OnDestroyWorld;

        private static readonly Dictionary<IntPtr, WorldState> Worlds = new Dictionary<IntPtr, WorldState>();

        private static bool attached;

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
                Debug.LogError($"{Tag}not attaching: {mismatch}. Rebuild the plugin with `RetooledCli native build`.");
                return;
            }

            NativeStore.ManagedCallbacks callbacks = new NativeStore.ManagedCallbacks
            {
                simulate = System.Runtime.InteropServices.Marshal.GetFunctionPointerForDelegate(SimulateDelegate),
                buildQueryWorld = System.Runtime.InteropServices.Marshal.GetFunctionPointerForDelegate(BuildQueryWorldDelegate),
                rayCast = System.Runtime.InteropServices.Marshal.GetFunctionPointerForDelegate(RayCastDelegate),
                destroyWorld = System.Runtime.InteropServices.Marshal.GetFunctionPointerForDelegate(DestroyWorldDelegate),
            };

            NativeStore.RetooledPhysics_Attach(&callbacks);
            attached = true;
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
                StateOf(world).Step(deltaTime);
            }
            catch (Exception e)
            {
                Debug.LogException(e);
            }
        }

        [MonoPInvokeCallback(typeof(NativeStore.BuildQueryWorldCallback))]
        private static void OnBuildQueryWorld(NativeStore.WorldData* world)
        {
            try
            {
                StateOf(world).BuildForQueries();
            }
            catch (Exception e)
            {
                Debug.LogException(e);
            }
        }

        [MonoPInvokeCallback(typeof(NativeStore.RayCastCallback))]
        private static int OnRayCast(NativeStore.WorldData* world, NativeStore.RayCastInput* input, NativeStore.RayCastHit* hit)
        {
            try
            {
                return StateOf(world).RayCast(input, hit) ? 1 : 0;
            }
            catch (Exception e)
            {
                Debug.LogException(e);
                return 0;
            }
        }

        [MonoPInvokeCallback(typeof(NativeStore.DestroyWorldCallback))]
        private static void OnDestroyWorld(NativeStore.WorldData* world)
        {
            try
            {
                ReleaseWorld(world);
            }
            catch (Exception e)
            {
                Debug.LogException(e);
            }
        }
    }
}
