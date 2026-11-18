using System;
using Unity.Physics.Base.Threading;

namespace Unity.Physics.Stepping
{
    internal unsafe partial struct StoreStepper
    {
        /// <summary>
        /// For tests: whether the next step throws right after it schedules the refit of the dynamic tree, while the
        /// solve's jobs still run. Only code that runs in Mono throws; Burst leaves the throw out.
        /// </summary>
        internal static bool ThrowAfterRefitScheduledForTests
        {
            get => s_ThrowAfterRefitScheduled.Data != 0;
            set => s_ThrowAfterRefitScheduled.Data = value ? 1 : 0;
        }

        /// <summary>For tests: the number of throws that <see cref="ThrowAfterRefitScheduledForTests"/> made.</summary>
        internal static int ThrowsAfterRefitScheduled => s_ThrowsAfterRefitScheduled;

        private static int s_ThrowsAfterRefitScheduled;

        [Unity.Burst.BurstDiscard]
        private static void ThrowAfterRefitScheduled()
        {
            s_ThrowAfterRefitScheduled.Data = 0;
            s_ThrowsAfterRefitScheduled++;
            throw new InvalidOperationException("a throw that a test injected into the step after the refit was scheduled");
        }

        private sealed class ThrowAfterRefitScheduledKey
        {
        }

        /// <summary>
        /// For tests: whether the next step throws before it applies the forces, which is also before it refreshes
        /// the pointers of the store's rows. Only code that runs in Mono throws; Burst leaves the throw out.
        /// </summary>
        internal static bool ThrowBeforeApplyForcesForTests
        {
            get => s_ThrowBeforeApplyForces.Data != 0;
            set => s_ThrowBeforeApplyForces.Data = value ? 1 : 0;
        }

        /// <summary>For tests: the number of throws that <see cref="ThrowBeforeApplyForcesForTests"/> made.</summary>
        internal static int ThrowsBeforeApplyForces => s_ThrowsBeforeApplyForces;

        private static int s_ThrowsBeforeApplyForces;

        internal static void ThrowBeforeApplyForcesIfRequested()
        {
            if (s_ThrowBeforeApplyForces.Data != 0)
            {
                ThrowBeforeApplyForces();
            }
        }

        [Unity.Burst.BurstDiscard]
        private static void ThrowBeforeApplyForces()
        {
            s_ThrowBeforeApplyForces.Data = 0;
            s_ThrowsBeforeApplyForces++;
            throw new InvalidOperationException("a throw that a test injected into the step before it applies the forces");
        }

        private sealed class ThrowBeforeApplyForcesKey
        {
        }

        private static readonly Unity.Burst.SharedStatic<int> s_ThrowBeforeApplyForces = Unity.Burst.SharedStatic<int>.GetOrCreate<StoreStepper, ThrowBeforeApplyForcesKey>();

        private static readonly Unity.Burst.SharedStatic<int> s_ThrowAfterRefitScheduled = Unity.Burst.SharedStatic<int>.GetOrCreate<StoreStepper, ThrowAfterRefitScheduledKey>();

        /// <summary>
        /// Set treeReady after a failed step unless a pending refit owns completion, preventing queries from
        /// waiting forever.
        /// </summary>
        internal void ReleaseTreeAfterThrow()
        {
            if (!Trees.refitPending)
            {
                System.Threading.Interlocked.Exchange(ref store->treeReady, 1);
                AddressWait.WakeAll(&store->treeReady);
            }
        }

    }
}
