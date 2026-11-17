using System.Threading;

namespace KSP.Retooled.Physics.Host
{
    internal static class BackendCounters
    {
        private static long managedQueries;
        private static long managedCooks;
        private static long managedCommands;
        private static long managedSteps;

        internal static long ManagedQueries => Interlocked.Read(ref managedQueries);
        internal static long ManagedCooks => Interlocked.Read(ref managedCooks);
        internal static long ManagedCommands => Interlocked.Read(ref managedCommands);
        internal static long ManagedSteps => Interlocked.Read(ref managedSteps);

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
    }
}
