using Unity.Collections;

namespace Unity.Physics.Stepping.State
{
    /// <summary>
    /// Owns the buffers used to group touching bodies and put them to sleep.
    /// The sleep jobs fill them from the current step.
    /// </summary>
    internal unsafe struct SleepState
    {
        internal NativeList<int> sleepParents;
        internal NativeList<int> sleepContacts;
        internal NativeList<bool> sleepTouchesStill;
        internal NativeList<int> fallingAsleep;
        internal NativeArray<int> sleepRunOut;

        internal void Initialize()
        {
            sleepParents = new NativeList<int>(Allocator.Persistent);
            sleepContacts = new NativeList<int>(Allocator.Persistent);
            sleepTouchesStill = new NativeList<bool>(Allocator.Persistent);
            fallingAsleep = new NativeList<int>(Allocator.Persistent);
            sleepRunOut = new NativeArray<int>(1, Allocator.Persistent);
        }

        internal void Dispose()
        {
            sleepParents.Dispose();
            sleepContacts.Dispose();
            sleepTouchesStill.Dispose();
            fallingAsleep.Dispose();
            sleepRunOut.Dispose();
        }
    }
}
