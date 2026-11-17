using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;

namespace Unity.Physics.Dynamics.Solver.Forest
{
    /// <summary>
    /// Sorted boundary row keys with the previous state in the low two bits and consecutive-state age in the next
    /// six bits.
    /// </summary>
    internal unsafe struct BoundaryMemory
    {
        public const int StateMask = 3;
        public const int AgeShift = 2;
        public const int MostAge = 63;

        /// <summary>The bits of an entry that hold the state and the age. A key has these bits zero.</summary>
        public const int ValueMask = 0xFF;

        public UnsafeList<ulong> States;

        /// <summary>How many times the caps have chosen this island's boundary rows. It seeds the hash that breaks ties.</summary>
        public uint CapCount;

        public void Dispose()
        {
            if (States.IsCreated)
            {
                States.Dispose();
            }
        }

        /// <summary>Returns the index of the entry for <paramref name="key"/>, or -1 if there is none.</summary>
        public static int Find(in UnsafeList<ulong> states, ulong key)
        {
            int low = 0, high = states.Length;
            while (low < high)
            {
                int middle = (low + high) >> 1;
                if ((states[middle] & ~(ulong)ValueMask) < key)
                {
                    low = middle + 1;
                }
                else
                {
                    high = middle;
                }
            }

            return low < states.Length && (states[low] & ~(ulong)ValueMask) == key ? low : -1;
        }

        /// <summary>The age that entry <paramref name="entry"/> holds.</summary>
        public static byte Age(ulong entry) => (byte)((entry & ValueMask) >> AgeShift);

        /// <summary>
        /// Store boundary keys and final states. Increment age when state is unchanged, otherwise reset it.
        /// </summary>
        public static void Remember(ref UnsafeList<ulong> states, ulong* keys, byte* rowStates, byte* ages, int count)
        {
            if (!states.IsCreated)
            {
                states = new UnsafeList<ulong>(math.max(16, count), Allocator.Persistent);
            }

            states.Clear();
            for (int i = 0; i < count; i++)
            {
                states.Add(keys[i] | (ulong)(rowStates[i] & StateMask) | ((ulong)math.min((int)ages[i], MostAge) << AgeShift));
            }

            NativeSortExtension.Sort(states.Ptr, states.Length);
        }
    }
}
