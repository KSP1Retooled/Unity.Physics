using System;
using Unity.Collections;

namespace Unity.Numerics.Linear
{
    /// <summary>
    /// Stores 64-bit priorities in a binary heap, so that the smallest priority comes out first. The sparse constraint
    /// solver uses this heap to pick the next node for its minimum-degree order.
    /// </summary>
    public struct MinHeap : IDisposable
    {
        NativeList<long> items;

        public MinHeap(int capacity, Allocator allocator)
        {
            items = new NativeList<long>(capacity, allocator);
        }

        public void Dispose()
        {
            items.Dispose();
        }

        /// <summary>Adds an item to the heap.</summary>
        public void Push(long item)
        {
            items.Add(item);
            int child = items.Length - 1;
            while (child > 0)
            {
                int parent = (child - 1) / 2;
                if (items[parent] <= items[child])
                {
                    break;
                }

                (items[parent], items[child]) = (items[child], items[parent]);
                child = parent;
            }
        }

        /// <summary>Removes the smallest item from the heap and returns it.</summary>
        public long Pop()
        {
            long top = items[0];
            items[0] = items[items.Length - 1];
            items.Length--;
            int parent = 0;
            while (true)
            {
                int left = parent * 2 + 1;
                int right = left + 1;
                int smallest = parent;
                if (left < items.Length && items[left] < items[smallest])
                {
                    smallest = left;
                }

                if (right < items.Length && items[right] < items[smallest])
                {
                    smallest = right;
                }

                if (smallest == parent)
                {
                    return top;
                }

                (items[parent], items[smallest]) = (items[smallest], items[parent]);
                parent = smallest;
            }
        }
    }
}
