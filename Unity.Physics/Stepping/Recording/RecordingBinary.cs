using System;
using System.IO;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;

namespace Unity.Physics.Stepping.Recording
{
    internal static unsafe class RecordingBinary
    {
        internal const string Magic = "Retooled flight recorder";
        internal const int Version = 9;
        internal const int EmptyVehicle = -2;

        internal static T ReadValue<T>(this BinaryReader reader) where T : unmanaged
        {
            T value = default;
            reader.ReadBlock(&value, sizeof(T));
            return value;
        }

        internal static void ReadBlock(this BinaryReader reader, void* destination, int length)
        {
            Span<byte> remaining = new Span<byte>(destination, length);
            while (!remaining.IsEmpty)
            {
                int count = reader.Read(remaining);
                if (count == 0)
                {
                    throw new EndOfStreamException();
                }

                remaining = remaining.Slice(count);
            }
        }

        internal static int ReadCount(this BinaryReader reader, int limit)
        {
            int count = reader.ReadInt32();
            if (count < 0 || count > limit)
            {
                throw new InvalidDataException($"Invalid count {count} (limit {limit}).");
            }

            return count;
        }

        internal static bool Same<T>(T first, T second) where T : unmanaged =>
            UnsafeUtility.MemCmp(&first, &second, sizeof(T)) == 0;

        internal static void ReadList<T>(this BinaryReader reader, ref UnsafeList<T> list) where T : unmanaged
        {
            int count = reader.ReadCount(int.MaxValue / sizeof(T));
            list.Resize(count, NativeArrayOptions.UninitializedMemory);
            reader.ReadBlock(list.Ptr, count * sizeof(T));
        }

        internal static void ReadList<T>(this BinaryReader reader, NativeList<T> list) where T : unmanaged
        {
            int count = reader.ReadCount(int.MaxValue / sizeof(T));
            list.ResizeUninitialized(count);
            reader.ReadBlock(list.GetUnsafePtr(), count * sizeof(T));
        }
    }
}
