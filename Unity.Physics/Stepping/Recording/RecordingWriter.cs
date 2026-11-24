using System;
using System.Runtime.CompilerServices;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;

namespace Unity.Physics.Stepping.Recording
{
    /// <summary>Writes a checkpoint into the current frame buffer or into reusable scratch storage.</summary>
    internal sealed unsafe class RecordingWriter
    {
        internal byte[] Buffer { get; private set; } = new byte[4096];
        internal int Length { get; private set; }

        internal void Clear() => Length = 0;

        internal void BeginFrame(byte[] buffer, int sizeHint)
        {
            Buffer = buffer ?? new byte[sizeHint];
            Length = 0;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal void Write<T>(T value) where T : unmanaged
        {
            int start = Reserve(sizeof(T));
            fixed (byte* target = &Buffer[start])
            {
                UnsafeUtility.CopyStructureToPtr(ref value, target);
            }
        }

        internal void WriteValue<T>(T value) where T : unmanaged => Write(value);

        internal void WriteBlock(void* source, int bytes)
        {
            if (bytes == 0)
            {
                return;
            }

            int start = Reserve(bytes);
            fixed (byte* destination = &Buffer[start])
            {
                UnsafeUtility.MemCpy(destination, source, bytes);
            }
        }

        internal void WriteList<T>(UnsafeList<T> list) where T : unmanaged
        {
            Write(list.Length);
            WriteBlock(list.Ptr, checked(list.Length * sizeof(T)));
        }

        internal void WriteList<T>(NativeList<T> list) where T : unmanaged
        {
            Write(list.Length);
            WriteBlock(list.GetUnsafeReadOnlyPtr(), checked(list.Length * sizeof(T)));
        }

        private int Reserve(int bytes)
        {
            int start = Length;
            Length = checked(start + bytes);
            if (Length > Buffer.Length)
            {
                byte[] buffer = Buffer;
                int growth = System.Math.Max(4096, buffer.Length / 8);
                Array.Resize(ref buffer, System.Math.Max(Length, checked(buffer.Length + growth)));
                Buffer = buffer;
            }

            return start;
        }
    }
}
