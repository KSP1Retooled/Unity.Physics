using System;
using System.Runtime.CompilerServices;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;

namespace Unity.Physics.Stepping.Recording
{
    /// <summary>Reuses a buffer for writing checkpoints and copies only the bytes that a frame keeps.</summary>
    internal sealed unsafe class RecordingWriter
    {
        internal byte[] Buffer { get; private set; } = new byte[4096];
        internal int Length { get; private set; }

        internal void Clear() => Length = 0;

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

        internal byte[] ToArray()
        {
            var result = new byte[Length];
            System.Buffer.BlockCopy(Buffer, 0, result, 0, Length);
            return result;
        }

        private int Reserve(int bytes)
        {
            int start = Length;
            Length = checked(start + bytes);
            if (Length > Buffer.Length)
            {
                byte[] buffer = Buffer;
                Array.Resize(ref buffer, System.Math.Max(Length, checked(buffer.Length * 2)));
                Buffer = buffer;
            }

            return start;
        }
    }
}
