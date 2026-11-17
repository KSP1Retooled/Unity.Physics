using System.Runtime.InteropServices;
using Unity.Mathematics;

namespace Unity.Physics.Store.World
{
    [StructLayout(LayoutKind.Sequential)]
    internal unsafe struct CommandArgs
    {
        public quaternion rotation;
        public float3 position;
        public uint flags;
        public float3 vector;
        public float scalar;
        public float3 vector2;
        public float scalar2;
        public void* pointer;
        public void* pointer2;
        public void* pointer3;
        public ulong size;
        public float scalar3;
        public uint mode;
        public uint enabled;
        public uint value;
        public uint value2;
        private readonly uint reserved0;
    }
}
