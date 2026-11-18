using System.Runtime.InteropServices;

namespace Unity.Physics.Store.Shapes
{
    [StructLayout(LayoutKind.Sequential)]
    internal unsafe struct HeightField
    {
        public short* heights;
        public byte* cells;
        public uint resolution;
        public uint holeCount;
        public int lowest;
        public int highest;
        public uint references;
        private readonly uint reserved0;
        public Shape* firstShape;
    }
}
