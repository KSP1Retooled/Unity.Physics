using System.Runtime.InteropServices;

namespace Unity.Physics.Store.Shapes
{
    [StructLayout(LayoutKind.Sequential)]
    internal unsafe struct CookOutput
    {
        public byte* bytes;
        public long size;
        public int status;
        private readonly int reserved0;
    }
}
