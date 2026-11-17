using System.Runtime.InteropServices;

namespace Unity.Physics.Store.Bodies
{
    /// <summary>Reserves space for the components of a body that has no entity.</summary>
    [StructLayout(LayoutKind.Sequential, Size = 304)]
    internal struct BodyHomeBlock
    {
        private readonly ulong alignment;
    }
}
