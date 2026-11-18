using System.Runtime.InteropServices;
using Unity.Physics.Store.Bodies;

namespace Unity.Physics.Store.Shapes
{
    /// <summary>
    /// Shape handle storage beginning with ShapeData, allowing either structure to use the same pointer.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    internal unsafe struct Shape
    {
        public ShapeData data;
        public Body* body;
        private fixed uint filter[4];
        public float contactOffset;
        public uint messages;
        public uint querySlot;

        /// <summary>
        /// The index of the shape's group record in collision group table plus one, or zero if the shape has no
        /// group. The plugin never reads this field.
        /// </summary>
        public uint groupRecord;
        public Shape* nextOnField;
    }
}
