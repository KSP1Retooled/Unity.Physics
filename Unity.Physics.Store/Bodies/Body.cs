using System.Runtime.InteropServices;
using Unity.Physics.Store.Shapes;
using Unity.Physics.Store.World;

namespace Unity.Physics.Store.Bodies
{
    [StructLayout(LayoutKind.Sequential)]
    internal unsafe struct Body
    {
        public WorldData* world;
        public ShapeData** shapes;
        public void* userData;
        public uint index;
        public uint shapeCount;
        public uint shapeCapacity;
        public int collisionDetectionMode;
        public uint positionIterations;
        public uint velocityIterations;

        // As in BodyData, the plugin only knows the fields above.
        public BodyHomeBlock home;

        // A number that no other body received. The store reuses the memory of a destroyed body for the next body
        // that is made, so the address alone doesn't tell two bodies apart.
        public uint serial;
    }
}
