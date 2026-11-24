using Unity.Burst;
using Unity.Physics.Store.Bodies;
using Unity.Physics.Store.Shapes;
using Unity.Physics.Store.World;

namespace Unity.Physics.Stepping.Recording
{
    [BurstCompile]
    internal unsafe struct RecordedStoreShape
    {
        internal ulong Address;
        internal ulong EngineId;
        internal int GeometryId;
        internal bool HasMaterial;
        internal MaterialData Material;
        internal Shape Shape;

        [BurstCompile]
        internal static void Capture(WorldData* world, uint engineIdOffset, RecordedStoreShape* shapes)
        {
            for (uint row = 0; row < world->bodyCount; row++)
            {
                BodyData* body = world->bodies + row;
                for (uint i = 0; i < body->shapeCount; i++)
                {
                    Shape* source = (Shape*)body->shapes[i];
                    Shape shape = *source;
                    var record = new RecordedStoreShape
                    {
                        Address = (ulong)source,
                        EngineId = FlightRecorder.EngineId(shape.data.userData, engineIdOffset),
                        HasMaterial = shape.data.material != null,
                    };

                    if (record.HasMaterial)
                    {
                        record.Material = *shape.data.material;
                        record.Material.userData = null;
                    }

                    shape.body = null;
                    shape.data.owner = null;
                    shape.data.mesh = null;
                    shape.data.material = null;
                    shape.data.userData = null;
                    shape.nextOnField = null;
                    shape.querySlot = 0;
                    record.Shape = shape;
                    *shapes++ = record;
                }
            }
        }
    }
}
