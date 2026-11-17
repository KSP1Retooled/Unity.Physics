using KSP.Retooled.Physics.Diagnostics;
using KSP.Retooled.Physics.Commands;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using Unity.Physics.Store.Memory;
using Unity.Physics.Store.Shapes;
using Unity.Physics.Store.World;
using Unity.Physics.Collision.Colliders.Building;

namespace KSP.Retooled.Physics.Terrain
{
    /// <summary>
    /// Store height field samples and attached shapes on the native heap across domain reloads.
    /// ColliderBuilder builds geometry for each shape's scale.
    /// </summary>
    internal static unsafe class HeightFields
    {
        internal static HeightField* Create(CommandArgs* a, StoreGlobals* globals)
        {
            using Unity.Profiling.ProfilerMarker.AutoScope span = BackendMarkers.CreateHeightField.Auto();

            // The descriptor's height count is often zero, so calculate the number of samples from the resolution.
            uint resolution = a->value;
            long sampleCount = resolution * (long)resolution;
            if (a->pointer == null || resolution < 2)
            {
                return null;
            }

            HeightField* field = StoreMemory.Allocate<HeightField>(globals);
            field->resolution = resolution;
            field->references = 1;
            field->heights = (short*)StoreMemory.Reallocate(globals, null, (ulong)(sampleCount * sizeof(short)));
            UnsafeUtility.MemCpy(field->heights, a->pointer, sampleCount * sizeof(short));

            field->lowest = short.MaxValue;
            field->highest = short.MinValue;
            WidenBounds(field);
            StoreHoles(field, (byte*)a->pointer2, a->mode, globals);
            return field;
        }

        /// <summary>
        /// Update height field samples and mark every attached body's collider for rebuilding.
        /// Unity supplies the entire edited grid, even for a partial update.
        /// </summary>
        internal static void Update(CommandArgs* a, StoreGlobals* globals)
        {
            using Unity.Profiling.ProfilerMarker.AutoScope span = BackendMarkers.UpdateHeightField.Auto();

            HeightField* field = (HeightField*)a->pointer3;
            if (field == null || a->pointer == null || a->value != field->resolution)
            {
                return;
            }

            UnsafeUtility.MemCpy(field->heights, a->pointer, field->resolution * (long)field->resolution * sizeof(short));
            WidenBounds(field);
            if (a->pointer2 != null)
            {
                StoreHoles(field, (byte*)a->pointer2, a->mode, globals);
            }

            for (Shape* shape = field->firstShape; shape != null; shape = shape->nextOnField)
            {
                if (shape->body != null)
                {
                    CommandStore.MarkColliderDirty(shape->body);
                }
            }
        }

        internal static void Retain(Shape* shape)
        {
            HeightField* field = (HeightField*)shape->data.mesh;
            if (field != null)
            {
                field->references++;
                shape->nextOnField = field->firstShape;
                field->firstShape = shape;
            }
        }

        internal static void Release(Shape* shape, StoreGlobals* globals)
        {
            HeightField* field = (HeightField*)shape->data.mesh;
            if (field == null)
            {
                return;
            }

            Shape** link = &field->firstShape;
            while (*link != shape)
            {
                link = &(*link)->nextOnField;
            }

            *link = shape->nextOnField;
            shape->nextOnField = null;
            Release(field, globals);
        }

        internal static void Release(HeightField* field, StoreGlobals* globals)
        {
            if (field != null && --field->references == 0)
            {
                StoreMemory.Release(globals, field->heights);
                StoreMemory.Release(globals, field->cells);
                StoreMemory.Release(globals, field);
            }
        }

        private static void WidenBounds(HeightField* field)
        {
            long sampleCount = field->resolution * (long)field->resolution;
            int lowest = field->lowest;
            int highest = field->highest;
            for (long i = 0; i < sampleCount; i++)
            {
                int height = field->heights[i];
                lowest = math.min(lowest, height);
                highest = math.max(highest, height);
            }

            field->lowest = lowest;
            field->highest = highest;
        }

        /// <summary>
        /// Stores the cell data only when at least one cell is a hole. The presence of holes determines which kind of
        /// collider to build.
        /// </summary>
        private static void StoreHoles(HeightField* field, byte* cells, uint cellCount, StoreGlobals* globals)
        {
            long expected = (field->resolution - 1L) * (field->resolution - 1L);
            uint holes = 0;
            if (cells != null && cellCount == expected)
            {
                for (long i = 0; i < expected; i++)
                {
                    holes += cells[i] == 0 ? 1u : 0u;
                }
            }

            field->holeCount = holes;
            if (holes == 0)
            {
                StoreMemory.Release(globals, field->cells);
                field->cells = null;
                return;
            }

            field->cells = (byte*)StoreMemory.Reallocate(globals, field->cells, (ulong)expected);
            UnsafeUtility.MemCpy(field->cells, cells, expected);
        }
    }
}
