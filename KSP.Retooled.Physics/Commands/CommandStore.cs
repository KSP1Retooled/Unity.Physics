using System.Threading;
using Unity.Entities;
using Unity.Physics;
using KSP.Retooled.Physics.Entities;
using Unity.Physics.Store.Bodies;
using Unity.Physics.Store.Memory;
using Unity.Physics.Store.Shapes;
using Unity.Physics.Store.Versions;
using Unity.Physics.Store.World;
using Unity.Physics.Stepping;
using Unity.Physics.Collision.Queries.Snapshot;

namespace KSP.Retooled.Physics.Commands
{
    /// <summary>Updates shape lists and invalidates collider and query data after commands.</summary>
    internal static unsafe class CommandStore
    {
        /// <summary>
        /// Mark static pose changes without weakening an existing full-rebuild request.
        /// </summary>
        internal static void MarkStaticMoved(Body* body)
        {
            ref int dirty = ref *(int*)&body->world->queryDirty;
            if (Interlocked.CompareExchange(ref dirty, (int)StoreMemory.QueryStaticsMoved, 0) == (int)StoreMemory.QueryTreePending)
            {
                Interlocked.CompareExchange(ref dirty, (int)StoreMemory.QueryStaticsMoved, (int)StoreMemory.QueryTreePending);
            }

            body->world->staticsVersion++;
        }

        internal static void MarkPoseChanged(Body* body, BodyFlags flags)
        {
            StoreMemory.MarkQueryDirty(body->world);
            if ((flags & BodyFlags.Static) != 0)
            {
                body->world->staticsVersion++;
            }
        }

        /// <summary>
        /// Mark collider geometry dirty and invalidate query snapshot data after shape changes.
        /// </summary>
        internal static void MarkColliderDirty(Body* body)
        {
            BodyData* hot = StoreRows.Hot(body);
            hot->flags |= BodyFlags.ColliderDirty;
            QuerySnapshotVersion.Changed();
            MarkPoseChanged(body, hot->flags);
        }

        /// <summary>
        /// Mark shape pose changes. Copy compounds with new child poses; rebuild colliders with baked poses.
        /// </summary>
        internal static void MarkCollidersMoved(Body* body)
        {
            BodyData* hot = StoreRows.Hot(body);
            BlobAssetReference<Collider> built = (*hot).Collider()->Value;
            if (built.IsCreated && built.Value.Type == ColliderType.Compound)
            {
                hot->flags |= BodyFlags.CollidersMoved;
                MarkPoseChanged(body, hot->flags);
            }
            else
            {
                MarkColliderDirty(body);
            }
        }

        private static void PublishShapes(Body* body)
        {
            BodyData* hot = StoreRows.Hot(body);
            hot->shapes = body->shapes;
            hot->shapeCount = body->shapeCount;
            MarkColliderDirty(body);
        }

        internal static void AddShape(Body* body, Shape* shape, StoreGlobals* globals)
        {
            if (body->shapeCount == body->shapeCapacity)
            {
                body->shapeCapacity = StoreMemory.Grown(body->shapeCapacity);
                body->shapes = (ShapeData**)StoreMemory.Reallocate(globals, body->shapes, (ulong)(body->shapeCapacity * sizeof(void*)));
            }

            shape->body = body;
            body->shapes[body->shapeCount++] = &shape->data;
            PublishShapes(body);
        }

        private static void NoteRemovedShape(WorldData* world, Shape* shape, StoreGlobals* globals)
        {
            if (world->removedShapeCount == world->removedShapeCapacity)
            {
                world->removedShapeCapacity = StoreMemory.Grown(world->removedShapeCapacity);
                world->removedShapes = (void**)StoreMemory.Reallocate(globals, world->removedShapes, world->removedShapeCapacity * (ulong)sizeof(void*));
            }

            world->removedShapes[world->removedShapeCount++] = shape;
            HideFromQueries(world, shape);
        }

        /// <summary>
        /// Hide a removed shape from the published query snapshot before its collider is destroyed.
        /// Verify that the snapshot slot still belongs to this shape.
        /// </summary>
        private static void HideFromQueries(WorldData* world, Shape* shape)
        {
            QueryWorld* query = (QueryWorld*)world->queryWorld;
            if (query != null && shape->querySlot != 0 && shape->querySlot <= (uint)query->ShapeCount)
            {
                QueryShape* published = query->Shapes + (shape->querySlot - 1);
                if (published->Handle == shape)
                {
                    published->Flags = 0;
                    published->UserData = null;
                }
            }

            shape->querySlot = 0;
        }

        /// <summary>
        /// Removes a shape from its body and preserves the order of the remaining shapes. The child indices in a
        /// compound collider depend on that order.
        /// </summary>
        internal static void RemoveShape(Body* body, Shape* shape, StoreGlobals* globals)
        {
            NoteRemovedShape(body->world, shape, globals);
            uint kept = 0;
            for (uint i = 0; i < body->shapeCount; i++)
            {
                if (body->shapes[i] != &shape->data)
                {
                    body->shapes[kept++] = body->shapes[i];
                }
            }

            body->shapeCount = kept;
            shape->body = null;
            PublishShapes(body);
        }

        /// <summary>
        /// Detaches a body's shapes when the body is destroyed. The shapes remain alive until Unity destroys them.
        /// </summary>
        internal static void DetachAllShapes(Body* body, StoreGlobals* globals)
        {
            for (uint i = 0; i < body->shapeCount; i++)
            {
                NoteRemovedShape(body->world, (Shape*)body->shapes[i], globals);
                ((Shape*)body->shapes[i])->body = null;
            }

            body->shapeCount = 0;
            StoreMemory.Release(globals, body->shapes);
            body->shapes = null;
            body->shapeCapacity = 0;
            PublishShapes(body);
        }
    }
}
