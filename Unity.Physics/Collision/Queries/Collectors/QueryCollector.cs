using Unity.Burst;
using Unity.Collections;
using Unity.Entities;
using Unity.Mathematics;
using Unity.Physics.Collision.Colliders.Building;
using Unity.Physics.Store.Bodies;
using Unity.Physics.Store.Queries;
using Unity.Physics.Store.World;
using Unity.Physics.Collision.Queries.Snapshot;

namespace Unity.Physics.Collision.Queries.Collectors
{
    /// <summary>
    /// Filter hits by body type, trigger/query flags and triangle side.
    /// When grouping by shape, combine triangle hits into one shape hit. The caller must call Finish after the
    /// query.
    /// </summary>
    internal unsafe struct QueryCollector<T, S> : ICollector<T>
        where T : unmanaged, IQueryResult
        where S : struct, IQuerySink<T>
    {
        internal S Sink;

        private readonly NativeArray<RigidBody> bodies;
        private readonly QueryBody* queryBodies;
        private readonly QueryShape* queryShapes;
        private readonly Body* ignoredBody;
        private readonly uint filterOptions;
        private readonly float3 direction;
        private readonly QueryHitRules rules;
        private readonly bool groupByShape;
        private readonly bool keepInitialOverlaps;

        private T run;
        private QueryShape* runShape;
        private uint runSerial;
        private bool runBackface;
        private bool runOverlapped;
        private bool stopped;

        internal QueryCollector(
            QueryWorld* world,
            S sink,
            uint filterOptions,
            float3 direction,
            QueryHitRules rules,
            bool groupByShape,
            bool keepInitialOverlaps = false,
            Body* ignoredBody = null)
        {
            Sink = sink;
            bodies = world->Collision.Bodies;
            queryBodies = world->Bodies;
            queryShapes = world->Shapes;
            this.ignoredBody = ignoredBody;
            this.filterOptions = filterOptions;
            this.direction = direction;
            this.rules = rules;
            this.groupByShape = groupByShape;
            this.keepInitialOverlaps = keepInitialOverlaps;
            run = default;
            runShape = null;
            runSerial = 0;
            runBackface = false;
            runOverlapped = false;
            stopped = false;
        }

        public bool EarlyOutOnFirstHit => Sink.EarlyOut || stopped;

        public float MaxFraction => Sink.MaxFraction;

        public int NumHits => Sink.NumHits + (runShape != null ? 1 : 0);

        public bool AddHit(T hit)
        {
            if (stopped)
            {
                return false;
            }

            RigidBody rigidBody = bodies[hit.RigidBodyIndex];
            if (rigidBody.Entity.Version == 0)
            {
                return false;
            }

            QueryBody* body = queryBodies + hit.RigidBodyIndex;
            if (ignoredBody != null && body->Owner == ignoredBody)
            {
                return false;
            }

            uint wanted = body->IsStatic != 0 ? StoreConstants.QueryUseStaticBodies : StoreConstants.QueryUseDynamicBodies;
            if ((filterOptions & wanted) == 0)
            {
                return false;
            }

            QueryShape* shape = queryShapes + body->FirstShape + ColliderBuilder.ShapeIndexOf(rigidBody.Collider, hit.ColliderKey, body->SingleShape);
            if ((shape->Flags & StoreConstants.ShapeFlagSceneQuery) == 0)
            {
                return false;
            }

            bool trigger = (shape->Flags & StoreConstants.ShapeFlagTrigger) != 0;
            if (trigger && (filterOptions & StoreConstants.QueryUseTriggerShapes) == 0)
            {
                return false;
            }

            if (rules == QueryHitRules.Ray && hit.Fraction <= 0.0f)
            {
                return false;
            }

            bool backface = false;
            if (rules != QueryHitRules.Overlap)
            {
                backface = QueryCollectors.IsBackface((Collider*)rigidBody.Collider.GetUnsafePtr(), hit.ColliderKey, rigidBody.WorldFromBody, direction);
                if (backface && (filterOptions & StoreConstants.QueryAllowBackfaces) == 0)
                {
                    return false;
                }
            }

            bool overlapped = rules == QueryHitRules.Sweep && hit.Fraction <= 0.0f;
            if (!groupByShape)
            {
                stopped = !Sink.Add(hit, shape, (uint)rigidBody.Entity.Version, backface, overlapped);
                return true;
            }

            if (shape != runShape)
            {
                FinishRun();
                run = hit;
                runShape = shape;
                runSerial = (uint)rigidBody.Entity.Version;
                runBackface = backface;
                runOverlapped = overlapped;
                return true;
            }

            runOverlapped |= overlapped;
            if (hit.Fraction < run.Fraction)
            {
                run = hit;
                runBackface = backface;
            }

            return true;
        }

        internal void Finish()
        {
            FinishRun();
        }

        private void FinishRun()
        {
            if (runShape != null && !stopped && (!runOverlapped || keepInitialOverlaps))
            {
                stopped = !Sink.Add(run, runShape, runSerial, runBackface, runOverlapped);
            }

            runShape = null;
        }
    }
}
