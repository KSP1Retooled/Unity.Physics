using Unity.Physics.Store.Bodies;

namespace KSP.Retooled.Physics.Entities
{
    internal static unsafe class StoreRows
    {
        /// <summary>
        /// Return the body's row with refreshed component pointers. Main thread only.
        /// The row may move when bodies are created or destroyed; entity pointers may change after structural
        /// changes.
        /// </summary>
        internal static BodyData* Hot(Body* body)
        {
            BodyData* hot = body->world->bodies + body->index;
            BodyEntities.Refresh(hot);
            return hot;
        }

        /// <summary>
        /// Points a row at the components stored in its body record. That component storage remains at the same
        /// address.
        /// </summary>
        internal static void PointAtHome(BodyData* hot, Body* body)
        {
            BodyHome* home = &body->home;
            hot->pose = &home->pose;
            hot->velocity = &home->velocity;
            hot->mass = &home->mass;
            hot->bodyMass = &home->bodyMass;
            hot->damping = &home->damping;
            hot->limits = &home->limits;
            hot->sleep = &home->sleep;
            hot->forces = &home->forces;
            hot->target = &home->target;
            hot->collider = &home->collider;
        }
    }
}
