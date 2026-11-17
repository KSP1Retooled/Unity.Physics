using Unity.Physics.Store.Bodies;
using Unity.Physics.Stepping;

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
            BodyHome* home = BodyHome.Of(body);
            (*hot).SetPosePointer(&home->pose);
            (*hot).SetVelocityPointer(&home->velocity);
            (*hot).SetMassPointer(&home->mass);
            (*hot).SetBodyMassPointer(&home->bodyMass);
            (*hot).SetDampingPointer(&home->damping);
            (*hot).SetLimitsPointer(&home->limits);
            (*hot).SetSleepPointer(&home->sleep);
            (*hot).SetForcesPointer(&home->forces);
            (*hot).SetTargetPointer(&home->target);
            (*hot).SetColliderPointer(&home->collider);
        }
    }
}
