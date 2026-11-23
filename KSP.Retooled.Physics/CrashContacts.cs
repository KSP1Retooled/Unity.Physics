using KSP.Retooled.Physics.Interop;
using UnityEngine;
using Unity.Physics.Collision.Filter.Groups;
using Unity.Physics.Store.World;

namespace KSP.Retooled.Physics
{
    /// <summary>
    /// Tell the backend which colliders the game destroys on a fast impact.
    /// A contact that begins faster than a collider's crash speed pushes with a limited impulse for its first step, so
    /// the doomed part cannot stop its vessel before the collision callback destroys it. The collision report is
    /// unchanged. A contact that the game leaves alone pushes without a limit from its second step.
    /// Call on the main thread, outside simulation.
    /// </summary>
    public static unsafe class CrashContacts
    {
        /// <summary>Whether the methods of this class have an effect.</summary>
        public static bool Available => BackendHost.IsAttached;

        /// <summary>
        /// Set the relative speed above which an impact destroys the collider's part. Zero removes the speed.
        /// The collider keeps the speed from before it is first simulated, and when Unity replaces its shape.
        /// Return false if the backend is detached.
        /// </summary>
        public static bool SetCrashSpeed(Collider collider, float speed)
        {
            if (collider == null || !Available)
            {
                return false;
            }

            StoreGlobals* globals = NativeLibrary.RetooledPhysics_Globals();
            GroupTable.SetCrashSpeed(globals, EntityId.ToULong(collider.GetEntityId()), speed);
            CollisionGroups.CheckOrphans(globals);
            return true;
        }
    }
}
