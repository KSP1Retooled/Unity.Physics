using KSP.Retooled.Physics.Entities;
using KSP.Retooled.Physics.Interop;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using UnityEngine;
using Unity.Physics.Store.Bodies;
using Unity.Physics.Store.Shapes;
using Unity.Physics.Store.World;
using Unity.Physics.Collision.Filter.Groups;
using Unity.Physics.Dynamics.Bodies;

namespace KSP.Retooled.Physics
{
    /// <summary>
    /// Control collisions within groups, such as parts of a vessel.
    /// Members of the same group collide only when both allow self-collision and have different member IDs,
    /// or when SetMembersCollide permits the pair. Physics.IgnoreCollision still takes precedence.
    /// Wheel ground queries use the same rules; Physics.GetIgnoreCollision reports only explicitly ignored
    /// pairs.
    /// Changes take effect on the next step. Call on the main thread, outside simulation.
    /// Methods return false without effect when the backend is detached; use Physics.IgnoreCollision as a
    /// fallback.
    /// </summary>
    public static unsafe class CollisionGroups
    {
        /// <summary>
        /// Initial count of grouped colliders without shapes that triggers removal of destroyed colliders.
        /// Double the threshold after each check to limit cleanup cost.
        /// </summary>
        private const uint OrphanCheckThreshold = 4096;

        private static uint nextOrphanCheck = OrphanCheckThreshold;

        /// <summary>Whether the methods of this class have an effect.</summary>
        public static bool Available => BackendHost.IsAttached;

        /// <summary>
        /// Set a collider's group, member and self-collision flag. Group zero removes membership.
        /// Return false if the backend is detached.
        /// </summary>
        public static bool Set(Collider collider, uint group, uint member, bool selfCollides)
        {
            if (collider == null || !Available)
            {
                return false;
            }

            StoreGlobals* globals = NativeLibrary.RetooledPhysics_Globals();
            ulong engineId = EntityId.ToULong(collider.GetEntityId());
            Shape* shape = GroupTable.Set(globals, engineId, group, member, selfCollides ? GroupTable.SelfCollides : 0u);
            Wake(shape);
            if (IgnoreTable.GroupsOf(globals)->Orphans >= nextOrphanCheck)
            {
                ForgetDestroyedColliders(globals);
                nextOrphanCheck = System.Math.Max(OrphanCheckThreshold, IgnoreTable.GroupsOf(globals)->Orphans * 2);
            }

            return true;
        }

        /// <summary>Removes a collider from its group. Returns false if the backend isn't attached.</summary>
        public static bool Clear(Collider collider)
        {
            return Set(collider, 0, 0, false);
        }

        /// <summary>
        /// Returns the collider's group, member and self-collision flag. Returns false if the collider has no group.
        /// </summary>
        public static bool TryGet(Collider collider, out uint group, out uint member, out bool selfCollides)
        {
            group = member = 0;
            selfCollides = false;
            if (collider == null || !Available)
            {
                return false;
            }

            if (!GroupTable.TryGet(NativeLibrary.RetooledPhysics_Globals(), EntityId.ToULong(collider.GetEntityId()), out GroupTable.Record record))
            {
                return false;
            }

            group = record.Group;
            member = record.Member;
            selfCollides = (record.Flags & GroupTable.SelfCollides) != 0;
            return true;
        }

        /// <summary>
        /// Enable or disable collisions between two members in the same group.
        /// Member IDs must be unique across groups. A member never collides with itself.
        /// Return false if the backend is detached.
        /// </summary>
        public static bool SetMembersCollide(uint first, uint second, bool collide)
        {
            if (!Available)
            {
                return false;
            }

            GroupTable.SetPair(NativeLibrary.RetooledPhysics_Globals(), first, second, collide);
            return true;
        }

        /// <summary>Returns whether <see cref="SetMembersCollide"/> lets two members collide.</summary>
        public static bool GetMembersCollide(uint first, uint second)
        {
            return Available && GroupTable.HasPair(NativeLibrary.RetooledPhysics_Globals(), first, second);
        }

        /// <summary>
        /// Returns whether the groups of two colliders stop them from colliding. This ignores Physics.IgnoreCollision.
        /// </summary>
        public static bool GroupsIgnore(Collider first, Collider second)
        {
            if (!TryGet(first, out uint groupA, out uint memberA, out bool selfA) || !TryGet(second, out uint groupB, out uint memberB, out bool selfB))
            {
                return false;
            }

            if (groupA != groupB)
            {
                return false;
            }

            if (memberA == memberB)
            {
                return true;
            }

            return !(selfA && selfB) && !GetMembersCollide(memberA, memberB);
        }

        /// <summary>
        /// Wake a shape's body after a group change so that contacts are reevaluated.
        /// </summary>
        private static void Wake(Shape* shape)
        {
            if (shape != null && shape->body != null && (StoreRows.Hot(shape->body)->flags & BodyFlags.Sleeping) != 0)
            {
                BodyKinematics.Wake(StoreRows.Hot(shape->body));
            }
        }

        internal static void ForgetDestroyedColliders(StoreGlobals* globals)
        {
            int capacity = (int)IgnoreTable.GroupsOf(globals)->Orphans;
            using var ids = new NativeArray<ulong>(capacity, Allocator.Temp);
            int count = GroupTable.ListOrphans(globals, (ulong*)ids.GetUnsafePtr(), capacity);
            var entities = new NativeArray<EntityId>(count, Allocator.Temp);
            var valid = new NativeArray<bool>(count, Allocator.Temp);
            for (int i = 0; i < count; i++)
            {
                entities[i] = EntityId.FromULong(ids[i]);
            }

            Resources.EntityIdsToValidArray(entities, valid);
            for (int i = 0; i < count; i++)
            {
                if (!valid[i])
                {
                    GroupTable.Set(globals, ids[i], 0, 0, 0);
                }
            }

            entities.Dispose();
            valid.Dispose();
        }
    }
}
