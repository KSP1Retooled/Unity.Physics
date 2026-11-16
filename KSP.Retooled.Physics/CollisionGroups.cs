using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using UnityEngine;

namespace KSP.Retooled.Physics
{
    /// <summary>
    /// Lets the game stop collisions between the parts of one vessel without calling Physics.IgnoreCollision for every
    /// pair of colliders. The game gives each collider a group and a member: the vessel and the part, for example. Two
    /// colliders in the same group don't collide, except in two cases:
    /// <list type="bullet">
    /// <item>Both colliders allow self-collision and belong to different members. This matches the part setting
    /// sameVesselCollision.</item>
    /// <item>The game called <see cref="SetMembersCollide"/> for their two members.</item>
    /// </list>
    /// Colliders in different groups, and colliders without a group, collide as before. Pairs that the game ignores
    /// through Physics.IgnoreCollision stay ignored whatever their groups are. A change takes effect on the next
    /// simulation step.
    ///
    /// A wheel collider follows its group in the same way when it looks for ground. Physics.GetIgnoreCollision still
    /// reports only the pairs that Physics.IgnoreCollision ignored, because Unity asks the backend about a pair of
    /// ignore ids rather than a pair of colliders.
    ///
    /// Every method does nothing and returns false when the retooled backend isn't attached. The game must then fall
    /// back to Physics.IgnoreCollision. Call these methods on the main thread, outside a simulation step.
    /// </summary>
    public static unsafe class CollisionGroups
    {
        /// <summary>
        /// Once this many colliders have a group but no shape, the next call to <see cref="Set"/> forgets the groups of
        /// colliders that have been destroyed. A wheel collider has no shape, so it always counts. After each check,
        /// the next check waits until the count doubles, so that the checks cost little per call.
        /// </summary>
        private const uint OrphanCheckThreshold = 4096;

        private static uint nextOrphanCheck = OrphanCheckThreshold;

        /// <summary>Whether the methods of this class have an effect.</summary>
        public static bool Available => BackendHost.IsAttached;

        /// <summary>
        /// Puts a collider in a group. A group of zero removes the collider from its group. The member and the
        /// self-collision flag matter only for colliders in the same group. Returns false if the backend isn't
        /// attached.
        /// </summary>
        public static bool Set(Collider collider, uint group, uint member, bool selfCollides)
        {
            if (collider == null || !Available)
            {
                return false;
            }

            NativeStore.StoreGlobals* globals = NativeStore.RetooledPhysics_Globals();
            ulong engineId = EntityId.ToULong(collider.GetEntityId());
            NativeStore.Shape* shape = GroupTable.Set(globals, engineId, group, member, selfCollides ? GroupTable.SelfCollides : 0u);
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

            if (!GroupTable.TryGet(NativeStore.RetooledPhysics_Globals(), EntityId.ToULong(collider.GetEntityId()), out GroupTable.Record record))
            {
                return false;
            }

            group = record.Group;
            member = record.Member;
            selfCollides = (record.Flags & GroupTable.SelfCollides) != 0;
            return true;
        }

        /// <summary>
        /// Lets the colliders of two members collide when they are in the same group, or stops them again. Members are
        /// matched by value alone, so the game must use member values that are unique across groups, such as part
        /// persistent ids. A member never collides with itself. Returns false if the backend isn't attached.
        /// </summary>
        public static bool SetMembersCollide(uint first, uint second, bool collide)
        {
            if (!Available)
            {
                return false;
            }

            GroupTable.SetPair(NativeStore.RetooledPhysics_Globals(), first, second, collide);
            return true;
        }

        /// <summary>Returns whether <see cref="SetMembersCollide"/> lets two members collide.</summary>
        public static bool GetMembersCollide(uint first, uint second)
        {
            return Available && GroupTable.HasPair(NativeStore.RetooledPhysics_Globals(), first, second);
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
        /// Wakes the body of a shape whose group changed, so that a sleeping body notices that it now touches a shape
        /// that it ignored before, or stops touching one.
        /// </summary>
        private static void Wake(NativeStore.Shape* shape)
        {
            if (shape != null && shape->body != null && (StoreOps.Hot(shape->body)->flags & NativeStore.BodyFlags.Sleeping) != 0)
            {
                JointKernel.WakeRow(StoreOps.Hot(shape->body));
            }
        }

        internal static void ForgetDestroyedColliders(NativeStore.StoreGlobals* globals)
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
