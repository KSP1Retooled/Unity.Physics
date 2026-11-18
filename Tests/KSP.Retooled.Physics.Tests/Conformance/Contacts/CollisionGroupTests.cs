using PhysicsScene = KSP.Retooled.Physics.Tests.Scenes.PhysicsScene;
using KSP.Retooled.Physics.Interop;
using NUnit.Framework;
using UnityEngine;
using Unity.Physics.Store.Memory;
using Unity.Physics.Store.World;
using Unity.Physics.Collision.Filter.Groups;

namespace KSP.Retooled.Physics.Tests.Conformance.Contacts
{
    /// <summary>
    /// Test collision groups by moving one box through another: collisions push it; ignored pairs leave it
    /// stationary.
    /// </summary>
    public class CollisionGroupTests : PhysicsScene
    {
        private Rigidbody moving;
        private Rigidbody standing;

        [SetUp]
        public void RequireBackend()
        {
            Assume.That(CollisionGroups.Available, "needs --backend retooled");
        }

        /// <summary>Member pairs stay in the store between tests, so remove the ones that tests add.</summary>
        [TearDown]
        public void ForgetMemberPairs()
        {
            CollisionGroups.SetMembersCollide(1, 2, false);
        }

        private Collider First => moving.GetComponent<Collider>();

        private Collider Second => standing.GetComponent<Collider>();

        private void TwoBoxes()
        {
            moving = WeightlessBox(new Vector3(-3.0f, 0.0f, 0.0f));
            standing = WeightlessBox(new Vector3(3.0f, 0.0f, 0.0f));
            moving.linearVelocity = new Vector3(5.0f, 0.0f, 0.0f);
        }

        private void AssertPassedThrough()
        {
            AssertVector(new Vector3(7.0f, 0.0f, 0.0f), moving.position, 0.05f, "moved through");
            AssertVector(new Vector3(3.0f, 0.0f, 0.0f), standing.position, 0.01f, "left alone");
        }

        private void AssertCollided()
        {
            Assert.That(standing.position.x, Is.GreaterThan(3.5f), "the second box was pushed");
        }

        [Test]
        public void TwoCollidersInOneGroupPassThrough()
        {
            TwoBoxes();
            CollisionGroups.Set(First, 7, 1, false);
            CollisionGroups.Set(Second, 7, 2, false);

            Simulate(100);

            AssertPassedThrough();
        }

        [Test]
        public void CollidersInDifferentGroupsCollide()
        {
            TwoBoxes();
            CollisionGroups.Set(First, 7, 1, false);
            CollisionGroups.Set(Second, 8, 2, false);

            Simulate(100);

            AssertCollided();
        }

        [Test]
        public void AColliderWithAGroupCollidesWithOneWithout()
        {
            TwoBoxes();
            CollisionGroups.Set(First, 7, 1, false);

            Simulate(100);

            AssertCollided();
        }

        [Test]
        public void MembersListedAsAPairCollideInsideTheirGroup()
        {
            TwoBoxes();
            CollisionGroups.Set(First, 7, 1, false);
            CollisionGroups.Set(Second, 7, 2, false);
            CollisionGroups.SetMembersCollide(2, 1, true);

            Simulate(100);

            AssertCollided();
            Assert.That(CollisionGroups.GetMembersCollide(1, 2), Is.True, "in either order");
            CollisionGroups.SetMembersCollide(1, 2, false);
            Assert.That(CollisionGroups.GetMembersCollide(1, 2), Is.False);
        }

        [Test]
        public void TwoSelfCollidingMembersOfOneGroupCollide()
        {
            TwoBoxes();
            CollisionGroups.Set(First, 7, 1, true);
            CollisionGroups.Set(Second, 7, 2, true);

            Simulate(100);

            AssertCollided();
        }

        [Test]
        public void SelfCollisionNeedsTheFlagOnBothMembers()
        {
            TwoBoxes();
            CollisionGroups.Set(First, 7, 1, true);
            CollisionGroups.Set(Second, 7, 2, false);

            Simulate(100);

            AssertPassedThrough();
        }

        [Test]
        public void TheCollidersOfOneMemberNeverCollideEvenWithSelfCollision()
        {
            TwoBoxes();
            CollisionGroups.Set(First, 7, 1, true);
            CollisionGroups.Set(Second, 7, 1, true);

            Simulate(100);

            AssertPassedThrough();
        }

        [Test]
        public void TryGetReturnsWhatWasSetUntilCleared()
        {
            TwoBoxes();
            Assert.That(CollisionGroups.TryGet(First, out _, out _, out _), Is.False, "nothing set yet");

            CollisionGroups.Set(First, 7, 3, true);
            Assert.That(CollisionGroups.TryGet(First, out uint group, out uint member, out bool selfCollides), Is.True);
            Assert.That((group, member, selfCollides), Is.EqualTo((7u, 3u, true)));

            CollisionGroups.Set(Second, 7, 4, false);
            Assert.That(CollisionGroups.GroupsIgnore(First, Second), Is.True);

            CollisionGroups.Clear(First);
            Assert.That(CollisionGroups.TryGet(First, out _, out _, out _), Is.False);
            Assert.That(CollisionGroups.GroupsIgnore(First, Second), Is.False);
        }

        [Test]
        public void SplittingAGroupCollidesOnTheNextStep()
        {
            // Two boxes of one vessel overlap. When the vessel splits, the next step pushes them apart.
            moving = WeightlessBox(new Vector3(0.0f, 0.0f, 0.0f));
            standing = WeightlessBox(new Vector3(0.5f, 0.0f, 0.0f));
            CollisionGroups.Set(First, 7, 1, false);
            CollisionGroups.Set(Second, 7, 2, false);
            Simulate(10);
            AssertVector(new Vector3(0.5f, 0.0f, 0.0f), standing.position, 0.001f, "one group, so left overlapping");

            CollisionGroups.Set(Second, 8, 2, false);
            Simulate(1);

            Assert.That(standing.position.x - moving.position.x, Is.GreaterThan(0.5f), "pushed apart in the first step");
        }

        [Test]
        public void JoiningAGroupStopsCollisionsOnTheNextStep()
        {
            // A box rests on the ground. When it docks with the ground's vessel, it falls through on the next step.
            Collider ground = Ground().GetComponent<Collider>();
            Rigidbody box = DynamicBox(new Vector3(0.0f, 0.5f, 0.0f));
            CollisionGroups.Set(ground, 7, 1, false);
            CollisionGroups.Set(box.GetComponent<Collider>(), 8, 2, false);
            Simulate(50);
            Assert.That(box.position.y, Is.EqualTo(0.5f).Within(0.02f), "different groups, so it rests");
            Assert.That(box.IsSleeping() || box.linearVelocity.magnitude < 0.01f, "at rest");

            CollisionGroups.Set(box.GetComponent<Collider>(), 7, 2, false);
            Simulate(1);
            Assert.That(box.linearVelocity.y, Is.LessThan(-0.1f), "falling in the first step");

            Simulate(40);
            Assert.That(box.position.y, Is.LessThan(-1.0f));
        }

        [Test]
        public void AnIgnoredPairStaysIgnoredInDifferentGroups()
        {
            TwoBoxes();
            CollisionGroups.Set(First, 7, 1, false);
            CollisionGroups.Set(Second, 8, 2, false);
            UnityEngine.Physics.IgnoreCollision(First, Second);

            Simulate(100);

            AssertPassedThrough();
        }

        [Test]
        public void RestoringAnIgnoredPairLeavesTheGroupRuleInPlace()
        {
            TwoBoxes();
            UnityEngine.Physics.IgnoreCollision(First, Second);
            CollisionGroups.Set(First, 7, 1, false);
            CollisionGroups.Set(Second, 7, 2, false);
            UnityEngine.Physics.IgnoreCollision(First, Second, false);

            Simulate(100);

            AssertPassedThrough();
            Assert.That(UnityEngine.Physics.GetIgnoreCollision(First, Second), Is.False, "the ignore table only knows its own pairs");
        }

        [Test]
        public void AnIgnoredPairInsideAnExceptionStaysIgnored()
        {
            TwoBoxes();
            CollisionGroups.Set(First, 7, 1, true);
            CollisionGroups.Set(Second, 7, 2, true);
            UnityEngine.Physics.IgnoreCollision(First, Second);

            Simulate(100);

            AssertPassedThrough();
        }

        [Test]
        public void OneColliderOfACompoundCanBeInAnotherGroup()
        {
            // A part with no physics of its own puts its collider on its parent's body, but it belongs to another
            // member. Here the child belongs to another group, so it lands on the ground while the root sinks.
            Collider ground = Ground().GetComponent<Collider>();
            GameObject root = Box(new Vector3(0.0f, 2.0f, 0.0f), Vector3.one);
            GameObject child = Track(new GameObject("child"));
            child.transform.SetParent(root.transform, false);
            child.transform.localPosition = new Vector3(0.0f, -1.0f, 0.0f);
            Collider lowest = child.AddComponent<BoxCollider>();
            Rigidbody body = root.AddComponent<Rigidbody>();
            body.constraints = RigidbodyConstraints.FreezeRotation;

            CollisionGroups.Set(ground, 7, 1, false);
            CollisionGroups.Set(root.GetComponent<Collider>(), 7, 2, false);
            CollisionGroups.Set(lowest, 8, 3, false);
            Simulate(200);

            Assert.That(body.position.y, Is.EqualTo(1.5f).Within(0.02f), "the child collider lands");
        }

        [Test]
        public void AGroupSurvivesDisablingTheCollider()
        {
            TwoBoxes();
            CollisionGroups.Set(First, 7, 1, false);
            CollisionGroups.Set(Second, 7, 2, false);
            Second.enabled = false;
            Simulate(1);
            Second.enabled = true;

            Simulate(99);

            AssertPassedThrough();
        }

        [Test]
        public void AGroupSetBeforeTheBodyExistsApplies()
        {
            // The collider gets its group before a Rigidbody joins it, as when a part's colliders are set up first.
            GameObject first = Box(new Vector3(-3.0f, 0.0f, 0.0f), Vector3.one);
            GameObject second = Box(new Vector3(3.0f, 0.0f, 0.0f), Vector3.one);
            CollisionGroups.Set(first.GetComponent<Collider>(), 7, 1, false);
            CollisionGroups.Set(second.GetComponent<Collider>(), 7, 2, false);
            moving = first.AddComponent<Rigidbody>();
            standing = second.AddComponent<Rigidbody>();
            moving.useGravity = standing.useGravity = false;
            moving.linearVelocity = new Vector3(5.0f, 0.0f, 0.0f);

            Simulate(100);

            AssertPassedThrough();
        }

        [Test]
        public unsafe void DestroyedCollidersAreForgotten()
        {
            TwoBoxes();
            CollisionGroups.Set(First, 7, 1, false);
            GroupTable* table = IgnoreTable.GroupsOf(NativeLibrary.RetooledPhysics_Globals());
            uint before = table->Orphans;

            Object.DestroyImmediate(First);
            Assert.That(table->Orphans, Is.EqualTo(before + 1), "the group outlives the shape for now");

            CollisionGroups.ForgetDestroyedColliders(NativeLibrary.RetooledPhysics_Globals());
            Assert.That(table->Orphans, Is.LessThanOrEqualTo(before), "and goes when the collider no longer exists");
        }

        /// <summary>
        /// Verify that Unity does not reuse destroyed collider engine IDs within a session, as required by group
        /// records.
        /// </summary>
        [Test]
        public void UnityNeverGivesADestroyedCollidersIdToANewOne()
        {
            const int count = 20000;
            var seen = new System.Collections.Generic.HashSet<ulong>();
            GameObject holder = Track(new GameObject("holder"));
            ulong first = 0, last = 0;
            for (int i = 0; i < count; i++)
            {
                BoxCollider box = holder.AddComponent<BoxCollider>();
                ulong id = EntityId.ToULong(box.GetEntityId());
                first = i == 0 ? id : first;
                last = id;
                Assert.That(seen.Add(id), Is.True, $"collider {i} on one game object has the id {id:X} of an earlier collider");
                Object.DestroyImmediate(box);
            }

            for (int i = 0; i < count; i++)
            {
                var owner = new GameObject("owner");
                BoxCollider box = owner.AddComponent<BoxCollider>();
                ulong id = EntityId.ToULong(box.GetEntityId());
                Assert.That(seen.Add(id), Is.True, $"collider {i} on its own game object has the id {id:X} of an earlier collider");
                Object.DestroyImmediate(owner);
            }

            Debug.Log($"[CollisionGroupTests] {seen.Count} collider ids, all different; on one game object from {first:X} to {last:X}");
        }

        /// <summary>
        /// Verify safe upgrade of a 64-byte ignore table with misleading bytes beyond its allocation, preserving
        /// ignored pairs.
        /// </summary>
        [Test]
        public unsafe void AnIgnoreTableFromBeforeTheGroupsGrowsAndKeepsItsPairs()
        {
            TwoBoxes();
            UnityEngine.Physics.IgnoreCollision(First, Second);
            StoreGlobals* globals = NativeLibrary.RetooledPhysics_Globals();
            IgnoreTable* current = globals->ignoreTable;

            // Seed bytes beyond the old table with a stale marker and pointer before shrinking the allocation.
            const int oldSize = IgnoreTable.OldSize;
            var block = (byte*)StoreMemory.Reallocate(globals, null, oldSize + 16);
            Unity.Collections.LowLevel.Unsafe.UnsafeUtility.MemCpy(block, current, oldSize);
            *(ulong*)(block + oldSize) = 0x3250554F52474954; // "TIGROUP2"
            *(ulong*)(block + oldSize + 8) = 0xDEADBEEF0;
            var old = (IgnoreTable*)StoreMemory.Reallocate(globals, block, oldSize);
            IgnoreTable.ClearLayoutMark(old);
            globals->ignoreTable = old;
            try
            {
                IgnoreTable.Upgrade(globals);
                Assert.That(IgnoreTable.GroupsOf(globals) == null, Is.True, "the grown table has no groups yet");
                Assert.That(UnityEngine.Physics.GetIgnoreCollision(First, Second), Is.True, "the grown table keeps its pairs");
            }
            finally
            {
                StoreMemory.Release(globals, globals->ignoreTable);
                globals->ignoreTable = current;
            }

            Assert.That(UnityEngine.Physics.GetIgnoreCollision(First, Second), Is.True);
        }

        private WheelCollider Unicycle(out Rigidbody body)
        {
            GameObject chassis = Track(new GameObject("chassis"));
            chassis.transform.position = new Vector3(0.0f, 0.95f, 0.0f);
            chassis.AddComponent<BoxCollider>().size = new Vector3(2.0f, 0.4f, 3.0f);
            body = chassis.AddComponent<Rigidbody>();
            body.mass = 100.0f;
            body.freezeRotation = true;

            GameObject holder = Track(new GameObject("wheel"));
            holder.transform.SetParent(chassis.transform, false);
            holder.transform.localPosition = new Vector3(0.0f, -0.2f, 0.0f);
            WheelCollider wheel = holder.AddComponent<WheelCollider>();
            wheel.radius = 0.5f;
            wheel.suspensionDistance = 0.3f;
            wheel.suspensionSpring = new JointSpring { spring = 35000.0f, damper = 4500.0f, targetPosition = 0.5f };
            return wheel;
        }

        [Test]
        public void AWheelPassesThroughAColliderInItsGroup()
        {
            Collider ground = Ground().GetComponent<Collider>();
            WheelCollider wheel = Unicycle(out Rigidbody body);
            CollisionGroups.Set(ground, 7, 1, false);
            CollisionGroups.Set(wheel, 7, 2, false);
            CollisionGroups.Set(body.GetComponent<Collider>(), 8, 2, false);

            Simulate(25);

            Assert.IsFalse(wheel.isGrounded);
        }

        [Test]
        public void AWheelStandsOnAColliderInAnotherGroup()
        {
            Collider ground = Ground().GetComponent<Collider>();
            WheelCollider wheel = Unicycle(out Rigidbody body);
            CollisionGroups.Set(ground, 8, 1, false);
            CollisionGroups.Set(wheel, 7, 2, false);
            CollisionGroups.Set(body.GetComponent<Collider>(), 7, 2, false);

            Simulate(25);

            Assert.IsTrue(wheel.isGrounded);
        }

        [Test]
        public void AWheelStandsOnAColliderOfAnExceptionPair()
        {
            Collider ground = Ground().GetComponent<Collider>();
            WheelCollider wheel = Unicycle(out _);
            CollisionGroups.Set(ground, 7, 1, false);
            CollisionGroups.Set(wheel, 7, 2, false);
            CollisionGroups.SetMembersCollide(1, 2, true);

            Simulate(25);

            Assert.IsTrue(wheel.isGrounded);
            CollisionGroups.SetMembersCollide(1, 2, false);
        }
    }
}
