using System.Collections.Generic;
using NUnit.Framework;
using UnityEngine;

namespace KSP.Retooled.Physics.Tests
{
    /// <summary>The collision and trigger messages, as a script sees them.</summary>
    public class CollisionEventConformanceTests : PhysicsScene
    {
        /// <summary>Records collision messages sent to a collider or body's GameObject, in order.</summary>
        [ExecuteAlways]
        internal class CollisionLog : MonoBehaviour
        {
            public readonly List<string> Messages = new List<string>();
            public readonly List<Collision> Enters = new List<Collision>();
            public readonly List<Collision> Stays = new List<Collision>();
            public readonly List<Collision> Exits = new List<Collision>();
            public bool DeactivateOnEnter;

            public int Count(string message)
            {
                return Messages.FindAll(m => m == message).Count;
            }

            private void OnCollisionEnter(Collision collision)
            {
                Messages.Add("Enter");
                Enters.Add(collision);
                if (DeactivateOnEnter)
                {
                    gameObject.SetActive(false);
                }
            }

            private void OnCollisionStay(Collision collision)
            {
                Messages.Add("Stay");
                Stays.Add(collision);
            }

            private void OnCollisionExit(Collision collision)
            {
                Messages.Add("Exit");
                Exits.Add(collision);
            }
        }

        /// <summary>Handles only collision entry messages, so Unity requests only those notifications.</summary>
        [ExecuteAlways]
        internal class EnterOnlyLog : MonoBehaviour
        {
            public int Enters;

            private void OnCollisionEnter(Collision collision)
            {
                Enters++;
            }
        }

        [ExecuteAlways]
        internal class TriggerLog : MonoBehaviour
        {
            public readonly List<Collider> Enters = new List<Collider>();
            public readonly List<Collider> Stays = new List<Collider>();
            public readonly List<Collider> Exits = new List<Collider>();

            private void OnTriggerEnter(Collider other)
            {
                Enters.Add(other);
            }

            private void OnTriggerStay(Collider other)
            {
                Stays.Add(other);
            }

            private void OnTriggerExit(Collider other)
            {
                Exits.Add(other);
            }
        }

        private bool reuseCallbacks;

        [SetUp]
        public void KeepCollisions()
        {
            reuseCallbacks = UnityEngine.Physics.reuseCollisionCallbacks;
            UnityEngine.Physics.reuseCollisionCallbacks = false;
        }

        [TearDown]
        public void RestoreReuse()
        {
            UnityEngine.Physics.reuseCollisionCallbacks = reuseCallbacks;
        }

        private CollisionLog FallingBox(Vector3 position, out Rigidbody body)
        {
            body = DynamicBox(position);
            return body.gameObject.AddComponent<CollisionLog>();
        }

        [Test]
        public void ABodyLandingEntersOnceThenStays()
        {
            Ground();
            CollisionLog log = FallingBox(new Vector3(0.0f, 1.0f, 0.0f), out _);

            Simulate(60);

            Assert.That(log.Count("Enter"), Is.EqualTo(1));
            Assert.That(log.Count("Stay"), Is.GreaterThan(10));
            Assert.That(log.Messages[0], Is.EqualTo("Enter"));
            Assert.That(log.Count("Exit"), Is.Zero);
        }

        [Test]
        public void TheGroundHearsAboutTheBodyToo()
        {
            CollisionLog ground = Ground().AddComponent<CollisionLog>();
            Rigidbody body = DynamicBox(new Vector3(0.0f, 1.0f, 0.0f));

            Simulate(60);

            Assert.That(ground.Count("Enter"), Is.EqualTo(1));
            Collision collision = ground.Enters[0];
            Assert.That(collision.rigidbody, Is.EqualTo(body));
            Assert.That(collision.collider, Is.EqualTo(body.GetComponent<Collider>()));
            Assert.That(collision.thisCollider, Is.EqualTo(ground.GetComponent<Collider>()));
            Assert.That(collision.gameObject, Is.EqualTo(body.gameObject));
        }

        [Test]
        public void ACollisionNamesBothSidesAndPointsItsNormalAtTheReceiver()
        {
            GameObject ground = Ground();
            CollisionLog log = FallingBox(new Vector3(0.0f, 1.0f, 0.0f), out Rigidbody body);

            Simulate(60);

            Collision collision = log.Enters[0];
            Assert.That(collision.rigidbody, Is.Null, "the ground has no body");
            Assert.That(collision.thisRigidbody, Is.EqualTo(body));
            Assert.That(collision.collider, Is.EqualTo(ground.GetComponent<Collider>()));
            Assert.That(collision.gameObject, Is.EqualTo(ground));
            Assert.That(collision.contactCount, Is.EqualTo(4), "a box lying flat touches at its corners");

            foreach (ContactPoint contact in collision.contacts)
            {
                AssertVector(Vector3.up, contact.normal, 0.01f, "normal");
                Assert.That(contact.point.y, Is.EqualTo(0.0f).Within(0.05f));
                Assert.That(Mathf.Abs(contact.point.x), Is.EqualTo(0.5f).Within(0.05f));
                Assert.That(Mathf.Abs(contact.point.z), Is.EqualTo(0.5f).Within(0.05f));
                Assert.That(contact.thisCollider, Is.EqualTo(body.GetComponent<Collider>()));
                Assert.That(contact.otherCollider, Is.EqualTo(ground.GetComponent<Collider>()));
            }
        }

        [Test]
        public void TheOtherSideSeesTheNormalReversed()
        {
            CollisionLog ground = Ground().AddComponent<CollisionLog>();
            DynamicBox(new Vector3(0.0f, 1.0f, 0.0f));

            Simulate(60);

            AssertVector(Vector3.down, ground.Enters[0].GetContact(0).normal, 0.01f, "normal");
        }

        [Test]
        public void TheFirstContactCarriesTheImpactsImpulse()
        {
            Ground();
            CollisionLog log = FallingBox(new Vector3(0.0f, 2.5f, 0.0f), out Rigidbody body);

            float speed = 0.0f;
            for (int i = 0; i < 200 && log.Count("Enter") == 0; i++)
            {
                speed = body.linearVelocity.magnitude;
                Simulate();
            }

            Collision collision = log.Enters[0];
            Assert.That(collision.impulse.magnitude, Is.EqualTo(speed * body.mass).Within(speed * 0.25f));
            AssertVector(Vector3.up, collision.impulse.normalized, 0.01f, "impulse direction");

            Vector3 sum = Vector3.zero;
            foreach (ContactPoint contact in collision.contacts)
            {
                sum += contact.impulse;
            }

            AssertVector(collision.impulse, sum, 0.01f, "points sum to the pair");
        }

        [Test]
        public void ARestingContactCarriesItsWeightEachStep()
        {
            Ground();
            CollisionLog log = FallingBox(new Vector3(0.0f, 0.51f, 0.0f), out Rigidbody body);
            body.sleepThreshold = 0.0f;

            Simulate(30);
            log.Stays.Clear();
            Simulate(5);

            float weightPerStep = body.mass * -Gravity * Step;
            Assert.That(log.Stays[log.Stays.Count - 1].impulse.magnitude, Is.EqualTo(weightPerStep).Within(weightPerStep * 0.1f));
        }

        [Test]
        public void ARestingContactNeverDriftsOffItsWeight()
        {
            Ground();
            CollisionLog log = FallingBox(new Vector3(0.0f, 0.51f, 0.0f), out Rigidbody body);
            body.sleepThreshold = 0.0f;

            Simulate(100);
            log.Stays.Clear();
            Simulate(50);

            float weightPerStep = body.mass * -Gravity * Step;
            Assert.That(log.Stays.Count, Is.EqualTo(50));
            foreach (Collision stay in log.Stays)
            {
                Assert.That(stay.impulse.magnitude, Is.EqualTo(weightPerStep).Within(weightPerStep * 0.02f));
            }
        }

        [Test]
        public void RelativeVelocityIsTheOtherMinusThisAtImpact()
        {
            Ground();
            CollisionLog log = FallingBox(new Vector3(0.0f, 2.5f, 0.0f), out Rigidbody body);

            float speed = 0.0f;
            for (int i = 0; i < 200 && log.Count("Enter") == 0; i++)
            {
                speed = body.linearVelocity.magnitude;
                Simulate();
            }

            Assert.That(speed, Is.GreaterThan(5.0f));
            AssertVector(Vector3.up * speed, log.Enters[0].relativeVelocity, 0.5f, "relative velocity");
        }

        [Test]
        public void TwoBodiesBothHearAndDisagreeOnRelativeVelocity()
        {
            CollisionLog first = FallingBox(new Vector3(-2.0f, 0.0f, 0.0f), out Rigidbody left);
            CollisionLog second = FallingBox(new Vector3(2.0f, 0.0f, 0.0f), out Rigidbody right);
            left.useGravity = false;
            right.useGravity = false;
            left.linearVelocity = new Vector3(4.0f, 0.0f, 0.0f);

            Simulate(100);

            Assert.That(first.Count("Enter"), Is.EqualTo(1));
            Assert.That(second.Count("Enter"), Is.EqualTo(1));
            Assert.That(first.Enters[0].rigidbody, Is.EqualTo(right));
            Assert.That(second.Enters[0].rigidbody, Is.EqualTo(left));
            AssertVector(first.Enters[0].relativeVelocity, -second.Enters[0].relativeVelocity, 0.001f, "opposite views");
            AssertVector(Vector3.right, first.Enters[0].GetContact(0).normal * -1.0f, 0.01f, "first's normal points back at it");
        }

        [Test]
        public void SeparatingExits()
        {
            GameObject ground = Ground();
            CollisionLog log = FallingBox(new Vector3(0.0f, 0.6f, 0.0f), out Rigidbody body);

            Simulate(30);
            Assert.That(log.Count("Enter"), Is.EqualTo(1));

            body.linearVelocity = new Vector3(0.0f, 8.0f, 0.0f);
            Simulate(10);

            Assert.That(log.Count("Exit"), Is.EqualTo(1));
            Assert.That(log.Exits[0].collider, Is.EqualTo(ground.GetComponent<Collider>()));
            Assert.That(log.Exits[0].contactCount, Is.Zero);
            int stays = log.Count("Stay");
            Simulate(5);
            Assert.That(log.Count("Stay"), Is.EqualTo(stays), "no stay after the exit");
        }

        [Test]
        public void IgnoringATouchingPairExits()
        {
            Collider ground = Ground().GetComponent<Collider>();
            CollisionLog log = FallingBox(new Vector3(0.0f, 0.6f, 0.0f), out Rigidbody body);

            Simulate(30);
            UnityEngine.Physics.IgnoreCollision(body.GetComponent<Collider>(), ground);
            Simulate(5);

            Assert.That(log.Count("Exit"), Is.EqualTo(1));
        }

        [Test]
        public void AnIgnoredPairSendsNothing()
        {
            Collider ground = Ground().GetComponent<Collider>();
            CollisionLog log = FallingBox(new Vector3(0.0f, 1.0f, 0.0f), out Rigidbody body);

            UnityEngine.Physics.IgnoreCollision(body.GetComponent<Collider>(), ground);
            Simulate(60);

            Assert.That(log.Messages, Is.Empty);
        }

        [Test]
        public void ABodyHearsEachChildOfItsCompoundWithTheRightCollider()
        {
            GameObject ground = Ground();
            GameObject root = Box(new Vector3(0.0f, 1.0f, 0.0f), Vector3.one);
            GameObject child = Track(new GameObject("child"));
            child.transform.SetParent(root.transform, false);
            child.transform.localPosition = new Vector3(3.0f, -0.25f, 0.0f);
            child.AddComponent<BoxCollider>().size = new Vector3(1.0f, 1.5f, 1.0f);
            Rigidbody body = root.AddComponent<Rigidbody>();
            body.constraints = RigidbodyConstraints.FreezeRotation;
            CollisionLog log = root.AddComponent<CollisionLog>();

            Simulate(60);

            Assert.That(log.Enters, Is.Not.Empty);
            Collision first = log.Enters[0];
            Assert.That(first.GetContact(0).thisCollider, Is.EqualTo(child.GetComponent<Collider>()), "the child is lower, so it lands");
            Assert.That(first.collider, Is.EqualTo(ground.GetComponent<Collider>()));
            Assert.That(first.rigidbody, Is.Null);
            Assert.That(first.thisRigidbody, Is.EqualTo(body));
            Assert.That(first.gameObject, Is.EqualTo(ground));
        }

        [Test]
        public void AChildColliderMessagesTheBodysObjectAlone()
        {
            Ground();
            GameObject root = Box(new Vector3(0.0f, 1.0f, 0.0f), Vector3.one);
            GameObject child = Track(new GameObject("child"));
            child.transform.SetParent(root.transform, false);
            child.transform.localPosition = new Vector3(0.0f, -0.25f, 0.0f);
            child.AddComponent<BoxCollider>();
            CollisionLog rootLog = root.AddComponent<CollisionLog>();
            CollisionLog childLog = child.AddComponent<CollisionLog>();
            root.AddComponent<Rigidbody>();

            Simulate(60);

            Assert.That(rootLog.Enters, Is.Not.Empty);
            Assert.That(childLog.Enters, Is.Empty);
        }

        [Test]
        public void OnlyTheMessagesAScriptTakesAreSent()
        {
            GameObject ground = Ground();
            Rigidbody body = DynamicBox(new Vector3(0.0f, 1.0f, 0.0f));
            EnterOnlyLog log = body.gameObject.AddComponent<EnterOnlyLog>();

            Simulate(60);

            Assert.That(log.Enters, Is.EqualTo(1));
            Assert.That(ground, Is.Not.Null);
        }

        [Test]
        public void KinematicAgainstStaticSendsNothing()
        {
            Ground();
            Rigidbody body = DynamicBox(new Vector3(0.0f, 0.4f, 0.0f));
            body.isKinematic = true;
            CollisionLog log = body.gameObject.AddComponent<CollisionLog>();

            Simulate(20);

            Assert.That(log.Messages, Is.Empty);
        }

        [Test]
        public void ABodyDeactivatedInItsCallbackLeavesTheStepFine()
        {
            CollisionLog ground = Ground().AddComponent<CollisionLog>();
            CollisionLog log = FallingBox(new Vector3(0.0f, 1.0f, 0.0f), out _);
            CollisionLog other = FallingBox(new Vector3(3.0f, 1.0f, 0.0f), out Rigidbody survivor);
            log.DeactivateOnEnter = true;

            Simulate(100);

            Assert.That(log.gameObject.activeSelf, Is.False);
            Assert.That(other.Count("Enter"), Is.EqualTo(1));
            Assert.That(survivor.position.y, Is.EqualTo(0.5f).Within(0.02f));
            Assert.That(ground.Count("Exit"), Is.Zero, "a body gone mid-touch sends no exit");
        }

        [Test]
        public void DisablingATouchingColliderSendsNoExit()
        {
            CollisionLog ground = Ground().AddComponent<CollisionLog>();
            CollisionLog log = FallingBox(new Vector3(0.0f, 0.6f, 0.0f), out Rigidbody body);

            Simulate(30);
            body.GetComponent<Collider>().enabled = false;
            Simulate(5);

            Assert.That(log.Count("Exit"), Is.Zero);
            Assert.That(ground.Count("Exit"), Is.Zero);

            body.GetComponent<Collider>().enabled = true;
            body.position = new Vector3(0.0f, 0.6f, 0.0f);
            body.linearVelocity = Vector3.zero;
            Simulate(30);

            Assert.That(log.Count("Enter"), Is.EqualTo(2), "enabled again, the pair starts over");
        }

        private GameObject Trigger(Vector3 position, Vector3 size)
        {
            GameObject trigger = Box(position, size);
            trigger.GetComponent<Collider>().isTrigger = true;
            return trigger;
        }

        [Test]
        public void ABodyFallingThroughATriggerEntersStaysAndExits()
        {
            TriggerLog zone = Trigger(new Vector3(0.0f, 0.0f, 0.0f), new Vector3(4.0f, 2.0f, 4.0f)).AddComponent<TriggerLog>();
            Rigidbody body = DynamicBox(new Vector3(0.0f, 3.0f, 0.0f));
            TriggerLog visitor = body.gameObject.AddComponent<TriggerLog>();

            Simulate(100);

            Assert.That(zone.Enters.Count, Is.EqualTo(1));
            Assert.That(zone.Enters[0], Is.EqualTo(body.GetComponent<Collider>()));
            Assert.That(zone.Stays.Count, Is.GreaterThan(3));
            Assert.That(zone.Exits.Count, Is.EqualTo(1));
            Assert.That(visitor.Enters.Count, Is.EqualTo(1));
            Assert.That(visitor.Enters[0], Is.EqualTo(zone.GetComponent<Collider>()));
            Assert.That(visitor.Exits.Count, Is.EqualTo(1));
        }

        [Test]
        public void ATriggerChildOfACompoundReportsItselfAndDoesNotPush()
        {
            Ground();
            GameObject root = Box(new Vector3(0.0f, 1.0f, 0.0f), Vector3.one);
            GameObject child = Track(new GameObject("sensor"));
            child.transform.SetParent(root.transform, false);
            child.transform.localPosition = new Vector3(0.0f, -1.0f, 0.0f);
            BoxCollider sensor = child.AddComponent<BoxCollider>();
            sensor.isTrigger = true;
            TriggerLog sensorLog = child.AddComponent<TriggerLog>();
            Rigidbody body = root.AddComponent<Rigidbody>();
            body.constraints = RigidbodyConstraints.FreezeRotation;
            TriggerLog rootLog = root.AddComponent<TriggerLog>();

            Simulate(60);

            Assert.That(body.position.y, Is.EqualTo(0.5f).Within(0.02f), "the trigger child doesn't hold the body up");
            Assert.That(sensorLog.Enters.Count, Is.EqualTo(1));
            Assert.That(rootLog.Enters.Count, Is.EqualTo(1), "the body's object hears its child's trigger too");
        }

        private Rigidbody[] JoinedPair(Vector3 position)
        {
            Rigidbody lower = DynamicBox(position);
            Rigidbody upper = DynamicBox(position + Vector3.up);
            upper.gameObject.AddComponent<FixedJoint>().connectedBody = lower;
            return new[] { lower, upper };
        }

        [Test]
        public void BodiesHeldByAJointHearATrigger()
        {
            TriggerLog zone = Trigger(Vector3.zero, new Vector3(4.0f, 2.0f, 4.0f)).AddComponent<TriggerLog>();
            Rigidbody[] pair = JoinedPair(new Vector3(0.0f, 3.0f, 0.0f));
            TriggerLog visitor = pair[0].gameObject.AddComponent<TriggerLog>();

            Simulate(100);

            Assert.That(zone.Enters.Count, Is.EqualTo(2));
            Assert.That(zone.Exits.Count, Is.EqualTo(2));
            Assert.That(visitor.Enters.Count, Is.EqualTo(1));
        }

        [Test]
        public void BodiesHeldByAJointLandInsideATrigger()
        {
            Ground();
            Trigger(new Vector3(0.0f, 2.0f, 0.0f), new Vector3(8.0f, 8.0f, 8.0f));
            Rigidbody[] pair = JoinedPair(new Vector3(0.0f, 3.0f, 0.0f));

            Simulate(150);

            Assert.That(pair[0].position.y, Is.EqualTo(0.5f).Within(0.02f));
            Assert.That(pair[1].position.y, Is.EqualTo(1.5f).Within(0.02f));
            Assert.That(pair[0].linearVelocity.magnitude, Is.LessThan(0.05f));
        }

        [Test]
        public void AnIgnoredPairDoesNotTrigger()
        {
            GameObject trigger = Trigger(Vector3.zero, new Vector3(4.0f, 2.0f, 4.0f));
            TriggerLog zone = trigger.AddComponent<TriggerLog>();
            Rigidbody body = DynamicBox(new Vector3(0.0f, 3.0f, 0.0f));

            UnityEngine.Physics.IgnoreCollision(body.GetComponent<Collider>(), trigger.GetComponent<Collider>());
            Simulate(100);

            Assert.That(zone.Enters, Is.Empty);
        }

        [Test]
        public void DisablingAColliderInsideATriggerEndsTheStays()
        {
            TriggerLog zone = Trigger(Vector3.zero, new Vector3(4.0f, 4.0f, 4.0f)).AddComponent<TriggerLog>();
            Rigidbody body = WeightlessBox(Vector3.zero);

            Simulate(5);
            Assert.That(zone.Enters.Count, Is.EqualTo(1));

            body.GetComponent<Collider>().enabled = false;
            Simulate(3);
            int stays = zone.Stays.Count;
            Simulate(5);

            Assert.That(zone.Stays.Count, Is.EqualTo(stays));
        }

        /// <summary>
        /// Unity keeps each trigger pair that entered, and it sends OnTriggerStay for the pair in every step until an
        /// event ends the pair. When a collider is disabled inside a trigger, the backend has to end the pair with the
        /// event for a removed shape, as PhysX does. Otherwise Unity keeps the pair, and when the collider is enabled
        /// again far from the trigger, Unity starts sending OnTriggerStay for it again.
        /// </summary>
        [Test]
        public void AColliderEnabledAgainOutsideATriggerGetsNoStays()
        {
            TriggerLog zone = Trigger(Vector3.zero, new Vector3(4.0f, 4.0f, 4.0f)).AddComponent<TriggerLog>();
            Rigidbody body = WeightlessBox(Vector3.zero);
            TriggerLog visitor = body.gameObject.AddComponent<TriggerLog>();

            Simulate(3);
            Assert.That(zone.Enters.Count, Is.EqualTo(1));
            Assert.That(zone.Stays.Count, Is.GreaterThan(0));

            body.GetComponent<Collider>().enabled = false;
            Simulate(2);
            body.position = new Vector3(20.0f, 0.0f, 0.0f);
            body.GetComponent<Collider>().enabled = true;
            int zoneStays = zone.Stays.Count;
            int visitorStays = visitor.Stays.Count;
            Simulate(5);

            Assert.That(zone.Stays.Count, Is.EqualTo(zoneStays), "the trigger hears no stays from a collider outside it");
            Assert.That(visitor.Stays.Count, Is.EqualTo(visitorStays), "the collider hears no stays from a trigger far away");
            Assert.That(zone.Enters.Count, Is.EqualTo(1));
            Assert.That(zone.Exits, Is.Empty, "a disabled collider sends no exit");
        }

        /// <summary>
        /// A new trigger pair whose two shapes took the addresses of a pair that ended because its shapes were
        /// destroyed. Unity keeps the trigger pairs by the addresses of their shapes, and it adds an entering pair only
        /// if it has no pair at those addresses. If the backend didn't end the old pair, Unity kept the old pair with
        /// the ids of the destroyed colliders, and the new pair got one OnTriggerStay and then no more. The store frees
        /// the memory of destroyed shapes, but the heap doesn't hand the same addresses back in any fixed order, so the
        /// test repeats the scene until the addresses were reused a few times.
        /// </summary>
        [Test]
        public unsafe void ANewTriggerPairAtTheAddressesOfADestroyedPairKeepsItsStays()
        {
            const int wantedReuses = 3;
            int reuses = 0;
            for (int iteration = 0; iteration < 200 && reuses < wantedReuses; iteration++)
            {
                GameObject oldZone = Trigger(Vector3.zero, new Vector3(4.0f, 2.0f, 4.0f));
                Rigidbody oldBody = WeightlessBox(Vector3.zero);
                Simulate(3);
                (long, long) oldShapes = (ShapeAddress(oldZone), ShapeAddress(oldBody.gameObject));
                DestroyAll();

                TriggerLog zone = Trigger(Vector3.zero, new Vector3(4.0f, 2.0f, 4.0f)).AddComponent<TriggerLog>();
                Rigidbody body = DynamicBox(new Vector3(0.0f, 3.0f, 0.0f));
                Simulate();
                bool reused = (ShapeAddress(zone.gameObject), ShapeAddress(body.gameObject)) == oldShapes;
                Simulate(100);

                Assert.That(zone.Enters.Count, Is.EqualTo(1), $"iteration {iteration}, addresses reused: {reused}");
                Assert.That(zone.Stays.Count, Is.GreaterThan(3), $"iteration {iteration}, addresses reused: {reused}");
                Assert.That(zone.Exits.Count, Is.EqualTo(1), $"iteration {iteration}, addresses reused: {reused}");
                reuses += reused ? 1 : 0;
                DestroyAll();
            }

            Assert.That(reuses, Is.EqualTo(wantedReuses), "the new shapes took the old shapes' addresses often enough");
        }

        /// <summary>The address of the backend's shape for the collider on an object, which Unity knows the shape by.</summary>
        private static unsafe long ShapeAddress(GameObject gameObject)
        {
            NativeStore.StoreGlobals* globals = NativeStore.RetooledPhysics_Globals();
            NativeStore.WorldData* store = StepKernel.Current != null ? StepKernel.Current->Store : null;
            if (globals == null || store == null)
            {
                Assert.Ignore("The test reads the shapes of the Retooled backend's store.");
            }

            ulong id = EntityId.ToULong(gameObject.GetComponent<Collider>().GetEntityId());
            for (uint row = 0; row < store->bodyCount; row++)
            {
                NativeStore.Body* owner = store->bodies[row].owner;
                for (uint i = 0; owner != null && i < owner->shapeCount; i++)
                {
                    NativeStore.ShapeData* shape = owner->shapes[i];
                    if (shape->userData != null && *(ulong*)((byte*)shape->userData + globals->engineObjectIdOffset) == id)
                    {
                        return (long)shape->owner;
                    }
                }
            }

            return 0;
        }

        [Test]
        public void AStaticTriggerAndAStaticColliderStaySilent()
        {
            TriggerLog zone = Trigger(Vector3.zero, Vector3.one).AddComponent<TriggerLog>();
            Box(Vector3.zero, Vector3.one);

            Simulate(5);

            Assert.That(zone.Enters, Is.Empty);
        }

        [Test]
        public void AKinematicBodyInAStaticTriggerEnters()
        {
            TriggerLog zone = Trigger(Vector3.zero, new Vector3(2.0f, 2.0f, 2.0f)).AddComponent<TriggerLog>();
            Rigidbody body = DynamicBox(new Vector3(0.0f, 5.0f, 0.0f));
            body.isKinematic = true;

            Simulate(2);
            body.MovePosition(Vector3.zero);
            Simulate(3);

            Assert.That(zone.Enters.Count, Is.EqualTo(1));
        }
    }
}
