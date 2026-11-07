using NUnit.Framework;
using UnityEngine;

namespace KSP.Retooled.Physics.Tests
{
    /// <summary>Automatic sleep: when bodies go to sleep, what wakes them, and what a sleeping body still does.</summary>
    public class SleepConformanceTests : PhysicsScene
    {
        private static int StepsUntilAsleep(Rigidbody body, int limit)
        {
            for (int i = 1; i <= limit; i++)
            {
                Simulate();
                if (body.IsSleeping())
                {
                    return i;
                }
            }

            return -1;
        }

        [Test]
        public void ABodyAtRestOnTheGroundFallsAsleep()
        {
            Ground();
            Rigidbody body = DynamicBox(new Vector3(0.0f, 0.5f, 0.0f));

            Assert.That(StepsUntilAsleep(body, 150), Is.GreaterThan(0));
            Assert.That(body.linearVelocity, Is.EqualTo(Vector3.zero));
            Assert.That(body.angularVelocity, Is.EqualTo(Vector3.zero));
        }

        [Test]
        public void AStillBodyInFreeSpaceSleepsAfterTheWakeTimer()
        {
            Rigidbody body = WeightlessBox(Vector3.zero);

            int steps = StepsUntilAsleep(body, 100);

            Assert.That(steps, Is.InRange(15, 25), "0.4 s of stillness at 0.02 s a step");
        }

        [Test]
        public void AFallingBodyStaysAwake()
        {
            Rigidbody body = DynamicBox(new Vector3(0.0f, 100.0f, 0.0f));

            Assert.That(StepsUntilAsleep(body, 100), Is.EqualTo(-1));
        }

        [Test]
        public void ABodyCoastingAboveItsThresholdStaysAwake()
        {
            Rigidbody body = WeightlessBox(Vector3.zero);
            body.linearVelocity = new Vector3(0.2f, 0.0f, 0.0f);

            Assert.That(StepsUntilAsleep(body, 100), Is.EqualTo(-1));
        }

        [Test]
        public void ASteadyDriftBelowTheThresholdStillKeepsABodyAwake()
        {
            Rigidbody body = WeightlessBox(Vector3.zero);
            body.sleepThreshold = 1.0f;
            body.linearVelocity = new Vector3(0.5f, 0.0f, 0.0f);

            Assert.That(StepsUntilAsleep(body, 100), Is.EqualTo(-1), "the energy of one step is 0.125, but velocities add up");
        }

        [Test]
        public void SpinInertiaCountsTowardsTheEnergy()
        {
            Rigidbody body = WeightlessBox(Vector3.zero);
            body.angularDamping = 0.0f;
            body.angularVelocity = new Vector3(0.0f, 0.5f, 0.0f);

            Assert.That(StepsUntilAsleep(body, 100), Is.EqualTo(-1), "0.5 * I * w^2 / m is about 0.02");
        }

        [Test]
        public void AThresholdOfZeroKeepsABodyAwake()
        {
            Rigidbody body = WeightlessBox(Vector3.zero);
            body.sleepThreshold = 0.0f;

            Assert.That(StepsUntilAsleep(body, 100), Is.EqualTo(-1));
        }

        [Test]
        public void AForceWakesASleepingBody()
        {
            Rigidbody body = WeightlessBox(Vector3.zero);
            body.Sleep();

            body.AddForce(new Vector3(10.0f, 0.0f, 0.0f));
            Assert.That(body.IsSleeping(), Is.False);

            Simulate(5);
            Assert.That(body.position.x, Is.GreaterThan(0.0f));
        }

        [Test]
        public void ATorqueWakesASleepingBody()
        {
            Rigidbody body = WeightlessBox(Vector3.zero);
            body.Sleep();

            body.AddTorque(new Vector3(10.0f, 0.0f, 0.0f));
            Simulate(5);

            Assert.That(body.IsSleeping(), Is.False);
            Assert.That(body.angularVelocity.magnitude, Is.GreaterThan(0.0f));
        }

        [Test]
        public void SettingAVelocityWakesASleepingBody()
        {
            Rigidbody body = WeightlessBox(Vector3.zero);
            body.Sleep();

            body.linearVelocity = new Vector3(1.0f, 0.0f, 0.0f);
            Simulate(5);

            Assert.That(body.IsSleeping(), Is.False);
            Assert.That(body.position.x, Is.EqualTo(0.1f).Within(0.01f));
        }

        [Test]
        public void SettingAnAngularVelocityWakesASleepingBody()
        {
            Rigidbody body = WeightlessBox(Vector3.zero);
            body.Sleep();

            body.angularVelocity = new Vector3(1.0f, 0.0f, 0.0f);
            Simulate(5);

            Assert.That(body.IsSleeping(), Is.False);
            Assert.That(body.rotation.eulerAngles.magnitude, Is.GreaterThan(1.0f));
        }

        [Test]
        public void ABodyStruckWhileAsleepTakesTheHit()
        {
            Rigidbody sleeper = WeightlessBox(Vector3.zero);
            Rigidbody striker = WeightlessBox(new Vector3(-3.0f, 0.0f, 0.0f));
            sleeper.Sleep();
            striker.linearVelocity = new Vector3(5.0f, 0.0f, 0.0f);

            Simulate(60);

            Assert.That(sleeper.IsSleeping(), Is.False);
            Vector3 momentum = sleeper.linearVelocity + striker.linearVelocity;
            AssertVector(new Vector3(5.0f, 0.0f, 0.0f), momentum, 0.25f, "momentum carries over");
            Assert.That(sleeper.linearVelocity.x, Is.EqualTo(striker.linearVelocity.x).Within(0.5f), "they move off together, without a bounce");
        }

        [Test]
        public void AStackAtRestSleepsTogether()
        {
            Ground();
            Rigidbody bottom = DynamicBox(new Vector3(0.0f, 0.5f, 0.0f));
            Rigidbody top = DynamicBox(new Vector3(0.0f, 1.5f, 0.0f));

            Assert.That(StepsUntilAsleep(top, 150), Is.GreaterThan(0));
            Assert.That(bottom.IsSleeping(), Is.True, "they sleep together");
        }

        [Test]
        public void AHitReachesAStackAsleep()
        {
            Ground();
            Rigidbody bottom = DynamicBox(new Vector3(0.0f, 0.5f, 0.0f));
            Rigidbody top = DynamicBox(new Vector3(0.0f, 1.5f, 0.0f));

            Simulate(30);
            bottom.Sleep();
            top.Sleep();
            Assert.That(top.IsSleeping(), Is.True);

            Rigidbody striker = WeightlessBox(new Vector3(-3.0f, 0.5f, 0.0f), 5.0f);
            striker.linearVelocity = new Vector3(10.0f, 0.0f, 0.0f);
            Simulate(40);

            Assert.That(bottom.position.x, Is.GreaterThan(0.2f), "the box struck moves");
            Assert.That(top.IsSleeping(), Is.False, "and what stands on it wakes");
        }

        [Test]
        public void TwoBodiesInAPileSleepOnlyWhenBothAreStill()
        {
            Ground();
            Rigidbody lower = DynamicBox(new Vector3(0.0f, 0.5f, 0.0f));
            Rigidbody upper = DynamicBox(new Vector3(0.0f, 1.5f, 0.0f));
            upper.sleepThreshold = 0.0f;

            Simulate(100);

            Assert.That(lower.IsSleeping(), Is.False, "the one above never lets its island sleep");
        }

        [Test]
        public void AKinematicMovedIntoASleepingBodyWakesAndPushesIt()
        {
            Rigidbody sleeper = WeightlessBox(Vector3.zero);
            Rigidbody pusher = DynamicBox(new Vector3(-1.2f, 0.0f, 0.0f));
            pusher.isKinematic = true;
            sleeper.Sleep();

            for (int i = 0; i < 20; i++)
            {
                pusher.MovePosition(pusher.position + new Vector3(0.05f, 0.0f, 0.0f));
                Simulate();
            }

            Assert.That(sleeper.IsSleeping(), Is.False);
            Assert.That(sleeper.position.x, Is.GreaterThan(0.1f));
        }

        [Test]
        public void ASleepingBodyStillCarriesWhatLandsOnItAsAFloor()
        {
            Ground();
            Rigidbody sleeper = DynamicBox(new Vector3(0.0f, 0.5f, 0.0f));
            Assert.That(StepsUntilAsleep(sleeper, 150), Is.GreaterThan(0));

            Rigidbody falling = DynamicBox(new Vector3(0.0f, 2.0f, 0.0f));
            Simulate(100);

            Assert.That(falling.position.y, Is.EqualTo(1.5f).Within(0.03f));
            Assert.That(sleeper.position.y, Is.EqualTo(0.5f).Within(0.02f));
        }

        [Test]
        public void ASleepingBodyStaysVisibleToQueries()
        {
            Ground();
            Rigidbody sleeper = DynamicBox(new Vector3(0.0f, 0.5f, 0.0f));
            Assert.That(StepsUntilAsleep(sleeper, 150), Is.GreaterThan(0));

            Assert.That(UnityEngine.Physics.Raycast(new Vector3(0.0f, 5.0f, 0.0f), Vector3.down, out RaycastHit hit, 10.0f), Is.True);
            Assert.That(hit.rigidbody, Is.EqualTo(sleeper));
            Assert.That(hit.point.y, Is.EqualTo(1.0f).Within(0.02f));
        }

        [Test]
        public void TeleportingASleepingBodyWakesIt()
        {
            Ground();
            Rigidbody sleeper = DynamicBox(new Vector3(0.0f, 0.5f, 0.0f));
            Assert.That(StepsUntilAsleep(sleeper, 150), Is.GreaterThan(0));

            sleeper.transform.position = new Vector3(0.0f, 3.0f, 0.0f);
            Simulate(10);

            Assert.That(sleeper.IsSleeping(), Is.False);
            Assert.That(sleeper.position.y, Is.LessThan(3.0f));
        }

        [Test]
        public void DisablingWhatABodySleepsOnLetsItFall()
        {
            GameObject ground = Ground();
            Rigidbody sleeper = DynamicBox(new Vector3(0.0f, 0.5f, 0.0f));
            Assert.That(StepsUntilAsleep(sleeper, 150), Is.GreaterThan(0));

            ground.GetComponent<Collider>().enabled = false;
            Simulate(30);

            Assert.That(sleeper.position.y, Is.LessThan(0.0f));
        }

        [Test]
        public void DestroyingWhatABodySleepsOnLetsItFall()
        {
            GameObject ground = Ground();
            Rigidbody sleeper = DynamicBox(new Vector3(0.0f, 0.5f, 0.0f));
            Assert.That(StepsUntilAsleep(sleeper, 150), Is.GreaterThan(0));

            Object.DestroyImmediate(ground);
            Simulate(30);

            Assert.That(sleeper.position.y, Is.LessThan(0.0f));
        }

        [Test]
        public void AJointedPairSleepsTogether()
        {
            Rigidbody first = WeightlessBox(new Vector3(0.0f, 0.0f, 0.0f));
            Rigidbody second = WeightlessBox(new Vector3(2.0f, 0.0f, 0.0f));
            first.gameObject.AddComponent<FixedJoint>().connectedBody = second;
            second.sleepThreshold = 0.0f;

            Simulate(100);

            Assert.That(first.IsSleeping(), Is.False, "a joint holds the two in one island");

            second.sleepThreshold = 0.005f;
            Simulate(100);

            Assert.That(first.IsSleeping(), Is.True);
            Assert.That(second.IsSleeping(), Is.True);
        }

        [Test]
        public void CollisionStaysStopWhileAsleepAndNoNewEnterFollowsWaking()
        {
            Ground();
            Rigidbody body = DynamicBox(new Vector3(0.0f, 0.5f, 0.0f));
            CollisionEventConformanceTests.CollisionLog log = body.gameObject.AddComponent<CollisionEventConformanceTests.CollisionLog>();

            Assert.That(StepsUntilAsleep(body, 150), Is.GreaterThan(0));
            Simulate(2);
            int stays = log.Count("Stay");
            Simulate(10);

            Assert.That(log.Count("Stay"), Is.EqualTo(stays));
            Assert.That(log.Count("Exit"), Is.Zero);

            body.WakeUp();
            Simulate(5);

            Assert.That(log.Count("Stay"), Is.GreaterThan(stays));
            Assert.That(log.Count("Enter"), Is.EqualTo(1));
        }
    }
}
