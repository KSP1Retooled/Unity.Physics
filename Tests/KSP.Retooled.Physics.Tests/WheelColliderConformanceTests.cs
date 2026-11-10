using NUnit.Framework;
using UnityEngine;

namespace KSP.Retooled.Physics.Tests
{
    /// <summary>
    /// What a WheelCollider does on its own: its suspension holds a body, its ray finds the ground and skips what it should,
    /// and it shares a body's mass among its wheels. The game turns tyre friction off and drives its wheels itself, so the
    /// tests do the same wherever friction would matter.
    /// </summary>
    public class WheelColliderConformanceTests : PhysicsScene
    {
        private const float Radius = 0.5f;
        private const float Travel = 0.3f;

        private Rigidbody Chassis(Vector3 position, float mass, bool freezeRotation = true)
        {
            GameObject chassis = Track(new GameObject("chassis"));
            chassis.transform.position = position;
            chassis.AddComponent<BoxCollider>().size = new Vector3(2.0f, 0.4f, 3.0f);
            Rigidbody body = chassis.AddComponent<Rigidbody>();
            body.mass = mass;
            body.freezeRotation = freezeRotation;
            return body;
        }

        /// <summary>A wheel hung 0.2 m under its body's origin, with friction off.</summary>
        private WheelCollider Wheel(Rigidbody chassis, Vector3 local)
        {
            GameObject holder = Track(new GameObject("wheel"));
            holder.transform.SetParent(chassis.transform, false);
            holder.transform.localPosition = local;
            WheelCollider wheel = holder.AddComponent<WheelCollider>();
            wheel.radius = Radius;
            wheel.suspensionDistance = Travel;
            wheel.suspensionSpring = new JointSpring { spring = 35000.0f, damper = 4500.0f, targetPosition = 0.5f };
            WheelFrictionCurve forward = wheel.forwardFriction;
            forward.stiffness = 0.0f;
            wheel.forwardFriction = forward;
            WheelFrictionCurve sideways = wheel.sidewaysFriction;
            sideways.stiffness = 0.0f;
            wheel.sidewaysFriction = sideways;
            return wheel;
        }

        /// <summary>A 100 kg body standing on one wheel under its centre, which can't tip over.</summary>
        private (Rigidbody Body, WheelCollider Wheel) Unicycle(float height = 0.95f)
        {
            Rigidbody body = Chassis(new Vector3(0.0f, height, 0.0f), 100.0f);
            return (body, Wheel(body, new Vector3(0.0f, -0.2f, 0.0f)));
        }

        [Test]
        public void AWheelHoldsItsBodyWithTheSuspensionAtItsTargetPosition()
        {
            Ground();
            (Rigidbody body, WheelCollider wheel) = Unicycle(1.5f);

            Simulate(200);

            // At rest the wheel sits halfway down its travel, its centre a radius above the ground.
            Assert.That(body.position.y, Is.EqualTo(0.2f + Radius + Travel * 0.5f).Within(0.01f));
            wheel.GetWorldPose(out Vector3 centre, out _);
            Assert.That(centre.y, Is.EqualTo(Radius).Within(0.01f));
            Assert.That(body.linearVelocity.magnitude, Is.LessThan(0.01f));
        }

        [Test]
        public void ABottomedOutSuspensionHoldsItsBodyButReportsOnlyTheSpring()
        {
            Ground();
            (Rigidbody body, WheelCollider wheel) = Unicycle();

            // Far more than the spring can hold at full compression, which is the sprung weight plus the spring there.
            for (int i = 0; i < 200; i++)
            {
                body.AddForce(Vector3.down * 10000.0f);
                Simulate();
            }

            // The body stops with the wheel at the top of its travel, well short of its box touching the ground.
            Assert.That(body.position.y, Is.EqualTo(0.2f + Radius).Within(0.01f));
            Assert.IsTrue(wheel.GetGroundHit(out WheelHit hit));
            Assert.That(hit.force, Is.EqualTo(100.0f * 9.81f + 35000.0f * Travel * 0.5f).Within(50.0f));
        }

        [TestCase(40.0f)]
        [TestCase(-40.0f)]
        public void ALeaningBodyLandingHardWhileRollingStopsAtFullCompression(float speed)
        {
            // PhysX holds the stop along the suspension against the world, so here it throws the body off the ground rolling
            // one way and lets it sink 7 cm past the stop rolling the other. Its iterative solver softens the throw enough that
            // a landing aircraft survives it, but our exact solve applies all of it.
            Assume.That(BackendInfo.IsRetooled);

            Box(new Vector3(0.0f, -0.5f, 0.0f), new Vector3(20.0f, 1.0f, 600.0f));
            (Rigidbody body, WheelCollider wheel) = Unicycle(1.2f);
            // Heavy enough to bottom the suspension out, and leaning like an aircraft touching down nose up.
            body.mass = 5000.0f;
            body.transform.rotation = Quaternion.Euler(-5.0f, 0.0f, 0.0f);
            body.linearVelocity = new Vector3(0.0f, -4.0f, speed);

            float lowest = float.MaxValue;
            float fastestUp = 0.0f;
            for (int i = 0; i < 30; i++)
            {
                Simulate();
                lowest = Mathf.Min(lowest, body.position.y);
                fastestUp = Mathf.Max(fastestUp, body.linearVelocity.y);
            }

            // Rolling either way, the stop holds the body where the wheel bottoms out, without throwing it back up.
            Assert.IsTrue(wheel.isGrounded);
            Assert.That(lowest, Is.EqualTo(0.2f + Radius).Within(0.02f));
            Assert.That(fastestUp, Is.LessThan(1.0f));
        }

        [Test]
        public void AWheelReportsTheGroundUnderIt()
        {
            GameObject ground = Ground();
            (_, WheelCollider wheel) = Unicycle();

            Simulate(25);

            Assert.IsTrue(wheel.isGrounded);
            Assert.IsTrue(wheel.GetGroundHit(out WheelHit hit));
            Assert.AreSame(ground.GetComponent<Collider>(), hit.collider);
            AssertVector(Vector3.zero, hit.point, 0.01f, "point");
            AssertVector(Vector3.up, hit.normal, 0.001f, "normal");
            AssertVector(Vector3.forward, hit.forwardDir, 0.001f, "forward");
            AssertVector(Vector3.right, hit.sidewaysDir, 0.001f, "sideways");
        }

        [Test]
        public void AWheelInTheAirReportsNoGround()
        {
            (_, WheelCollider wheel) = Unicycle();

            Simulate(5);

            Assert.IsFalse(wheel.isGrounded);
            Assert.IsFalse(wheel.GetGroundHit(out WheelHit hit));
            Assert.IsNull(hit.collider);
        }

        [Test]
        public void AWheelPassesThroughAColliderItIgnores()
        {
            GameObject ground = Ground();
            (_, WheelCollider wheel) = Unicycle();
            UnityEngine.Physics.IgnoreCollision(wheel, ground.GetComponent<Collider>());

            Simulate(25);

            Assert.IsFalse(wheel.isGrounded);
        }

        [Test]
        public void AWheelPassesThroughALayerItDoesNotCollideWith()
        {
            GameObject ground = Ground();
            ground.layer = 30;
            (_, WheelCollider wheel) = Unicycle();
            wheel.gameObject.layer = 26;

            // The layer matrix is a project setting, which the editor saves, so the test leaves it as it found it.
            bool ignored = UnityEngine.Physics.GetIgnoreLayerCollision(26, 30);
            UnityEngine.Physics.IgnoreLayerCollision(26, 30, true);
            try
            {
                Simulate(25);
                Assert.IsFalse(wheel.isGrounded);
            }
            finally
            {
                UnityEngine.Physics.IgnoreLayerCollision(26, 30, ignored);
            }
        }

        [Test]
        public void AWheelPassesThroughATrigger()
        {
            Ground().GetComponent<Collider>().isTrigger = true;
            (_, WheelCollider wheel) = Unicycle();

            Simulate(25);

            Assert.IsFalse(wheel.isGrounded);
        }

        [Test]
        public void AWheelPushesDownADynamicBodyItStandsOn()
        {
            Rigidbody floor = DynamicBox(new Vector3(0.0f, -0.5f, 0.0f), 1000.0f);
            floor.transform.localScale = new Vector3(6.0f, 1.0f, 6.0f);
            floor.useGravity = false;
            floor.constraints = RigidbodyConstraints.FreezeRotation | RigidbodyConstraints.FreezePositionX | RigidbodyConstraints.FreezePositionZ;
            (_, WheelCollider wheel) = Unicycle();

            Simulate(25);

            Assert.IsTrue(wheel.isGrounded);
            Assert.That(floor.position.y, Is.LessThan(-0.55f), "the wheel's push sends the weightless floor down");
        }

        [Test]
        public void AWheelWithoutFrictionSlidesDownASlope()
        {
            GameObject slope = Box(Vector3.zero, new Vector3(200.0f, 1.0f, 200.0f));
            slope.transform.rotation = Quaternion.Euler(0.0f, 0.0f, 15.0f);
            slope.transform.position = slope.transform.rotation * new Vector3(0.0f, -0.5f, 0.0f);
            (Rigidbody body, WheelCollider wheel) = Unicycle(2.0f);

            Simulate(50);
            Vector3 before = body.linearVelocity;
            Simulate(50);

            // The suspension pushes along the ground's normal, so the level body slides as on frictionless ice.
            Assert.IsTrue(wheel.isGrounded);
            float acceleration = (body.linearVelocity - before).magnitude / (50 * Step);
            Assert.That(acceleration, Is.EqualTo(-Gravity * Mathf.Sin(15.0f * Mathf.Deg2Rad)).Within(0.1f));
        }

        [Test]
        public void SprungMassesBalanceFourWheelsAboutTheCentreOfMass()
        {
            Ground();
            Rigidbody body = Chassis(new Vector3(0.0f, 1.0f, 0.0f), 400.0f, false);
            body.centerOfMass = new Vector3(0.2f, 0.0f, 0.5f);
            WheelCollider[] wheels =
            {
                Wheel(body, new Vector3(-1.0f, -0.2f, 1.2f)),
                Wheel(body, new Vector3(1.0f, -0.2f, 1.2f)),
                Wheel(body, new Vector3(-1.0f, -0.2f, -1.2f)),
                Wheel(body, new Vector3(1.0f, -0.2f, -1.2f)),
            };

            Simulate();

            // Of the sets that sum to the body's mass and balance it, PhysX takes the smallest.
            float[] expected = { 121.6667f, 161.6667f, 38.3333f, 78.3333f };
            for (int i = 0; i < wheels.Length; i++)
            {
                Assert.That(wheels[i].sprungMass, Is.EqualTo(expected[i]).Within(0.01f), $"wheel {i}");
            }
        }

        [Test]
        public void SprungMassesBalanceWheelsInALine()
        {
            Ground();
            Rigidbody body = Chassis(new Vector3(0.0f, 1.0f, 0.0f), 300.0f, false);
            body.centerOfMass = new Vector3(0.0f, 0.0f, 0.4f);
            WheelCollider front = Wheel(body, new Vector3(0.0f, -0.2f, 1.2f));
            WheelCollider back = Wheel(body, new Vector3(0.0f, -0.2f, -1.2f));

            Simulate();

            Assert.That(front.sprungMass, Is.EqualTo(200.0f).Within(0.01f));
            Assert.That(back.sprungMass, Is.EqualTo(100.0f).Within(0.01f));
        }

        [Test]
        public void ALoneWheelCarriesTheWholeBody()
        {
            (Rigidbody body, WheelCollider wheel) = Unicycle();

            Simulate();
            Assert.That(wheel.sprungMass, Is.EqualTo(100.0f).Within(0.01f));

            body.mass = 250.0f;
            Simulate();
            Assert.That(wheel.sprungMass, Is.EqualTo(250.0f).Within(0.01f));
        }

        [TestCase(2.0f, 0.0f, 1.1071f)]
        [TestCase(1.0f, 5.0f, 0.1651f)]
        [TestCase(0.0f, 5.0f, 0.0f)]
        public void SidewaysSlipIsTheAngleOfTheWheelsSlide(float across, float along, float slip)
        {
            Ground();
            (Rigidbody body, WheelCollider wheel) = Unicycle();
            Simulate(30);

            body.linearVelocity = new Vector3(across, 0.0f, along);
            Simulate();

            Assert.IsTrue(wheel.GetGroundHit(out WheelHit hit));
            Assert.That(hit.sidewaysSlip, Is.EqualTo(slip).Within(0.005f));
        }

        [Test]
        public void SteeringTurnsTheWheelsHeading()
        {
            Ground();
            (_, WheelCollider wheel) = Unicycle();
            wheel.steerAngle = 30.0f;

            Simulate(5);

            Assert.IsTrue(wheel.GetGroundHit(out WheelHit hit));
            AssertVector(new Vector3(0.5f, 0.0f, 0.866f), hit.forwardDir, 0.001f, "forward");
            AssertVector(new Vector3(0.866f, 0.0f, -0.5f), hit.sidewaysDir, 0.001f, "sideways");
        }

        [Test]
        public void AMotorSpinsAWheelByItsInertia()
        {
            Ground();
            (_, WheelCollider wheel) = Unicycle();
            wheel.wheelDampingRate = 0.0f;
            wheel.motorTorque = 10.0f;

            Simulate(50);

            // A 20 kg wheel of radius 0.5 has an inertia of 2.5 kg m², so 10 N m spins it up by 4 rad/s a second.
            Assert.That(wheel.rpm, Is.EqualTo(4.0f * 60.0f / (2.0f * Mathf.PI)).Within(0.2f));
        }

        [Test]
        public void WheelsKeepWorkingAsOthersComeAndGo()
        {
            Ground();
            Rigidbody body = Chassis(new Vector3(0.0f, 0.95f, 0.0f), 300.0f);
            WheelCollider first = Wheel(body, new Vector3(0.0f, -0.2f, 1.0f));
            WheelCollider second = Wheel(body, new Vector3(-1.0f, -0.2f, -1.0f));
            WheelCollider third = Wheel(body, new Vector3(1.0f, -0.2f, -1.0f));
            Simulate(25);
            Assert.IsTrue(first.isGrounded && second.isGrounded && third.isGrounded);

            Object.DestroyImmediate(second);
            Simulate(25);
            Assert.IsTrue(first.isGrounded && third.isGrounded);

            first.enabled = false;
            Simulate();
            first.enabled = true;
            Simulate(25);
            Assert.IsTrue(first.isGrounded && third.isGrounded);
        }

        [Test]
        public void DestroyingTheGroundUnderAWheelLeavesItHanging()
        {
            GameObject ground = Ground();
            (_, WheelCollider wheel) = Unicycle();
            Simulate(25);
            Assert.IsTrue(wheel.isGrounded);

            Object.DestroyImmediate(ground.GetComponent<Collider>());
            Simulate();

            Assert.IsFalse(wheel.isGrounded);
        }
    }
}
