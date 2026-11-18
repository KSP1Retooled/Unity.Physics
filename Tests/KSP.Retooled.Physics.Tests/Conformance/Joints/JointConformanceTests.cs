using PhysicsScene = KSP.Retooled.Physics.Tests.Scenes.PhysicsScene;
using System.Collections.Generic;
using NUnit.Framework;
using UnityEngine;

namespace KSP.Retooled.Physics.Tests.Conformance.Joints
{
    /// <summary>
    /// Test joint behavior through Unity's public API, using PhysX observations and tolerances shared by both
    /// solvers.
    /// </summary>
    public class JointConformanceTests : PhysicsScene
    {
        private const float AnchorTolerance = 5e-3f;

        private static ConfigurableJoint Lock(Rigidbody owner, Rigidbody connected, Vector3 anchor, Vector3 connectedAnchor)
        {
            ConfigurableJoint joint = owner.gameObject.AddComponent<ConfigurableJoint>();
            joint.connectedBody = connected;
            joint.autoConfigureConnectedAnchor = false;
            joint.anchor = anchor;
            joint.connectedAnchor = connectedAnchor;
            joint.xMotion = joint.yMotion = joint.zMotion = ConfigurableJointMotion.Locked;
            joint.angularXMotion = joint.angularYMotion = joint.angularZMotion = ConfigurableJointMotion.Locked;
            return joint;
        }

        private static float AnchorGap(ConfigurableJoint joint)
        {
            Vector3 own = joint.transform.TransformPoint(joint.anchor);
            Vector3 other = joint.connectedBody != null ? joint.connectedBody.transform.TransformPoint(joint.connectedAnchor) : joint.connectedAnchor;
            return (own - other).magnitude;
        }

        private Rigidbody Weightless(Vector3 position)
        {
            Rigidbody body = WeightlessBox(position);
            body.GetComponent<Collider>().enabled = false;
            body.linearDamping = 0.0f;
            body.angularDamping = 0.0f;
            return body;
        }

        [Test]
        public void LockedJointToWorldHoldsABodyAgainstGravity()
        {
            Rigidbody body = DynamicBox(new Vector3(0.0f, 5.0f, 0.0f));
            ConfigurableJoint joint = Lock(body, null, new Vector3(0.0f, 1.0f, 0.0f), new Vector3(0.0f, 6.0f, 0.0f));

            Simulate(100);

            Assert.That(AnchorGap(joint), Is.LessThan(AnchorTolerance));
            AssertVector(new Vector3(0.0f, 5.0f, 0.0f), body.position, AnchorTolerance);
            Assert.That(Quaternion.Angle(Quaternion.identity, body.rotation), Is.LessThan(0.1f));
        }

        [Test]
        public void LockedJointMovesTwoBodiesAsOne()
        {
            Rigidbody first = Weightless(new Vector3(0.0f, 0.0f, 0.0f));
            Rigidbody second = Weightless(new Vector3(0.0f, 2.0f, 0.0f));
            ConfigurableJoint joint = Lock(second, first, new Vector3(0.0f, -1.0f, 0.0f), new Vector3(0.0f, 1.0f, 0.0f));

            first.AddForce(new Vector3(0.0f, 0.0f, 10.0f), ForceMode.Impulse);
            Simulate(50);

            Assert.That(AnchorGap(joint), Is.LessThan(AnchorTolerance));
            Assert.That((second.position - first.position).magnitude, Is.EqualTo(2.0f).Within(AnchorTolerance));
            Assert.That(first.position.z, Is.GreaterThan(1.0f));
        }

        [Test]
        public void FreeMotionsLeaveTheBodyFree()
        {
            Rigidbody body = Weightless(Vector3.zero);
            ConfigurableJoint joint = Lock(body, null, Vector3.zero, Vector3.zero);
            joint.xMotion = ConfigurableJointMotion.Free;

            body.linearVelocity = new Vector3(1.0f, 1.0f, 0.0f);
            Simulate(50);

            Assert.That(body.position.x, Is.EqualTo(1.0f).Within(0.01f));
            Assert.That(body.position.y, Is.EqualTo(0.0f).Within(AnchorTolerance));
        }

        [Test]
        public void FreeAngularMotionTurnsOnlyAboutTheJointAxis()
        {
            Rigidbody body = Weightless(Vector3.zero);
            ConfigurableJoint joint = Lock(body, null, Vector3.zero, Vector3.zero);
            joint.axis = Vector3.up;
            joint.secondaryAxis = Vector3.right;
            joint.angularXMotion = ConfigurableJointMotion.Free;

            body.angularVelocity = new Vector3(1.0f, 1.0f, 0.0f);
            Simulate(25);

            Assert.That(body.angularVelocity.y, Is.EqualTo(1.0f).Within(0.05f));
            Assert.That(body.angularVelocity.x, Is.EqualTo(0.0f).Within(0.01f));
        }

        [Test]
        public void LinearLimitStopsAtItsDistance()
        {
            Rigidbody body = Weightless(Vector3.zero);
            ConfigurableJoint joint = Lock(body, null, Vector3.zero, Vector3.zero);
            joint.xMotion = ConfigurableJointMotion.Limited;
            joint.linearLimit = new SoftJointLimit { limit = 0.5f };

            body.linearVelocity = new Vector3(2.0f, 0.0f, 0.0f);
            Simulate(100);

            Assert.That(body.position.x, Is.EqualTo(0.5f).Within(0.02f));
        }

        [Test]
        public void LinearLimitIsOneDistanceAcrossItsAxes()
        {
            Rigidbody body = Weightless(Vector3.zero);
            ConfigurableJoint joint = Lock(body, null, Vector3.zero, Vector3.zero);
            joint.xMotion = ConfigurableJointMotion.Limited;
            joint.yMotion = ConfigurableJointMotion.Limited;
            joint.linearLimit = new SoftJointLimit { limit = 1.0f };

            body.linearVelocity = new Vector3(2.0f, 2.0f, 0.0f);
            Simulate(100);

            Assert.That(new Vector2(body.position.x, body.position.y).magnitude, Is.EqualTo(1.0f).Within(0.03f));
        }

        [Test]
        public void TwistLimitStopsWithinItsRangeOnTheRightSide()
        {
            foreach (float direction in new[] { 1.0f, -1.0f })
            {
                Rigidbody body = Weightless(Vector3.zero);
                ConfigurableJoint joint = Lock(body, null, Vector3.zero, Vector3.zero);
                joint.axis = Vector3.right;
                joint.secondaryAxis = Vector3.up;
                joint.angularXMotion = ConfigurableJointMotion.Limited;
                joint.lowAngularXLimit = new SoftJointLimit { limit = -10.0f };
                joint.highAngularXLimit = new SoftJointLimit { limit = 40.0f };

                body.angularVelocity = new Vector3(direction, 0.0f, 0.0f);
                Simulate(100);

                float angle = Mathf.DeltaAngle(0.0f, body.rotation.eulerAngles.x);
                float expected = direction > 0.0f ? 10.0f : -40.0f;
                Assert.That(angle, Is.EqualTo(expected).Within(2.0f), $"turning {direction}");
            }
        }

        [Test]
        public void AngularLimitsStopABodyOnTheLimit()
        {
            // A limit that only acts once it is passed lets the body in a little and then pushes it back out, slowly enough
            // that nothing stops it, so it ends short of the limit or still moving.
            var limits = new Dictionary<string, float> { ["twist"] = 10.0f, ["cone"] = 20.0f, ["hinge"] = 40.0f };
            foreach (string kind in limits.Keys)
            {
                Rigidbody body = Weightless(Vector3.zero);
                if (kind == "hinge")
                {
                    HingeJoint hinge = body.gameObject.AddComponent<HingeJoint>();
                    hinge.axis = Vector3.right;
                    hinge.useLimits = true;
                    hinge.limits = new JointLimits { min = -10.0f, max = 40.0f };
                    body.angularVelocity = new Vector3(1.0f, 0.0f, 0.0f);
                }
                else
                {
                    ConfigurableJoint joint = Lock(body, null, Vector3.zero, Vector3.zero);
                    joint.axis = Vector3.right;
                    joint.secondaryAxis = Vector3.up;
                    if (kind == "twist")
                    {
                        joint.angularXMotion = ConfigurableJointMotion.Limited;
                        joint.lowAngularXLimit = new SoftJointLimit { limit = -10.0f };
                        joint.highAngularXLimit = new SoftJointLimit { limit = 40.0f };
                        body.angularVelocity = new Vector3(1.0f, 0.0f, 0.0f);
                    }
                    else
                    {
                        joint.angularYMotion = ConfigurableJointMotion.Limited;
                        joint.angularZMotion = ConfigurableJointMotion.Limited;
                        joint.angularYLimit = new SoftJointLimit { limit = 20.0f };
                        joint.angularZLimit = new SoftJointLimit { limit = 20.0f };
                        body.angularVelocity = new Vector3(0.0f, 1.0f, 0.0f);
                    }
                }

                Simulate(60);

                Assert.That(Quaternion.Angle(Quaternion.identity, body.rotation), Is.EqualTo(limits[kind]).Within(0.02f), kind);
                Assert.That(body.angularVelocity.magnitude, Is.LessThan(1e-3f), $"{kind} has stopped");
                Object.DestroyImmediate(body.gameObject);
            }
        }

        [Test]
        public void SwingLimitStopsAtTheCone()
        {
            Rigidbody body = Weightless(Vector3.zero);
            ConfigurableJoint joint = Lock(body, null, Vector3.zero, Vector3.zero);
            joint.angularYMotion = ConfigurableJointMotion.Limited;
            joint.angularZMotion = ConfigurableJointMotion.Limited;
            joint.angularYLimit = new SoftJointLimit { limit = 20.0f };
            joint.angularZLimit = new SoftJointLimit { limit = 20.0f };

            body.angularVelocity = new Vector3(0.0f, 1.0f, 0.0f);
            Simulate(100);

            Assert.That(Quaternion.Angle(Quaternion.identity, body.rotation), Is.EqualTo(20.0f).Within(2.0f));
        }

        [Test]
        public void LinearDriveSettlesAtItsTargetPosition()
        {
            Rigidbody body = Weightless(Vector3.zero);
            ConfigurableJoint joint = Lock(body, null, Vector3.zero, Vector3.zero);
            joint.xMotion = ConfigurableJointMotion.Free;
            joint.xDrive = new JointDrive { positionSpring = 100.0f, positionDamper = 20.0f, maximumForce = float.MaxValue };
            joint.targetPosition = new Vector3(1.0f, 0.0f, 0.0f);

            Simulate(300);

            Assert.That(body.position.x, Is.EqualTo(-1.0f).Within(0.02f));
        }

        [Test]
        public void LinearDriveFollowsItsTargetVelocity()
        {
            Rigidbody body = Weightless(Vector3.zero);
            ConfigurableJoint joint = Lock(body, null, Vector3.zero, Vector3.zero);
            joint.xMotion = ConfigurableJointMotion.Free;
            joint.xDrive = new JointDrive { positionDamper = 50.0f, maximumForce = float.MaxValue };
            joint.targetVelocity = new Vector3(2.0f, 0.0f, 0.0f);

            Simulate(100);

            Assert.That(body.linearVelocity.x, Is.EqualTo(-2.0f).Within(0.1f));
        }

        [Test]
        public void DriveMaximumForceLimitsAcceleration()
        {
            Rigidbody body = Weightless(Vector3.zero);
            ConfigurableJoint joint = Lock(body, null, Vector3.zero, Vector3.zero);
            joint.xMotion = ConfigurableJointMotion.Free;
            joint.xDrive = new JointDrive { positionSpring = 1e6f, positionDamper = 1e3f, maximumForce = 5.0f };
            joint.targetPosition = new Vector3(100.0f, 0.0f, 0.0f);

            Simulate(50);

            Assert.That(Mathf.Abs(body.linearVelocity.x), Is.EqualTo(5.0f).Within(0.25f));
        }

        [Test]
        public void ADriveWithNoForceDoesNothing()
        {
            Rigidbody body = Weightless(Vector3.zero);
            ConfigurableJoint joint = Lock(body, null, Vector3.zero, Vector3.zero);
            joint.xMotion = ConfigurableJointMotion.Free;
            joint.angularXMotion = ConfigurableJointMotion.Free;
            JointDrive drive = new JointDrive { positionSpring = 1e6f, positionDamper = 1e3f, maximumForce = 0.0f };
            joint.xDrive = drive;
            joint.angularXDrive = drive;
            joint.rotationDriveMode = RotationDriveMode.XYAndZ;
            body.linearVelocity = new Vector3(2.0f, 0.0f, 0.0f);
            body.angularVelocity = new Vector3(3.0f, 0.0f, 0.0f);

            Simulate(25);

            Assert.That(body.linearVelocity.x, Is.EqualTo(2.0f).Within(0.01f));
            Assert.That(body.angularVelocity.x, Is.EqualTo(3.0f).Within(0.01f));
        }

        [Test]
        public void AngularDriveSettlesAtItsTargetRotation()
        {
            foreach (RotationDriveMode mode in new[] { RotationDriveMode.XYAndZ, RotationDriveMode.Slerp })
            {
                Rigidbody body = Weightless(Vector3.zero);
                ConfigurableJoint joint = Lock(body, null, Vector3.zero, Vector3.zero);
                joint.angularXMotion = joint.angularYMotion = joint.angularZMotion = ConfigurableJointMotion.Free;
                JointDrive drive = new JointDrive { positionSpring = 50.0f, positionDamper = 10.0f, maximumForce = float.MaxValue };
                joint.rotationDriveMode = mode;
                joint.angularXDrive = drive;
                joint.angularYZDrive = drive;
                joint.slerpDrive = drive;
                joint.targetRotation = Quaternion.Euler(30.0f, 0.0f, 0.0f);

                Simulate(300);

                float angle = Mathf.DeltaAngle(0.0f, body.rotation.eulerAngles.x);
                Assert.That(angle, Is.EqualTo(-30.0f).Within(1.0f), mode.ToString());
            }
        }

        [Test]
        public void AngularDriveFollowsItsTargetVelocity()
        {
            Rigidbody body = Weightless(Vector3.zero);
            ConfigurableJoint joint = Lock(body, null, Vector3.zero, Vector3.zero);
            joint.angularXMotion = ConfigurableJointMotion.Free;
            joint.angularXDrive = new JointDrive { positionDamper = 50.0f, maximumForce = float.MaxValue };
            joint.targetAngularVelocity = new Vector3(1.0f, 0.0f, 0.0f);

            Simulate(100);

            Assert.That(body.angularVelocity.x, Is.EqualTo(1.0f).Within(0.1f));
        }

        [Test]
        public void FixedJointKeepsTheOffset()
        {
            Rigidbody first = DynamicBox(new Vector3(0.0f, 5.0f, 0.0f));
            first.isKinematic = true;
            Rigidbody second = DynamicBox(new Vector3(0.0f, 3.0f, 0.0f));
            second.gameObject.AddComponent<FixedJoint>().connectedBody = first;

            Simulate(100);

            AssertVector(new Vector3(0.0f, 3.0f, 0.0f), second.position, 0.01f);
        }

        [Test]
        public void WorldJointSurvivesOtherBodiesBeingRemovedAndAdded()
        {
            Assume.That(UnityEngine.Physics.GetCurrentIntegrationInfo().isFallback, Is.False);
            Rigidbody debris = WeightlessBox(new Vector3(20.0f, 5.0f, 0.0f));
            Rigidbody held = WeightlessBox(new Vector3(0.0f, 5.0f, 0.0f));
            MakeWorldClamp(held);
            GameObject oldStatic = Box(new Vector3(0.0f, -10.0f, 0.0f), Vector3.one);
            Simulate();

            Vector3 expected = new Vector3(0.0f, 5.0f, 0.0f);
            Object.DestroyImmediate(debris.gameObject);
            Object.DestroyImmediate(oldStatic);
            AssertHeldForThreeSteps(held, expected);

            for (int i = 0; i < 256; i++)
            {
                WeightlessBox(new Vector3(20.0f + 2.0f * i, 5.0f, 0.0f));
            }
            Box(new Vector3(0.0f, -10.0f, 0.0f), Vector3.one);
            AssertHeldForThreeSteps(held, expected);
        }

        [TestCase(false)]
        [TestCase(true)]
        public void WorldJointAfterAnOriginShift(bool moveConnectedAnchor)
        {
            IntegrationInfo backend = UnityEngine.Physics.GetCurrentIntegrationInfo();
            Assume.That(backend.isFallback, Is.False);
            Rigidbody held = WeightlessBox(new Vector3(0.0f, 5.0f, 0.0f));
            ConfigurableJoint clamp = MakeWorldClamp(held);
            GameObject ground = Ground();
            Simulate();
            AssertVector(new Vector3(0.0f, 5.0f, 0.0f), held.position, 0.001f, "initial clamped pose");

            Vector3 shift = new Vector3(-512.0f, 0.0f, 0.0f);
            Vector3 shiftedPosition = held.position + shift;
            held.transform.position += shift;
            ground.transform.position += shift;
            if (moveConnectedAnchor)
            {
                clamp.connectedAnchor += shift;
            }
            UnityEngine.Physics.SyncTransforms();

            if (moveConnectedAnchor)
            {
                AssertHeldForThreeSteps(held, shiftedPosition);
            }
            else
            {
                Simulate();
                TestContext.WriteLine($"{backend.name} first step after shift: position {held.position}, velocity {held.linearVelocity}");
                Assert.That(held.position.x - shiftedPosition.x, Is.GreaterThan(0.01f));
                Assert.That(clamp.connectedAnchor.x, Is.EqualTo(0.0f));
            }
        }

        private static ConfigurableJoint MakeWorldClamp(Rigidbody held)
        {
            ConfigurableJoint clamp = held.gameObject.AddComponent<ConfigurableJoint>();
            clamp.autoConfigureConnectedAnchor = false;
            clamp.connectedAnchor = held.position;
            clamp.xMotion = clamp.yMotion = clamp.zMotion = ConfigurableJointMotion.Locked;
            clamp.angularXMotion = clamp.angularYMotion = clamp.angularZMotion = ConfigurableJointMotion.Locked;
            return clamp;
        }

        private static void AssertHeldForThreeSteps(Rigidbody held, Vector3 position)
        {
            for (int step = 0; step < 3; step++)
            {
                Simulate();
                Assert.That(held.linearVelocity.magnitude, Is.LessThan(0.001f), $"step {step}");
                AssertVector(position, held.position, 0.001f, $"step {step} position");
            }
        }

        [Test]
        public void HingeSwingsAboutItsAxisOnly()
        {
            Rigidbody body = DynamicBox(new Vector3(0.0f, 0.0f, 1.0f));
            HingeJoint hinge = body.gameObject.AddComponent<HingeJoint>();
            hinge.anchor = new Vector3(0.0f, 0.0f, -1.0f);
            hinge.axis = Vector3.right;

            Simulate(50);

            Assert.That(Mathf.Abs(body.position.x), Is.LessThan(AnchorTolerance));
            Assert.That(body.position.y, Is.LessThan(-0.2f));
            Assert.That((body.position - Vector3.zero).magnitude, Is.EqualTo(1.0f).Within(0.01f));
        }

        [Test]
        public void HingeLimitStopsTheSwing()
        {
            Rigidbody body = DynamicBox(new Vector3(0.0f, 0.0f, 1.0f));
            HingeJoint hinge = body.gameObject.AddComponent<HingeJoint>();
            hinge.anchor = new Vector3(0.0f, 0.0f, -1.0f);
            hinge.axis = Vector3.right;
            hinge.useLimits = true;
            hinge.limits = new JointLimits { min = -30.0f, max = 30.0f };

            Simulate(200);

            Assert.That(Mathf.Abs(Mathf.DeltaAngle(0.0f, body.rotation.eulerAngles.x)), Is.EqualTo(30.0f).Within(2.0f));
        }

        [Test]
        public void HingeMotorReachesItsSpeed()
        {
            Rigidbody body = Weightless(Vector3.zero);
            HingeJoint hinge = body.gameObject.AddComponent<HingeJoint>();
            hinge.axis = Vector3.right;
            hinge.useMotor = true;
            hinge.motor = new JointMotor { targetVelocity = 90.0f, force = 100.0f };

            Simulate(50);

            Assert.That(body.angularVelocity.x, Is.EqualTo(Mathf.PI / 2.0f).Within(0.05f));
        }

        [Test]
        public void HingeLimitsKeepTheirSides()
        {
            Rigidbody body = Weightless(Vector3.zero);
            HingeJoint hinge = body.gameObject.AddComponent<HingeJoint>();
            hinge.axis = Vector3.right;
            hinge.useLimits = true;
            hinge.limits = new JointLimits { min = -10.0f, max = 40.0f };

            body.angularVelocity = new Vector3(1.0f, 0.0f, 0.0f);
            Simulate(100);

            Assert.That(Mathf.DeltaAngle(0.0f, body.rotation.eulerAngles.x), Is.EqualTo(40.0f).Within(2.0f));
        }

        [Test]
        public void HingeMotorIsLimitedByItsForce()
        {
            Rigidbody body = Weightless(Vector3.zero);
            HingeJoint hinge = body.gameObject.AddComponent<HingeJoint>();
            hinge.axis = Vector3.right;
            hinge.useMotor = true;
            hinge.motor = new JointMotor { targetVelocity = 5000.0f, force = 0.5f };
            body.inertiaTensor = Vector3.one;
            body.maxAngularVelocity = 100.0f;

            Simulate(50);

            Assert.That(Mathf.Abs(body.angularVelocity.x), Is.EqualTo(0.5f).Within(0.03f));
        }

        [Test]
        public void JointOnARotatedBodyWithAnOffsetCenterOfMassHolds()
        {
            Rigidbody body = DynamicBox(new Vector3(1.0f, 5.0f, 2.0f));
            body.rotation = Quaternion.Euler(30.0f, 40.0f, 50.0f);
            body.transform.rotation = body.rotation;
            body.centerOfMass = new Vector3(0.3f, -0.2f, 0.1f);
            ConfigurableJoint joint = Lock(body, null, new Vector3(0.1f, 0.2f, 0.3f), body.transform.TransformPoint(0.1f, 0.2f, 0.3f));
            joint.axis = new Vector3(1.0f, 1.0f, 0.0f);
            joint.secondaryAxis = Vector3.forward;
            Quaternion start = body.rotation;

            Simulate(100);

            Assert.That(AnchorGap(joint), Is.LessThan(AnchorTolerance));
            Assert.That(Quaternion.Angle(start, body.rotation), Is.LessThan(0.1f));
        }

        [Test]
        public void KinematicBodyCarriesAJointedBody()
        {
            Rigidbody carrier = DynamicBox(Vector3.zero);
            carrier.isKinematic = true;
            Rigidbody load = WeightlessBox(new Vector3(0.0f, 2.0f, 0.0f));
            ConfigurableJoint joint = Lock(load, carrier, new Vector3(0.0f, -1.0f, 0.0f), new Vector3(0.0f, 1.0f, 0.0f));

            for (int i = 0; i < 50; i++)
            {
                carrier.MovePosition(new Vector3(i * 0.02f, 0.0f, 0.0f));
                Simulate();
            }

            Assert.That(AnchorGap(joint), Is.LessThan(0.03f));
            Assert.That(load.position.x, Is.EqualTo(0.98f).Within(0.05f));
        }

        [Test]
        public void SleepingBodyWakesWhenItsPartnerMoves()
        {
            // PhysX keeps the pair asleep after one step here, at x = 0.005. A sleeping body is immovable in our step, so the
            // awake one would tear the joint instead.
            Assume.That(BackendInfo.IsRetooled);

            Rigidbody first = Weightless(Vector3.zero);
            Rigidbody second = Weightless(new Vector3(0.0f, 2.0f, 0.0f));
            Lock(second, first, new Vector3(0.0f, -1.0f, 0.0f), new Vector3(0.0f, 1.0f, 0.0f));
            Simulate();
            second.Sleep();

            first.linearVelocity = new Vector3(1.0f, 0.0f, 0.0f);
            Simulate(25);

            Assert.That(second.position.x, Is.GreaterThan(0.1f));
        }

        [Test]
        public void ChangingTheConnectedBodyMovesTheJoint()
        {
            Rigidbody first = Weightless(Vector3.zero);
            Rigidbody second = Weightless(new Vector3(0.0f, 2.0f, 0.0f));
            Rigidbody third = Weightless(new Vector3(0.0f, 4.0f, 0.0f));
            ConfigurableJoint joint = Lock(second, first, new Vector3(0.0f, -1.0f, 0.0f), new Vector3(0.0f, 1.0f, 0.0f));
            Simulate();

            joint.connectedBody = third;
            joint.connectedAnchor = new Vector3(0.0f, -3.0f, 0.0f);
            third.linearVelocity = new Vector3(1.0f, 0.0f, 0.0f);
            Simulate(25);

            Assert.That(first.position.x, Is.EqualTo(0.0f).Within(AnchorTolerance));
            Assert.That(AnchorGap(joint), Is.LessThan(AnchorTolerance));
            Assert.That(second.position.x, Is.GreaterThan(0.1f));
        }

        [Test]
        public void ConnectedBodyFeelsTheReactionOfTheBreakForce()
        {
            Rigidbody anchor = DynamicBox(new Vector3(0.0f, 5.0f, 0.0f), 3.0f);
            anchor.isKinematic = true;
            Rigidbody body = DynamicBox(new Vector3(0.0f, 3.0f, 0.0f), 3.0f);
            ConfigurableJoint joint = Lock(body, anchor, new Vector3(0.0f, 1.0f, 0.0f), new Vector3(0.0f, -1.0f, 0.0f));

            Simulate(50);

            AssertVector(new Vector3(0.0f, 3.0f * 9.81f, 0.0f), joint.currentForce, 0.5f, "force on the owner");
        }

        [Test]
        public void LinearDriveCountsForNothingInCurrentForce()
        {
            Rigidbody body = DynamicBox(new Vector3(0.0f, 5.0f, 0.0f), 3.0f);
            ConfigurableJoint joint = Lock(body, null, Vector3.zero, new Vector3(0.0f, 5.0f, 0.0f));
            joint.yMotion = ConfigurableJointMotion.Free;
            joint.yDrive = new JointDrive { positionSpring = 1e5f, positionDamper = 1e3f, maximumForce = 1000.0f };

            Simulate(100);

            Assert.That(body.position.y, Is.EqualTo(5.0f).Within(0.01f));
            AssertVector(Vector3.zero, joint.currentForce, 0.5f, "force");
        }

        [Test]
        public void AngularDriveCountsForNothingInCurrentTorque()
        {
            Rigidbody body = DynamicBox(new Vector3(2.0f, 5.0f, 0.0f), 3.0f);
            ConfigurableJoint joint = Lock(body, null, new Vector3(-2.0f, 0.0f, 0.0f), new Vector3(0.0f, 5.0f, 0.0f));
            joint.axis = Vector3.forward;
            joint.secondaryAxis = Vector3.up;
            joint.angularXMotion = ConfigurableJointMotion.Free;
            joint.angularXDrive = new JointDrive { positionSpring = 1e5f, positionDamper = 1e3f, maximumForce = 1000.0f };

            Simulate(100);

            Assert.That(body.position.x, Is.EqualTo(2.0f).Within(0.01f));
            AssertVector(Vector3.zero, joint.currentTorque, 0.5f, "torque");
        }

        [Test]
        public void SoftRotationDriveCountsForNothingInCurrentTorque()
        {
            Rigidbody body = DynamicBox(new Vector3(2.0f, 5.0f, 0.0f), 3.0f);
            ConfigurableJoint joint = Lock(body, null, new Vector3(-2.0f, 0.0f, 0.0f), new Vector3(0.0f, 5.0f, 0.0f));
            joint.angularXMotion = joint.angularYMotion = joint.angularZMotion = ConfigurableJointMotion.Free;
            JointDrive drive = new JointDrive { positionSpring = 1e5f, positionDamper = 1e3f, maximumForce = float.MaxValue };
            joint.angularXDrive = drive;
            joint.angularYZDrive = drive;

            Simulate(100);

            Assert.That(body.position.x, Is.EqualTo(2.0f).Within(0.01f));
            AssertVector(Vector3.zero, joint.currentTorque, 0.5f, "torque");
        }

        [Test]
        public void FixedJointBetweenMovingBodiesReportsNoForceWithoutLoad()
        {
            Rigidbody first = Weightless(Vector3.zero);
            Rigidbody second = Weightless(new Vector3(0.0f, 2.0f, 0.0f));
            FixedJoint joint = second.gameObject.AddComponent<FixedJoint>();
            joint.connectedBody = first;
            first.linearVelocity = second.linearVelocity = new Vector3(1.0f, 0.0f, 0.0f);

            Simulate(10);

            Assert.That(joint.currentForce.magnitude, Is.LessThan(0.01f));
        }

        [Test]
        public void JoinedBodiesDoNotCollideByDefault()
        {
            Rigidbody first = WeightlessBox(Vector3.zero);
            Rigidbody second = WeightlessBox(new Vector3(0.5f, 0.0f, 0.0f));
            ConfigurableJoint joint = second.gameObject.AddComponent<ConfigurableJoint>();
            joint.connectedBody = first;

            Simulate(25);

            Assert.That((second.position - first.position).magnitude, Is.EqualTo(0.5f).Within(AnchorTolerance));
        }

        [Test]
        public void EnableCollisionLetsJoinedBodiesCollide()
        {
            Rigidbody first = WeightlessBox(Vector3.zero);
            Rigidbody second = WeightlessBox(new Vector3(0.5f, 0.0f, 0.0f));
            ConfigurableJoint joint = second.gameObject.AddComponent<ConfigurableJoint>();
            joint.connectedBody = first;
            joint.xMotion = ConfigurableJointMotion.Free;
            joint.enableCollision = true;

            Simulate(100);

            Assert.That((second.position - first.position).magnitude, Is.GreaterThan(0.9f));
        }

        [Test]
        public void OneJointThatEnablesCollisionIsEnough()
        {
            Rigidbody first = WeightlessBox(Vector3.zero);
            Rigidbody second = WeightlessBox(new Vector3(0.5f, 0.0f, 0.0f));
            ConfigurableJoint allowing = second.gameObject.AddComponent<ConfigurableJoint>();
            allowing.connectedBody = first;
            allowing.xMotion = ConfigurableJointMotion.Free;
            allowing.enableCollision = true;
            ConfigurableJoint disabling = second.gameObject.AddComponent<ConfigurableJoint>();
            disabling.connectedBody = first;
            disabling.xMotion = ConfigurableJointMotion.Free;

            Simulate(100);

            Assert.That((second.position - first.position).magnitude, Is.GreaterThan(0.9f));
        }

        [Test]
        public void DestroyingTheConnectedBodyFreesTheOwner()
        {
            Rigidbody first = DynamicBox(new Vector3(0.0f, 5.0f, 0.0f));
            first.isKinematic = true;
            Rigidbody second = DynamicBox(new Vector3(0.0f, 3.0f, 0.0f));
            ConfigurableJoint joint = Lock(second, first, new Vector3(0.0f, 1.0f, 0.0f), new Vector3(0.0f, -1.0f, 0.0f));
            Simulate(5);

            Object.DestroyImmediate(first.gameObject);
            Simulate(25);

            Assert.That(second.position.y, Is.LessThan(2.8f));
            Assert.That(joint == null || joint.connectedBody == null, Is.True);
        }

        [Test]
        public void DestroyingABodyTheJointWasMovedToFreesTheOwner()
        {
            Rigidbody first = DynamicBox(new Vector3(0.0f, 5.0f, 0.0f));
            first.isKinematic = true;
            Rigidbody moved = DynamicBox(new Vector3(3.0f, 5.0f, 0.0f));
            moved.isKinematic = true;
            Rigidbody second = DynamicBox(new Vector3(0.0f, 3.0f, 0.0f));
            ConfigurableJoint joint = Lock(second, first, new Vector3(0.0f, 1.0f, 0.0f), new Vector3(0.0f, -1.0f, 0.0f));
            Simulate(5);

            joint.connectedBody = moved;
            Simulate(5);
            float held = second.position.y;
            Object.DestroyImmediate(moved.gameObject);
            Simulate(25);

            Assert.That(second.position.y, Is.LessThan(held - 0.5f));
        }

        [Test]
        public void DestroyingTheOwnerLeavesTheConnectedBodySimulated()
        {
            Rigidbody first = DynamicBox(new Vector3(0.0f, 5.0f, 0.0f));
            Rigidbody second = WeightlessBox(new Vector3(0.0f, 3.0f, 0.0f));
            Lock(second, first, new Vector3(0.0f, 1.0f, 0.0f), new Vector3(0.0f, -1.0f, 0.0f));
            Simulate(5);

            Object.DestroyImmediate(second.gameObject);
            Simulate(5);

            Assert.That(first.position.y, Is.LessThan(5.0f));
        }

        [Test]
        public void HeldBodyReportsItsWeightAsJointForce()
        {
            Rigidbody body = DynamicBox(new Vector3(0.0f, 5.0f, 0.0f), 3.0f);
            ConfigurableJoint joint = Lock(body, null, Vector3.zero, new Vector3(0.0f, 5.0f, 0.0f));

            Simulate(50);

            AssertVector(new Vector3(0.0f, 3.0f * 9.81f, 0.0f), joint.currentForce, 0.5f, "force");
            AssertVector(Vector3.zero, joint.currentTorque, 0.5f, "torque");
        }

        [Test]
        public void CantileverReportsItsMomentAsJointTorque()
        {
            Rigidbody body = DynamicBox(new Vector3(2.0f, 5.0f, 0.0f), 3.0f);
            ConfigurableJoint joint = Lock(body, null, new Vector3(-2.0f, 0.0f, 0.0f), new Vector3(0.0f, 5.0f, 0.0f));

            Simulate(50);

            AssertVector(new Vector3(0.0f, 3.0f * 9.81f, 0.0f), joint.currentForce, 0.5f, "force");
            AssertVector(new Vector3(0.0f, 0.0f, 2.0f * 3.0f * 9.81f), joint.currentTorque, 1.0f, "torque");
        }

        [Test]
        public void JointBreaksOverItsBreakForce()
        {
            Rigidbody body = DynamicBox(new Vector3(0.0f, 5.0f, 0.0f), 3.0f);
            ConfigurableJoint joint = Lock(body, null, Vector3.zero, new Vector3(0.0f, 5.0f, 0.0f));
            joint.breakForce = 10.0f;

            Simulate(50);

            Assert.That(body.position.y, Is.LessThan(4.5f));
        }

        [Test]
        public void AnImpulseFarOverTheBreakForceBreaksAJointInItsStep()
        {
            // Under PhysX a one-step impulse breaks a joint once its force over the step passes the break force. The retooled
            // backend counts only part of a short jolt, so these two sit well either side of both.
            foreach (float impulse in new[] { 10.0f, 300.0f })
            {
                Rigidbody body = WeightlessBox(new Vector3(impulse, 0.0f, 0.0f));
                ConfigurableJoint joint = Lock(body, null, Vector3.zero, new Vector3(impulse, 0.0f, 0.0f));
                joint.breakForce = 1000.0f;
                Simulate();

                body.AddForce(new Vector3(0.0f, impulse, 0.0f), ForceMode.Impulse);
                Simulate();

                // The joint soaked up that impulse in the step it broke, so a push shows whether it is still there.
                body.AddForce(new Vector3(0.0f, 1.0f, 0.0f), ForceMode.Impulse);
                Simulate(25);

                Assert.That(body.position.y > 0.1f, Is.EqualTo(impulse > 100.0f), $"impulse {impulse}");
            }
        }

        [Test]
        public void JointHoldsUnderItsBreakForce()
        {
            Rigidbody body = DynamicBox(new Vector3(0.0f, 5.0f, 0.0f), 3.0f);
            ConfigurableJoint joint = Lock(body, null, Vector3.zero, new Vector3(0.0f, 5.0f, 0.0f));
            joint.breakForce = 60.0f;

            Simulate(50);

            Assert.That(body.position.y, Is.EqualTo(5.0f).Within(AnchorTolerance));
        }

        [Test]
        public void JointBreaksOverItsBreakTorque()
        {
            Rigidbody body = DynamicBox(new Vector3(2.0f, 5.0f, 0.0f), 3.0f);
            ConfigurableJoint joint = Lock(body, null, new Vector3(-2.0f, 0.0f, 0.0f), new Vector3(0.0f, 5.0f, 0.0f));
            joint.breakTorque = 20.0f;

            Simulate(50);

            Assert.That(body.position.y, Is.LessThan(4.5f));
        }

        [Test]
        public void LinearDriveToAConnectedBodyActsOnBoth()
        {
            Rigidbody first = Weightless(Vector3.zero);
            Rigidbody second = Weightless(new Vector3(3.0f, 0.0f, 0.0f));
            ConfigurableJoint joint = Lock(second, first, Vector3.zero, new Vector3(3.0f, 0.0f, 0.0f));
            joint.xMotion = ConfigurableJointMotion.Free;
            joint.xDrive = new JointDrive { positionSpring = 100.0f, positionDamper = 20.0f, maximumForce = float.MaxValue };
            joint.targetPosition = new Vector3(1.0f, 0.0f, 0.0f);

            Simulate(300);

            Assert.That(second.position.x - first.position.x, Is.EqualTo(2.0f).Within(0.02f));
            Assert.That(first.position.x + second.position.x, Is.EqualTo(3.0f).Within(0.02f));
        }

        [Test]
        public void RigidbodyOnASoftLimitComesBackToIt()
        {
            Rigidbody body = Weightless(Vector3.zero);
            ConfigurableJoint joint = Lock(body, null, Vector3.zero, Vector3.zero);
            joint.xMotion = ConfigurableJointMotion.Limited;
            joint.linearLimit = new SoftJointLimit { limit = 0.5f };
            joint.linearLimitSpring = new SoftJointLimitSpring { spring = 50.0f, damper = 14.0f };

            body.linearVelocity = new Vector3(5.0f, 0.0f, 0.0f);
            Simulate(25);
            Assert.That(body.position.x, Is.GreaterThan(0.55f), "past the limit, on its spring");

            Simulate(300);
            Assert.That(Mathf.Abs(body.position.x), Is.LessThan(0.53f), "back inside the limit");
        }

        [Test]
        public void SpringJointPullsBackToItsDistance()
        {
            Rigidbody body = Weightless(new Vector3(3.0f, 0.0f, 0.0f));
            SpringJoint spring = body.gameObject.AddComponent<SpringJoint>();
            spring.autoConfigureConnectedAnchor = false;
            spring.connectedAnchor = Vector3.zero;
            spring.spring = 50.0f;
            spring.damper = 14.0f;
            spring.minDistance = 0.0f;
            spring.maxDistance = 1.0f;

            Simulate(300);

            Assert.That(Mathf.Abs(body.position.x), Is.LessThan(1.03f), "back inside its distance");
        }

        [Test]
        public void CharacterJointStopsAtItsSwingLimit()
        {
            Rigidbody body = Weightless(Vector3.zero);
            CharacterJoint joint = body.gameObject.AddComponent<CharacterJoint>();
            joint.axis = Vector3.right;
            joint.swingAxis = Vector3.up;
            joint.swing1Limit = new SoftJointLimit { limit = 20.0f };
            joint.swing2Limit = new SoftJointLimit { limit = 20.0f };

            body.angularVelocity = new Vector3(0.0f, 1.0f, 0.0f);
            Simulate(100);

            Assert.That(Quaternion.Angle(Quaternion.identity, body.rotation), Is.EqualTo(20.0f).Within(2.0f));
        }
    }
}
