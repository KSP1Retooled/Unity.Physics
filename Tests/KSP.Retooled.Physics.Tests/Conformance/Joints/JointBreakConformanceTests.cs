using PhysicsScene = KSP.Retooled.Physics.Tests.Scenes.PhysicsScene;
using NUnit.Framework;
using UnityEngine;

namespace KSP.Retooled.Physics.Tests.Conformance.Joints
{
    /// <summary>What a script sees when a joint breaks.</summary>
    public class JointBreakConformanceTests : PhysicsScene
    {
        [ExecuteAlways]
        internal class BreakLog : MonoBehaviour
        {
            public int Breaks;
            public float Force;

            private void OnJointBreak(float breakForce)
            {
                Breaks++;
                Force = breakForce;
            }
        }

        private static FixedJoint HangFromTheWorld(Rigidbody body, float breakForce)
        {
            FixedJoint joint = body.gameObject.AddComponent<FixedJoint>();
            joint.breakForce = breakForce;
            return joint;
        }

        [Test]
        public void OnJointBreakFiresOnceWithTheForceThatBrokeIt()
        {
            Rigidbody body = DynamicBox(new Vector3(0.0f, 5.0f, 0.0f), 3.0f);
            BreakLog log = body.gameObject.AddComponent<BreakLog>();
            HangFromTheWorld(body, 10.0f);

            Simulate(20);

            Assert.That(log.Breaks, Is.EqualTo(1));
            Assert.That(log.Force, Is.GreaterThan(10.0f));
        }

        [Test]
        public void JointToWorldBreaksWhenLockedAndLimitedAxesAreLoadedTogether()
        {
            Rigidbody body = WeightlessBox(Vector3.up * 0.5f);
            body.GetComponent<Collider>().enabled = false;
            body.linearDamping = 0.0f;
            BreakLog log = body.gameObject.AddComponent<BreakLog>();
            ConfigurableJoint joint = body.gameObject.AddComponent<ConfigurableJoint>();
            joint.autoConfigureConnectedAnchor = false;
            joint.connectedAnchor = Vector3.zero;
            joint.xMotion = ConfigurableJointMotion.Locked;
            joint.yMotion = joint.zMotion = ConfigurableJointMotion.Limited;
            joint.linearLimit = new SoftJointLimit { limit = 0.5f };
            const float breakForce = 15.0f;
            joint.breakForce = breakForce;
            body.linearVelocity = new Vector3(2.0f, 2.0f, 0.0f);

            Simulate();

            Assert.That(log.Breaks, Is.EqualTo(1));
            Assert.That(log.Force, Is.GreaterThan(breakForce));
            TestContext.WriteLine($"Break force reported by OnJointBreak: {log.Force:R} N");
        }

        [Test]
        public void AJointUnderItsLimitStaysQuiet()
        {
            Rigidbody body = DynamicBox(new Vector3(0.0f, 5.0f, 0.0f), 3.0f);
            BreakLog log = body.gameObject.AddComponent<BreakLog>();
            HangFromTheWorld(body, 1e6f);

            Simulate(20);

            Assert.That(log.Breaks, Is.Zero);
        }

        [Test]
        public void ABrokenJointIsRemovedFromItsObject()
        {
            Rigidbody body = DynamicBox(new Vector3(0.0f, 5.0f, 0.0f), 3.0f);
            HangFromTheWorld(body, 10.0f);

            Simulate(20);

            Assert.That(body.GetComponent<FixedJoint>() == null, Is.True);
        }
    }
}
