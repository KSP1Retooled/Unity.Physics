using NUnit.Framework;
using UnityEngine;

namespace KSP.Retooled.Physics.Tests
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
