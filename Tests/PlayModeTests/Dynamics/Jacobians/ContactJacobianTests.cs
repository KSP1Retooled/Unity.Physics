using System;
using NUnit.Framework;
using Unity.Collections;
using Assert = UnityEngine.Assertions.Assert;
using PhysicsSolver = Unity.Physics.Solver;

namespace Unity.Physics.Tests.Dynamics.Jacobians
{
    class ContactJacobiansTests
    {
        [Test]
        public void SolveTest()
        {
            var jacobian = new ContactJacobian();

            var jacHeader = new JacobianHeader();
            var velocityA = MotionVelocity.Zero;
            var velocityB = MotionVelocity.Zero;
            var stepInput = new PhysicsSolver.StepInput();
            var collisionEventsWriter = new NativeStream.Writer();

            jacobian.SolveContact(ref jacHeader, ref velocityA, ref velocityB, stepInput, ref collisionEventsWriter,
                false, PhysicsSolver.MotionStabilizationInput.Default, PhysicsSolver.MotionStabilizationInput.Default);

            Assert.AreEqual(new JacobianHeader(), jacHeader);
            Assert.AreEqual(MotionVelocity.Zero, velocityA);
            Assert.AreEqual(MotionVelocity.Zero, velocityB);
            Assert.AreEqual(new NativeStream.Writer(), collisionEventsWriter);
        }
    }
}
