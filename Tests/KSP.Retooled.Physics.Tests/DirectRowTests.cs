using System;
using System.Collections.Generic;
using NUnit.Framework;
using Unity.Physics;
using Unity.Physics.Dynamics.Solver.Sparse;
using UnityEngine;
using Unity.Physics.Configuration;

namespace KSP.Retooled.Physics.Tests
{
    /// <summary>
    /// Require bitwise-identical direct solver inputs from optimized and reference row builders on sparse and
    /// forest paths.
    /// </summary>
    public unsafe class DirectRowTests : PhysicsScene
    {
        private const float PartHeight = 2.0f;

        private static readonly object Gate = new object();
        private static List<string> problems;

        private bool wasEnabled;
        private bool wasLocking;
        private bool wasLegacy;

        [SetUp]
        public void RememberSwitches()
        {
            if (!BackendInfo.IsRetooled)
            {
                Assert.Ignore("The direct rows are part of the Unity Physics backend.");
            }

            // The environment's switches apply when the first world attaches; read them now, so they can't land on top of
            // what this fixture sets.
            ForestSwitches.Initialize();
            wasEnabled = ForestSwitches.Enabled;
            wasLocking = ForestSwitches.Locking;
            wasLegacy = Solver.LegacyRowBuild;

            // A locked section depends on what the steps before it carried, so the second of two runs would start from what
            // the first left. Locking only reads the rows, which are the same either way.
            ForestSwitches.Locking = false;
        }

        [TearDown]
        public void RestoreSwitches()
        {
            SparseProblemObserver.Set(null);
            ForestSwitches.Enabled = wasEnabled;
            ForestSwitches.Locking = wasLocking;
            Solver.LegacyRowBuild = wasLegacy;
        }

        private static void Observe(SparseConstraintSolver.Problem* problem, float* impulses, byte bounded)
        {
            // Runs on a worker in the middle of a solve, so it only hashes.
            int rows = problem->RowCount;
            ulong hash = 14695981039346656037ul;
            hash = Hash(hash, (byte*)problem->Jacobian, rows * SparseConstraintSolver.ColumnsPerRow * sizeof(float));
            hash = Hash(hash, (byte*)problem->Epsilon, rows * sizeof(float));
            hash = Hash(hash, (byte*)problem->RightHandSide, rows * sizeof(float));
            hash = Hash(hash, (byte*)problem->Lower, rows * sizeof(float));
            hash = Hash(hash, (byte*)problem->Upper, rows * sizeof(float));
            if (problem->InitialStates != null)
            {
                hash = Hash(hash, problem->InitialStates, rows);
            }

            lock (Gate)
            {
                problems?.Add($"{rows}:{bounded}:{hash:X16}");
            }
        }

        private static ulong Hash(ulong hash, byte* bytes, int count)
        {
            for (int i = 0; i < count; i++)
            {
                hash = (hash ^ bytes[i]) * 1099511628211ul;
            }

            return hash;
        }

        private static readonly SparseProblemObserver.Callback Observer = Observe;

        private static ConfigurableJoint Joint(Rigidbody host, Rigidbody target, Vector3 anchor)
        {
            ConfigurableJoint joint = host.gameObject.AddComponent<ConfigurableJoint>();
            joint.connectedBody = target;
            joint.autoConfigureConnectedAnchor = false;
            joint.anchor = host.transform.InverseTransformPoint(anchor);
            joint.connectedAnchor = target != null ? target.transform.InverseTransformPoint(anchor) : anchor;
            return joint;
        }

        /// <summary>A part joint: limits far away, drives doing the holding, three over a node so they group.</summary>
        private static void PartJoint(Rigidbody host, Rigidbody target, Vector3 node)
        {
            for (int i = 0; i < 3; i++)
            {
                ConfigurableJoint joint = Joint(host, target, node + Quaternion.AngleAxis(120.0f * i, Vector3.up) * new Vector3(0.5f, 0.0f, 0.0f));
                joint.xMotion = joint.yMotion = joint.zMotion = ConfigurableJointMotion.Limited;
                joint.angularXMotion = joint.angularYMotion = joint.angularZMotion = ConfigurableJointMotion.Limited;
                joint.linearLimit = new SoftJointLimit { limit = 1.0f };
                joint.highAngularXLimit = new SoftJointLimit { limit = 177.0f };
                joint.lowAngularXLimit = new SoftJointLimit { limit = -177.0f };
                joint.angularYLimit = new SoftJointLimit { limit = 177.0f };
                joint.angularZLimit = new SoftJointLimit { limit = 177.0f };
                joint.xDrive = joint.yDrive = joint.zDrive = new JointDrive { positionSpring = 1e20f, maximumForce = 1e20f };
                joint.rotationDriveMode = RotationDriveMode.XYAndZ;
                joint.angularXDrive = joint.angularYZDrive = new JointDrive { positionSpring = 20000.0f, maximumForce = 1e20f };
            }
        }

        /// <summary>
        /// Build a landing column covering part joints, struts, motors, hinges, swing limits, linear drives and
        /// distance limits.
        /// </summary>
        private List<string> Run(bool forest, bool legacy, bool ground)
        {
            TearDownScene();
            SetUpScene();
            ForestSwitches.Enabled = forest;
            Solver.LegacyRowBuild = legacy;
            if (ground)
            {
                Ground();
            }

            float bottom = ground ? 0.3f : 6.0f;
            var bodies = new List<Rigidbody>();
            for (int i = 0; i < 7; i++)
            {
                Rigidbody body = Box(new Vector3(0.0f, bottom + PartHeight * (i + 0.5f), 0.0f), new Vector3(1.0f, PartHeight, 1.0f))
                    .AddComponent<Rigidbody>();
                body.mass = 4.0f;
                bodies.Add(body);
            }

            Vector3 Node(int i) => new Vector3(0.0f, bottom + PartHeight * i, 0.0f);
            PartJoint(bodies[1], bodies[0], Node(1));
            PartJoint(bodies[2], bodies[1], Node(2));

            HingeJoint motor = bodies[3].gameObject.AddComponent<HingeJoint>();
            motor.connectedBody = bodies[2];
            motor.anchor = new Vector3(0.0f, -0.5f, 0.0f);
            motor.axis = Vector3.up;
            motor.useMotor = true;
            motor.motor = new JointMotor { targetVelocity = 90.0f, force = 50.0f };

            HingeJoint limited = bodies[4].gameObject.AddComponent<HingeJoint>();
            limited.connectedBody = bodies[3];
            limited.anchor = new Vector3(0.0f, -0.5f, 0.0f);
            limited.axis = Vector3.right;
            limited.useLimits = true;
            limited.limits = new JointLimits { min = -10.0f, max = 15.0f };

            ConfigurableJoint swing = Joint(bodies[5], bodies[4], Node(5));
            swing.xMotion = swing.yMotion = swing.zMotion = ConfigurableJointMotion.Locked;
            swing.angularXMotion = ConfigurableJointMotion.Locked;
            swing.angularYMotion = swing.angularZMotion = ConfigurableJointMotion.Limited;
            swing.angularYLimit = new SoftJointLimit { limit = 20.0f };
            swing.angularZLimit = new SoftJointLimit { limit = 30.0f };
            swing.angularYZLimitSpring = new SoftJointLimitSpring { spring = 500.0f, damper = 5.0f };

            ConfigurableJoint slider = Joint(bodies[6], bodies[5], Node(6));
            slider.xMotion = ConfigurableJointMotion.Free;
            slider.yMotion = slider.zMotion = ConfigurableJointMotion.Locked;
            slider.angularXMotion = slider.angularYMotion = slider.angularZMotion = ConfigurableJointMotion.Locked;
            slider.xDrive = new JointDrive { positionSpring = 300.0f, positionDamper = 10.0f, maximumForce = 1000.0f };
            slider.targetPosition = new Vector3(0.2f, 0.0f, 0.0f);

            Strut(bodies[0], bodies[2]);
            SpringJoint spring = bodies[6].gameObject.AddComponent<SpringJoint>();
            spring.connectedBody = bodies[3];
            spring.minDistance = 0.0f;
            spring.maxDistance = 0.5f;
            spring.spring = 100.0f;

            bodies[6].linearVelocity = new Vector3(1.0f, 0.0f, 0.5f);
            bodies[0].angularVelocity = new Vector3(0.0f, 0.0f, 1.0f);

            var seen = new List<string>();
            lock (Gate)
            {
                problems = seen;
            }

            SparseProblemObserver.Set(Observer);
            Simulate(60);
            SparseProblemObserver.Set(null);
            lock (Gate)
            {
                problems = null;
            }

            return seen;
        }

        private static void Strut(Rigidbody host, Rigidbody target)
        {
            ConfigurableJoint joint = host.gameObject.AddComponent<ConfigurableJoint>();
            joint.connectedBody = target;
            joint.xMotion = joint.yMotion = joint.zMotion = ConfigurableJointMotion.Locked;
            joint.angularXMotion = joint.angularYMotion = joint.angularZMotion = ConfigurableJointMotion.Locked;
        }

        [TestCase(false, false)]
        [TestCase(false, true)]
        [TestCase(true, false)]
        [TestCase(true, true)]
        public void RowsMatchTheSeparateBuildBitForBit(bool forest, bool ground)
        {
            List<string> legacy = Run(forest, true, ground);
            List<string> fast = Run(forest, false, ground);
            Debug.Log($"[DirectRowTests] forest {forest}, ground {ground}: {legacy.Count} problems compared");
            Assert.That(legacy.Count, Is.GreaterThan(0), "no direct problem was solved");
            Assert.That(fast, Is.EqualTo(legacy));
        }
    }
}
