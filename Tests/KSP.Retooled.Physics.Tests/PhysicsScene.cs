using System.Collections.Generic;
using NUnit.Framework;
using UnityEngine;

namespace KSP.Retooled.Physics.Tests
{
    /// <summary>
    /// Provides a scene that conformance fixtures step explicitly at a fixed rate under Earth-like gravity. Each test
    /// checks one behaviour, with the same assertions for PhysX and the Unity Physics backend.
    /// </summary>
    public abstract class PhysicsScene
    {
        protected const float Step = 0.02f;
        protected const float Gravity = -9.81f;

        private readonly List<GameObject> created = new List<GameObject>();
        private SimulationMode simulationMode;
        private Vector3 gravity;
        private bool autoSyncTransforms;

        [SetUp]
        public void SetUpScene()
        {
            simulationMode = UnityEngine.Physics.simulationMode;
            gravity = UnityEngine.Physics.gravity;
            autoSyncTransforms = UnityEngine.Physics.autoSyncTransforms;

            UnityEngine.Physics.simulationMode = SimulationMode.Script;
            UnityEngine.Physics.gravity = new Vector3(0.0f, Gravity, 0.0f);
            UnityEngine.Physics.autoSyncTransforms = true;
        }

        [TearDown]
        public void TearDownScene()
        {
            foreach (GameObject gameObject in created)
            {
                Object.DestroyImmediate(gameObject);
            }

            created.Clear();
            UnityEngine.Physics.simulationMode = simulationMode;
            UnityEngine.Physics.gravity = gravity;
            UnityEngine.Physics.autoSyncTransforms = autoSyncTransforms;
        }

        protected GameObject Track(GameObject gameObject)
        {
            created.Add(gameObject);
            return gameObject;
        }

        protected GameObject Box(Vector3 position, Vector3 size)
        {
            GameObject box = Track(new GameObject("box"));
            box.transform.position = position;
            box.AddComponent<BoxCollider>().size = size;
            return box;
        }

        protected Rigidbody DynamicBox(Vector3 position, float mass = 1.0f)
        {
            Rigidbody body = Box(position, Vector3.one).AddComponent<Rigidbody>();
            body.mass = mass;
            return body;
        }

        protected Rigidbody WeightlessBox(Vector3 position, float mass = 1.0f)
        {
            Rigidbody body = DynamicBox(position, mass);
            body.useGravity = false;
            return body;
        }

        /// <summary>A static box whose top face lies at y = 0.</summary>
        protected GameObject Ground()
        {
            return Box(new Vector3(0.0f, -0.5f, 0.0f), new Vector3(20.0f, 1.0f, 20.0f));
        }

        protected static void Simulate(int steps = 1)
        {
            for (int i = 0; i < steps; i++)
            {
                UnityEngine.Physics.Simulate(Step);
            }
        }

        protected static void AssertVector(Vector3 expected, Vector3 actual, float tolerance, string what = "value")
        {
            Assert.That((expected - actual).magnitude, Is.LessThanOrEqualTo(tolerance), $"{what}: expected {expected:F5}, got {actual:F5}");
        }
    }
}
