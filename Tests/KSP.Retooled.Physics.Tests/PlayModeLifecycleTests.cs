using System.Collections;
using System.Collections.Generic;
using System.Linq;
using NUnit.Framework;
using UnityEditor;
using UnityEngine;
using UnityEngine.TestTools;

namespace KSP.Retooled.Physics.Tests
{
    /// <summary>
    /// Tests scenes during script reloads and play mode using the engine's own loop. Other fixtures step scenes
    /// explicitly and skip that loop. A reload clears the fixture's fields, so these tests store state in the scene and
    /// SessionState.
    /// </summary>
    public class PlayModeLifecycleTests
    {
        private const string BodyName = "lifecycle-body";
        private const string GroundName = "lifecycle-ground";
        private const string HeightKey = "PlayModeLifecycleTests.Height";
        private const string ModeKey = "PlayModeLifecycleTests.Mode";
        private const string GravityKey = "PlayModeLifecycleTests.Gravity";

        private const float Speed = 10.0f;

        // A fixed step takes many frames here. Interpolation has nothing to blend until the body has stepped twice.
        private const float SettlingTime = 0.1f;

        /// <summary>The project's gravity is zero, because KSP applies its own.</summary>
        private static readonly Vector3 TestGravity = new Vector3(0.0f, -9.81f, 0.0f);

        private static Rigidbody BodyInScene()
        {
            GameObject found = GameObject.Find(BodyName);
            Assert.That(found, Is.Not.Null, "the body is still in the scene");
            return found.GetComponent<Rigidbody>();
        }

        private static void BuildScene()
        {
            GameObject ground = new GameObject(GroundName);
            ground.transform.position = new Vector3(0.0f, -0.5f, 0.0f);
            ground.AddComponent<BoxCollider>().size = new Vector3(20.0f, 1.0f, 20.0f);

            GameObject body = new GameObject(BodyName);
            body.transform.position = new Vector3(0.0f, 2.0f, 0.0f);
            body.AddComponent<BoxCollider>();
            body.AddComponent<Rigidbody>();
        }

        private static void DestroyScene()
        {
            foreach (string name in new[] { BodyName, GroundName })
            {
                GameObject found = GameObject.Find(name);
                if (found != null)
                {
                    Object.DestroyImmediate(found);
                }
            }
        }

        private static void Step(int steps)
        {
            for (int i = 0; i < steps; i++)
            {
                UnityEngine.Physics.Simulate(0.02f);
            }
        }

        /// <summary>
        /// Returns the fixed time after <paramref name="steps"/> additional steps. Edit mode tests run only their own
        /// iterator, so they wait within that iterator.
        /// </summary>
        private static float FixedTimeAfter(int steps)
        {
            return Time.fixedTime + steps * Time.fixedDeltaTime;
        }

        /// <summary>The runner sets a test up again after a reload, so only the first time counts.</summary>
        [SetUp]
        public void RememberSettings()
        {
            if (SessionState.GetInt(ModeKey, -1) == -1)
            {
                SessionState.SetInt(ModeKey, (int)UnityEngine.Physics.simulationMode);
                SessionState.SetVector3(GravityKey, UnityEngine.Physics.gravity);
            }

            UnityEngine.Physics.gravity = TestGravity;
        }

        /// <summary>Runs on a new fixture after a reload, so it works from the scene and SessionState alone.</summary>
        [TearDown]
        public void RestoreSettings()
        {
            DestroyScene();
            UnityEngine.Physics.simulationMode = (SimulationMode)SessionState.GetInt(ModeKey, (int)SimulationMode.FixedUpdate);
            UnityEngine.Physics.gravity = SessionState.GetVector3(GravityKey, Vector3.zero);
            SessionState.EraseInt(ModeKey);
        }

        [UnityTest]
        public IEnumerator ABodyKeepsFallingAcrossAScriptReload()
        {
            UnityEngine.Physics.simulationMode = SimulationMode.Script;
            BuildScene();
            UnityEngine.Physics.SyncTransforms();
            Step(10);
            SessionState.SetFloat(HeightKey, BodyInScene().position.y);

            EditorUtility.RequestScriptReload();
            yield return new WaitForDomainReload();

            Rigidbody body = BodyInScene();
            float height = SessionState.GetFloat(HeightKey, float.NaN);
            Assert.That(body.position.y, Is.EqualTo(height).Within(1e-4f), "the pose survives");
            Assert.That(body.linearVelocity.y, Is.LessThan(-0.5f), "the velocity survives");

            Step(1);
            Assert.That(body.position.y, Is.LessThan(height), "the step still runs");

            Step(200);
            Assert.That(body.position.y, Is.EqualTo(0.5f).Within(0.03f), "colliders work again");
            Assert.That(UnityEngine.Physics.Raycast(new Vector3(0.0f, 5.0f, 0.0f), Vector3.down, 10.0f), Is.True);
        }

        [UnityTest]
        public IEnumerator ASceneWithBodiesSurvivesPlayingAndStopping()
        {
            BuildScene();

            yield return new EnterPlayMode();

            Assert.That(Application.isPlaying, Is.True);
            UnityEngine.Physics.gravity = TestGravity;
            Rigidbody body = BodyInScene();
            float start = body.position.y;
            float end = FixedTimeAfter(5);
            while (Time.fixedTime < end)
            {
                yield return null;
            }

            Assert.That(body.position.y, Is.LessThan(start), "the engine's fixed update steps the world");

            yield return new ExitPlayMode();

            Assert.That(Application.isPlaying, Is.False);
            body = BodyInScene();
            Assert.That(body.position.y, Is.EqualTo(2.0f).Within(1e-4f), "edit mode has its own scene again");

            UnityEngine.Physics.gravity = TestGravity;
            UnityEngine.Physics.simulationMode = SimulationMode.Script;
            Step(10);
            Assert.That(body.position.y, Is.LessThan(2.0f), "edit mode still steps");
        }

        private static InterpolationSampler StartMotion(RigidbodyInterpolation mode)
        {
            UnityEngine.Physics.gravity = Vector3.zero;
            GameObject moving = new GameObject("moving");
            Rigidbody body = moving.AddComponent<Rigidbody>();
            body.useGravity = false;
            body.interpolation = mode;
            body.linearVelocity = new Vector3(Speed, 0.0f, 0.0f);
            return new InterpolationSampler(body);
        }

        private static IEnumerable<InterpolationSampler.Sample> Settled(InterpolationSampler sampler)
        {
            float start = sampler.Samples[0].Time + SettlingTime;
            return sampler.Samples.Where(sample => sample.Time > start);
        }

        private static float StepFraction(InterpolationSampler.Sample sample)
        {
            return (sample.Time - sample.FixedTime) / Time.fixedDeltaTime;
        }

        private static void AssertSampledBetweenSteps(InterpolationSampler sampler)
        {
            int between = 0;
            foreach (InterpolationSampler.Sample sample in sampler.Samples)
            {
                float fraction = StepFraction(sample);
                if (fraction > 0.1f && fraction < 0.9f)
                {
                    between++;
                }
            }

            Assert.That(between, Is.GreaterThan(20), $"{sampler.Samples.Count} frames, too few of them between fixed steps");
        }

        internal class DestroyOnLanding : MonoBehaviour
        {
            private void OnCollisionEnter(Collision collision)
            {
                Destroy(gameObject);
            }
        }

        internal class DestroyJointOwnerOnBreak : MonoBehaviour
        {
            private void OnJointBreak(float breakForce)
            {
                Destroy(gameObject);
            }
        }

        [UnityTest]
        public IEnumerator TheEngineLoopDeliversACollisionThatDestroysItsBody()
        {
            yield return new EnterPlayMode();

            UnityEngine.Physics.gravity = TestGravity;
            BuildScene();
            BodyInScene().gameObject.AddComponent<DestroyOnLanding>();

            float end = FixedTimeAfter(100);
            while (Time.fixedTime < end && GameObject.Find(BodyName) != null)
            {
                yield return null;
            }

            Assert.That(GameObject.Find(BodyName), Is.Null);

            yield return new ExitPlayMode();
        }

        [UnityTest]
        public IEnumerator TheEngineLoopDeliversAJointBreak()
        {
            yield return new EnterPlayMode();

            UnityEngine.Physics.gravity = TestGravity;
            BuildScene();
            GameObject body = BodyInScene().gameObject;
            body.AddComponent<DestroyJointOwnerOnBreak>();
            body.AddComponent<FixedJoint>().breakForce = 0.1f;

            float end = FixedTimeAfter(20);
            while (Time.fixedTime < end && GameObject.Find(BodyName) != null)
            {
                yield return null;
            }

            Assert.That(GameObject.Find(BodyName), Is.Null);

            yield return new ExitPlayMode();
        }

        [UnityTest]
        public IEnumerator InterpolationLagsTheBodyByThePartOfAStepNotYetShown()
        {
            yield return new EnterPlayMode();

            InterpolationSampler sampler = StartMotion(RigidbodyInterpolation.Interpolate);
            float end = FixedTimeAfter(100);
            while (Time.fixedTime < end)
            {
                yield return null;
            }

            sampler.Dispose();

            AssertSampledBetweenSteps(sampler);
            float stepLength = Speed * Time.fixedDeltaTime;
            foreach (InterpolationSampler.Sample sample in Settled(sampler))
            {
                float expected = sample.Body - stepLength * (1.0f - StepFraction(sample));
                Assert.That(sample.Transform, Is.EqualTo(expected).Within(0.1f * stepLength), $"at {sample.Time}, step at {sample.FixedTime}");
            }

            yield return new ExitPlayMode();
        }

        [UnityTest]
        public IEnumerator ExtrapolationLeadsTheBodyByTheTimeSinceTheStep()
        {
            yield return new EnterPlayMode();

            InterpolationSampler sampler = StartMotion(RigidbodyInterpolation.Extrapolate);
            float end = FixedTimeAfter(100);
            while (Time.fixedTime < end)
            {
                yield return null;
            }

            sampler.Dispose();

            AssertSampledBetweenSteps(sampler);
            float stepLength = Speed * Time.fixedDeltaTime;
            foreach (InterpolationSampler.Sample sample in Settled(sampler))
            {
                float expected = sample.Body + stepLength * StepFraction(sample);
                Assert.That(sample.Transform, Is.EqualTo(expected).Within(0.1f * stepLength), $"at {sample.Time}, step at {sample.FixedTime}");
            }

            yield return new ExitPlayMode();
        }

        [UnityTest]
        public IEnumerator WithoutInterpolationTheTransformFollowsTheBody()
        {
            yield return new EnterPlayMode();

            InterpolationSampler sampler = StartMotion(RigidbodyInterpolation.None);
            float end = FixedTimeAfter(100);
            while (Time.fixedTime < end)
            {
                yield return null;
            }

            sampler.Dispose();

            foreach (InterpolationSampler.Sample sample in Settled(sampler))
            {
                Assert.That(sample.Transform, Is.EqualTo(sample.Body).Within(1e-3f), $"at {sample.Time}, step at {sample.FixedTime}");
            }

            yield return new ExitPlayMode();
        }
    }
}
