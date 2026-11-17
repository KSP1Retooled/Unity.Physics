using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using NUnit.Framework;
using Unity.Collections;
using Unity.Jobs;
using UnityEngine;

namespace KSP.Retooled.Physics.Tests
{
    /// <summary>
    /// Compare queries and bitwise bounds with cached shape geometry enabled and disabled.
    /// </summary>
    public unsafe class ShapeBoundsQueryTests : PhysicsScene
    {
        [SetUp]
        public void RequireTheBackend()
        {
            Assume.That(BackendInfo.IsRetooled, "needs --backend retooled");
        }

        private sealed class Scene
        {
            public readonly List<Rigidbody> Bodies = new List<Rigidbody>();
            public readonly Dictionary<UnityEngine.Collider, int> Index = new Dictionary<UnityEngine.Collider, int>();
        }

        private Mesh CubeMesh()
        {
            GameObject cube = GameObject.CreatePrimitive(PrimitiveType.Cube);
            Mesh mesh = cube.GetComponent<MeshFilter>().sharedMesh;
            Object.DestroyImmediate(cube);
            return mesh;
        }

        /// <summary>
        /// Adds a weightless body of one of five kinds: a box, a sphere, a capsule, a compound of two boxes, and a convex
        /// mesh. The body spins and drifts, so that its pose changes in every step.
        /// </summary>
        private Rigidbody AddBody(Scene scene, int i, Vector3 position, Mesh cube)
        {
            GameObject body = Track(new GameObject($"body {i}"));
            body.transform.SetPositionAndRotation(position, Quaternion.Euler(13.0f * i, 7.0f * i, 3.0f * i));
            switch (i % 5)
            {
                case 0:
                    body.AddComponent<BoxCollider>().size = new Vector3(1.0f, 0.6f, 0.8f);
                    break;
                case 1:
                    body.AddComponent<SphereCollider>().radius = 0.5f;
                    break;
                case 2:
                    CapsuleCollider capsule = body.AddComponent<CapsuleCollider>();
                    capsule.radius = 0.3f;
                    capsule.height = 1.2f;
                    capsule.direction = i % 3;
                    break;
                case 3:
                    for (int c = 0; c < 2; c++)
                    {
                        var child = new GameObject("child");
                        child.transform.SetParent(body.transform, false);
                        child.transform.localPosition = new Vector3(c == 0 ? -0.4f : 0.4f, 0.0f, 0.0f);
                        child.transform.localRotation = Quaternion.Euler(0.0f, 30.0f * c, 0.0f);
                        child.AddComponent<BoxCollider>().size = new Vector3(0.5f, 0.5f, 0.5f);
                    }

                    break;
                default:
                    MeshCollider mesh = body.AddComponent<MeshCollider>();
                    mesh.sharedMesh = cube;
                    mesh.convex = true;
                    break;
            }

            Rigidbody rigidbody = body.AddComponent<Rigidbody>();
            rigidbody.useGravity = false;
            rigidbody.sleepThreshold = 0.0f;
            rigidbody.linearVelocity = new Vector3(0.1f * (i % 3 - 1), 0.05f * (i % 4), -0.1f);
            rigidbody.angularVelocity = new Vector3(0.7f, 1.3f * (i % 2 == 0 ? 1.0f : -1.0f), 0.4f);
            foreach (UnityEngine.Collider collider in body.GetComponentsInChildren<UnityEngine.Collider>())
            {
                scene.Index[collider] = scene.Bodies.Count;
            }

            scene.Bodies.Add(rigidbody);
            return rigidbody;
        }

        private static string Hit(Scene scene, bool hit, RaycastHit result)
        {
            if (!hit || result.collider == null)
            {
                return "none";
            }

            return string.Format(CultureInfo.InvariantCulture, "{0} {1:R} {2:R} {3:R} {4:R} {5:R} {6:R} {7:R}",
                scene.Index.TryGetValue(result.collider, out int index) ? index : -1,
                result.point.x, result.point.y, result.point.z, result.normal.x, result.normal.y, result.normal.z, result.distance);
        }

        /// <summary>
        /// Record centre-directed ray/sphere casts, sphere overlaps and worker ray hits after the step.
        /// </summary>
        private static void Query(Scene scene, StringBuilder log)
        {
            var rays = new NativeArray<RaycastCommand>(scene.Bodies.Count, Allocator.TempJob);
            var hits = new NativeArray<RaycastHit>(scene.Bodies.Count, Allocator.TempJob);
            try
            {
                for (int i = 0; i < scene.Bodies.Count; i++)
                {
                    Rigidbody body = scene.Bodies[i];
                    if (body == null)
                    {
                        // A destroyed body's ray points at nothing.
                        rays[i] = new RaycastCommand(new Vector3(0.0f, 1000.0f, 0.0f), Vector3.up, QueryParameters.Default, 1.0f);
                        continue;
                    }

                    Vector3 above = body.position + new Vector3(0.0f, 3.0f, 0.0f);
                    rays[i] = new RaycastCommand(above, Vector3.down, QueryParameters.Default, 6.0f);
                    bool hit = UnityEngine.Physics.Raycast(above, Vector3.down, out RaycastHit ray, 6.0f);
                    log.Append($"ray {i}: {Hit(scene, hit, ray)}\n");
                    hit = UnityEngine.Physics.SphereCast(above, 0.2f, Vector3.down, out RaycastHit sphere, 6.0f);
                    log.Append($"sphere cast {i}: {Hit(scene, hit, sphere)}\n");
                    IEnumerable<int> overlaps = UnityEngine.Physics.OverlapSphere(body.position, 0.6f)
                        .Select(c => scene.Index.TryGetValue(c, out int index) ? index : -1).Distinct().OrderBy(index => index);
                    log.Append($"overlap {i}: {string.Join(",", overlaps)}\n");
                }

                RaycastCommand.ScheduleBatch(rays, hits, 4).Complete();
                for (int i = 0; i < scene.Bodies.Count; i++)
                {
                    log.Append($"worker ray {i}: {Hit(scene, hits[i].collider != null, hits[i])}\n");
                }
            }
            finally
            {
                rays.Dispose();
                hits.Dispose();
            }
        }

        /// <summary>
        /// Runs the scene for 40 steps and returns what the queries found after the steps. Two steps in every five have
        /// no query after them. Between steps, the test moves a body by setting its pose, adds a body, and destroys one.
        /// </summary>
        private string Run(bool shapes)
        {
            DestroyAll();
            bool wasEnabled = Unity.Physics.DynamicBoundsCache.ShapesEnabled;
            bool wasChecking = Unity.Physics.DynamicBoundsCache.CheckEnabled;
            Unity.Physics.DynamicBoundsCache.ShapesEnabled = shapes;
            Unity.Physics.DynamicBoundsCache.CheckEnabled = true;
            try
            {
                var scene = new Scene();
                var log = new StringBuilder();
                Mesh cube = CubeMesh();
                for (int i = 0; i < 40; i++)
                {
                    AddBody(scene, i, new Vector3(2.5f * (i % 8), 0.0f, 2.5f * (i / 8)), cube);
                }

                for (int step = 0; step < 40; step++)
                {
                    Simulate(1);
                    if (step == 10)
                    {
                        // The game moves a body between the steps.
                        scene.Bodies[7].position += new Vector3(0.3f, 0.5f, -0.2f);
                        scene.Bodies[7].rotation *= Quaternion.Euler(10.0f, 0.0f, 0.0f);
                    }
                    else if (step == 15)
                    {
                        AddBody(scene, 40, new Vector3(-3.0f, 0.0f, -3.0f), cube);
                    }
                    else if (step == 20)
                    {
                        Object.DestroyImmediate(scene.Bodies[12].gameObject);
                        scene.Bodies[12] = null;
                    }

                    if (step % 5 == 1 || step % 5 == 2)
                    {
                        continue;
                    }

                    log.Append($"step {step}\n");
                    Query(scene, log);
                }

                return log.ToString();
            }
            finally
            {
                Unity.Physics.DynamicBoundsCache.ShapesEnabled = wasEnabled;
                Unity.Physics.DynamicBoundsCache.CheckEnabled = wasChecking;
            }
        }

        [Test]
        public void QueriesBetweenStepsFindTheSameHitsWithTheShapesOnOrOff()
        {
            long hits = Unity.Physics.DynamicBoundsCache.Counts->ShapeHits;
            long mismatches = Unity.Physics.DynamicBoundsCache.Counts->Mismatches;
            string on = Run(true);
            long shapeHits = Unity.Physics.DynamicBoundsCache.Counts->ShapeHits - hits;
            string off = Run(false);
            string again = Run(true);
            Assume.That(again, Is.EqualTo(on), "the scene gives the same queries twice in a row");

            Assert.That(Unity.Physics.DynamicBoundsCache.Counts->Mismatches - mismatches, Is.Zero, "bodies whose bounds differed from the collider's");
            Assert.That(shapeHits, Is.GreaterThan(0), "bodies that computed their bounds from the cache's shapes");
            string[] onLines = on.Split('\n'), offLines = off.Split('\n');
            for (int i = 0; i < System.Math.Min(onLines.Length, offLines.Length); i++)
            {
                Assert.That(onLines[i], Is.EqualTo(offLines[i]), $"line {i}");
            }

            Assert.That(onLines.Length, Is.EqualTo(offLines.Length));
            Debug.Log($"[ShapeBoundsQueryTests] {shapeHits} bodies computed their bounds from the cache's shapes, {onLines.Length} query lines");
        }
    }
}
