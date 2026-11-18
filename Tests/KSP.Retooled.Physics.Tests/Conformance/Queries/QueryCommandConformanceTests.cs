using KSP.Retooled.Physics.Tests.Scenes;
using NUnit.Framework;
using Unity.Collections;
using UnityEngine;

namespace KSP.Retooled.Physics.Tests.Conformance.Queries
{
    /// <summary>
    /// Tests Physics.RaycastAll, queries against meshes, and batched query commands running on job workers.
    /// </summary>
    public class QueryCommandConformanceTests : QueryScene
    {
        private const float Tolerance = 3e-3f;

        private static QueryParameters Parameters(bool hitMultipleFaces = false, bool hitBackfaces = false)
        {
            QueryParameters parameters = QueryParameters.Default;
            parameters.hitMultipleFaces = hitMultipleFaces;
            parameters.hitBackfaces = hitBackfaces;
            return parameters;
        }

        /// <summary>Two parallel squares in one mesh, facing -Z, at z = 0 and z = 2.</summary>
        private static Mesh TwoLayers()
        {
            Mesh mesh = QuadMesh();
            Vector3[] quad = mesh.vertices;
            int[] triangles = mesh.triangles;
            Vector3[] vertices = new Vector3[8];
            int[] both = new int[12];
            for (int i = 0; i < 4; i++)
            {
                vertices[i] = quad[i];
                vertices[i + 4] = quad[i] + new Vector3(0.0f, 0.0f, 2.0f);
            }

            for (int i = 0; i < 6; i++)
            {
                both[i] = triangles[i];
                both[i + 6] = triangles[i] + 4;
            }

            mesh.vertices = vertices;
            mesh.triangles = both;
            return mesh;
        }

        [Test]
        public void RaycastAllFindsEveryColliderOnTheWay()
        {
            Box(new Vector3(0.0f, 0.0f, 5.0f), Vector3.one);
            Box(new Vector3(0.0f, 0.0f, 9.0f), Vector3.one);

            RaycastHit[] hits = UnityEngine.Physics.RaycastAll(Vector3.zero, Vector3.forward, 100.0f);

            Assert.That(Distances(hits), Is.EqualTo(new[] { 4.5f, 8.5f }).Within(Tolerance));
        }

        [Test]
        public void RaycastAllMissesWhatItStartsIn()
        {
            Box(Vector3.zero, Vector3.one);
            GameObject next = Box(Vector3.forward, Vector3.one);

            RaycastHit[] hits = UnityEngine.Physics.RaycastAll(Vector3.zero, Vector3.forward, 100.0f);

            Assert.That(hits.Length, Is.EqualTo(1));
            Assert.That(hits[0].collider, Is.SameAs(next.GetComponent<Collider>()));
        }

        [Test]
        public void RaycastAllIsTheNearestFaceOfEachMesh()
        {
            MeshObject(TwoLayers(), Vector3.zero, false);

            RaycastHit[] hits = UnityEngine.Physics.RaycastAll(new Vector3(0.5f, 0.25f, -3.0f), Vector3.forward, 100.0f);

            Assert.That(Distances(hits), Is.EqualTo(new[] { 3.0f }).Within(Tolerance));
        }

        [Test]
        public void RaycastNonAllocFillsTheArrayAndStops()
        {
            Box(new Vector3(0.0f, 0.0f, 5.0f), Vector3.one);
            Box(new Vector3(0.0f, 0.0f, 9.0f), Vector3.one);

            Assert.That(UnityEngine.Physics.RaycastNonAlloc(Vector3.zero, Vector3.forward, new RaycastHit[1], 100.0f), Is.EqualTo(1));
            Assert.That(UnityEngine.Physics.RaycastNonAlloc(Vector3.zero, Vector3.forward, new RaycastHit[4], 100.0f), Is.EqualTo(2));
        }

        [Test]
        public void ARayHittingABackfaceGetsTheTrianglesOwnNormal()
        {
            TriangleMeshQuad(Vector3.zero);
            bool backfaces = UnityEngine.Physics.queriesHitBackfaces;
            UnityEngine.Physics.queriesHitBackfaces = true;
            try
            {
                Assert.That(UnityEngine.Physics.Raycast(new Vector3(0.0f, 0.0f, 3.0f), Vector3.back, out RaycastHit hit, 100.0f), Is.True);
                AssertVector(Vector3.back, hit.normal, Tolerance);
            }
            finally
            {
                UnityEngine.Physics.queriesHitBackfaces = backfaces;
            }
        }

        [Test]
        public void ACompletedRayBatchLetsItsColliderBeDestroyed()
        {
            Collider collider = Box(Vector3.forward * 5.0f, Vector3.one).GetComponent<Collider>();
            UnityEngine.Physics.SyncTransforms();
            BackendHost.SyncQueryWorlds();

            using var commands = new NativeArray<RaycastCommand>(1, Allocator.TempJob);
            using var hits = new NativeArray<RaycastHit>(1, Allocator.TempJob);
            NativeArray<RaycastCommand> rays = commands;
            rays[0] = new RaycastCommand(Vector3.zero, Vector3.forward, QueryParameters.Default, 10.0f);

            RaycastCommand.ScheduleBatch(commands, hits, 1).Complete();

            Assert.That(hits[0].collider, Is.SameAs(collider));
            DestroyAll();
            Assert.That(collider == null, Is.True);
        }

        [Test]
        public void ARayCommandCanHitEveryFaceOfAMesh()
        {
            MeshObject(TwoLayers(), Vector3.zero, false);
            UnityEngine.Physics.SyncTransforms();
            BackendHost.SyncQueryWorlds();

            using NativeArray<RaycastCommand> commands = new NativeArray<RaycastCommand>(1, Allocator.TempJob);
            using NativeArray<RaycastHit> hits = new NativeArray<RaycastHit>(4, Allocator.TempJob);
            NativeArray<RaycastCommand> rays = commands;

            rays[0] = new RaycastCommand(new Vector3(0.5f, 0.25f, -3.0f), Vector3.forward, Parameters(hitMultipleFaces: true), 100.0f);
            RaycastCommand.ScheduleBatch(commands, hits, 1, 4).Complete();

            Assert.That(hits[0].collider, Is.Not.Null);
            Assert.That(hits[1].collider, Is.Not.Null);
            Assert.That(hits[2].collider, Is.Null);
            Assert.That(Mathf.Min(hits[0].distance, hits[1].distance), Is.EqualTo(3.0f).Within(Tolerance));
            Assert.That(Mathf.Max(hits[0].distance, hits[1].distance), Is.EqualTo(5.0f).Within(Tolerance));
        }

        [Test]
        public void ARayCommandCanHitBackfaces()
        {
            TriangleMeshQuad(Vector3.zero);
            UnityEngine.Physics.SyncTransforms();
            BackendHost.SyncQueryWorlds();

            using NativeArray<RaycastCommand> commands = new NativeArray<RaycastCommand>(1, Allocator.TempJob);
            using NativeArray<RaycastHit> hits = new NativeArray<RaycastHit>(1, Allocator.TempJob);
            NativeArray<RaycastCommand> rays = commands;

            rays[0] = new RaycastCommand(new Vector3(0.0f, 0.0f, 3.0f), Vector3.back, Parameters(hitBackfaces: true), 100.0f);
            RaycastCommand.ScheduleBatch(commands, hits, 1).Complete();

            Assert.That(hits[0].collider, Is.Not.Null);
            Assert.That(hits[0].distance, Is.EqualTo(3.0f).Within(Tolerance));
        }

        [Test]
        public void ACastCommandSweepsOnAWorker()
        {
            GameObject box = Box(new Vector3(0.0f, 0.0f, 5.0f), Vector3.one);
            UnityEngine.Physics.SyncTransforms();
            BackendHost.SyncQueryWorlds();

            using NativeArray<SpherecastCommand> spheres = new NativeArray<SpherecastCommand>(1, Allocator.TempJob);
            using NativeArray<BoxcastCommand> boxes = new NativeArray<BoxcastCommand>(1, Allocator.TempJob);
            using NativeArray<CapsulecastCommand> capsules = new NativeArray<CapsulecastCommand>(1, Allocator.TempJob);
            using NativeArray<RaycastHit> hits = new NativeArray<RaycastHit>(1, Allocator.TempJob);
            NativeArray<SpherecastCommand> sphere = spheres;
            NativeArray<BoxcastCommand> boxCommand = boxes;
            NativeArray<CapsulecastCommand> capsule = capsules;

            sphere[0] = new SpherecastCommand(Vector3.zero, 0.25f, Vector3.forward, QueryParameters.Default, 100.0f);
            SpherecastCommand.ScheduleBatch(spheres, hits, 1).Complete();
            Assert.That(hits[0].collider, Is.SameAs(box.GetComponent<Collider>()));
            Assert.That(hits[0].distance, Is.EqualTo(4.25f).Within(Tolerance));

            boxCommand[0] = new BoxcastCommand(Vector3.zero, Vector3.one * 0.25f, Quaternion.identity, Vector3.forward, QueryParameters.Default, 100.0f);
            BoxcastCommand.ScheduleBatch(boxes, hits, 1).Complete();
            Assert.That(hits[0].distance, Is.EqualTo(4.25f).Within(Tolerance));

            capsule[0] = new CapsulecastCommand(Vector3.zero, Vector3.up * 0.2f, 0.25f, Vector3.forward, QueryParameters.Default, 100.0f);
            CapsulecastCommand.ScheduleBatch(capsules, hits, 1).Complete();
            Assert.That(hits[0].distance, Is.EqualTo(4.25f).Within(Tolerance));
        }

        [Test]
        public void AnOverlapCommandListsShapesOnAWorker()
        {
            GameObject near = Box(Vector3.zero, Vector3.one);
            Box(new Vector3(0.0f, 0.0f, 9.0f), Vector3.one);
            UnityEngine.Physics.SyncTransforms();
            BackendHost.SyncQueryWorlds();

            using NativeArray<OverlapSphereCommand> spheres = new NativeArray<OverlapSphereCommand>(1, Allocator.TempJob);
            using NativeArray<OverlapBoxCommand> boxes = new NativeArray<OverlapBoxCommand>(1, Allocator.TempJob);
            using NativeArray<OverlapCapsuleCommand> capsules = new NativeArray<OverlapCapsuleCommand>(1, Allocator.TempJob);
            using NativeArray<ColliderHit> hits = new NativeArray<ColliderHit>(4, Allocator.TempJob);
            NativeArray<OverlapSphereCommand> sphere = spheres;
            NativeArray<OverlapBoxCommand> box = boxes;
            NativeArray<OverlapCapsuleCommand> capsule = capsules;

            sphere[0] = new OverlapSphereCommand(Vector3.zero, 1.0f, QueryParameters.Default);
            OverlapSphereCommand.ScheduleBatch(spheres, hits, 1, 4).Complete();
            Assert.That(hits[0].collider, Is.SameAs(near.GetComponent<Collider>()));
            Assert.That(hits[1].collider, Is.Null);

            box[0] = new OverlapBoxCommand(Vector3.zero, Vector3.one, Quaternion.identity, QueryParameters.Default);
            OverlapBoxCommand.ScheduleBatch(boxes, hits, 1, 4).Complete();
            Assert.That(hits[0].collider, Is.SameAs(near.GetComponent<Collider>()));

            capsule[0] = new OverlapCapsuleCommand(Vector3.zero, Vector3.up, 1.0f, QueryParameters.Default);
            OverlapCapsuleCommand.ScheduleBatch(capsules, hits, 1, 4).Complete();
            Assert.That(hits[0].collider, Is.SameAs(near.GetComponent<Collider>()));
        }

        [Test]
        public void AClosestPointCommandAnswersOnAWorker()
        {
            GameObject box = Box(new Vector3(0.0f, 0.0f, 5.0f), Vector3.one);
            UnityEngine.Physics.SyncTransforms();
            BackendHost.SyncQueryWorlds();

            using NativeArray<ClosestPointCommand> commands = new NativeArray<ClosestPointCommand>(1, Allocator.TempJob);
            using NativeArray<Vector3> results = new NativeArray<Vector3>(1, Allocator.TempJob);
            NativeArray<ClosestPointCommand> command = commands;

            command[0] = new ClosestPointCommand(Vector3.zero, box.GetComponent<Collider>().GetEntityId(), box.transform.position, box.transform.rotation, box.transform.lossyScale);
            ClosestPointCommand.ScheduleBatch(commands, results, 1).Complete();

            AssertVector(new Vector3(0.0f, 0.0f, 4.5f), results[0], Tolerance);
        }

        [Test]
        public void ParticlesLandOnAWorldCollider()
        {
            Box(new Vector3(0.0f, -2.0f, 0.0f), new Vector3(20.0f, 1.0f, 20.0f));
            ParticleSystem particles = Track(new GameObject("particles")).AddComponent<ParticleSystem>();

            Assert.That(ParticleAfterASecond(particles, Vector3.zero).position.y, Is.EqualTo(-1.5f).Within(0.1f));
        }

        [Test]
        public void ParticlesStartingInAColliderComeOutOfIt()
        {
            Box(new Vector3(0.0f, -2.0f, 0.0f), new Vector3(20.0f, 1.0f, 20.0f));
            ParticleSystem particles = Track(new GameObject("particles")).AddComponent<ParticleSystem>();

            Assert.That(ParticleAfterASecond(particles, new Vector3(0.0f, -1.6f, 0.0f)).position.y, Is.EqualTo(-1.5f).Within(0.01f));
        }

        [Test]
        public void AnOverlapFromAJobSkipsAColliderDestroyedWithoutASync()
        {
            GameObject near = Box(Vector3.zero, Vector3.one);
            GameObject far = Box(new Vector3(0.0f, 0.0f, 1.0f), Vector3.one);
            UnityEngine.Physics.SyncTransforms();
            BackendHost.SyncQueryWorlds();

            Object.DestroyImmediate(near);

            using NativeArray<OverlapSphereCommand> spheres = new NativeArray<OverlapSphereCommand>(1, Allocator.TempJob);
            using NativeArray<ColliderHit> hits = new NativeArray<ColliderHit>(4, Allocator.TempJob);
            NativeArray<OverlapSphereCommand> sphere = spheres;
            sphere[0] = new OverlapSphereCommand(Vector3.zero, 1.0f, QueryParameters.Default);
            OverlapSphereCommand.ScheduleBatch(spheres, hits, 1, 4).Complete();

            Assert.That(hits[0].collider, Is.SameAs(far.GetComponent<Collider>()));
            Assert.That(hits[1].collider, Is.Null);
        }
    }
}
