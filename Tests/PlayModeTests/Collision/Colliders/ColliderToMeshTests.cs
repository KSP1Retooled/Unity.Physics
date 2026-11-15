using System.Collections;
using System.Collections.Generic;
using NUnit.Framework;
using Unity.Collections;
using Unity.Entities;
using Unity.Mathematics;
using Unity.Physics.Extensions;
using Unity.Physics.Tests.Utils;
using UnityEngine;
using UnityEngine.TestTools;

namespace Unity.Physics.Tests.Collision.Colliders
{
    /// <summary>
    /// Tests for the lifetime of the meshes created by <see cref="ColliderExtensions.ToMesh"/>
    /// </summary>
    class ColliderToMeshTests : PhysicsTestFixtureBase
    {
        public enum TestColliderType
        {
            Box,
            Convex,
            Mesh,
            Terrain,
            CompoundOfBoxAndSphere
        }

        static BlobAssetReference<Collider> CreateCollider(TestColliderType type)
        {
            switch (type)
            {
                case TestColliderType.Box:
                {
                    return CreateBox();
                }
                case TestColliderType.Convex:
                {
                    using var points = new NativeArray<float3>(ConvexColliderTests.k_TestPoints, Allocator.Temp);
                    return ConvexCollider.Create(points, ConvexHullGenerationParameters.Default);
                }
                case TestColliderType.Mesh:
                {
                    using var vertices = new NativeArray<float3>(new[] { new float3(0, 0, 0), new float3(1, 0, 0), new float3(0, 0, 1) }, Allocator.Temp);
                    using var triangles = new NativeArray<int3>(new[] { new int3(0, 1, 2) }, Allocator.Temp);
                    return MeshCollider.Create(vertices, triangles);
                }
                case TestColliderType.Terrain:
                {
                    using var heights = new NativeArray<float>(16, Allocator.Temp);
                    return TerrainCollider.Create(heights, new int2(4, 4), 1, TerrainCollider.CollisionMethod.Triangles);
                }
                case TestColliderType.CompoundOfBoxAndSphere:
                {
                    using var box = CreateBox();
                    using var sphere = SphereCollider.Create(new SphereGeometry { Center = float3.zero, Radius = 0.5f });
                    using var children = new NativeArray<CompoundCollider.ColliderBlobInstance>(new[]
                    {
                        new CompoundCollider.ColliderBlobInstance { Collider = box, CompoundFromChild = RigidTransform.Translate(new float3(-1, 0, 0)) },
                        new CompoundCollider.ColliderBlobInstance { Collider = sphere, CompoundFromChild = RigidTransform.Translate(new float3(1, 0, 0)) }
                    }, Allocator.Temp);
                    return CompoundCollider.Create(children);
                }
                default:
                    throw new System.NotImplementedException();
            }
        }

        static BlobAssetReference<Collider> CreateBox()
        {
            return BoxCollider.Create(new BoxGeometry
            {
                Center = float3.zero,
                Orientation = quaternion.identity,
                Size = new float3(1)
            });
        }

        // ToMesh() combines intermediate meshes into the returned mesh. These intermediate meshes are HideAndDontSave,
        // so Resources.UnloadUnusedAssets() never frees them, and ToMesh() must destroy them itself.
        [UnityTest]
        public IEnumerator ToMesh_DestroysIntermediateMeshes([Values] TestColliderType colliderType)
        {
            using var collider = CreateCollider(colliderType);

            // ToMesh() runs synchronously on the main thread, so any mesh that appears between these two snapshots was created by it.
            // The ToMesh() call must stay between the snapshots, otherwise no intermediate meshes are detected.
            var meshesBefore = new HashSet<UnityEngine.Mesh>(Resources.FindObjectsOfTypeAll<UnityEngine.Mesh>());
            var mesh = AddForTearDown(collider.Value.ToMesh());
            var createdMeshes = new List<UnityEngine.Mesh>();
            foreach (var m in Resources.FindObjectsOfTypeAll<UnityEngine.Mesh>())
            {
                if (m != mesh && !meshesBefore.Contains(m))
                {
                    createdMeshes.Add(m);
                }
            }

            // Make sure this test actually exercises the intermediate mesh cleanup
            Assert.That(createdMeshes, Is.Not.Empty, "Expected ToMesh() to create at least one intermediate mesh for this collider type");

            // Object.Destroy() is deferred until the end of the frame
            yield return null;

            for (int i = 0; i < createdMeshes.Count; i++)
            {
                Assert.IsTrue(createdMeshes[i] == null, $"Intermediate mesh {i} of {createdMeshes.Count} created by ToMesh() was not destroyed");
            }
        }

        // Sphere and capsule colliders reuse Unity's builtin sphere and cylinder meshes. ToMesh() must not destroy
        // these along with its intermediate meshes, or every later conversion of these colliders would be broken.
        [UnityTest]
        public IEnumerator ToMesh_DoesNotDestroySharedReferenceMeshes()
        {
            using var sphere = SphereCollider.Create(new SphereGeometry { Center = float3.zero, Radius = 1 });
            using var capsule = CapsuleCollider.Create(new CapsuleGeometry { Vertex0 = new float3(0, -1, 0), Vertex1 = new float3(0, 1, 0), Radius = 1 });

            for (int i = 0; i < 2; i++)
            {
                var sphereMesh = AddForTearDown(sphere.Value.ToMesh());
                var capsuleMesh = AddForTearDown(capsule.Value.ToMesh());

                Assert.Greater(sphereMesh.vertexCount, 0);
                Assert.Greater(capsuleMesh.vertexCount, 0);
                TestUtils.AreEqual(new float3(2), sphereMesh.bounds.size, 1e-5f);
                TestUtils.AreEqual(new float3(2, 4, 2), capsuleMesh.bounds.size, 1e-5f);

                // Let any deferred Object.Destroy() calls run before converting again
                yield return null;
            }
        }
    }
}
