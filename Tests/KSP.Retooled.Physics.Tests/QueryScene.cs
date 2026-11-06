using System.Linq;
using UnityEngine;

namespace KSP.Retooled.PhysicsBackend.Tests
{
    /// <summary>What the query fixtures build their scenes from, on top of <see cref="PhysicsScene"/>.</summary>
    public abstract class QueryScene : PhysicsScene
    {
        /// <summary>Hit distances, sorted, because neither backend promises an order.</summary>
        protected static float[] Distances(RaycastHit[] hits)
        {
            return hits.Select(hit => hit.distance).OrderBy(distance => distance).ToArray();
        }

        /// <summary>A flat square of side 4 in the XY plane, whose front faces -Z.</summary>
        protected static Mesh QuadMesh()
        {
            Mesh mesh = new Mesh
            {
                vertices = new[]
                {
                    new Vector3(-2.0f, -2.0f, 0.0f),
                    new Vector3(2.0f, -2.0f, 0.0f),
                    new Vector3(2.0f, 2.0f, 0.0f),
                    new Vector3(-2.0f, 2.0f, 0.0f),
                },
            };

            mesh.triangles = new[] { 0, 2, 1, 0, 3, 2 };
            return mesh;
        }

        /// <summary>A closed unit cube whose triangles face out.</summary>
        protected static Mesh CubeMesh()
        {
            GameObject primitive = GameObject.CreatePrimitive(PrimitiveType.Cube);
            Mesh mesh = Object.Instantiate(primitive.GetComponent<MeshFilter>().sharedMesh);
            Object.DestroyImmediate(primitive);
            return mesh;
        }

        protected GameObject MeshObject(Mesh mesh, Vector3 position, bool convex)
        {
            GameObject gameObject = Track(new GameObject("mesh"));
            gameObject.transform.position = position;
            MeshCollider collider = gameObject.AddComponent<MeshCollider>();
            collider.convex = convex;
            collider.sharedMesh = mesh;
            return gameObject;
        }

        protected GameObject TriangleMeshQuad(Vector3 position)
        {
            return MeshObject(QuadMesh(), position, false);
        }

        protected GameObject TriangleMeshCube(Vector3 position)
        {
            return MeshObject(CubeMesh(), position, false);
        }

        protected GameObject ConvexCube(Vector3 position)
        {
            return MeshObject(CubeMesh(), position, true);
        }

        protected SphereCollider Sphere(Vector3 position, float radius)
        {
            GameObject gameObject = Track(new GameObject("sphere"));
            gameObject.transform.position = position;
            SphereCollider sphere = gameObject.AddComponent<SphereCollider>();
            sphere.radius = radius;
            return sphere;
        }
    }
}
