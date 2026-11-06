using System.Linq;
using NUnit.Framework;
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

        /// <summary>One particle of a system that collides with the world, dropped from a point and left to fall for a second.</summary>
        protected static ParticleSystem.Particle ParticleAfterASecond(ParticleSystem particles, Vector3 start)
        {
            particles.Stop(true, ParticleSystemStopBehavior.StopEmittingAndClear);
            particles.useAutoRandomSeed = false;
            ParticleSystem.MainModule main = particles.main;
            main.startLifetime = 10.0f;
            main.startSpeed = 0.0f;
            main.gravityModifier = 1.0f;
            main.simulationSpace = ParticleSystemSimulationSpace.World;
            ParticleSystem.ShapeModule shape = particles.shape;
            shape.enabled = false;
            ParticleSystem.EmissionModule emission = particles.emission;
            emission.rateOverTime = 0.0f;
            ParticleSystem.CollisionModule collision = particles.collision;
            collision.enabled = true;
            collision.type = ParticleSystemCollisionType.World;
            collision.mode = ParticleSystemCollisionMode.Collision3D;
            collision.quality = ParticleSystemCollisionQuality.High;
            collision.bounce = 0.0f;
            collision.dampen = 0.9f;
            collision.radiusScale = 0.01f;
            Physics.SyncTransforms();
            BackendHost.SyncQueryWorlds();

            particles.Play();
            particles.Emit(new ParticleSystem.EmitParams { position = start, startSize = 0.1f, startLifetime = 10.0f, applyShapeToPosition = false }, 1);
            for (int i = 0; i < 50; i++)
            {
                particles.Simulate(0.02f, true, false, false);
            }

            ParticleSystem.Particle[] alive = new ParticleSystem.Particle[1];
            Assert.That(particles.GetParticles(alive), Is.EqualTo(1));
            return alive[0];
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
