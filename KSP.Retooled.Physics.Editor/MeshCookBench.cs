using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using UnityEditor;
using UnityEngine;
using Debug = UnityEngine.Debug;

namespace KSP.Retooled.Physics.Editor
{
    /// <summary>
    /// What a MeshCollider costs on the selected backend: the cook, and the first use that builds its collider. Meshes are the
    /// game's own, plus a PQS-sized grid and a sphere. Run with `unity exec --backend`; lines start with [MeshBench].
    /// </summary>
    public static class MeshCookBench
    {
        private const int Rounds = 3;
        private const double BurstWaitSeconds = 45.0;

        // The sample the medians come from, and the size above which a mesh isn't one a part would use.
        private const int SampleSize = 200;
        private const int MaximumVertices = 2000;

        public static void Run()
        {
            UnityEngine.Physics.simulationMode = SimulationMode.Script;
            UnityEngine.Physics.autoSyncTransforms = false;

            List<Mesh> game = GameCollisionMeshes();
            Debug.Log($"[MeshBench] {game.Count} game meshes, median {Median(game.Select(m => (double)m.vertexCount)):F0} vertices, "
                + $"largest {game.Select(m => m.vertexCount).DefaultIfEmpty(0).Max()}");

            Mesh largest = game.OrderByDescending(m => m.vertexCount).FirstOrDefault();
            game = game.Where(m => m.vertexCount <= MaximumVertices).OrderBy(m => m.GetEntityId().GetHashCode()).Take(SampleSize).ToList();
            Debug.Log($"[MeshBench] sample of {game.Count}, median {Median(game.Select(m => (double)m.vertexCount)):F0} vertices");

            Mesh grid = Grid(15);
            Mesh sphere = Sphere();

            // Cooks and builds run in Mono until Burst delivers them.
            Stopwatch wait = Stopwatch.StartNew();
            long managed = -1;
            while (wait.Elapsed.TotalSeconds < BurstWaitSeconds && managed != BackendHost.ManagedCooks)
            {
                managed = BackendHost.ManagedCooks;
                MeasureCook("warming up", new List<Mesh> { grid }, false, false);
                BuildOnce(grid, false);
                System.Threading.Thread.Sleep(2000);
            }

            Debug.Log($"[MeshBench] Burst wait {wait.Elapsed.TotalSeconds:F0} s");

            long managedBefore = BackendHost.ManagedCooks;
            MeasureCook("cook convex, game meshes", game, true);
            MeasureCook("cook triangle, game meshes", game, false);
            MeasureCook("cook triangle, 15x15 grid", new List<Mesh> { grid }, false);
            MeasureCook("cook convex, sphere", new List<Mesh> { sphere }, true);
            MeasureBuild("first use convex, game meshes", game, true);
            MeasureBuild("first use triangle, game meshes", game, false);
            MeasureBuild("first use triangle, 15x15 grid", new List<Mesh> { grid }, false);
            MeasureBuild("first use convex, sphere", new List<Mesh> { sphere }, true);
            if (largest != null)
            {
                MeasureCook($"cook convex, largest mesh of {largest.vertexCount}", new List<Mesh> { largest }, true);
            }

            Debug.Log($"[MeshBench] {BackendHost.ManagedCooks - managedBefore} of the measured cooks ran in Mono");
        }

        private static List<Mesh> GameCollisionMeshes()
        {
            HashSet<Mesh> meshes = new HashSet<Mesh>();
            foreach (string guid in AssetDatabase.FindAssets("t:Prefab", new[] { "Assets/KSP" }))
            {
                GameObject prefab = AssetDatabase.LoadAssetAtPath<GameObject>(AssetDatabase.GUIDToAssetPath(guid));
                if (prefab == null)
                {
                    continue;
                }

                foreach (MeshCollider collider in prefab.GetComponentsInChildren<MeshCollider>(true))
                {
                    if (collider.sharedMesh != null)
                    {
                        meshes.Add(collider.sharedMesh);
                    }
                }
            }

            return meshes.ToList();
        }

        /// <summary>Unity keeps a mesh's bake, so each round cooks fresh copies.</summary>
        private static void MeasureCook(string name, List<Mesh> meshes, bool convex, bool report = true)
        {
            if (meshes.Count == 0)
            {
                return;
            }

            List<double> samples = new List<double>();
            for (int round = 0; round < Rounds; round++)
            {
                List<Mesh> copies = meshes.Select(Object.Instantiate).ToList();
                Stopwatch watch = Stopwatch.StartNew();
                foreach (Mesh mesh in copies)
                {
                    UnityEngine.Physics.BakeMesh(mesh.GetEntityId(), convex);
                }

                samples.Add(watch.Elapsed.TotalMilliseconds * 1000.0 / meshes.Count);
                copies.ForEach(Object.DestroyImmediate);
            }

            if (report)
            {
                Debug.Log($"[MeshBench] {name}: {Median(samples):F1} us per mesh over {meshes.Count}");
            }
        }

        private static void MeasureBuild(string name, List<Mesh> meshes, bool convex)
        {
            if (meshes.Count == 0)
            {
                return;
            }

            List<double> samples = new List<double>();
            for (int round = 0; round < Rounds; round++)
            {
                double total = 0.0;
                foreach (Mesh mesh in meshes)
                {
                    total += BuildOnce(mesh, convex);
                }

                samples.Add(total * 1000.0 / meshes.Count);
            }

            Debug.Log($"[MeshBench] {name}: {Median(samples):F1} us per mesh over {meshes.Count}");
        }

        /// <summary>Milliseconds the first query after adding a collider took, which is its blob plus one rebuild of an empty world.</summary>
        private static double BuildOnce(Mesh mesh, bool convex)
        {
            GameObject gameObject = new GameObject("bench");
            MeshCollider collider = gameObject.AddComponent<MeshCollider>();
            collider.convex = convex;
            collider.sharedMesh = mesh;
            UnityEngine.Physics.SyncTransforms();

            Stopwatch watch = Stopwatch.StartNew();
            UnityEngine.Physics.Raycast(new Vector3(0.0f, 100.0f, 0.0f), Vector3.down, 200.0f);
            double elapsed = watch.Elapsed.TotalMilliseconds;

            Object.DestroyImmediate(gameObject);
            return elapsed;
        }

        private static double Median(IEnumerable<double> values)
        {
            List<double> sorted = values.OrderBy(v => v).ToList();
            return sorted.Count == 0 ? 0.0 : sorted[sorted.Count / 2];
        }

        private static Mesh Grid(int cells)
        {
            List<Vector3> vertices = new List<Vector3>();
            List<int> triangles = new List<int>();
            for (int z = 0; z <= cells; z++)
            {
                for (int x = 0; x <= cells; x++)
                {
                    vertices.Add(new Vector3(x, Mathf.PerlinNoise(x * 0.3f, z * 0.3f) * 2.0f, z));
                }
            }

            for (int z = 0; z < cells; z++)
            {
                for (int x = 0; x < cells; x++)
                {
                    int corner = z * (cells + 1) + x;
                    int next = corner + cells + 1;
                    triangles.AddRange(new[] { corner, next, next + 1, corner, next + 1, corner + 1 });
                }
            }

            Mesh mesh = new Mesh { vertices = vertices.ToArray(), triangles = triangles.ToArray() };
            return mesh;
        }

        private static Mesh Sphere()
        {
            GameObject primitive = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            Mesh mesh = primitive.GetComponent<MeshFilter>().sharedMesh;
            Object.DestroyImmediate(primitive);
            return mesh;
        }
    }
}
