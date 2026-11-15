using System;
using System.Collections.Generic;
using NUnit.Framework;
using UnityEngine;
using Object = UnityEngine.Object;

namespace KSP.Retooled.Physics.Tests
{
    /// <summary>What a TerrainCollider does: its surface, contacts, queries, edits and lifetime.</summary>
    public class TerrainConformanceTests : PhysicsScene
    {
        // Unity's smallest height map, with 32 cells to a side.
        private const int Resolution = 33;

        private readonly List<TerrainData> terrains = new List<TerrainData>();

        [TearDown]
        public void DestroyTerrains()
        {
            foreach (TerrainData data in terrains)
            {
                Object.DestroyImmediate(data);
            }

            terrains.Clear();
        }

        private TerrainData Data(Vector3 size, Func<int, int, float> height)
        {
            TerrainData data = new TerrainData();
            terrains.Add(data);
            data.heightmapResolution = Resolution;
            data.size = size;
            data.SetHeights(0, 0, Heights(height));
            return data;
        }

        /// <summary>
        /// Calls the height function with x and z for each sample, and stores normalized heights in [z, x] order.
        /// </summary>
        private static float[,] Heights(Func<int, int, float> height)
        {
            float[,] heights = new float[Resolution, Resolution];
            for (int z = 0; z < Resolution; z++)
            {
                for (int x = 0; x < Resolution; x++)
                {
                    heights[z, x] = height(x, z);
                }
            }

            return heights;
        }

        private TerrainCollider Terrain(TerrainData data, Vector3 position = default)
        {
            GameObject gameObject = Track(new GameObject("terrain"));
            gameObject.transform.position = position;
            TerrainCollider collider = gameObject.AddComponent<TerrainCollider>();
            collider.terrainData = data;
            UnityEngine.Physics.SyncTransforms();
            return collider;
        }

        /// <summary>A plane rising along +x: y runs from 4 to 14 over x from 3 to 103, with z from 5 to 45.</summary>
        private TerrainCollider Slope()
        {
            return Terrain(Data(new Vector3(100.0f, 20.0f, 40.0f), (x, z) => x / 32.0f * 0.5f), new Vector3(3.0f, 4.0f, 5.0f));
        }

        private static readonly Vector3 SlopeMiddle = new Vector3(53.0f, 9.0f, 25.0f);

        private static float SurfaceY(Vector3 above)
        {
            UnityEngine.Physics.SyncTransforms();
            return UnityEngine.Physics.Raycast(above, Vector3.down, out RaycastHit hit, 1000.0f) ? hit.point.y : float.NaN;
        }

        private Rigidbody Ball(Vector3 position, float radius)
        {
            GameObject gameObject = Track(new GameObject("ball"));
            gameObject.transform.position = position;
            gameObject.AddComponent<SphereCollider>().radius = radius;
            return gameObject.AddComponent<Rigidbody>();
        }

        [Test]
        public void HeightsFollowTheGridAtANonUniformScale()
        {
            TerrainCollider alongX = Slope();
            for (float x = 8.0f; x < 100.0f; x += 13.3f)
            {
                for (float z = 7.0f; z < 44.0f; z += 7.7f)
                {
                    Assert.That(SurfaceY(new Vector3(x, 100.0f, z)), Is.EqualTo(4.0f + (x - 3.0f) * 0.1f).Within(1e-3f));
                }
            }

            Object.DestroyImmediate(alongX.gameObject);

            Terrain(Data(new Vector3(100.0f, 20.0f, 40.0f), (x, z) => z / 32.0f * 0.5f), new Vector3(3.0f, 4.0f, 5.0f));
            for (float x = 8.0f; x < 100.0f; x += 13.3f)
            {
                for (float z = 7.0f; z < 44.0f; z += 7.7f)
                {
                    Assert.That(SurfaceY(new Vector3(x, 100.0f, z)), Is.EqualTo(4.0f + (z - 5.0f) * 0.25f).Within(1e-3f));
                }
            }
        }

        [Test]
        public void ARayReportsTheTerrainAndItsSlope()
        {
            TerrainCollider collider = Slope();

            Assert.That(UnityEngine.Physics.Raycast(new Vector3(53.0f, 100.0f, 25.0f), Vector3.down, out RaycastHit hit, 1000.0f), Is.True);

            Assert.That(hit.collider, Is.SameAs(collider));
            Assert.That(hit.distance, Is.EqualTo(91.0f).Within(1e-3f));
            AssertVector(new Vector3(-0.0995f, 0.995f, 0.0f), hit.normal, 2e-3f, "normal");
            Assert.That(hit.triangleIndex, Is.EqualTo(-1));
            Assert.That(hit.textureCoord.x, Is.EqualTo(0.5f).Within(1e-3f));
            Assert.That(hit.textureCoord.y, Is.EqualTo(0.5f).Within(1e-3f));
        }

        [Test]
        public void TheColliderAloneAnswersARay()
        {
            TerrainCollider collider = Slope();

            Ray ray = new Ray(new Vector3(53.0f, 100.0f, 25.0f), Vector3.down);
            Assert.That(collider.Raycast(ray, out RaycastHit hit, 1000.0f), Is.True);
            Assert.That(hit.distance, Is.EqualTo(91.0f).Within(1e-3f));

            Ray beside = new Ray(new Vector3(-53.0f, 100.0f, 25.0f), Vector3.down);
            Assert.That(collider.Raycast(beside, out _, 1000.0f), Is.False);
        }

        [Test]
        public void ARayMissesBesideTheGridAndHitsItsCorner()
        {
            Slope();

            Assert.That(SurfaceY(new Vector3(-53.0f, 100.0f, 25.0f)), Is.NaN);
            Assert.That(SurfaceY(new Vector3(103.5f, 100.0f, 45.5f)), Is.NaN);
            Assert.That(SurfaceY(new Vector3(3.01f, 100.0f, 5.01f)), Is.EqualTo(4.0f).Within(1e-2f));
        }

        [TestCase(10, 10)]
        [TestCase(11, 10)]
        [TestCase(10, 11)]
        [TestCase(11, 11)]
        public void EveryCellSplitsFromItsLowCornerToItsHighCorner(int raisedX, int raisedZ)
        {
            Terrain(Data(new Vector3(32.0f, 1.0f, 32.0f), (x, z) => x == raisedX && z == raisedZ ? 1.0f : 0.0f));

            // The diagonal runs through the raised sample in the cells where it is the low or the high corner.
            Assert.That(SurfaceY(new Vector3(raisedX + 0.5f, 5.0f, raisedZ + 0.5f)), Is.EqualTo(0.5f).Within(1e-3f));
            Assert.That(SurfaceY(new Vector3(raisedX - 0.5f, 5.0f, raisedZ - 0.5f)), Is.EqualTo(0.5f).Within(1e-3f));
            Assert.That(SurfaceY(new Vector3(raisedX + 0.5f, 5.0f, raisedZ - 0.5f)), Is.EqualTo(0.0f).Within(1e-3f));
            Assert.That(SurfaceY(new Vector3(raisedX - 0.5f, 5.0f, raisedZ + 0.5f)), Is.EqualTo(0.0f).Within(1e-3f));
        }

        [Test]
        public void TheTerrainHasNoBack()
        {
            Slope();
            Vector3 below = new Vector3(53.0f, -100.0f, 25.0f);

            Assert.That(UnityEngine.Physics.Raycast(below, Vector3.up, 1000.0f), Is.False);
            Assert.That(UnityEngine.Physics.SphereCast(below, 1.0f, Vector3.up, out _, 1000.0f), Is.False);

            bool hitBackfaces = UnityEngine.Physics.queriesHitBackfaces;
            UnityEngine.Physics.queriesHitBackfaces = true;
            try
            {
                Assert.That(UnityEngine.Physics.Raycast(below, Vector3.up, out RaycastHit hit, 1000.0f), Is.True);
                Assert.That(hit.distance, Is.EqualTo(109.0f).Within(1e-3f));
                AssertVector(new Vector3(-0.0995f, 0.995f, 0.0f), hit.normal, 2e-3f, "normal");
            }
            finally
            {
                UnityEngine.Physics.queriesHitBackfaces = hitBackfaces;
            }
        }

        [Test]
        public void ALayerMaskLeavesTheTerrainOut()
        {
            TerrainCollider collider = Slope();
            collider.gameObject.layer = 8;
            UnityEngine.Physics.SyncTransforms();
            Vector3 above = new Vector3(53.0f, 100.0f, 25.0f);

            Assert.That(UnityEngine.Physics.Raycast(above, Vector3.down, 1000.0f, 1 << 8), Is.True);
            Assert.That(UnityEngine.Physics.Raycast(above, Vector3.down, 1000.0f, ~(1 << 8)), Is.False);
            Assert.That(UnityEngine.Physics.CheckSphere(SlopeMiddle, 1.0f, 1 << 8), Is.True);
            Assert.That(UnityEngine.Physics.CheckSphere(SlopeMiddle, 1.0f, ~(1 << 8)), Is.False);
        }

        [Test]
        public void ADisabledTerrainIsGone()
        {
            TerrainCollider collider = Slope();

            collider.enabled = false;
            Assert.That(SurfaceY(new Vector3(53.0f, 100.0f, 25.0f)), Is.NaN);

            collider.enabled = true;
            Assert.That(SurfaceY(new Vector3(53.0f, 100.0f, 25.0f)), Is.EqualTo(9.0f).Within(1e-3f));
        }

        [Test]
        public void ShapesOverlapTheSurfaceAndNothingBelowIt()
        {
            TerrainCollider collider = Slope();

            Assert.That(UnityEngine.Physics.OverlapSphere(SlopeMiddle, 1.0f), Is.EqualTo(new Collider[] { collider }));
            Assert.That(UnityEngine.Physics.CheckSphere(SlopeMiddle, 1.0f), Is.True);
            Assert.That(UnityEngine.Physics.CheckBox(SlopeMiddle, Vector3.one), Is.True);
            Assert.That(UnityEngine.Physics.CheckSphere(SlopeMiddle + Vector3.up * 1.1f, 1.0f), Is.False);
            Assert.That(UnityEngine.Physics.CheckSphere(SlopeMiddle + Vector3.down * 1.1f, 1.0f), Is.False);
            Assert.That(UnityEngine.Physics.CheckBox(SlopeMiddle + Vector3.down * 1.1f, Vector3.one * 0.5f), Is.False);
            Assert.That(UnityEngine.Physics.CheckSphere(SlopeMiddle + Vector3.down * 3.0f, 1.0f), Is.False);
            Vector3 deep = SlopeMiddle + Vector3.down * 3.0f;
            Assert.That(UnityEngine.Physics.CheckCapsule(deep, deep + Vector3.up * 0.5f, 0.5f), Is.False);
        }

        [Test]
        public void ASweepLandsOnTheSlope()
        {
            Slope();
            Vector3 above = new Vector3(53.0f, 100.0f, 25.0f);

            Assert.That(UnityEngine.Physics.SphereCast(above, 1.0f, Vector3.down, out RaycastHit sphere, 200.0f), Is.True);
            Assert.That(sphere.distance, Is.EqualTo(89.995f).Within(2e-3f));
            AssertVector(new Vector3(53.0993f, 9.0102f, 25.0f), sphere.point, 2e-3f, "sphere point");
            AssertVector(new Vector3(-0.0993f, 0.9951f, 0.0f), sphere.normal, 2e-3f, "sphere normal");

            Assert.That(UnityEngine.Physics.BoxCast(above, Vector3.one, Vector3.down, out RaycastHit box, Quaternion.identity, 200.0f), Is.True);

            // The corner reaches the surface at 89.9, and PhysX stops a box 5 mm past that.
            Assert.That(box.distance, Is.EqualTo(89.8975f).Within(3e-3f));
            AssertVector(new Vector3(-0.0993f, 0.9951f, 0.0f), box.normal, 2e-3f, "box normal");
        }

        [Test]
        public void ASweepThatStartsOnTheSurfaceIsAnOverlap()
        {
            TerrainCollider collider = Slope();
            Vector3 touching = SlopeMiddle + Vector3.up * 0.5f;

            Assert.That(UnityEngine.Physics.SphereCast(touching, 1.0f, Vector3.down, out _, 20.0f), Is.False);

            RaycastHit[] hits = UnityEngine.Physics.SphereCastAll(touching, 1.0f, Vector3.down, 20.0f);
            Assert.That(hits.Length, Is.EqualTo(1));
            Assert.That(hits[0].collider, Is.SameAs(collider));
            Assert.That(hits[0].distance, Is.EqualTo(0.0f));
            AssertVector(Vector3.up, hits[0].normal, 1e-4f, "normal");
        }

        [Test]
        public void PenetrationComesOutOfTheFrontOnly()
        {
            TerrainCollider collider = Slope();
            SphereCollider sphere = Track(new GameObject("query")).AddComponent<SphereCollider>();
            sphere.radius = 1.0f;
            sphere.gameObject.layer = 2;
            Vector3 position = collider.transform.position;

            bool above = UnityEngine.Physics.ComputePenetration(
                sphere, SlopeMiddle + Vector3.up * 0.5f, Quaternion.identity, collider, position, Quaternion.identity, out Vector3 direction, out float distance);

            Assert.That(above, Is.True);
            AssertVector(new Vector3(-0.0995f, 0.995f, 0.0f), direction, 2e-3f, "direction");
            Assert.That(distance, Is.EqualTo(0.5025f).Within(1e-3f));

            bool below = UnityEngine.Physics.ComputePenetration(
                sphere, SlopeMiddle + Vector3.down * 0.5f, Quaternion.identity, collider, position, Quaternion.identity, out _, out _);

            Assert.That(below, Is.False);
        }

        [Test]
        public void ATerrainTakesNoRotationOrScaleFromItsTransform()
        {
            TerrainCollider collider = Slope();

            collider.transform.rotation = Quaternion.Euler(0.0f, 45.0f, 0.0f);
            collider.transform.localScale = new Vector3(2.0f, 2.0f, 2.0f);
            UnityEngine.Physics.SyncTransforms();

            AssertVector(new Vector3(53.0f, 9.0f, 25.0f), collider.bounds.center, 1e-3f, "bounds centre");
            AssertVector(new Vector3(50.0f, 5.0f, 20.0f), collider.bounds.extents, 1e-3f, "bounds extents");
            Assert.That(SurfaceY(new Vector3(53.0f, 100.0f, 25.0f)), Is.EqualTo(9.0f).Within(1e-3f));
        }

        [Test]
        public void MovingATerrainMovesItsSurface()
        {
            TerrainCollider collider = Slope();

            collider.transform.position = new Vector3(13.0f, 6.0f, 5.0f);

            Assert.That(SurfaceY(new Vector3(53.0f, 100.0f, 25.0f)), Is.EqualTo(10.0f).Within(1e-3f));
        }

        [Test]
        public void TheBoundsAreTheGridAndTheHeightsItHasHeld()
        {
            TerrainCollider collider = Slope();

            AssertVector(new Vector3(53.0f, 9.0f, 25.0f), collider.bounds.center, 1e-3f, "bounds centre");
            AssertVector(new Vector3(50.0f, 5.0f, 20.0f), collider.bounds.extents, 1e-3f, "bounds extents");

            // Editing heights can expand the bounds, but lowering the terrain does not shrink them.
            collider.terrainData.SetHeights(0, 0, new float[Resolution, Resolution]);
            UnityEngine.Physics.SyncTransforms();

            AssertVector(new Vector3(53.0f, 9.0f, 25.0f), collider.bounds.center, 1e-3f, "bounds centre after lowering");
            AssertVector(new Vector3(50.0f, 5.0f, 20.0f), collider.bounds.extents, 1e-3f, "bounds extents after lowering");
        }

        [Test]
        public void ResizingTheDataResizesTheSurface()
        {
            TerrainCollider collider = Slope();

            collider.terrainData.size = new Vector3(50.0f, 20.0f, 40.0f);
            UnityEngine.Physics.SyncTransforms();

            AssertVector(new Vector3(28.0f, 9.0f, 25.0f), collider.bounds.center, 1e-3f, "bounds centre");
            AssertVector(new Vector3(25.0f, 5.0f, 20.0f), collider.bounds.extents, 1e-3f, "bounds extents");
            Assert.That(SurfaceY(new Vector3(28.0f, 100.0f, 25.0f)), Is.EqualTo(9.0f).Within(1e-3f));
        }

        [Test]
        public void AFlatTerrainOfZerosHasASurface()
        {
            TerrainCollider collider = Terrain(Data(new Vector3(40.0f, 10.0f, 80.0f), (x, z) => 0.0f), new Vector3(100.0f, 5.0f, 200.0f));

            AssertVector(new Vector3(20.0f, 0.0f, 40.0f), collider.bounds.extents, 1e-3f, "bounds extents");
            Assert.That(SurfaceY(new Vector3(110.0f, 50.0f, 230.0f)), Is.EqualTo(5.0f).Within(1e-3f));

            Rigidbody box = DynamicBox(new Vector3(120.0f, 7.0f, 240.0f));
            Simulate(100);

            Assert.That(box.position.y, Is.EqualTo(5.5f).Within(0.02f));
        }

        [Test]
        public void ABoxSettlesOnTheSlope()
        {
            Slope();
            Rigidbody box = DynamicBox(new Vector3(53.0f, 16.0f, 25.0f));

            Simulate(300);

            // The box lies on the slope's plane, half its height above it along the normal.
            Assert.That(box.position.x, Is.EqualTo(53.0f).Within(0.3f));
            Assert.That(box.position.z, Is.EqualTo(25.0f).Within(0.05f));
            float surface = 9.0f + (box.position.x - 53.0f) * 0.1f;
            Assert.That(box.position.y - surface, Is.EqualTo(0.5f / 0.995f).Within(0.03f));
            Assert.That(Quaternion.Angle(box.rotation, Quaternion.Euler(0.0f, 0.0f, 5.71f)), Is.LessThan(1.5f));
            Assert.That(box.IsSleeping(), Is.True);
        }

        [Test]
        public void ASphereRestsOnTheSurface()
        {
            Terrain(Data(new Vector3(32.0f, 10.0f, 32.0f), (x, z) => 0.5f));
            Rigidbody ball = Ball(new Vector3(24.0f, 20.0f, 24.0f), 0.5f);

            Simulate(200);

            Assert.That(ball.position.y, Is.EqualTo(5.5f).Within(0.02f));
        }

        [Test]
        public void ASphereAt60MetresASecondStopsAtTheSurface()
        {
            Terrain(Data(new Vector3(32.0f, 10.0f, 32.0f), (x, z) => 0.5f));
            Rigidbody ball = Ball(new Vector3(24.0f, 20.0f, 24.0f), 0.5f);
            ball.linearVelocity = new Vector3(0.0f, -60.0f, 0.0f);

            Simulate(100);

            Assert.That(ball.position.y, Is.EqualTo(5.5f).Within(0.02f));
        }

        [Test]
        public void ABodyBelowTheSurfaceIsNotPushedOut()
        {
            Terrain(Data(new Vector3(32.0f, 10.0f, 32.0f), (x, z) => 0.5f));
            Rigidbody straddling = WeightlessBox(new Vector3(8.0f, 4.8f, 8.0f));
            Rigidbody buried = WeightlessBox(new Vector3(16.0f, 3.0f, 16.0f));

            Simulate(100);

            AssertVector(new Vector3(8.0f, 4.8f, 8.0f), straddling.position, 1e-3f, "straddling box");
            AssertVector(new Vector3(16.0f, 3.0f, 16.0f), buried.position, 1e-3f, "buried box");
        }

        [Test]
        public void BouncinessComesFromTheTerrainsMaterial()
        {
            TerrainCollider collider = Terrain(Data(new Vector3(32.0f, 10.0f, 32.0f), (x, z) => 0.5f));
            PhysicsMaterial material = new PhysicsMaterial { bounciness = 1.0f, bounceCombine = PhysicsMaterialCombine.Maximum };
            collider.sharedMaterial = material;
            Rigidbody ball = Ball(new Vector3(16.0f, 8.0f, 16.0f), 0.5f);
            ball.linearVelocity = new Vector3(0.0f, -10.0f, 0.0f);

            Simulate(30);

            Assert.That(ball.linearVelocity.y, Is.GreaterThan(4.0f));
            Object.DestroyImmediate(material);
        }

        [TestCase(1e4f)]
        [TestCase(1e5f)]
        [TestCase(1e6f)]
        public void ContactHoldsFarFromTheOrigin(float offset)
        {
            Vector3 origin = new Vector3(offset, 0.0f, -offset);
            Terrain(Data(new Vector3(320.0f, 10.0f, 320.0f), (x, z) => 0.25f + x / 32.0f * 0.1f), origin);
            Rigidbody box = DynamicBox(origin + new Vector3(160.0f, 10.0f, 160.0f));

            Simulate(150);

            Assert.That(box.position.y, Is.EqualTo(3.5f).Within(0.02f));
            Assert.That(SurfaceY(origin + new Vector3(160.0f, 30.0f, 100.0f)), Is.EqualTo(3.0f).Within(1e-2f));
        }

        [Test]
        public void SetHeightsChangesTheSurfaceAtOnce()
        {
            TerrainCollider collider = Slope();
            Vector3 above = new Vector3(83.0f, 100.0f, 25.0f);
            Assert.That(SurfaceY(above), Is.EqualTo(12.0f).Within(1e-3f));

            collider.terrainData.SetHeights(0, 0, new float[Resolution, Resolution]);

            Assert.That(SurfaceY(above), Is.EqualTo(4.0f).Within(1e-3f));
        }

        [Test]
        public void SetHeightsChangesOnlyItsRegion()
        {
            TerrainCollider collider = Terrain(Data(new Vector3(100.0f, 20.0f, 40.0f), (x, z) => 0.0f), new Vector3(3.0f, 4.0f, 5.0f));

            collider.terrainData.SetHeights(10, 20, new[,] { { 0.25f, 0.25f, 0.25f }, { 0.25f, 0.25f, 0.25f } });

            // Samples 10 to 12 along x and 20 and 21 along z, with cells of 3.125 by 1.25.
            Assert.That(SurfaceY(new Vector3(38.0f, 100.0f, 30.5f)), Is.EqualTo(9.0f).Within(1e-3f));
            Assert.That(SurfaceY(new Vector3(33.0f, 100.0f, 26.0f)), Is.EqualTo(4.0f).Within(1e-3f));
            Assert.That(SurfaceY(new Vector3(60.0f, 100.0f, 30.5f)), Is.EqualTo(4.0f).Within(1e-3f));
        }

        [Test]
        public void DelayedHeightsReachTheSurfaceBeforeTheSync()
        {
            TerrainCollider collider = Slope();
            Vector3 above = new Vector3(83.0f, 100.0f, 25.0f);

            collider.terrainData.SetHeightsDelayLOD(0, 0, new float[Resolution, Resolution]);
            Assert.That(SurfaceY(above), Is.EqualTo(4.0f).Within(1e-3f));

            collider.terrainData.SyncHeightmap();
            Assert.That(SurfaceY(above), Is.EqualTo(4.0f).Within(1e-3f));
        }

        [Test]
        public void AnEditLeavesASleepingBodyWhereItIs()
        {
            TerrainCollider collider = Slope();
            Rigidbody box = DynamicBox(new Vector3(53.0f, 16.0f, 25.0f));
            Simulate(300);
            Assert.That(box.IsSleeping(), Is.True);
            Vector3 rested = box.position;

            collider.terrainData.SetHeights(0, 0, new float[Resolution, Resolution]);
            Simulate(50);

            Assert.That(box.IsSleeping(), Is.True);
            AssertVector(rested, box.position, 1e-5f, "sleeping box");

            box.WakeUp();
            Simulate(100);

            Assert.That(box.position.y, Is.EqualTo(4.5f).Within(0.05f));
        }

        [Test]
        public void AnEditReachesAnAwakeBody()
        {
            TerrainCollider collider = Terrain(Data(new Vector3(32.0f, 10.0f, 32.0f), (x, z) => 0.5f));
            Rigidbody ball = Ball(new Vector3(16.0f, 5.5f, 16.0f), 0.5f);
            Simulate(10);

            collider.terrainData.SetHeights(0, 0, Heights((x, z) => 0.25f));
            Simulate(100);

            Assert.That(ball.position.y, Is.EqualTo(3.0f).Within(0.02f));
        }

        [Test]
        public void TwoTerrainsShareTheirDataAndItsEdits()
        {
            TerrainData data = Data(new Vector3(32.0f, 10.0f, 32.0f), (x, z) => 0.5f);
            Terrain(data);
            TerrainCollider second = Terrain(data, new Vector3(100.0f, 0.0f, 0.0f));

            data.SetHeights(0, 0, Heights((x, z) => 0.25f));

            Assert.That(SurfaceY(new Vector3(16.0f, 100.0f, 16.0f)), Is.EqualTo(2.5f).Within(1e-3f));
            Assert.That(SurfaceY(new Vector3(116.0f, 100.0f, 16.0f)), Is.EqualTo(2.5f).Within(1e-3f));

            Object.DestroyImmediate(second.gameObject);
            data.SetHeights(0, 0, Heights((x, z) => 0.75f));

            Assert.That(SurfaceY(new Vector3(16.0f, 100.0f, 16.0f)), Is.EqualTo(7.5f).Within(1e-3f));
            Assert.That(SurfaceY(new Vector3(116.0f, 100.0f, 16.0f)), Is.NaN);
        }

        [Test]
        public void ANewResolutionFlattensTheSurface()
        {
            TerrainCollider collider = Slope();

            collider.terrainData.heightmapResolution = 65;
            UnityEngine.Physics.SyncTransforms();

            AssertVector(new Vector3(100.0f, 0.0f, 40.0f), collider.bounds.extents, 1e-3f, "bounds extents");
            Assert.That(SurfaceY(new Vector3(83.0f, 100.0f, 25.0f)), Is.EqualTo(4.0f).Within(1e-3f));
        }

        [Test]
        public void SwappingTheDataSwapsTheSurface()
        {
            TerrainCollider collider = Slope();
            TerrainData other = Data(new Vector3(10.0f, 10.0f, 10.0f), (x, z) => 0.5f);

            collider.terrainData = other;
            UnityEngine.Physics.SyncTransforms();

            AssertVector(new Vector3(8.0f, 6.5f, 10.0f), collider.bounds.center, 1e-3f, "bounds centre");
            Assert.That(SurfaceY(new Vector3(8.0f, 100.0f, 10.0f)), Is.EqualTo(9.0f).Within(1e-3f));
            Assert.That(SurfaceY(new Vector3(83.0f, 100.0f, 25.0f)), Is.NaN);

            collider.terrainData = null;
            UnityEngine.Physics.SyncTransforms();

            Assert.That(SurfaceY(new Vector3(8.0f, 100.0f, 10.0f)), Is.NaN);
        }

        [Test]
        public void DestroyingTheDataLeavesTheColliderItsSurface()
        {
            TerrainCollider collider = Slope();
            TerrainData data = collider.terrainData;

            Object.DestroyImmediate(data);
            terrains.Remove(data);
            UnityEngine.Physics.SyncTransforms();

            Assert.That(SurfaceY(new Vector3(53.0f, 100.0f, 25.0f)), Is.EqualTo(9.0f).Within(1e-3f));
        }

        [Test]
        public void DestroyingTheColliderLeavesTheDataUsable()
        {
            TerrainCollider collider = Slope();
            TerrainData data = collider.terrainData;

            Object.DestroyImmediate(collider.gameObject);
            UnityEngine.Physics.SyncTransforms();
            Assert.That(SurfaceY(new Vector3(53.0f, 100.0f, 25.0f)), Is.NaN);

            data.SetHeights(0, 0, new float[Resolution, Resolution]);
            Terrain(data);

            Assert.That(SurfaceY(new Vector3(10.0f, 100.0f, 10.0f)), Is.EqualTo(0.0f).Within(1e-3f));
        }

        [Test]
        public void HolesLetThroughRaysShapesAndBodies()
        {
            TerrainData data = Data(new Vector3(32.0f, 1.0f, 32.0f), (x, z) => 0.5f);
            Terrain(data);
            data.SetHoles(3, 2, new[,] { { false } });
            UnityEngine.Physics.SyncTransforms();

            Assert.That(SurfaceY(new Vector3(3.2f, 5.0f, 2.2f)), Is.NaN);
            Assert.That(SurfaceY(new Vector3(3.8f, 5.0f, 2.8f)), Is.NaN);
            Assert.That(SurfaceY(new Vector3(2.5f, 5.0f, 2.5f)), Is.EqualTo(0.5f).Within(1e-3f));
            Assert.That(SurfaceY(new Vector3(3.5f, 5.0f, 3.5f)), Is.EqualTo(0.5f).Within(1e-3f));
            Assert.That(UnityEngine.Physics.CheckSphere(new Vector3(3.5f, 0.5f, 2.5f), 0.1f), Is.False);
            Assert.That(UnityEngine.Physics.CheckSphere(new Vector3(5.5f, 0.5f, 5.5f), 0.1f), Is.True);

            Rigidbody falling = Ball(new Vector3(3.5f, 2.0f, 2.5f), 0.1f);
            Rigidbody resting = Ball(new Vector3(5.5f, 2.0f, 5.5f), 0.1f);
            Simulate(100);

            Assert.That(falling.position.y, Is.LessThan(-5.0f));
            Assert.That(resting.position.y, Is.EqualTo(0.6f).Within(0.02f));
        }

        [Test]
        public void AFieldWithHolesKeepsItsShape()
        {
            TerrainData data = Data(new Vector3(100.0f, 20.0f, 40.0f), (x, z) => x / 32.0f * 0.5f + (x == 20 && z == 20 ? 0.25f : 0.0f));
            Terrain(data, new Vector3(3.0f, 4.0f, 5.0f));

            data.SetHoles(3, 2, new[,] { { false } });
            UnityEngine.Physics.SyncTransforms();

            Assert.That(SurfaceY(new Vector3(53.0f, 100.0f, 7.0f)), Is.EqualTo(9.0f).Within(1e-3f));

            // Sample quarter points in the four cells around the raised sample. Each cell's diagonal determines how it
            // splits into triangles.
            float cellX = 100.0f / 32.0f;
            float cellZ = 40.0f / 32.0f;
            float sampleX = 3.0f + 20.0f * cellX;
            float sampleZ = 5.0f + 20.0f * cellZ;
            float slopeAt = 4.0f + 20.0f * cellX * 0.1f;
            Assert.That(SurfaceY(new Vector3(sampleX + cellX * 0.5f, 100.0f, sampleZ + cellZ * 0.5f)), Is.EqualTo(slopeAt + cellX * 0.05f + 2.5f).Within(1e-3f));
            Assert.That(SurfaceY(new Vector3(sampleX + cellX * 0.5f, 100.0f, sampleZ - cellZ * 0.5f)), Is.EqualTo(slopeAt + cellX * 0.05f).Within(1e-3f));
        }

        [Test]
        public void FillingAHoleRestoresTheSurface()
        {
            TerrainData data = Data(new Vector3(32.0f, 1.0f, 32.0f), (x, z) => 0.5f);
            Terrain(data);

            data.SetHoles(3, 2, new[,] { { false } });
            UnityEngine.Physics.SyncTransforms();
            Assert.That(SurfaceY(new Vector3(3.5f, 5.0f, 2.5f)), Is.NaN);

            data.SetHoles(3, 2, new[,] { { true } });
            UnityEngine.Physics.SyncTransforms();
            Assert.That(SurfaceY(new Vector3(3.5f, 5.0f, 2.5f)), Is.EqualTo(0.5f).Within(1e-3f));
        }

        [Test]
        public void AFieldOfNothingButHolesHasNoSurface()
        {
            TerrainData data = Data(new Vector3(32.0f, 1.0f, 32.0f), (x, z) => 0.5f);
            Terrain(data);

            data.SetHoles(0, 0, new bool[32, 32]);
            UnityEngine.Physics.SyncTransforms();

            Assert.That(SurfaceY(new Vector3(3.5f, 5.0f, 2.5f)), Is.NaN);
        }

        [Test]
        public void HeightEditsKeepHolesOpen()
        {
            TerrainData data = Data(new Vector3(32.0f, 1.0f, 32.0f), (x, z) => 0.5f);
            Terrain(data);
            data.SetHoles(3, 2, new[,] { { false } });

            data.SetHeights(0, 0, Heights((x, z) => 0.25f));
            UnityEngine.Physics.SyncTransforms();

            Assert.That(SurfaceY(new Vector3(3.5f, 5.0f, 2.5f)), Is.NaN);
            Assert.That(SurfaceY(new Vector3(2.5f, 5.0f, 2.5f)), Is.EqualTo(0.25f).Within(1e-3f));
        }

        [Test]
        public void ATerrainSharesItsGameObjectWithABox()
        {
            TerrainCollider terrain = Slope();
            BoxCollider box = terrain.gameObject.AddComponent<BoxCollider>();
            box.center = new Vector3(10.0f, 50.0f, 10.0f);
            box.size = Vector3.one * 2.0f;
            UnityEngine.Physics.SyncTransforms();

            Assert.That(UnityEngine.Physics.Raycast(new Vector3(13.0f, 100.0f, 15.0f), Vector3.down, out RaycastHit top, 1000.0f), Is.True);
            Assert.That(top.collider, Is.SameAs(box));
            Assert.That(top.point.y, Is.EqualTo(55.0f).Within(1e-3f));

            Assert.That(UnityEngine.Physics.Raycast(new Vector3(53.0f, 100.0f, 25.0f), Vector3.down, out RaycastHit ground, 1000.0f), Is.True);
            Assert.That(ground.collider, Is.SameAs(terrain));
            Assert.That(ground.point.y, Is.EqualTo(9.0f).Within(1e-3f));
        }

        [Test]
        public void ATerrainUnderAStaticParentKeepsItsPlace()
        {
            GameObject parent = Track(new GameObject("parent"));
            parent.transform.position = new Vector3(10.0f, 0.0f, 0.0f);
            parent.transform.rotation = Quaternion.Euler(0.0f, 90.0f, 0.0f);
            TerrainCollider collider = Slope();
            collider.transform.SetParent(parent.transform, true);
            UnityEngine.Physics.SyncTransforms();

            Assert.That(SurfaceY(new Vector3(53.0f, 100.0f, 25.0f)), Is.EqualTo(9.0f).Within(1e-3f));
            AssertVector(new Vector3(53.0f, 9.0f, 25.0f), collider.bounds.center, 1e-3f, "bounds centre");
        }

        [Test]
        public void ReattachingBringsTheSurfaceBackAndKeepsEditing()
        {
            Assume.That(BackendInfo.IsRetooled, "needs --backend retooled");
            TerrainCollider collider = Slope();
            Rigidbody box = DynamicBox(new Vector3(53.0f, 16.0f, 25.0f));
            Simulate(100);
            float rested = box.position.y;

            BackendHost.Detach();
            BackendHost.Attach();

            Assert.That(SurfaceY(new Vector3(83.0f, 100.0f, 25.0f)), Is.EqualTo(12.0f).Within(1e-3f));
            Simulate(50);
            Assert.That(box.position.y, Is.EqualTo(rested).Within(0.02f), "the terrain came back under the box");

            collider.terrainData.SetHeights(0, 0, new float[Resolution, Resolution]);
            Assert.That(SurfaceY(new Vector3(83.0f, 100.0f, 25.0f)), Is.EqualTo(4.0f).Within(1e-3f));
        }
    }
}
