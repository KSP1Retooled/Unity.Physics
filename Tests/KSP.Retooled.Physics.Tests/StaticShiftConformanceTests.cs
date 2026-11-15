using NUnit.Framework;
using UnityEngine;

namespace KSP.Retooled.Physics.Tests
{
    /// <summary>
    /// Statics moved all together, as KSP's floating origin moves them, far enough from the origin that each one's pose rounds
    /// a little differently.
    /// </summary>
    public class StaticShiftConformanceTests : PhysicsScene
    {
        private static readonly Vector3 Far = new Vector3(3000.0f, 0.0f, -2000.0f);
        private static readonly Vector3 Shift = new Vector3(5.0f, 0.25f, -3.0f);

        private GameObject scenery;

        [SetUp]
        public void BuildScenery()
        {
            scenery = Track(new GameObject("scenery"));
            scenery.transform.position = Far;
            for (int x = 0; x < 10; x++)
            {
                for (int z = 0; z < 10; z++)
                {
                    GameObject tile = Box(Far + new Vector3(x * 5.0f, -0.5f, z * 5.0f), new Vector3(4.0f, 1.0f, 4.0f));
                    tile.transform.SetParent(scenery.transform, true);
                }
            }
        }

        [Test]
        public void ABodyStaysOnStaticsThatShiftEveryStep()
        {
            Rigidbody body = DynamicBox(Far + new Vector3(0.0f, 0.5f, 0.0f));
            Simulate(25);

            for (int i = 0; i < 300; i++)
            {
                scenery.transform.position += Shift;
                body.transform.position += Shift;
                Simulate(1);

                float height = body.position.y - scenery.transform.position.y;
                Assert.That(height, Is.EqualTo(0.5f).Within(0.02f), $"step {i}");
            }
        }

        [Test]
        public void ARayFindsAShiftedStaticWhereItWent()
        {
            GameObject marker = Box(Far + new Vector3(-20.0f, 0.0f, 0.0f), Vector3.one);
            marker.transform.SetParent(scenery.transform, true);
            Simulate(1);

            for (int i = 0; i < 300; i++)
            {
                Vector3 before = marker.transform.position;
                scenery.transform.position += Shift;
                Simulate(1);

                Vector3 after = marker.transform.position;
                Assert.That(UnityEngine.Physics.Raycast(after + Vector3.up * 5.0f, Vector3.down, out RaycastHit hit, 10.0f), Is.True, $"step {i}");
                Assert.That(hit.collider.gameObject, Is.SameAs(marker), $"step {i}");
                Assert.That(UnityEngine.Physics.Raycast(before + Vector3.up * 5.0f, Vector3.down, 10.0f), Is.False, $"left its place at step {i}");
            }
        }

        [Test]
        public void AStaticThatTurnsWhileTheRestShiftIsFoundWhereItTurned()
        {
            GameObject bar = Box(Far + new Vector3(-30.0f, 0.0f, 0.0f), new Vector3(10.0f, 1.0f, 1.0f));
            bar.transform.SetParent(scenery.transform, true);
            Simulate(1);

            scenery.transform.position += Shift;
            bar.transform.localRotation = Quaternion.Euler(0.0f, 90.0f, 0.0f);
            Simulate(1);

            Vector3 centre = bar.transform.position;
            Assert.That(UnityEngine.Physics.Raycast(centre + new Vector3(4.0f, 5.0f, 0.0f), Vector3.down, 10.0f), Is.False, "the old end");
            Assert.That(UnityEngine.Physics.Raycast(centre + new Vector3(0.0f, 5.0f, 4.0f), Vector3.down, out RaycastHit hit, 10.0f), Is.True, "the new end");
            Assert.That(hit.collider.gameObject, Is.SameAs(bar));
        }

        // KSP's scaled planets move their own way while the floating origin shifts everything else.
        [Test]
        public void AStaticThatMovesItsOwnWayIsFoundWhereItWent()
        {
            GameObject planet = Box(Far + new Vector3(-30.0f, 0.0f, 0.0f), Vector3.one);
            Simulate(1);

            for (int i = 0; i < 20; i++)
            {
                Vector3 before = planet.transform.position;
                scenery.transform.position += Shift;
                planet.transform.position += new Vector3(0.0f, 0.0f, 3.0f);
                Simulate(1);

                Vector3 after = planet.transform.position;
                Assert.That(UnityEngine.Physics.Raycast(after + Vector3.up * 5.0f, Vector3.down, out RaycastHit hit, 10.0f), Is.True, $"step {i}");
                Assert.That(hit.collider.gameObject, Is.SameAs(planet), $"step {i}");
                Assert.That(UnityEngine.Physics.Raycast(before + Vector3.up * 5.0f, Vector3.down, 10.0f), Is.False, $"left its place at step {i}");
            }
        }

        // A scene change destroys the static objects and creates new ones, possibly at the same positions. The new
        // colliders may also reuse the memory addresses of the old colliders.
        [Test]
        public void StaticsMadeAgainInTheSamePlaceHaveTheirNewLayer()
        {
            Simulate(1);
            Object.DestroyImmediate(scenery);
            BuildScenery();
            foreach (Transform tile in scenery.transform)
            {
                tile.gameObject.layer = 8;
            }

            Simulate(1);

            Vector3 above = Far + new Vector3(0.0f, 5.0f, 0.0f);
            Assert.That(UnityEngine.Physics.Raycast(above, Vector3.down, 10.0f, 1 << 8), Is.True);
            Assert.That(UnityEngine.Physics.Raycast(above, Vector3.down, 10.0f, ~(1 << 8)), Is.False);
        }

        [Test]
        public void AStaticThatGrowsWhileTheRestShiftIsFoundToItsNewEdge()
        {
            GameObject block = Box(Far + new Vector3(-30.0f, 0.0f, 0.0f), Vector3.one);
            block.transform.SetParent(scenery.transform, true);
            Simulate(1);

            scenery.transform.position += Shift;
            block.GetComponent<BoxCollider>().size = new Vector3(8.0f, 1.0f, 1.0f);
            Simulate(1);

            Vector3 centre = block.transform.position;
            Assert.That(UnityEngine.Physics.Raycast(centre + new Vector3(3.5f, 5.0f, 0.0f), Vector3.down, out RaycastHit hit, 10.0f), Is.True);
            Assert.That(hit.collider.gameObject, Is.SameAs(block));
        }
    }
}
