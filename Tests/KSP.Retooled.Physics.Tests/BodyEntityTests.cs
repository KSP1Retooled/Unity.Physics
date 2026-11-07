using KSP.Retooled.Components.Physics;
using NUnit.Framework;
using Unity.Collections;
using Unity.Entities;
using Unity.Mathematics;
using Unity.Physics;
using Unity.Transforms;
using UnityEngine;

namespace KSP.Retooled.Physics.Tests
{
    public class BodyEntityTests : PhysicsScene
    {
        private struct Moved : IComponentData
        {
            public int Value;
        }

        private static EntityManager Manager => BodyEntityWorld.World.EntityManager;

        [SetUp]
        public void RequireTheBackend()
        {
            Assume.That(BackendInfo.IsRetooled, "needs --backend retooled");
        }

        [TearDown]
        public void RebindTheWorld()
        {
            if (!BodyEntities.Bound && BackendInfo.IsRetooled)
            {
                BodyEntities.Bind(Manager);
            }
        }

        private static int EntityCount()
        {
            using EntityQuery query = Manager.CreateEntityQuery(ComponentType.ReadOnly<LocalTransform>());
            return query.CalculateEntityCount();
        }

        private static Entity EntityAt(float3 position)
        {
            using EntityQuery query = Manager.CreateEntityQuery(ComponentType.ReadOnly<LocalTransform>());
            using NativeArray<Entity> entities = query.ToEntityArray(Allocator.Temp);
            foreach (Entity entity in entities)
            {
                float3 at = Manager.GetComponentData<LocalTransform>(entity).Position;
                if (math.distance(at, position) < 1e-3f)
                {
                    return entity;
                }
            }

            return Entity.Null;
        }

        [Test]
        public void ABodysStateIsOnItsEntity()
        {
            float3 position = new float3(100.0f, 0.0f, 0.0f);
            Rigidbody body = WeightlessBox(position, 7.0f);
            body.linearVelocity = new Vector3(1.0f, 2.0f, 3.0f);

            Entity entity = EntityAt(position);

            Assert.That(entity, Is.Not.EqualTo(Entity.Null));
            Assert.That(Manager.GetComponentData<BodyMass>(entity).Mass, Is.EqualTo(7.0f));
            Assert.That(Manager.GetComponentData<PhysicsVelocity>(entity).Linear.y, Is.EqualTo(2.0f));
        }

        [Test]
        public void AStepMovesTheEntity()
        {
            float3 position = new float3(200.0f, 0.0f, 0.0f);
            Rigidbody body = WeightlessBox(position);
            Entity entity = EntityAt(position);
            body.linearVelocity = new Vector3(0.0f, 5.0f, 0.0f);

            Simulate();

            float3 moved = Manager.GetComponentData<LocalTransform>(entity).Position;
            Assert.That(moved.y, Is.EqualTo(5.0f * Step).Within(1e-3f));
            Assert.That(body.position.y, Is.EqualTo(moved.y));
        }

        [Test]
        public void WritingTheComponentReachesTheRigidbody()
        {
            float3 position = new float3(300.0f, 0.0f, 0.0f);
            Rigidbody body = WeightlessBox(position);

            Manager.SetComponentData(EntityAt(position), new PhysicsVelocity { Linear = new float3(0.0f, 0.0f, 9.0f) });

            Assert.That(body.linearVelocity.z, Is.EqualTo(9.0f));
        }

        [Test]
        public void ARigidbodyFollowsItsEntityToAnotherChunk()
        {
            float3 position = new float3(400.0f, 0.0f, 0.0f);
            Rigidbody body = WeightlessBox(position);
            Entity entity = EntityAt(position);
            body.linearVelocity = Vector3.one;

            Manager.AddComponentData(entity, new Moved { Value = 1 });
            body.linearVelocity = Vector3.up;
            Simulate();

            Assert.That(Manager.GetComponentData<PhysicsVelocity>(entity).Linear.y, Is.EqualTo(1.0f).Within(1e-3f));
            Assert.That(Manager.GetComponentData<LocalTransform>(entity).Position.y, Is.EqualTo(Step).Within(1e-3f));
        }

        [Test]
        public void DestroyingABodyLeavesTheOthersAlone()
        {
            float3 kept = new float3(500.0f, 0.0f, 0.0f);
            float3 gone = new float3(510.0f, 0.0f, 0.0f);
            Rigidbody first = WeightlessBox(gone);
            Rigidbody second = WeightlessBox(kept);
            second.linearVelocity = Vector3.up;

            Object.DestroyImmediate(first);

            Assert.That(second.linearVelocity.y, Is.EqualTo(1.0f));
            Assert.That(Manager.GetComponentData<PhysicsVelocity>(EntityAt(kept)).Linear.y, Is.EqualTo(1.0f));
        }

        [Test]
        public void ABodyKeepsItsStateWithNoWorld()
        {
            float3 position = new float3(600.0f, 0.0f, 0.0f);
            Rigidbody body = WeightlessBox(position);
            body.linearVelocity = Vector3.right;

            BodyEntities.Unbind();

            Assert.That(EntityCount(), Is.Zero);
            Assert.That(body.linearVelocity.x, Is.EqualTo(1.0f));

            body.linearVelocity = Vector3.up;
            BodyEntities.Bind(Manager);

            Assert.That(Manager.GetComponentData<PhysicsVelocity>(EntityAt(position)).Linear.y, Is.EqualTo(1.0f));
        }

        [Test]
        public void ABodyMadeBeforeTheWorldGetsAnEntity()
        {
            BodyEntities.Unbind();
            float3 position = new float3(700.0f, 0.0f, 0.0f);
            WeightlessBox(position, 3.0f);

            BodyEntities.Bind(Manager);

            Assert.That(Manager.GetComponentData<BodyMass>(EntityAt(position)).Mass, Is.EqualTo(3.0f));
        }

        [Test]
        public void TheWorldsOwnUpdateDoesNotStepBodies()
        {
            float3 position = new float3(750.0f, 0.0f, 0.0f);
            Rigidbody body = WeightlessBox(position);
            body.linearVelocity = Vector3.up;

            BodyEntityWorld.World.GetExistingSystemManaged<Unity.Physics.Systems.PhysicsSystemGroup>().Update();

            Assert.That(body.position.y, Is.Zero);
        }

        [Test]
        public void TurningABodyKeepsItsWorldAngularVelocity()
        {
            Rigidbody body = WeightlessBox(new float3(850.0f, 0.0f, 0.0f));
            Vector3 angular = new Vector3(1.0f, 2.0f, 3.0f);
            body.angularVelocity = angular;

            body.rotation = Quaternion.Euler(30.0f, 40.0f, 50.0f);

            Assert.That(Vector3.Distance(body.angularVelocity, angular), Is.LessThan(1e-5f));
        }

        [Test]
        public void TurningTheInertiaAxesKeepsTheWorldAngularVelocity()
        {
            Rigidbody body = WeightlessBox(new float3(860.0f, 0.0f, 0.0f));
            Vector3 angular = new Vector3(1.0f, 2.0f, 3.0f);
            body.angularVelocity = angular;

            body.inertiaTensorRotation = Quaternion.Euler(30.0f, 40.0f, 50.0f);

            Assert.That(Vector3.Distance(body.angularVelocity, angular), Is.LessThan(1e-5f));
        }

        [Test]
        public void TheEntityHoldsAngularVelocityAboutThePrincipalAxes()
        {
            float3 position = new float3(870.0f, 0.0f, 0.0f);
            Rigidbody body = WeightlessBox(position);
            Quaternion rotation = Quaternion.Euler(0.0f, 90.0f, 0.0f);
            body.rotation = rotation;
            Entity entity = EntityAt(position);

            body.angularVelocity = Vector3.right;

            float3 held = Manager.GetComponentData<PhysicsVelocity>(entity).Angular;
            float3 expected = math.rotate(math.inverse((quaternion)rotation), new float3(1.0f, 0.0f, 0.0f));
            Assert.That(math.distance(held, expected), Is.LessThan(1e-5f));
        }

        [Test]
        public void AWorldThatIsGoneStillLetsItsBodiesGo()
        {
            float3 position = new float3(900.0f, 0.0f, 0.0f);
            Rigidbody body = WeightlessBox(position, 5.0f);
            BodyEntities.Unbind();
            World gone = new World("Gone");
            BodyEntities.Bind(gone.EntityManager);
            gone.Dispose();

            Assert.That(() => BodyEntities.Unbind(), Throws.Exception);

            Assert.That(BodyEntities.Bound, Is.False);
            Assert.That(body.mass, Is.EqualTo(5.0f));
        }

        [Test]
        public void ADetachLeavesOneEntityPerBody()
        {
            Rigidbody body = WeightlessBox(new float3(800.0f, 0.0f, 0.0f), 4.0f);
            int before = EntityCount();

            BackendHost.Detach();
            Assert.That(EntityCount(), Is.Zero);
            BackendHost.Attach();

            Assert.That(EntityCount(), Is.EqualTo(before));
            Assert.That(body.mass, Is.EqualTo(4.0f));
        }
    }
}
