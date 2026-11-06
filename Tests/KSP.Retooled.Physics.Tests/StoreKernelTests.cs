using NUnit.Framework;
using Unity.Collections;
using Unity.Mathematics;
using Unity.Physics;

namespace KSP.Retooled.PhysicsBackend.Tests
{
    /// <summary>
    /// The command kernels against a world of their own, for cases the Unity API can't reach, or can't show the result of.
    /// The kernels run as plain methods here, so a failure has a stack.
    /// </summary>
    public unsafe class StoreKernelTests
    {
        private const NativeStore.BodyFlags Dynamic = NativeStore.BodyFlags.Simulated;
        private const NativeStore.BodyFlags Kinematic = NativeStore.BodyFlags.Simulated | NativeStore.BodyFlags.Kinematic;

        private NativeStore.StoreGlobals* globals;
        private NativeStore.WorldData* world;

        [SetUp]
        public void CreateWorld()
        {
            Assume.That(BackendInfo.IsRetooled, "needs --backend retooled");
            globals = NativeStore.RetooledPhysics_Globals();
            world = StoreOps.Allocate<NativeStore.WorldData>(globals);
        }

        [TearDown]
        public void DestroyWorld()
        {
            if (world != null)
            {
                WorldKernel.FreeBodies(world, globals);
                StoreOps.Release(globals, world->bodies);
                StoreOps.Release(globals, world);
                world = null;
            }
        }

        private NativeStore.Body* CreateBody(NativeStore.BodyFlags flags, bool isStatic = false)
        {
            NativeStore.CommandArgs args = default;
            args.rotation = quaternion.identity;
            args.flags = (uint)flags;
            args.enabled = isStatic ? 1u : 0u;
            args.scalar = 1.0f;
            WorldKernel.Run((uint)WorldCommand.CreateBody, world, &args, globals);
            return (NativeStore.Body*)args.pointer;
        }

        private NativeStore.Shape* CreateShape(uint geometryType, float3 extents)
        {
            NativeStore.CommandArgs args = default;
            args.mode = geometryType;
            args.vector = extents;
            ShapeKernel.Run((uint)ShapeCommand.CreateShape, null, &args, globals);
            return (NativeStore.Shape*)args.pointer;
        }

        private void DestroyShape(NativeStore.Shape* shape)
        {
            NativeStore.CommandArgs args = default;
            ShapeKernel.Run((uint)ShapeCommand.DestroyShape, shape, &args, globals);
        }

        private void Attach(NativeStore.Body* body, NativeStore.Shape* shape)
        {
            NativeStore.CommandArgs args = default;
            args.rotation = quaternion.identity;
            args.pointer = shape;
            args.enabled = 1;
            BodyKernel.Run((uint)BodyCommand.AttachShape, body, &args, globals);
        }

        private void SetTarget(NativeStore.Body* body, float3 position)
        {
            NativeStore.CommandArgs args = default;
            args.rotation = quaternion.identity;
            args.position = position;
            BodyKernel.Run((uint)BodyCommand.SetKinematicTarget, body, &args, globals);
        }

        private bool HasTarget(NativeStore.Body* body)
        {
            return (StoreOps.Hot(body)->flags & NativeStore.BodyFlags.HasKinematicTarget) != 0;
        }

        [Test]
        public void AttachingAShapeTakesItOffItsPreviousBody()
        {
            NativeStore.Body* first = CreateBody(Dynamic);
            NativeStore.Body* second = CreateBody(Dynamic);
            NativeStore.Shape* shape = CreateShape(NativeStore.GeometryBox, new float3(0.5f));

            Attach(first, shape);
            Attach(second, shape);

            Assert.That(first->shapeCount, Is.EqualTo(0u));
            Assert.That(StoreOps.Hot(first)->shapeCount, Is.EqualTo(0u));
            Assert.That(second->shapeCount, Is.EqualTo(1u));
            Assert.That(shape->body == second, Is.True);

            DestroyShape(shape);
            Assert.That(second->shapeCount, Is.EqualTo(0u));
        }

        [Test]
        public void AttachingAShapeTwiceListsItOnce()
        {
            NativeStore.Body* body = CreateBody(Dynamic);
            NativeStore.Shape* shape = CreateShape(NativeStore.GeometryBox, new float3(0.5f));

            Attach(body, shape);
            Attach(body, shape);

            Assert.That(body->shapeCount, Is.EqualTo(1u));
            DestroyShape(shape);
        }

        [Test]
        public void AZeroVolumeCapsuleTakesNoMassAndLeavesInertiaFinite()
        {
            NativeStore.Body* body = CreateBody(Dynamic);
            NativeStore.Shape* box = CreateShape(NativeStore.GeometryBox, new float3(0.5f));
            NativeStore.Shape* capsule = CreateShape(NativeStore.GeometryCapsule, float3.zero);
            Attach(body, box);
            Attach(body, capsule);

            NativeStore.CommandArgs args = default;
            BodyKernel.Run((uint)BodyCommand.RecomputeMassProperties, body, &args, globals);

            float3 inertia = StoreOps.Hot(body)->inertia;
            Assert.That(math.all(math.isfinite(inertia)), Is.True, $"inertia {inertia}");
            Assert.That(inertia.x, Is.EqualTo(1.0f / 6.0f).Within(1e-5f), "the box alone");

            DestroyShape(box);
            DestroyShape(capsule);
        }

        [Test]
        public void ATargetIsNotKeptForABodyThatWontStep()
        {
            NativeStore.Body* body = CreateBody(Kinematic);
            NativeStore.CommandArgs args = default;

            SetTarget(body, new float3(1.0f, 0.0f, 0.0f));
            Assert.That(HasTarget(body), Is.True);
            BodyKernel.Run((uint)BodyCommand.Sleep, body, &args, globals);
            Assert.That(HasTarget(body), Is.False, "asleep");

            SetTarget(body, new float3(1.0f, 0.0f, 0.0f));
            args.flags = (uint)NativeStore.BodyFlags.Simulated;
            BodyKernel.Run((uint)BodyCommand.SetFlags, body, &args, globals);
            Assert.That(HasTarget(body), Is.False, "taken out of the simulation");

            SetTarget(body, new float3(1.0f, 0.0f, 0.0f));
            Assert.That(HasTarget(body), Is.False, "set while out of the simulation");
        }

        private bool IsIgnored(uint first, uint second)
        {
            NativeStore.CommandArgs args = default;
            args.value = first;
            args.value2 = second;
            ShapeKernel.Run((uint)ShapeCommand.ShouldIgnoreCollision, null, &args, globals);
            return args.enabled != 0;
        }

        private void SetIgnored(uint first, uint second, bool ignored)
        {
            NativeStore.CommandArgs args = default;
            args.value = first;
            args.value2 = second;
            args.enabled = ignored ? 1u : 0u;
            ShapeKernel.Run((uint)ShapeCommand.IgnoreCollision, null, &args, globals);
        }

        [Test]
        public void IgnoredPairsAreRememberedInEitherOrderUntilCleared()
        {
            SetIgnored(7, 3, true);
            SetIgnored(3, 9, true);
            SetIgnored(1, 2, true);
            SetIgnored(7, 3, true);

            Assert.That(IsIgnored(3, 7), Is.True);
            Assert.That(IsIgnored(9, 3), Is.True);
            Assert.That(IsIgnored(1, 2), Is.True);
            Assert.That(IsIgnored(3, 8), Is.False);

            SetIgnored(3, 7, false);
            Assert.That(IsIgnored(7, 3), Is.False);
            Assert.That(IsIgnored(3, 9), Is.True, "only that pair");

            NativeStore.CommandArgs cleanup = default;
            cleanup.value = 3;
            ShapeKernel.Run((uint)ShapeCommand.CleanupIgnoredColliders, null, &cleanup, globals);
            Assert.That(IsIgnored(3, 9), Is.False, "every pair of a destroyed collider goes");
            Assert.That(IsIgnored(1, 2), Is.True, "and no other");

            SetIgnored(1, 2, false);
        }

        [Test]
        public void StatsCountEachKindOfBodyOnce()
        {
            CreateBody(Dynamic);
            CreateBody(Dynamic);
            CreateBody(Kinematic);
            CreateBody(Dynamic, true);
            CreateBody(Dynamic, true);
            CreateBody(Dynamic, true);

            NativeStore.CommandArgs sleep = default;
            BodyKernel.Run((uint)BodyCommand.Sleep, world->bodies[1].owner, &sleep, globals);

            WorldKernel.BodyCounts counts = default;
            NativeStore.CommandArgs args = default;
            args.pointer = &counts;
            WorldKernel.Run((uint)WorldCommand.GetLastSimulationStats, world, &args, globals);

            Assert.That(counts.DynamicBodies, Is.EqualTo(2u));
            Assert.That(counts.ActiveDynamicBodies, Is.EqualTo(1u));
            Assert.That(counts.KinematicBodies, Is.EqualTo(1u));
            Assert.That(counts.ActiveKinematicBodies, Is.EqualTo(1u));
            Assert.That(counts.StaticBodies, Is.EqualTo(3u));
        }

        [Test]
        public void DestroyingABodyMovesTheLastIntoItsRow()
        {
            NativeStore.Body* first = CreateBody(Dynamic);
            NativeStore.Body* last = CreateBody(Dynamic);

            NativeStore.CommandArgs args = default;
            args.pointer = first;
            WorldKernel.Run((uint)WorldCommand.DestroyBody, world, &args, globals);

            Assert.That(world->bodyCount, Is.EqualTo(1u));
            Assert.That(last->index, Is.EqualTo(0u));
            Assert.That(world->bodies[0].owner == last, Is.True);
        }

        [Test]
        public void ShapesByLayerMaskListOnlyTheLayer()
        {
            NativeStore.Body* body = CreateBody(Dynamic);
            NativeStore.Shape* wanted = CreateShape(NativeStore.GeometryBox, new float3(0.5f));
            NativeStore.Shape* other = CreateShape(NativeStore.GeometryBox, new float3(0.5f));
            wanted->data.belongsTo = 1;
            wanted->data.userData = wanted;
            other->data.belongsTo = 2;
            Attach(body, wanted);
            Attach(body, other);

            NativeStore.CommandArgs args = default;
            args.value = 1;
            WorldKernel.Run((uint)WorldCommand.GetAllShapesByLayerMask, world, &args, globals);

            Assert.That(args.size, Is.EqualTo(1ul));
            Assert.That(((void**)args.pointer)[0] == wanted, Is.True);

            StoreOps.Release(globals, args.pointer);
            DestroyShape(wanted);
            DestroyShape(other);
        }

        [Test]
        public void AKinematicTargetIsReachedByTheCentreOfMass()
        {
            const float deltaTime = 0.02f;
            float3 centerOfMass = new float3(1.0f, 0.0f, 0.0f);
            quaternion targetRotation = quaternion.RotateY(0.5f);
            float3 targetPosition = new float3(0.0f, 2.0f, 0.0f);

            NativeStore.BodyData body = default;
            body.rotation = quaternion.identity;
            body.inertiaRotation = quaternion.identity;
            body.localCenterOfMass = centerOfMass;
            body.flags = Kinematic | NativeStore.BodyFlags.HasKinematicTarget;
            body.targetPosition = targetPosition;
            body.targetRotation = targetRotation;

            using NativeArray<int> dynamicIndices = new NativeArray<int>(new[] { 0 }, Allocator.TempJob);
            using NativeArray<int> staticIndices = new NativeArray<int>(0, Allocator.TempJob);
            using NativeArray<RigidBody> bodies = new NativeArray<RigidBody>(2, Allocator.TempJob);
            using NativeArray<MotionData> motions = new NativeArray<MotionData>(1, Allocator.TempJob);
            using NativeArray<MotionVelocity> velocities = new NativeArray<MotionVelocity>(1, Allocator.TempJob);

            new BuildBodiesJob
            {
                Store = &body,
                DynamicIndices = dynamicIndices,
                StaticIndices = staticIndices,
                Bodies = bodies,
                MotionDatas = motions,
                MotionVelocities = velocities,
                DeltaTime = deltaTime,
            }.Execute();

            float3 reached = motions[0].WorldFromMotion.pos + velocities[0].LinearVelocity * deltaTime;
            float3 expected = targetPosition + math.rotate(targetRotation, centerOfMass);
            Assert.That(math.distance(expected, reached), Is.LessThan(1e-4f), $"reached {reached}, expected {expected}");
        }
    }
}
