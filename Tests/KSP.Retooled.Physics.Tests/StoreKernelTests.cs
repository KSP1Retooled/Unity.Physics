using NUnit.Framework;
using Unity.Collections;
using Unity.Mathematics;
using Unity.Physics;
using Unity.Transforms;
using Unity.Physics.Store.Bodies;
using Unity.Physics.Store.Memory;
using Unity.Physics.Store.Shapes;
using Unity.Physics.Store.World;
using Unity.Physics.Stepping;

namespace KSP.Retooled.Physics.Tests
{
    /// <summary>
    /// Test command kernels directly in an isolated world for cases not exposed through Unity's public API.
    /// </summary>
    public unsafe class StoreKernelTests
    {
        private const BodyFlags Dynamic = BodyFlags.Simulated;
        private const BodyFlags Kinematic = BodyFlags.Simulated | BodyFlags.Kinematic;

        private StoreGlobals* globals;
        private WorldData* world;

        [SetUp]
        public void CreateWorld()
        {
            Assume.That(BackendInfo.IsRetooled, "needs --backend retooled");
            globals = NativeStore.RetooledPhysics_Globals();
            world = StoreMemory.Allocate<WorldData>(globals);
        }

        [TearDown]
        public void DestroyWorld()
        {
            if (world != null)
            {
                WorldKernel.FreeBodies(world, globals);
                StoreMemory.Release(globals, world->bodies);
                StoreMemory.Release(globals, world);
                world = null;
            }
        }

        private Body* CreateBody(BodyFlags flags, bool isStatic = false)
        {
            CommandArgs args = default;
            args.rotation = quaternion.identity;
            args.flags = (uint)flags;
            args.enabled = isStatic ? 1u : 0u;
            args.scalar = 1.0f;
            WorldKernel.Run((uint)WorldCommand.CreateBody, world, &args, globals);
            return (Body*)args.pointer;
        }

        private Shape* CreateShape(uint geometryType, float3 extents)
        {
            CommandArgs args = default;
            args.mode = geometryType;
            args.vector = extents;
            ShapeKernel.Run((uint)ShapeCommand.CreateShape, null, &args, globals);
            return (Shape*)args.pointer;
        }

        private void DestroyShape(Shape* shape)
        {
            CommandArgs args = default;
            ShapeKernel.Run((uint)ShapeCommand.DestroyShape, shape, &args, globals);
        }

        private void Attach(Body* body, Shape* shape)
        {
            CommandArgs args = default;
            args.rotation = quaternion.identity;
            args.pointer = shape;
            args.enabled = 1;
            BodyKernel.Run((uint)BodyCommand.AttachShape, body, &args, globals);
        }

        private void SetTarget(Body* body, float3 position)
        {
            CommandArgs args = default;
            args.rotation = quaternion.identity;
            args.position = position;
            BodyKernel.Run((uint)BodyCommand.SetKinematicTarget, body, &args, globals);
        }

        private bool HasTarget(Body* body)
        {
            return (StoreOps.Hot(body)->flags & BodyFlags.HasKinematicTarget) != 0;
        }

        [Test]
        public void AttachingAShapeTakesItOffItsPreviousBody()
        {
            Body* first = CreateBody(Dynamic);
            Body* second = CreateBody(Dynamic);
            Shape* shape = CreateShape(StoreConstants.GeometryBox, new float3(0.5f));

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
            Body* body = CreateBody(Dynamic);
            Shape* shape = CreateShape(StoreConstants.GeometryBox, new float3(0.5f));

            Attach(body, shape);
            Attach(body, shape);

            Assert.That(body->shapeCount, Is.EqualTo(1u));
            DestroyShape(shape);
        }

        [Test]
        public void AZeroVolumeCapsuleTakesNoMassAndLeavesInertiaFinite()
        {
            Body* body = CreateBody(Dynamic);
            Shape* box = CreateShape(StoreConstants.GeometryBox, new float3(0.5f));
            Shape* capsule = CreateShape(StoreConstants.GeometryCapsule, float3.zero);
            Attach(body, box);
            Attach(body, capsule);

            CommandArgs args = default;
            BodyKernel.Run((uint)BodyCommand.RecomputeMassProperties, body, &args, globals);

            float3 inertia = (*StoreOps.Hot(body)).BodyMass()->Inertia;
            Assert.That(math.all(math.isfinite(inertia)), Is.True, $"inertia {inertia}");
            Assert.That(inertia.x, Is.EqualTo(1.0f / 6.0f).Within(1e-5f), "the box alone");

            DestroyShape(box);
            DestroyShape(capsule);
        }

        [Test]
        public void ATargetIsNotKeptForABodyThatWontStep()
        {
            Body* body = CreateBody(Kinematic);
            CommandArgs args = default;

            SetTarget(body, new float3(1.0f, 0.0f, 0.0f));
            Assert.That(HasTarget(body), Is.True);
            BodyKernel.Run((uint)BodyCommand.Sleep, body, &args, globals);
            Assert.That(HasTarget(body), Is.False, "asleep");

            SetTarget(body, new float3(1.0f, 0.0f, 0.0f));
            args.flags = (uint)BodyFlags.Simulated;
            BodyKernel.Run((uint)BodyCommand.SetFlags, body, &args, globals);
            Assert.That(HasTarget(body), Is.False, "taken out of the simulation");

            SetTarget(body, new float3(1.0f, 0.0f, 0.0f));
            Assert.That(HasTarget(body), Is.False, "set while out of the simulation");
        }

        private bool IsIgnored(uint first, uint second)
        {
            CommandArgs args = default;
            args.value = first;
            args.value2 = second;
            ShapeKernel.Run((uint)ShapeCommand.ShouldIgnoreCollision, null, &args, globals);
            return args.enabled != 0;
        }

        private void SetIgnored(uint first, uint second, bool ignored)
        {
            CommandArgs args = default;
            args.value = first;
            args.value2 = second;
            args.enabled = ignored ? 1u : 0u;
            ShapeKernel.Run((uint)ShapeCommand.IgnoreCollision, null, &args, globals);
        }

        // The store's table of ignored collider pairs is shared by all live colliders. Allocate IDs from that table and
        // return them after use.
        private uint[] NewIds(int count)
        {
            uint[] ids = new uint[count];
            for (int i = 0; i < count; i += 2)
            {
                CommandArgs args = default;
                args.enabled = 1;
                ShapeKernel.Run((uint)ShapeCommand.IgnoreCollision, null, &args, globals);
                ids[i] = args.value;
                if (i + 1 < count)
                {
                    ids[i + 1] = args.value2;
                }

                SetIgnored(args.value, args.value2, false);
            }

            return ids;
        }

        private void Destroyed(uint id)
        {
            CommandArgs cleanup = default;
            cleanup.value = id;
            ShapeKernel.Run((uint)ShapeCommand.CleanupIgnoredColliders, null, &cleanup, globals);
        }

        [Test]
        public void IgnoredPairsAreRememberedInEitherOrderUntilCleared()
        {
            uint[] ids = NewIds(6);
            uint a = ids[0];
            uint b = ids[1];
            uint c = ids[2];
            uint d = ids[3];
            uint e = ids[4];
            uint f = ids[5];

            SetIgnored(e, c, true);
            SetIgnored(c, f, true);
            SetIgnored(a, b, true);
            SetIgnored(e, c, true);

            Assert.That(IsIgnored(c, e), Is.True);
            Assert.That(IsIgnored(f, c), Is.True);
            Assert.That(IsIgnored(a, b), Is.True);
            Assert.That(IsIgnored(c, d), Is.False);

            SetIgnored(c, e, false);
            Assert.That(IsIgnored(e, c), Is.False);
            Assert.That(IsIgnored(c, f), Is.True, "only that pair");

            Destroyed(c);
            Assert.That(IsIgnored(c, f), Is.False, "every pair of a destroyed collider goes");
            Assert.That(IsIgnored(a, b), Is.True, "and no other");

            Destroyed(a);
            Destroyed(b);
            Destroyed(d);
            Destroyed(e);
            Destroyed(f);
        }

        [Test]
        public void IgnoringWithoutIdsHandsOutOnesThatDestroyingFreesAgain()
        {
            CommandArgs args = default;
            args.enabled = 1;
            ShapeKernel.Run((uint)ShapeCommand.IgnoreCollision, null, &args, globals);
            uint first = args.value;
            uint second = args.value2;

            Assert.That(first, Is.Not.Zero);
            Assert.That(second, Is.Not.Zero.And.Not.EqualTo(first));
            Assert.That(IsIgnored(first, second), Is.True);

            CommandArgs cleanup = default;
            cleanup.value = second;
            ShapeKernel.Run((uint)ShapeCommand.CleanupIgnoredColliders, null, &cleanup, globals);
            Assert.That(IsIgnored(first, second), Is.False);

            CommandArgs another = default;
            another.value = first;
            another.enabled = 1;
            ShapeKernel.Run((uint)ShapeCommand.IgnoreCollision, null, &another, globals);
            Assert.That(another.value2, Is.EqualTo(second), "a destroyed collider's id goes back in use");

            cleanup.value = second;
            ShapeKernel.Run((uint)ShapeCommand.CleanupIgnoredColliders, null, &cleanup, globals);
            cleanup.value = first;
            ShapeKernel.Run((uint)ShapeCommand.CleanupIgnoredColliders, null, &cleanup, globals);
        }

        [Test]
        public void ManyIgnoredPairsSurviveGrowingAndRestoring()
        {
            const int Count = 400;
            uint[] ids = NewIds(Count);
            for (int i = 0; i < Count; i++)
            {
                for (int j = i + 1; j < Count; j += 3)
                {
                    SetIgnored(ids[i], ids[j], true);
                }
            }

            for (int i = 0; i < Count; i += 2)
            {
                for (int j = i + 1; j < Count; j += 3)
                {
                    SetIgnored(ids[j], ids[i], false);
                }
            }

            Assert.That(IsIgnored(ids[0], ids[1]), Is.False);
            Assert.That(IsIgnored(ids[1], ids[2]), Is.True);
            Assert.That(IsIgnored(ids[1], ids[3]), Is.False);

            foreach (uint id in ids)
            {
                Destroyed(id);
            }

            Assert.That(IsIgnored(ids[1], ids[2]), Is.False);
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

            CommandArgs sleep = default;
            BodyKernel.Run((uint)BodyCommand.Sleep, world->bodies[1].owner, &sleep, globals);

            WorldKernel.BodyCounts counts = default;
            CommandArgs args = default;
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
            Body* first = CreateBody(Dynamic);
            Body* last = CreateBody(Dynamic);

            CommandArgs args = default;
            args.pointer = first;
            WorldKernel.Run((uint)WorldCommand.DestroyBody, world, &args, globals);

            Assert.That(world->bodyCount, Is.EqualTo(1u));
            Assert.That(last->index, Is.EqualTo(0u));
            Assert.That(world->bodies[0].owner == last, Is.True);
        }

        [Test]
        public void ShapesByLayerMaskListOnlyTheLayer()
        {
            Body* body = CreateBody(Dynamic);
            Shape* wanted = CreateShape(StoreConstants.GeometryBox, new float3(0.5f));
            Shape* other = CreateShape(StoreConstants.GeometryBox, new float3(0.5f));
            wanted->data.belongsTo = 1;
            wanted->data.userData = wanted;
            other->data.belongsTo = 2;
            Attach(body, wanted);
            Attach(body, other);

            CommandArgs args = default;
            args.value = 1;
            WorldKernel.Run((uint)WorldCommand.GetAllShapesByLayerMask, world, &args, globals);

            Assert.That(args.size, Is.EqualTo(1ul));
            Assert.That(((void**)args.pointer)[0] == wanted, Is.True);

            StoreMemory.Release(globals, args.pointer);
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

            LocalTransform pose = new LocalTransform { Rotation = quaternion.identity, Scale = 1.0f };
            PhysicsVelocity bodyVelocity = default;
            PhysicsMass mass = new PhysicsMass { Transform = new RigidTransform(quaternion.identity, centerOfMass) };
            BodyMass bodyMass = default;
            PhysicsDamping damping = default;
            BodyLimits limits = default;
            BodySleep sleep = default;
            BodyForces forces = default;
            BodyTarget target = new BodyTarget { Position = targetPosition, Rotation = targetRotation };
            PhysicsCollider collider = default;

            BodyData body = default;
            body.SetPosePointer(&pose);
            body.SetVelocityPointer(&bodyVelocity);
            body.SetMassPointer(&mass);
            body.SetBodyMassPointer(&bodyMass);
            body.SetDampingPointer(&damping);
            body.SetLimitsPointer(&limits);
            body.SetSleepPointer(&sleep);
            body.SetForcesPointer(&forces);
            body.SetTargetPointer(&target);
            body.SetColliderPointer(&collider);
            body.flags = Kinematic | BodyFlags.HasKinematicTarget;

            using NativeArray<int> dynamicIndices = new NativeArray<int>(new[] { 0 }, Allocator.TempJob);
            using NativeArray<int> staticIndices = new NativeArray<int>(0, Allocator.TempJob);
            using NativeArray<RigidBody> bodies = new NativeArray<RigidBody>(2, Allocator.TempJob);
            using NativeArray<MotionData> motions = new NativeArray<MotionData>(1, Allocator.TempJob);
            using NativeArray<MotionVelocity> velocities = new NativeArray<MotionVelocity>(1, Allocator.TempJob);

            var job = new StoreBodies.BuildBodiesJob
            {
                Store = &body,
                DynamicIndices = dynamicIndices,
                StaticIndices = staticIndices,
                Bodies = bodies,
                MotionDatas = motions,
                MotionVelocities = velocities,
                DeltaTime = deltaTime,
            };

            job.Execute(0);

            float3 reached = motions[0].WorldFromMotion.pos + velocities[0].LinearVelocity * deltaTime;
            float3 expected = targetPosition + math.rotate(targetRotation, centerOfMass);
            Assert.That(math.distance(expected, reached), Is.LessThan(1e-4f), $"reached {reached}, expected {expected}");
        }
    }
}
