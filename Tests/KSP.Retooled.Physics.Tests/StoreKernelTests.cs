using KSP.Retooled.Components.Physics;
using NUnit.Framework;
using Unity.Collections;
using Unity.Mathematics;
using Unity.Physics;
using Unity.Transforms;

namespace KSP.Retooled.Physics.Tests
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

            float3 inertia = StoreOps.Hot(body)->bodyMass->Inertia;
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

        // The ignore table is the store's, shared with every live collider, so ids have to come from it and go back to it.
        private uint[] NewIds(int count)
        {
            uint[] ids = new uint[count];
            for (int i = 0; i < count; i += 2)
            {
                NativeStore.CommandArgs args = default;
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
            NativeStore.CommandArgs cleanup = default;
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
            NativeStore.CommandArgs args = default;
            args.enabled = 1;
            ShapeKernel.Run((uint)ShapeCommand.IgnoreCollision, null, &args, globals);
            uint first = args.value;
            uint second = args.value2;

            Assert.That(first, Is.Not.Zero);
            Assert.That(second, Is.Not.Zero.And.Not.EqualTo(first));
            Assert.That(IsIgnored(first, second), Is.True);

            NativeStore.CommandArgs cleanup = default;
            cleanup.value = second;
            ShapeKernel.Run((uint)ShapeCommand.CleanupIgnoredColliders, null, &cleanup, globals);
            Assert.That(IsIgnored(first, second), Is.False);

            NativeStore.CommandArgs another = default;
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

            NativeStore.BodyData body = default;
            body.pose = &pose;
            body.velocity = &bodyVelocity;
            body.mass = &mass;
            body.bodyMass = &bodyMass;
            body.damping = &damping;
            body.limits = &limits;
            body.sleep = &sleep;
            body.forces = &forces;
            body.target = &target;
            body.collider = &collider;
            body.flags = Kinematic | NativeStore.BodyFlags.HasKinematicTarget;

            using NativeArray<int> dynamicIndices = new NativeArray<int>(new[] { 0 }, Allocator.TempJob);
            using NativeArray<int> staticIndices = new NativeArray<int>(0, Allocator.TempJob);
            using NativeArray<RigidBody> bodies = new NativeArray<RigidBody>(2, Allocator.TempJob);
            using NativeArray<MotionData> motions = new NativeArray<MotionData>(1, Allocator.TempJob);
            using NativeArray<MotionVelocity> velocities = new NativeArray<MotionVelocity>(1, Allocator.TempJob);

            var job = new BuildBodiesJob
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
