using Unity.Physics.Configuration;
using System;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Physics.Store.World;
using Unity.Physics.Store.Bodies;
using Unity.Mathematics;

namespace Unity.Physics.Stepping.State
{
    /// <summary>
    /// Reuses static body rows when the static version, row order, and body storage still match.
    /// </summary>
    internal unsafe struct StaticRows
    {
        internal bool staticsChanged;
        // Store rows used for the retained static bodies.
        internal NativeList<int> staticRows;
        internal long keptStaticRowsCount;
        private uint builtStaticsVersion;
        private UnsafeList<RetiredBodies> retiredBodies;
        private ulong buildStep;

        private struct RetiredBodies
        {
            internal NativeArray<RigidBody> Bodies;
            internal ulong ReleaseStep;
        }

        /// <summary>The builds that kept the static bodies' rows of the last build. Used by tests.</summary>
        internal readonly long KeptStaticRowsCount => keptStaticRowsCount;

        internal void NoteBuild(WorldData* store, ref bool builtWhole)
        {
            staticsChanged = !builtWhole || store->staticsVersion != builtStaticsVersion;
            builtStaticsVersion = store->staticsVersion;
        }

        internal void NotePoseUpdate(WorldData* store)
        {
            builtStaticsVersion = store->staticsVersion;
            staticsChanged = true;
        }

        internal void Initialize()
        {
            staticRows = new NativeList<int>(Allocator.Persistent);
        }

        internal void Dispose()
        {
            staticRows.Dispose();
            if (retiredBodies.IsCreated)
            {
                for (int i = 0; i < retiredBodies.Length; i++)
                {
                    retiredBodies[i].Bodies.Dispose();
                }
                retiredBodies.Dispose();
            }
        }

        internal void QuarantineReplacedBodies(NativeArray<RigidBody> replaced, bool step)
        {
            const ulong lifetime = 4;
            if (step)
            {
                buildStep++;
            }

            for (int i = retiredBodies.Length - 1; i >= 0; i--)
            {
                if (retiredBodies[i].ReleaseStep <= buildStep)
                {
                    retiredBodies[i].Bodies.Dispose();
                    retiredBodies.RemoveAtSwapBack(i);
                }
            }

            if (!replaced.IsCreated)
            {
                return;
            }

            // All bits set is a quiet NaN for both single and double precision.
            UnsafeUtility.MemSet(replaced.GetUnsafePtr(), 0xff, (long)replaced.Length * sizeof(RigidBody));
            if (!retiredBodies.IsCreated)
            {
                retiredBodies = new UnsafeList<RetiredBodies>(4, Allocator.Persistent);
            }
            retiredBodies.Add(new RetiredBodies { Bodies = replaced, ReleaseStep = buildStep + lifetime });
        }

        /// <summary>
        /// Reuse static body rows when the previous build completed, staticsVersion is unchanged,
        /// and static row order matches. The caller must also verify that body storage was retained.
        /// </summary>
        internal readonly bool KeepsStaticRows(ref PhysicsWorld world, ref BodyPartition partition)
        {
            return StepCacheSwitches.StaticReuseEnabled && !staticsChanged && world.NumStaticBodies == partition.staticIndices.Length + 1
                && BodyPartition.SameRows(staticRows, partition.staticIndices);
        }

        /// <summary>
        /// Move retained static rows after the new dynamic body range when its size changes.
        /// Call only while the built world is inaccessible to queries and jobs.
        /// </summary>
        internal void MoveStaticRows(ref PhysicsWorld world, ref BodyPartition partition, int dynamicBefore)
        {
            if (dynamicBefore == partition.dynamicIndices.Length)
            {
                return;
            }

            RigidBody* bodies = (RigidBody*)NativeArrayUnsafeUtility.GetUnsafeBufferPointerWithoutChecks(world.Bodies);
            UnsafeUtility.MemMove(bodies + partition.dynamicIndices.Length, bodies + dynamicBefore, (long)partition.staticIndices.Length * sizeof(RigidBody));
        }

        internal bool RetainOrRemember(ref PhysicsWorld world, ref BodyPartition partition,
            int dynamicBefore, bool keepStaticRows, bool keptBodies)
        {
            keepStaticRows &= keptBodies;
            if (keepStaticRows)
            {
                keptStaticRowsCount++;
                MoveStaticRows(ref world, ref partition, dynamicBefore);
                NativeArray<RigidBody> bodies = world.Bodies;
                bodies[bodies.Length - 1] = RigidBody.Zero;
            }
            else if (StepCacheSwitches.StaticReuseEnabled)
            {
                // With the reuse off, every build writes the rows, so the rows that they come from don't matter.
                staticRows.CopyFrom(partition.staticIndices.AsArray());
            }
            return keepStaticRows;
        }

        internal static void Validate(WorldData* store, ref PhysicsWorld world, ref BodyPartition partition)
        {
            var bodies = world.StaticBodies;
            RigidBody anchor = bodies[bodies.Length - 1];
            if (anchor.Collider.IsCreated || anchor.Entity != RigidBody.Zero.Entity
                || !anchor.WorldFromBody.Equals(RigidTransform.identity) || anchor.Scale != RigidBody.Zero.Scale)
            {
                throw new InvalidOperationException("The synthetic world anchor differs from the identity body.");
            }

            int index = 0;
            for (int row = 0; row < store->bodyCount; row++)
            {
                ref BodyData source = ref store->bodies[row];
                const BodyFlags staticAndSimulated = BodyFlags.Static | BodyFlags.Simulated;
                if ((source.flags & staticAndSimulated) != staticAndSimulated)
                {
                    continue;
                }

                if (index >= partition.staticIndices.Length || partition.staticIndices[index] != row)
                {
                    throw new InvalidOperationException($"Static body {index} does not map to store row {row}.");
                }

                RigidBody body = bodies[index++];
                RigidTransform pose = StoreBodies.BuildBodiesJob.WorldFromBody(ref source);
                int serial = source.serial != 0 ? (int)source.serial : -(row + 1);
                if (body.Entity.Index != row || body.Entity.Version != serial
                    || body.Collider.GetUnsafePtr() != source.collider->Value.GetUnsafePtr()
                    || !math.all(math.isfinite(body.WorldFromBody.pos))
                    || !math.all(math.isfinite(body.WorldFromBody.rot.value))
                    || !math.all(math.abs(pose.pos - body.WorldFromBody.pos) <= 0.00001f + 0.000001f * math.abs(pose.pos))
                    || !math.all(math.abs(pose.rot.value - body.WorldFromBody.rot.value) <= 0.00001f))
                {
                    throw new InvalidOperationException($"Built static body at store row {row} differs from the store.");
                }
            }

            if (index + 1 != bodies.Length)
            {
                throw new InvalidOperationException("The number of built statics and the anchor differs from the store.");
            }
        }
    }
}
