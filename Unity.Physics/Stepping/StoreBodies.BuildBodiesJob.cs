using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Entities;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Physics.Dynamics.Bodies;
using Unity.Physics.Store.Bodies;
using Unity.Physics.Store.Shapes;
using Unity.Physics.Store.World;
using Unity.Physics.Stepping;
using Unity.Physics.Collision.Colliders.Building;
using Unity.Physics.Dynamics.Joint.Translation;

namespace Unity.Physics.Stepping
{
    internal static partial class StoreBodies
    {
        /// <summary>
        /// Copy store bodies and motion data into the built world, with dynamic bodies first and the world anchor last.
        /// Treat sleeping and kinematic bodies as infinite mass. Use zero kinematic velocities when DeltaTime is zero.
        /// </summary>
        [BurstCompile]
        internal unsafe struct BuildBodiesJob : IJobParallelFor
        {
            [NativeDisableUnsafePtrRestriction]
            public BodyData* Store;

            [ReadOnly]
            public NativeArray<int> DynamicIndices;

            [ReadOnly]
            public NativeArray<int> StaticIndices;

            public NativeArray<RigidBody> Bodies;
            public NativeArray<MotionData> MotionDatas;
            public NativeArray<MotionVelocity> MotionVelocities;

            public float DeltaTime;

            /// <summary>
            /// Optional per-body bounds cache flags. Set when CompareBodies is true and the overwritten body's pose and
            /// collider match exactly.
            /// </summary>
            [NativeDisableUnsafePtrRestriction]
            public byte* Same;

            /// <summary>Whether the bodies that this job overwrites are the bodies that the bounds cache holds.</summary>
            public bool CompareBodies;

            /// <summary>
            /// Read serials and expansion factors from their original sources instead of cached row fields.
            /// Used by tests to verify identical built worlds.
            /// </summary>
            public bool ReadSources;

            /// <summary>
            /// Reuse retained static rows and the anchor, writing only dynamic bodies when true.
            /// </summary>
            public bool KeepStatics;

            /// <summary>
            /// Number of bodies to write. Include statics and the anchor unless KeepStatics is true.
            /// </summary>
            public int Count => KeepStatics ? DynamicIndices.Length : DynamicIndices.Length + StaticIndices.Length + 1;

            public void Execute(int builtIndex)
            {
                if (builtIndex < DynamicIndices.Length)
                {
                    int index = DynamicIndices[builtIndex];
                    ref BodyData body = ref Store[index];
                    RigidBody built = RigidBodyOf(ref body, index, ReadSources);
                    if (Same != null)
                    {
                        // This job writes the body's memory anyway, so reading it first costs little.
                        Same[builtIndex] = CompareBodies && DynamicBoundsCache.View.Matches(Bodies[builtIndex], built) ? (byte)1 : (byte)0;
                    }

                    Bodies[builtIndex] = built;
                    WriteMotion(builtIndex, ref body);
                }
                else if (builtIndex < DynamicIndices.Length + StaticIndices.Length)
                {
                    int index = StaticIndices[builtIndex - DynamicIndices.Length];
                    Bodies[builtIndex] = RigidBodyOf(ref Store[index], index, ReadSources);
                }
                else
                {
                    Bodies[builtIndex] = RigidBody.Zero;
                }
            }

            /// <summary>Builds the Unity Physics body for a store row. See <see cref="EntityVersion"/> for its entity.</summary>
            private static RigidBody RigidBodyOf(ref BodyData body, int index, bool readSources)
            {
                return new RigidBody
                {
                    Collider = body.collider->Value,
                    WorldFromBody = WorldFromBody(ref body),
                    Entity = new Entity { Index = index, Version = EntityVersion(ref body, index, readSources) },
                    Scale = 1.0f,
                };
            }

            /// <summary>
            /// Return a stable, nonzero body identity from its serial number, independent of its store row.
            /// Ownerless test rows use negative identities derived from their indices. Read the cached serial unless
            /// readSources is true.
            /// </summary>
            private static int EntityVersion(ref BodyData body, int index, bool readSources)
            {
                uint serial = body.serial;
                if (readSources)
                {
                    serial = body.owner != null ? body.owner->serial : 0u;
                }

                return serial != 0 ? (int)serial : -(index + 1);
            }

            internal static RigidTransform WorldFromBody(ref BodyData body)
            {
                RigidTransform worldFromBody = new RigidTransform(body.pose->Rotation, body.pose->Position);
                ShapeData* posed = ColliderBuilder.PosedShape((BodyData*)UnsafeUtility.AddressOf(ref body));
                if (posed != null)
                {
                    worldFromBody = math.mul(worldFromBody, new RigidTransform(posed->localRotation, posed->localPosition));
                }

                return worldFromBody;
            }

            private void WriteMotion(int motionIndex, ref BodyData body)
            {
                RigidTransform bodyFromMotion = body.mass->Transform;
                RigidTransform worldFromMotion = math.mul(new RigidTransform(body.pose->Rotation, body.pose->Position), bodyFromMotion);

                bool kinematic = (body.flags & BodyFlags.Kinematic) != 0;
                bool sleeping = (body.flags & BodyFlags.Sleeping) != 0;
                bool heldInPlace = kinematic || sleeping;

                float3 linear = body.velocity->Linear;
                float3 angular = body.velocity->Angular;
                if (kinematic)
                {
                    linear = float3.zero;
                    angular = float3.zero;
                    if ((body.flags & BodyFlags.HasKinematicTarget) != 0 && DeltaTime > 0.0f)
                    {
                        // Motion velocities describe movement of the centre of mass. Rotating around the body origin also
                        // moves the centre of mass.
                        float3 targetCenter = body.target->Position + math.rotate(body.target->Rotation, body.mass->Transform.pos);
                        linear = (targetCenter - worldFromMotion.pos) / DeltaTime;
                        quaternion turn = math.normalize(math.mul(body.target->Rotation, math.conjugate(body.pose->Rotation)));
                        angular = math.rotate(math.conjugate(worldFromMotion.rot), JointConstraints.AngleAxis(turn) / DeltaTime);
                    }
                }

                // The row holds a copy of the collider's expansion factor, which is 0 when the body has no collider. Adding
                // 0 to the length gives the same bits as not adding it, because the length is never -0.
                float colliderExpansion = body.colliderExpansion;
                if (ReadSources)
                {
                    colliderExpansion = body.collider->Value.IsCreated ? body.collider->Value.Value.MassProperties.AngularExpansionFactor : 0.0f;
                }

                // The build doesn't write the result back into the body's PhysicsMass, because no code in this package reads it
                // there.
                float expansion = math.length(body.mass->Transform.pos) + colliderExpansion;

                MotionDatas[motionIndex] = new MotionData
                {
                    WorldFromMotion = worldFromMotion,
                    BodyFromMotion = bodyFromMotion,
                    LinearDamping = 0.0f,
                    AngularDamping = 0.0f,
                };

                MotionVelocities[motionIndex] = new MotionVelocity
                {
                    LinearVelocity = linear,
                    AngularVelocity = angular,
                    InverseInertia = heldInPlace ? float3.zero : body.mass->InverseInertia,
                    InverseMass = heldInPlace ? 0.0f : body.mass->InverseMass,
                    AngularExpansionFactor = expansion,
                    GravityFactor = 0.0f,
                };
            }
        }
    }
}
