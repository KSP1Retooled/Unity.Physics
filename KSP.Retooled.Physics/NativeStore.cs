using System;
using System.Runtime.InteropServices;
using KSP.Retooled.Components.Physics;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Entities;
using Unity.Mathematics;
using Unity.Physics;
using Unity.Transforms;

namespace KSP.Retooled.Physics
{
    /// <summary>
    /// The managed view of the plugin's store, field for field the "mirrored" structs of Native/RetooledPhysics/src/Store.h,
    /// which documents them. Everything is blittable so jobs can read the native rows directly. Joints are defined here
    /// alone: the plugin only keeps them on its heap, so that they outlive a domain reload.
    /// </summary>
    internal static unsafe class NativeStore
    {
        internal const string Library = "RetooledPhysics";

        // The geometry types and flags of the plugin API.
        internal const uint GeometrySphere = 1;
        internal const uint GeometryCapsule = 3;
        internal const uint GeometryBox = 4;
        internal const uint GeometryConvexMesh = 5;
        internal const uint GeometryTriangleMesh = 6;
        internal const uint GeometryHeightField = 7;
        internal const uint ShapeFlagSceneQuery = 1 << 0;
        internal const uint ShapeFlagTrigger = 1 << 1;

        internal const uint QueryUseDynamicBodies = 1 << 0;
        internal const uint QueryUseStaticBodies = 1 << 1;
        internal const uint QueryUseTriggerShapes = 1 << 2;
        internal const uint QueryAllowBackfaces = 1 << 4;
        internal const uint QueryAllowMultipleMeshHits = 1 << 5;

        // QueryWorldBuild in Store.h.
        internal const int QueryWorldFailed = 0;
        internal const int QueryWorldBuilt = 1;
        internal const int QueryWorldDeferred = 2;

        internal const uint CombineAverage = 0;
        internal const uint CombineMultiply = 1;
        internal const uint CombineMinimum = 2;
        internal const uint CombineMaximum = 3;

        [Flags]
        internal enum BodyFlags : uint
        {
            Gravity = 1 << 0,
            Kinematic = 1 << 1,
            Simulated = 1 << 2,
            Static = 1 << 8,
            Sleeping = 1 << 9,
            HasKinematicTarget = 1 << 10,
            ColliderDirty = 1 << 11,

            // Only shapes' poses changed since the collider was built, so a compound can keep its children.
            CollidersMoved = 1 << 12,
            PublicMask = Gravity | Kinematic | Simulated,
        }

        [StructLayout(LayoutKind.Sequential)]
        internal struct BodyData
        {
            public Body* owner;
            public ShapeData** shapes;
            public BodyFlags flags;
            public uint shapeCount;

            // The plugin only knows the fields above, so everything from here on is free to change without building it.
            public LocalTransform* pose;
            public PhysicsVelocity* velocity;
            public PhysicsMass* mass;
            public BodyMass* bodyMass;
            public PhysicsDamping* damping;
            public BodyLimits* limits;
            public BodySleep* sleep;
            public BodyForces* forces;
            public BodyTarget* target;

            /// <summary>The body's Unity.Physics collider. A null blob means no collider, or that a rebuild is due.</summary>
            public PhysicsCollider* collider;

            public Entity entity;
            public int orderVersion;

            /// <summary>How many joint ends point at the body, so destroying one that none do skips searching the joints.</summary>
            public int jointEnds;
        }

        /// <summary>A body's components, for as long as they aren't on an entity.</summary>
        [StructLayout(LayoutKind.Sequential)]
        internal struct BodyHome
        {
            public LocalTransform pose;
            public PhysicsVelocity velocity;
            public PhysicsMass mass;
            public BodyMass bodyMass;
            public PhysicsDamping damping;
            public BodyLimits limits;
            public BodySleep sleep;
            public BodyForces forces;
            public BodyTarget target;
            public PhysicsCollider collider;
        }

        [StructLayout(LayoutKind.Sequential)]
        internal struct Body
        {
            public WorldData* world;
            public ShapeData** shapes;
            public void* userData;
            public uint index;
            public uint shapeCount;
            public uint shapeCapacity;
            public int collisionDetectionMode;
            public uint positionIterations;
            public uint velocityIterations;

            // As in BodyData, the plugin only knows the fields above.
            public BodyHome home;
        }

        /// <summary>A shape's handle points at this, and its first field makes that a <see cref="ShapeData"/> as well.</summary>
        [StructLayout(LayoutKind.Sequential)]
        internal struct Shape
        {
            public ShapeData data;
            public Body* body;
            private fixed uint filter[4];
            public float contactOffset;
            public uint messages;
            public uint querySlot;
            private readonly uint reserved0;
            public Shape* nextOnField;
        }

        [StructLayout(LayoutKind.Sequential)]
        internal struct MaterialData
        {
            public float dynamicFriction;
            public float staticFriction;
            public float bounciness;
            public uint frictionCombine;
            public uint bounceCombine;
            public uint references;
            public void* userData;
        }

        [StructLayout(LayoutKind.Sequential)]
        internal struct ShapeData
        {
            public quaternion localRotation;
            public float3 localPosition;
            public uint geometryType;

            public float3 extents;
            public uint flags;

            public uint belongsTo;
            public uint collidesWith;
            public uint ignorePairId;
            public uint collisionDisabled;

            public void* mesh;
            public MaterialData* material;
            public void* userData;
            public Shape* owner;
        }

        [StructLayout(LayoutKind.Sequential)]
        internal struct MeshData
        {
            public uint magic;
            public uint version;
            public uint convex;
            public uint vertexCount;
            public uint triangleCount;
            public float3 lower;
            public float3 upper;
        }

        [StructLayout(LayoutKind.Sequential)]
        internal struct MeshMoments
        {
            public float volume;
            public float3 centroid;
            public float3 covarianceDiagonal;
            public float3 covarianceOffDiagonal;
        }

        /// <summary>The head of the handle a shape holds. Native keeps the reference count after it, which is never read here.</summary>
        [StructLayout(LayoutKind.Sequential)]
        internal struct Mesh
        {
            public MeshData* data;
            public MeshMoments moments;
            public uint momentsKnown;
        }

        [StructLayout(LayoutKind.Sequential)]
        internal struct HeightField
        {
            public short* heights;
            public byte* cells;
            public uint resolution;
            public uint holeCount;
            public int lowest;
            public int highest;
            public uint references;
            private readonly uint reserved0;
            public Shape* firstShape;
        }

        [StructLayout(LayoutKind.Sequential)]
        internal struct CookInput
        {
            public void* vertices;
            public void* indices;
            public uint vertexCount;
            public uint indexCount;
            public uint vertexStride;
            public uint indexStride;
            public uint convex;
            public uint flippedNormals;
        }

        [StructLayout(LayoutKind.Sequential)]
        internal struct CookOutput
        {
            public byte* bytes;
            public long size;
            public int status;
            private readonly int reserved0;
        }

        /// <summary>The cooked mesh of a shape, or null while it has none.</summary>
        internal static MeshData* MeshOf(ShapeData* shape)
        {
            return shape->mesh != null ? ((Mesh*)shape->mesh)->data : null;
        }

        internal static float3* VerticesOf(MeshData* mesh)
        {
            return (float3*)(mesh + 1);
        }

        internal static int3* TrianglesOf(MeshData* mesh)
        {
            return (int3*)(VerticesOf(mesh) + mesh->vertexCount);
        }

        // Physics::ConstraintAxis.
        internal const int AxisAngularX = 0;
        internal const int AxisAngularY = 1;
        internal const int AxisAngularZ = 2;
        internal const int AxisAngularYZ = 3;
        internal const int AxisLinearX = 4;
        internal const int AxisDistance = 7;
        internal const int AxisSlerp = 8;
        internal const int JointAxisCount = 9;

        // Physics::JointDofLock.
        internal const uint DofLocked = 0;
        internal const uint DofLimited = 1;
        internal const uint DofFree = 2;

        // Physics::JointType.
        internal const uint JointTypeFixed = 0;
        internal const uint JointTypePrismatic = 1;
        internal const uint JointTypeHinge = 2;
        internal const uint JointTypeBallAndSocket = 3;
        internal const uint JointTypeDistance = 5;
        internal const uint JointTypeDof6 = 6;

        // Physics::JointFlags.
        internal const uint JointFlagCollisionEnabled = 1 << 2;

        [Flags]
        internal enum JointState : uint
        {
            Broken = 1 << 0,
            HingeLimitEnabled = 1 << 1,
            HingeMotorFreeSpin = 1 << 2,
            BodyDestroyed = 1 << 3,
        }

        /// <summary>
        /// One axis of a joint, at the index Physics::ConstraintAxis gives it. min and max are radians on an angular axis and
        /// metres on a linear one, in the units Unity sends them.
        /// </summary>
        [StructLayout(LayoutKind.Sequential)]
        internal struct JointAxis
        {
            public float min;
            public float max;
            public float bounciness;
            public float bounceMinVelocity;
            public float contactDistance;
            public float limitSpring;
            public float limitDamping;
            public float driveSpring;
            public float driveDamper;
            public float driveMaxForce;
            public uint dofLock; // Physics::JointDofLock
            public uint driveUsesAcceleration;
        }

        /// <summary>
        /// A joint's settings as Unity sent them and what the last step measured. A joint's handle points here, and the world's
        /// list points back. A null body is the world, which the joint's pose is then relative to.
        /// </summary>
        [StructLayout(LayoutKind.Sequential)]
        internal struct JointData
        {
            public quaternion rotation0;
            public float3 position0;
            public uint type; // Physics::JointType

            public quaternion rotation1;
            public float3 position1;
            public uint flags; // Physics::JointFlags

            public quaternion targetRotation;
            public float3 targetPosition;
            public JointState state;

            public float3 targetLinearVelocity;
            public float breakForce;
            public float3 targetAngularVelocity;
            public float breakTorque;

            // What the joint's constraints applied in the last step, as a force in newtons and a torque in newton metres, in
            // world space.
            public float3 force;
            public float inverseMassScale0;
            public float3 torque;
            public float inverseMassScale1;

            public float projectionLinearTolerance;
            public float projectionAngularTolerance;
            public float distanceErrorTolerance;
            public float hingeMotorMaxForce;

            public float hingeMotorTargetVelocity;
            public uint index; // In the world's list.

            // The force and torque magnitudes that breaking compares with, low-passed over a few steps.
            public float breakingForce;
            public float breakingTorque;

            public Body* body0;
            public Body* body1;
            public void* userData;

            // JointAxisCount of them, which AxisOf reaches from the first.
            public JointAxis axes0;
            private readonly JointAxis axes1;
            private readonly JointAxis axes2;
            private readonly JointAxis axes3;
            private readonly JointAxis axes4;
            private readonly JointAxis axes5;
            private readonly JointAxis axes6;
            private readonly JointAxis axes7;
            private readonly JointAxis axes8;
        }

        /// <summary>The axis of a joint at a Physics::ConstraintAxis index. The record is on the store's heap, so it doesn't move.</summary>
        internal static JointAxis* AxisOf(JointData* joint, int axis)
        {
            return &joint->axes0 + axis;
        }

        [StructLayout(LayoutKind.Sequential)]
        internal struct JointBreak
        {
            public ulong engineObjectId;
            public float linearMagnitude;
            public float angularMagnitude;
        }

        /// <summary>The blittable head of Physics::ActorDescriptor, whose layout Joint.cpp checks.</summary>
        [StructLayout(LayoutKind.Sequential)]
        internal struct ActorDescriptor
        {
            public Body* actor;
            public void* userData;
            public ushort type;
            private readonly ushort reserved0;
            private readonly uint reserved1;
        }

        [StructLayout(LayoutKind.Sequential)]
        internal struct ContactPoint
        {
            public float3 position;
            public float separation;
            public float3 normal;
            public uint faceIndex0;
            public float3 impulse;
            public uint faceIndex1;
        }

        [StructLayout(LayoutKind.Sequential)]
        internal struct ContactShapePair
        {
            public ulong thisCollider;
            public ulong otherCollider;
            public ContactPoint* points;
            public uint pointCount;
            public ushort flags;
            public ushort events;
            public float3 impulseSum;
            private readonly uint reserved0;
        }

        [StructLayout(LayoutKind.Sequential)]
        internal struct ContactBodyPair
        {
            public ulong thisBody;
            public ulong otherBody;
            public ContactShapePair* shapes;
            public uint shapeCount;
            public ushort flags;
            private readonly ushort reserved0;
            public float3 thisLinearVelocity;
            public float3 thisAngularVelocity;
            public float3 otherLinearVelocity;
            public float3 otherAngularVelocity;
        }

        internal const uint TriggerEventEnter = 1;
        internal const uint TriggerEventExit = 2;

        [StructLayout(LayoutKind.Sequential)]
        internal struct TriggerEvent
        {
            public Shape* triggerShape;
            public Shape* colliderShape;
            public ulong trigger;
            public ulong collider;
            public uint type;
            private readonly uint reserved0;
        }

        [StructLayout(LayoutKind.Sequential)]
        internal struct WorldData
        {
            public BodyData* bodies;
            public uint bodyCount;
            public uint bodyCapacity;

            public JointData** joints;
            public uint jointCount;
            public uint jointCapacity;

            public JointBreak* jointBreaks;
            public uint jointBreakCount;
            public uint jointBreakCapacity;

            public ContactBodyPair* contactBodies;
            public uint contactBodyCount;
            public uint contactBodyCapacity;

            public ContactShapePair* contactShapes;
            public uint contactShapeCount;
            public uint contactShapeCapacity;

            public ContactPoint* contactPoints;
            public uint contactPointCount;
            public uint contactPointCapacity;

            public TriggerEvent* triggers;
            public uint triggerCount;
            public uint triggerCapacity;

            public void** removedShapes;
            public uint removedShapeCount;
            public uint removedShapeCapacity;

            public void** deadColliders;
            public uint deadColliderCount;
            public uint deadColliderCapacity;

            public float3 gravity;
            public float bounceThreshold;

            public uint staticsVersion;
            public uint queryDirty;

            public IntPtr owner;
            public QueryWorld* queryWorld;
        }

        [StructLayout(LayoutKind.Sequential)]
        internal struct StoreGlobals
        {
            public uint materialVersion;
            public uint engineObjectIdOffset;

            public uint jointDataSize;
            public uint bodyDataSize;
            public uint bodySize;

            /// <summary>Bumped whenever the ignored pairs change, so that what was worked out from them can be kept.</summary>
            public uint ignoreVersion;

            public IgnoreTable* ignoreTable;

            public MaterialData* defaultMaterial;

            public IntPtr reallocate;
            public IntPtr release;
            public IntPtr retainMesh;
            public IntPtr releaseMesh;
            public IntPtr retainMaterial;
            public IntPtr releaseMaterial;
        }

        [StructLayout(LayoutKind.Sequential)]
        internal struct CommandArgs
        {
            public quaternion rotation;
            public float3 position;
            public uint flags;
            public float3 vector;
            public float scalar;
            public float3 vector2;
            public float scalar2;
            public void* pointer;
            public void* pointer2;
            public void* pointer3;
            public ulong size;
            public float scalar3;
            public uint mode;
            public uint enabled;
            public uint value;
            public uint value2;
            private readonly uint reserved0;
        }

        /// <summary>QueryKind in Store.h.</summary>
        internal enum QueryKind : uint
        {
            Ray = 0,
            RayAll = 1,
            RayAgainstShape = 2,
            OverlapAny = 3,
            OverlapAll = 4,
            OverlapBroadphase = 5,
            Cast = 6,
            CastAll = 7,
            BodyCast = 8,
            BodyCastAll = 9,
            ClosestPoint = 10,
            Penetration = 11,
            CastAgainstShape = 12,
        }

        [StructLayout(LayoutKind.Sequential)]
        internal struct QueryInput
        {
            public quaternion rotation;
            public float3 position;
            public uint geometryType;

            public float3 extents;
            public int layerMask;

            public float3 direction;
            public float maxDistance;

            public uint filterOptions;
            public uint keepInitialOverlaps;

            public quaternion rotation2;
            public float3 position2;
            private readonly uint reserved0;

            public Shape* shape;
            public Shape* shape2;
            public Body* body;
            public void* collector;
            public void* collectorContext;
        }

        [StructLayout(LayoutKind.Sequential)]
        internal struct RayCastHit
        {
            public float3 point;
            public float3 normal;
            public uint faceId;
            public float distance;
            public float2 uv;
            public ulong engineObjectId;
        }

        [StructLayout(LayoutKind.Sequential)]
        internal struct QueryOutput
        {
            public RayCastHit hit;
            public void* shapeUserData;
        }

        /// <summary>Physics::QueryHitResultCollectorCallback. The answer is a C++ bool, so it is one byte.</summary>
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        internal delegate byte HitCollectorFunction(void* context, RayCastHit* hits, ulong count);

        /// <summary>Physics::QueryShapesResultCollectorCallback.</summary>
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        internal delegate byte ShapeCollectorFunction(void* context, void** shapeUserDatas, ulong count);

        /// <summary>Physics::BroadPhaseQueryShapeCollectorCallback.</summary>
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        internal delegate byte BroadphaseCollectorFunction(void* context, void* shape, void* body, void* shapeUserData, ushort geometryType);

        internal delegate void SimulateCallback(WorldData* world, float deltaTime);

        internal delegate int BuildQueryWorldCallback(WorldData* world);

        internal delegate void DestroyWorldCallback(WorldData* world);

        /// <summary>A category's commands, in a Burst function. The command is the plugin API's id and the handle what it addresses.</summary>
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        internal delegate void CommandFunction(uint command, void* handle, CommandArgs* args, StoreGlobals* globals);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        internal delegate void* ReallocateFunction(void* memory, ulong size);

        /// <summary>The plugin's functions that take one pointer: freeing memory, and retaining or releasing a record.</summary>
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        internal delegate void PointerFunction(void* pointer);

        [StructLayout(LayoutKind.Sequential)]
        internal struct ManagedCallbacks
        {
            public IntPtr simulate;
            public IntPtr buildQueryWorld;
            public IntPtr query;
            public IntPtr destroyWorld;
            public IntPtr cookMesh;
            public IntPtr validateMesh;
            public IntPtr bodyCommand;
            public IntPtr shapeCommand;
            public IntPtr worldCommand;
            public IntPtr jointCommand;
        }

        [DllImport(Library)]
        private static extern int RetooledPhysics_StructSize(int id);

        [DllImport(Library)]
        internal static extern void RetooledPhysics_Attach(ManagedCallbacks* callbacks);

        [DllImport(Library)]
        internal static extern void RetooledPhysics_Detach();

        [DllImport(Library)]
        internal static extern StoreGlobals* RetooledPhysics_Globals();

        [DllImport(Library)]
        internal static extern int RetooledPhysics_WorldCount();

        [DllImport(Library)]
        internal static extern WorldData* RetooledPhysics_GetWorld(int index);

        [DllImport(Library)]
        internal static extern void RetooledPhysics_SyncQueryWorlds();

        /// <summary>Non-zero on the thread that attached, which is the main thread.</summary>
        [DllImport(Library)]
        internal static extern int RetooledPhysics_OnMainThread();

        /// <summary>
        /// What stops this domain attaching, or null when nothing does: the first struct or command enum whose size differs
        /// from the plugin's, or a joint layout other than the one the store's records were written with.
        /// </summary>
        internal static string FirstSizeMismatch()
        {
            int[] sizes =
            {
                UnsafeUtility.GetFieldOffset(typeof(BodyData).GetField(nameof(BodyData.pose))),
                sizeof(ShapeData),
                sizeof(MaterialData),
                sizeof(WorldData),
                sizeof(StoreGlobals),
                sizeof(QueryInput),
                sizeof(RayCastHit),
                sizeof(ManagedCallbacks),
                sizeof(MeshData),
                sizeof(CookInput),
                sizeof(CookOutput),
                UnsafeUtility.GetFieldOffset(typeof(Body).GetField(nameof(Body.home))),
                sizeof(Shape),
                sizeof(CommandArgs),
                (int)BodyCommand.Count,
                (int)ShapeCommand.Count,
                (int)WorldCommand.Count,
                sizeof(JointBreak),
                (int)JointCommand.Count,
                sizeof(HeightField),
            };

            string[] names =
            {
                nameof(BodyData),
                nameof(ShapeData),
                nameof(MaterialData),
                nameof(WorldData),
                nameof(StoreGlobals),
                nameof(QueryInput),
                nameof(RayCastHit),
                nameof(ManagedCallbacks),
                nameof(MeshData),
                nameof(CookInput),
                nameof(CookOutput),
                nameof(Body),
                nameof(Shape),
                nameof(CommandArgs),
                nameof(BodyCommand),
                nameof(ShapeCommand),
                nameof(WorldCommand),
                nameof(JointBreak),
                nameof(JointCommand),
                nameof(HeightField),
            };

            for (int i = 0; i < sizes.Length; i++)
            {
                int native = RetooledPhysics_StructSize(i);
                if (native != sizes[i])
                {
                    return $"{names[i]} is {sizes[i]} here and {native} in the plugin. Rebuild the plugin with `RetooledCli native build`";
                }
            }

            StoreGlobals* globals = RetooledPhysics_Globals();
            if (globals->jointDataSize != 0 && globals->jointDataSize != sizeof(JointData))
            {
                return $"{nameof(JointData)} is {sizeof(JointData)} here and {globals->jointDataSize} in the store's joints. Restart Unity";
            }

            if (globals->bodyDataSize != 0 && (globals->bodyDataSize != sizeof(BodyData) || globals->bodySize != sizeof(Body)))
            {
                return $"Body records are {sizeof(BodyData)} and {sizeof(Body)} bytes here and {globals->bodyDataSize} and {globals->bodySize} in the store. Restart Unity";
            }

            globals->jointDataSize = (uint)sizeof(JointData);
            globals->bodyDataSize = (uint)sizeof(BodyData);
            globals->bodySize = (uint)sizeof(Body);
            return null;
        }
    }
}
