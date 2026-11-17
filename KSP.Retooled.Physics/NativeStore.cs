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
    /// Managed view of the native store. Blittable structures match Native~/src/Store.h field for field.
    /// Joints are defined here and allocated on the native heap to survive domain reloads.
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

            // Only the shapes' poses have changed since the collider was built, so the children of a compound collider
            // can be reused.
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

            /// <summary>
            /// Pointer to the current body pose. The plugin reads it directly only where Body.cpp permits.
            /// </summary>
            public LocalTransform* pose;

            // The plugin uses only the fields above. The fields below can change without rebuilding the plugin.
            public PhysicsVelocity* velocity;
            public PhysicsMass* mass;
            public BodyMass* bodyMass;
            public PhysicsDamping* damping;
            public BodyLimits* limits;
            public BodySleep* sleep;
            public BodyForces* forces;
            public BodyTarget* target;

            /// <summary>
            /// Stores the body's Unity.Physics collider. A null blob means that the body has no collider or that its
            /// collider needs to be rebuilt.
            /// </summary>
            public PhysicsCollider* collider;

            public Entity entity;
            public int orderVersion;

            /// <summary>
            /// Counts the joint ends that point to this body. When the count is zero, the body can be destroyed without
            /// searching the joints.
            /// </summary>
            public int jointEnds;

            /// <summary>
            /// Cached owner serial, or zero for an unowned row. Copied with the row when it moves.
            /// </summary>
            public uint serial;

            /// <summary>
            /// Cached collider AngularExpansionFactor, or zero without a collider.
            /// Updated together with the collider by ColliderBuilder.SetCollider.
            /// </summary>
            public float colliderExpansion;
        }

        /// <summary>Stores a body's components while they are not on an entity.</summary>
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

            // A number that no other body received. The store reuses the memory of a destroyed body for the next body
            // that is made, so the address alone doesn't tell two bodies apart.
            public uint serial;
        }

        /// <summary>
        /// Shape handle storage beginning with ShapeData, allowing either structure to use the same pointer.
        /// </summary>
        [StructLayout(LayoutKind.Sequential)]
        internal struct Shape
        {
            public ShapeData data;
            public Body* body;
            private fixed uint filter[4];
            public float contactOffset;
            public uint messages;
            public uint querySlot;

            /// <summary>
            /// The index of the shape's group record in <see cref="GroupTable"/> plus one, or zero if the shape has no
            /// group. The plugin never reads this field.
            /// </summary>
            public uint groupRecord;
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

        /// <summary>
        /// Mirrored prefix of a mesh handle. The native reference count follows these fields.
        /// </summary>
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

        /// <summary>Returns the cooked mesh of a shape, or null if the shape has no cooked mesh.</summary>
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
        /// Describes one axis of a joint at the index defined by Physics::ConstraintAxis. The min and max values are in
        /// radians for an angular axis and metres for a linear axis. These are the units that Unity supplies.
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
        /// Joint settings and last-step measurements. Handles and the world list point to this record.
        /// A null body represents the world, with that end's pose in world space.
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
            public float inverseMassScale0;

            public float3 targetLinearVelocity;
            public float inverseMassScale1;
            public float3 targetAngularVelocity;
            public uint index; // In the world's list.

            // Keep per-step joint load fields together to limit cache lines read by SettleAwake.
            public JointState state;
            public float breakForce;
            public float breakTorque;

            // The magnitudes of force and torque used to decide whether the joint breaks. A low-pass filter smooths
            // them over a few steps.
            public float breakingForce;

            // The force and torque applied by the joint's constraints during the last step. Both are in world space,
            // with force in newtons and torque in newton metres.
            public float3 force;
            public float breakingTorque;
            public float3 torque;

            public float hingeMotorTargetVelocity;
            public float projectionLinearTolerance;
            public float projectionAngularTolerance;
            public float distanceErrorTolerance;
            public float hingeMotorMaxForce;

            public Body* body0;
            public Body* body1;
            public void* userData;

            // JointSettings.Version when this joint last changed. Vessel fill jobs compare it with their cached
            // version.
            public uint settingsStamp;

            // A number that no other joint received. The store reuses the memory of a destroyed joint for the next
            // joint that is made, so the address alone doesn't tell two joints apart.
            public uint serial;

            // There are JointAxisCount consecutive axes. AxisOf accesses them by adding an offset to the address of the
            // first axis.
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

        /// <summary>
        /// Returns the joint axis at a Physics::ConstraintAxis index. The record is allocated on the store's heap, so
        /// its address does not change.
        /// </summary>
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

        /// <summary>
        /// Defines the blittable beginning of Physics::ActorDescriptor. Joint.cpp checks this layout.
        /// </summary>
        [StructLayout(LayoutKind.Sequential)]
        internal struct ActorDescriptor
        {
            public Body* actor;
            public void* userData;
            public ushort type;
            private readonly ushort reserved0;
            private readonly uint reserved1;
        }

        /// <summary>
        /// Vehicle state followed by wheel slots in the same native allocation.
        /// Survives domain reloads and is freed with the world.
        /// </summary>
        [StructLayout(LayoutKind.Sequential)]
        internal struct VehicleData
        {
            public Body* body;
            public uint wheelCount;
            public uint implicitSprungMasses;
            public uint expansionLimit;
            public float substepSpeed;
            public uint substepsBelow;
            public uint substepsAbove;

            // The body's velocity in world space before the forces for this step are applied. This velocity is used to
            // calculate suspension damping.
            public float3 startLinear;
            public float3 startAngular;
        }

        internal static WheelSlot* WheelsOf(VehicleData* vehicle)
        {
            return (WheelSlot*)(vehicle + 1);
        }

        /// <summary>
        /// Stores the settings that Unity supplied for one wheel and the results of the query beneath that wheel during
        /// the last step.
        /// </summary>
        [StructLayout(LayoutKind.Sequential)]
        internal struct WheelSlot
        {
            public void* userData;

            // The shape recorded by the last hit. When the shape is destroyed, its recorded hit can be cleared too.
            public Shape* hitShape;
            public WheelQueryResult hit;
            public WheelSuspension suspension;
            public WheelBody wheel;

            // The wheel's centre at rest and the point where its force is applied. Both are offsets from the body's
            // centre of mass, expressed along the body's axes.
            public float3 centreOffset;
            public float3 forceOffset;

            // The plugin API's FilterData, stored as four words. The upper half of the last word contains the ignore
            // id.
            public uint collidesWith;
            public uint belongsTo;
            public uint filterFlags;
            public uint filterIds;

            public float rotationSpeed;
            public float motorTorque;
            public float brakeTorque;
            public float steerAngle;
            public float rotationAngle;

            // The height of the wheel above its rest position. It ranges from minus the maximum droop to the maximum
            // compression.
            public float jounce;
            public uint enabled;
            public uint grounded;
        }

        /// <summary>Matches Physics::PhysXExt::WheelQueryResult. Vehicle.cpp checks this layout.</summary>
        [StructLayout(LayoutKind.Sequential)]
        internal struct WheelQueryResult
        {
            public float3 point;
            public float3 normal;
            public float3 forwardDir;
            public float3 sidewaysDir;
            public float force;
            public float forwardSlip;
            public float sidewaysSlip;
            public float suspensionCompression;
            public float steerAngle;
            private readonly uint reserved0;
            public void* userData;
        }

        /// <summary>Matches Physics::PhysXExt::WheelSuspensionData.</summary>
        [StructLayout(LayoutKind.Sequential)]
        internal struct WheelSuspension
        {
            public float springStrength;
            public float springDamperRate;
            public float maxCompression;
            public float maxDroop;
            public float sprungMass;
        }

        /// <summary>Matches Physics::PhysXExt::WheelData.</summary>
        [StructLayout(LayoutKind.Sequential)]
        internal struct WheelBody
        {
            public float radius;
            public float mass;
            public float momentOfInertia;
            public float dampingRate;
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

        /// <summary>
        /// Trigger removal event type. Zero EngineObjectIds clear Unity's cached pair without sending a message.
        /// </summary>
        internal const uint TriggerEventShapeRemoved = 0;

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

            public VehicleData** vehicles;
            public uint vehicleCount;
            public uint vehicleCapacity;

            /// <summary>
            /// Zero while the previous step's tree refit is pending, otherwise one.
            /// </summary>
            public int treeReady;
            private readonly uint reserved0;

            /// <summary>The refit that a query helps to finish while <see cref="treeReady"/> is 0.</summary>
            public SharedDynamicTreeRefit* refit;
        }

        [StructLayout(LayoutKind.Sequential)]
        internal struct StoreGlobals
        {
            public uint materialVersion;
            public uint engineObjectIdOffset;

            public uint jointDataSize;
            public uint bodyDataSize;
            public uint bodySize;

            /// <summary>
            /// Increases whenever the ignored pairs change. Cached results derived from those pairs can be reused while
            /// this version stays the same.
            /// </summary>
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

        /// <summary>Matches QueryKind in Store.h.</summary>
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

        /// <summary>
        /// Matches Physics::QueryHitResultCollectorCallback. The return value is a C++ bool, so it occupies one byte.
        /// </summary>
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        internal delegate byte HitCollectorFunction(void* context, RayCastHit* hits, ulong count);

        /// <summary>Matches Physics::QueryShapesResultCollectorCallback.</summary>
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        internal delegate byte ShapeCollectorFunction(void* context, void** shapeUserDatas, ulong count);

        /// <summary>Matches Physics::BroadPhaseQueryShapeCollectorCallback.</summary>
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        internal delegate byte BroadphaseCollectorFunction(void* context, void* shape, void* body, void* shapeUserData, ushort geometryType);

        internal delegate void SimulateCallback(WorldData* world, float deltaTime);

        internal delegate int BuildQueryWorldCallback(WorldData* world);

        internal delegate void DestroyWorldCallback(WorldData* world);

        /// <summary>
        /// Burst command entry point. command is the plugin API ID; handle identifies the target object.
        /// </summary>
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        internal delegate void CommandFunction(uint command, void* handle, CommandArgs* args, StoreGlobals* globals);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        internal delegate void* ReallocateFunction(void* memory, ulong size);

        /// <summary>
        /// Represents plugin functions that take one pointer to free memory or to retain or release a record.
        /// </summary>
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
            public IntPtr vehicleCommand;
            public IntPtr helpRefit;
        }

        /// <summary>Matches ManagedCallbacks::helpRefit: does the work that is left of a pending refit.</summary>
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        internal delegate void RefitFunction(void* refit);

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

        /// <summary>
        /// Returns a non-zero value on the thread that attached the plugin, which is the main thread.
        /// </summary>
        [DllImport(Library)]
        internal static extern int RetooledPhysics_OnMainThread();

        /// <summary>
        /// Get native address wait/wake functions, if the plugin provides them.
        /// </summary>
        [DllImport(Library)]
        private static extern void RetooledPhysics_WaitFunctions(IntPtr* wait, IntPtr* wake);

        /// <summary>
        /// Install native address wait/wake functions. Without them, solver workers spin while waiting.
        /// </summary>
        internal static void InstallAddressWait()
        {
            IntPtr wait = IntPtr.Zero;
            IntPtr wake = IntPtr.Zero;
            try
            {
                RetooledPhysics_WaitFunctions(&wait, &wake);
            }
            catch (EntryPointNotFoundException)
            {
                return;
            }

            Unity.Physics.AddressWait.Install(wait, wake);
        }

        /// <summary>
        /// Return the first struct size, command count or stored layout mismatch that prevents attach, or null.
        /// </summary>
        internal static string FirstSizeMismatch()
        {
            int[] sizes =
            {
                UnsafeUtility.GetFieldOffset(typeof(BodyData).GetField(nameof(BodyData.velocity))),
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
                (int)VehicleCommand.Count,
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
                nameof(VehicleCommand),
            };

            for (int i = 0; i < sizes.Length; i++)
            {
                int native = RetooledPhysics_StructSize(i);
                if (native != sizes[i])
                {
                    return $"{names[i]} is {sizes[i]} here and {native} in the plugin. Rebuild the plugin with `RetooledCli native build`";
                }
            }

            // The plugin reads a pose as BodyPose in Store.h: the position, the scale and the rotation.
            if (sizeof(LocalTransform) != 32 || UnsafeUtility.GetFieldOffset(typeof(LocalTransform).GetField(nameof(LocalTransform.Rotation))) != 16)
            {
                return $"{nameof(LocalTransform)} doesn't have the layout of BodyPose in the plugin";
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
