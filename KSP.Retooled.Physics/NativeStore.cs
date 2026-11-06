using System;
using System.Runtime.InteropServices;
using Unity.Mathematics;

namespace KSP.Retooled.PhysicsBackend
{
    /// <summary>
    /// The managed view of the plugin's store, field for field the "mirrored" structs of Native/RetooledPhysics/src/Store.h,
    /// which documents them. Everything is blittable so jobs can read the native rows directly.
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
        internal const uint ShapeFlagSceneQuery = 1 << 0;
        internal const uint ShapeFlagTrigger = 1 << 1;

        internal const uint QueryUseDynamicBodies = 1 << 0;
        internal const uint QueryUseStaticBodies = 1 << 1;
        internal const uint QueryUseTriggerShapes = 1 << 2;
        internal const uint QueryAllowBackfaces = 1 << 4;

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
        }

        [StructLayout(LayoutKind.Sequential)]
        internal struct BodyData
        {
            public quaternion rotation;
            public float3 position;
            public BodyFlags flags;

            public float3 linearVelocity;
            public float mass;

            public float3 angularVelocity;
            public float inverseMass;

            public float3 inertia;
            public float linearDamping;

            public float3 inverseInertia;
            public float angularDamping;

            public quaternion inertiaRotation;

            public float3 localCenterOfMass;
            public float maxLinearVelocity;

            public float maxAngularVelocity;
            public float maxDepenetrationVelocity;
            public float wakeCounter;
            public float sleepThreshold;

            public float3 force;
            public uint shapeCount;
            public float3 torque;
            private readonly uint reserved0;
            public float3 acceleration;
            private readonly uint reserved1;
            public float3 angularAcceleration;
            private readonly uint reserved2;
            public float3 impulse;
            private readonly uint reserved3;
            public float3 angularImpulse;
            private readonly uint reserved4;
            public float3 velocityChange;
            private readonly uint reserved5;
            public float3 angularVelocityChange;
            private readonly uint reserved6;

            public quaternion targetRotation;
            public float3 targetPosition;
            private readonly uint reserved7;

            public IntPtr owner;
            public ShapeData** shapes;
            public void* collider;
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
            public IntPtr owner;
        }

        [StructLayout(LayoutKind.Sequential)]
        internal struct MeshMoments
        {
            public float volume;
            public float3 centroid;
            public float3 covarianceDiagonal;
            public float3 covarianceOffDiagonal;
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
            public MeshMoments moments;
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
            public long capacity;
            public long size;
            public int status;
            private readonly int reserved0;
        }

        /// <summary>The cooked mesh of a shape, or null while it has none. A shape's mesh handle starts with this pointer.</summary>
        internal static MeshData* MeshOf(ShapeData* shape)
        {
            return shape->mesh != null ? *(MeshData**)shape->mesh : null;
        }

        internal static float3* VerticesOf(MeshData* mesh)
        {
            return (float3*)(mesh + 1);
        }

        internal static int3* TrianglesOf(MeshData* mesh)
        {
            return (int3*)(VerticesOf(mesh) + mesh->vertexCount);
        }

        [StructLayout(LayoutKind.Sequential)]
        internal struct WorldData
        {
            public BodyData* bodies;
            public uint bodyCount;
            public uint bodyCapacity;

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
        }

        [StructLayout(LayoutKind.Sequential)]
        internal struct RayCastInput
        {
            public float3 origin;
            public float3 direction;
            public float maxDistance;
            public int layerMask;
            public uint filterOptions;
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

        internal delegate void SimulateCallback(WorldData* world, float deltaTime);

        internal delegate int BuildQueryWorldCallback(WorldData* world);

        internal delegate void DestroyWorldCallback(WorldData* world);

        [StructLayout(LayoutKind.Sequential)]
        internal struct ManagedCallbacks
        {
            public IntPtr simulate;
            public IntPtr buildQueryWorld;
            public IntPtr rayCast;
            public IntPtr destroyWorld;
            public IntPtr cookMesh;
            public IntPtr validateMesh;
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

        /// <summary>The first struct whose size differs from the plugin's, or null when all agree.</summary>
        internal static string FirstSizeMismatch()
        {
            int[] sizes =
            {
                sizeof(BodyData),
                sizeof(ShapeData),
                sizeof(MaterialData),
                sizeof(WorldData),
                sizeof(StoreGlobals),
                sizeof(RayCastInput),
                sizeof(RayCastHit),
                sizeof(ManagedCallbacks),
                sizeof(MeshData),
                sizeof(CookInput),
                sizeof(CookOutput),
            };

            string[] names =
            {
                nameof(BodyData),
                nameof(ShapeData),
                nameof(MaterialData),
                nameof(WorldData),
                nameof(StoreGlobals),
                nameof(RayCastInput),
                nameof(RayCastHit),
                nameof(ManagedCallbacks),
                nameof(MeshData),
                nameof(CookInput),
                nameof(CookOutput),
            };

            for (int i = 0; i < sizes.Length; i++)
            {
                int native = RetooledPhysics_StructSize(i);
                if (native != sizes[i])
                {
                    return $"{names[i]} is {sizes[i]} bytes here and {native} in the plugin";
                }
            }

            return null;
        }
    }
}
