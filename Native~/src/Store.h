// The native store: everything the backend knows about the physics scene, held in plain memory that outlives a managed
// domain reload. The structs marked "mirrored" are read and written directly by managed code and Burst jobs through
// NativeStore.cs, so their layout is a contract: only 4-byte fields and 8-byte pointers, padding spelled out, and a size
// and offset check here and in NativeStore.FirstSizeMismatch. RetooledPhysics_StructSize reports the sizes, and managed
// attach refuses to run on a mismatch.
//
// Commands reach the store through the Burst kernels in BodyKernel.cs, ShapeKernel.cs and WorldKernel.cs. Native code keeps
// only what comes before managed code exists: worlds, their settings and materials.
//
// SDK handles are stable pointers to the cold records (Body*, Shape*, Joint*, MaterialData*, Mesh*, World*). Hot body
// state lives in a dense swap-remove BodyData array per world, so jobs walk it linearly. A BodyData moves when another body
// is destroyed; a Body never does.
//
// Threads: store mutation happens on the main thread outside Simulate. Queries can arrive from worker threads, and they
// read only the query world, a snapshot managed builds on the main thread and holds nothing of the live store. Replacing
// that snapshot takes World::queryLock exclusively, and queries hold it shared. Mesh cooking can arrive from worker
// threads too: a cook touches only its own new record, a mesh's reference count is the one thing threads share, and
// g_MeshLock keeps the cook functions from going away under a cook.

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <shared_mutex>
#include <thread>
#include <vector>

#include "PhysicsFilterData.h"

namespace Retooled
{
    struct Body;
    struct Mesh;
    struct Shape;
    struct World;

    // NativeStore.cs alone defines this. A joint is a pointer here, on the store's heap, which the world frees with it.
    struct JointData;

    struct Float3
    {
        float x, y, z;
    };

    struct Float4
    {
        float x, y, z, w;
    };

    // The low bits equal Physics::BodyFlags.
    enum BodyStateFlags : uint32_t
    {
        kBodyGravity = 1 << 0,
        kBodyKinematic = 1 << 1,
        kBodySimulated = 1 << 2,

        kBodyStatic = 1 << 8,
        kBodySleeping = 1 << 9,
        kBodyHasKinematicTarget = 1 << 10,
        // The body's collider no longer matches its shapes, so managed rebuilds it before the next use.
        kBodyColliderDirty = 1 << 11,
    };

    // Mirrored. One rigid body, static or not. All vectors are world space unless named local. Velocities are those of
    // the centre of mass, angular in radians per second. The accumulators hold what AddForce and AddTorque collected
    // since the last step, sorted by ForceMode because each mode integrates differently, and they are zeroed by the
    // step that consumes them.
    struct BodyData
    {
        Float4 rotation;
        Float3 position;
        uint32_t flags; // BodyStateFlags

        Float3 linearVelocity;
        float mass;

        Float3 angularVelocity;
        float inverseMass;

        // Principal moments of inertia, in the frame given by localCenterOfMass and inertiaRotation.
        Float3 inertia;
        float linearDamping;

        Float3 inverseInertia;
        float angularDamping;

        Float4 inertiaRotation;

        Float3 localCenterOfMass;
        float maxLinearVelocity;

        float maxAngularVelocity;
        float maxDepenetrationVelocity;
        float wakeCounter;
        float sleepThreshold;

        Float3 force;                 // ForceMode.Force: N, scaled by dt / mass
        uint32_t shapeCount;
        Float3 torque;                // ForceMode.Force: N m, scaled by dt / inertia
        uint32_t reserved0;
        Float3 acceleration;          // ForceMode.Acceleration: m/s^2, scaled by dt
        uint32_t reserved1;
        Float3 angularAcceleration;   // ForceMode.Acceleration: rad/s^2, scaled by dt
        uint32_t reserved2;
        Float3 impulse;               // ForceMode.Impulse: N s, scaled by 1 / mass
        uint32_t reserved3;
        Float3 angularImpulse;        // ForceMode.Impulse: N m s, scaled by 1 / inertia
        uint32_t reserved4;
        Float3 velocityChange;        // ForceMode.VelocityChange: m/s
        uint32_t reserved5;
        Float3 angularVelocityChange; // ForceMode.VelocityChange: rad/s
        uint32_t reserved6;

        Float4 targetRotation;
        Float3 targetPosition;
        uint32_t reserved7;

        // The velocities summed since the body last showed enough energy to stay awake, the angular one in the inertia frame.
        Float3 sleepLinearSum;
        uint32_t reserved8;
        Float3 sleepAngularSum;
        uint32_t reserved9;

        Body* owner;
        Shape* const* shapes;  // shapeCount entries, owned by the Body
        void* collider;        // Unity.Physics blob, owned by managed; null means no collider or a rebuild is due
    };

    static_assert(sizeof(BodyData) == 360, "NativeStore.BodyData mirrors this layout");
    static_assert(offsetof(BodyData, force) == 144);
    static_assert(offsetof(BodyData, targetRotation) == 272);
    static_assert(offsetof(BodyData, owner) == 336);

    // Mirrored. Friction and bounciness of a PhysicsMaterial; combine modes hold Physics::MaterialCombineMode. Unity holds
    // one reference and each shape using the material another, because Unity releases a material while shapes still use it.
    struct MaterialData
    {
        float dynamicFriction;
        float staticFriction;
        float bounciness;
        uint32_t frictionCombine;
        uint32_t bounceCombine;
        uint32_t references;
        void* userData;
    };

    static_assert(sizeof(MaterialData) == 32, "NativeStore.MaterialData mirrors this layout");

    // Mirrored. One collision shape.
    struct ShapeData
    {
        Float4 localRotation;
        Float3 localPosition;
        uint32_t geometryType; // Physics::GeometryType

        // Box half extents; x is the radius of a sphere; x the radius and y the half length of a capsule; or the mesh scale.
        Float3 extents;
        uint32_t flags; // Physics::ShapeFlags

        uint32_t belongsTo;
        uint32_t collidesWith;
        uint32_t ignorePairId;
        uint32_t collisionDisabled;

        void* mesh; // the Mesh or height field of the geometry, when it has one. A shape holds a reference to a Mesh.
        MaterialData* material; // never null, and the shape holds a reference
        void* userData;
        Shape* owner;
    };

    static_assert(sizeof(ShapeData) == 96, "NativeStore.ShapeData mirrors this layout");

    // Mirrored, and written by MeshCook.cs. The head of a cooked mesh: one block with no pointers in it, followed by
    // vertexCount Float3 and then triangleCount triples of uint32_t, which is also the stream stored in a baked mesh asset.
    // The vertices are the source mesh's, unscaled, and doubled into a slab for a flat convex mesh. A convex mesh may have no
    // triangles, since its hull comes from its vertices. A mesh with no vertices has no collider.
    struct MeshData
    {
        uint32_t magic;
        uint32_t version;
        uint32_t convex;
        uint32_t vertexCount;
        uint32_t triangleCount;
        Float3 lower;
        Float3 upper;
    };

    static_assert(sizeof(MeshData) == 44, "NativeStore.MeshData mirrors this layout");

    // Mirrored. The solid a convex mesh's hull bounds, unscaled. Volume is 0 when it bounds none. The covariance is the mean
    // of (x - centroid)(x - centroid)^T over the solid, which is what a scale transforms simply.
    struct MeshMoments
    {
        float volume;
        Float3 centroid;
        Float3 covarianceDiagonal;    // xx yy zz
        Float3 covarianceOffDiagonal; // xy xz yz
    };

    static_assert(sizeof(MeshMoments) == 40, "NativeStore.MeshMoments mirrors this layout");

    // Mirrored. A cook's source mesh, as the descriptor gave it.
    struct CookInput
    {
        const void* vertices;
        const void* indices;
        uint32_t vertexCount;
        uint32_t indexCount;
        uint32_t vertexStride;
        uint32_t indexStride;
        uint32_t convex;
        uint32_t flippedNormals;
    };

    static_assert(sizeof(CookInput) == 40, "NativeStore.CookInput mirrors this layout");

    enum MeshCookStatus : int32_t
    {
        kMeshCookOk = 0,
        kMeshCookEmpty = 1,
        kMeshCookBadVertices = 2,
        kMeshCookBadIndices = 3,
    };

    // Mirrored. With null bytes the cook reports in size a capacity that is enough. Otherwise it fills bytes, and sets size
    // to what it wrote, and status.
    struct CookOutput
    {
        uint8_t* bytes;
        int64_t size;
        int32_t status; // MeshCookStatus
        int32_t reserved0;
    };

    static_assert(sizeof(CookOutput) == 24, "NativeStore.CookOutput mirrors this layout");

    // Mirrored. A joint that broke in the last step, in the shape of Physics::JointBreakEvent.
    struct JointBreak
    {
        uint64_t engineObjectId;
        float linearMagnitude;
        float angularMagnitude;
    };

    static_assert(sizeof(JointBreak) == 16, "NativeStore.JointBreak mirrors this layout");

    // Mirrored. What Unity reads back after a step, laid out as the plugin API's Physics::ContactPoint, ShapePair, BodyPair
    // and TriggerEvent. World.cpp checks that they match.
    struct ContactPoint
    {
        Float3 position;
        float separation;
        Float3 normal;
        uint32_t faceIndex0;
        Float3 impulse;
        uint32_t faceIndex1;
    };

    static_assert(sizeof(ContactPoint) == 48, "NativeStore.ContactPoint mirrors this layout");

    struct ContactShapePair
    {
        uint64_t thisCollider;
        uint64_t otherCollider;
        ContactPoint* points;
        uint32_t pointCount;
        uint16_t flags;
        uint16_t events;
        Float3 impulseSum;
        uint32_t reserved0;
    };

    static_assert(sizeof(ContactShapePair) == 48, "NativeStore.ContactShapePair mirrors this layout");

    struct ContactBodyPair
    {
        uint64_t thisBody;
        uint64_t otherBody;
        ContactShapePair* shapes;
        uint32_t shapeCount;
        uint16_t flags;
        uint16_t reserved0;
        Float3 thisLinearVelocity;
        Float3 thisAngularVelocity;
        Float3 otherLinearVelocity;
        Float3 otherAngularVelocity;
    };

    static_assert(sizeof(ContactBodyPair) == 80, "NativeStore.ContactBodyPair mirrors this layout");

    struct TriggerEvent
    {
        void* triggerShape;
        void* colliderShape;
        uint64_t trigger;
        uint64_t collider;
        uint32_t type;
        uint32_t reserved0;
    };

    static_assert(sizeof(TriggerEvent) == 40, "NativeStore.TriggerEvent mirrors this layout");

    // Mirrored. One physics scene.
    struct WorldData
    {
        BodyData* bodies;
        uint32_t bodyCount;
        uint32_t bodyCapacity;

        JointData** joints;
        uint32_t jointCount;
        uint32_t jointCapacity;

        // The step's results, held until the next step, or until Unity releases them.
        JointBreak* jointBreaks;
        uint32_t jointBreakCount;
        uint32_t jointBreakCapacity;

        ContactBodyPair* contactBodies;
        uint32_t contactBodyCount;
        uint32_t contactBodyCapacity;

        ContactShapePair* contactShapes;
        uint32_t contactShapeCount;
        uint32_t contactShapeCapacity;

        ContactPoint* contactPoints;
        uint32_t contactPointCount;
        uint32_t contactPointCapacity;

        TriggerEvent* triggers;
        uint32_t triggerCount;
        uint32_t triggerCapacity;

        // Shapes taken off their bodies since the last step, which end their touches without an exit.
        void** removedShapes;
        uint32_t removedShapeCount;
        uint32_t removedShapeCapacity;

        // Colliders of destroyed bodies, freed by managed once no built world refers to them.
        void** deadColliders;
        uint32_t deadColliderCount;
        uint32_t deadColliderCapacity;

        Float3 gravity;
        float bounceThreshold;

        // Bumped whenever a static body or the set of bodies changes, so the static tree is rebuilt only then.
        uint32_t staticsVersion;
        // Non-zero when store state moved on from the query world, so the main thread must rebuild it before reading it.
        uint32_t queryDirty;

        World* owner;
        // What managed hands the query function, null while no built world can be read. Guarded by queryLock.
        void* queryWorld;
    };

    static_assert(sizeof(WorldData) == 184, "NativeStore.WorldData mirrors this layout");

    // Mirrored. What every world shares, and what a kernel needs from the plugin, which reaches it as function pointers
    // because a Burst function gets everything through its arguments.
    struct StoreGlobals
    {
        // Bumped when any material changes: colliders embed their materials, so all of them are rebuilt.
        uint32_t materialVersion;
        // Where the EngineObjectId sits inside the object a userData pointer refers to.
        uint32_t engineObjectIdOffset;

        // sizeof(JointData) in the domain that first attached. Managed keeps it here because joints outlive a domain reload.
        uint32_t jointDataSize;
        uint32_t reserved0;

        // Which collision pairs Physics.IgnoreCollision switched off. IgnoreTable.cs owns the layout, and it stays on the
        // store's heap across a reload.
        void* ignoreTable;

        MaterialData* defaultMaterial;

        // The heap everything in the store comes from, because it has to outlive a domain reload.
        void* (*reallocate)(void* memory, uint64_t size);
        void (*release)(void* memory);

        void (*retainMesh)(Mesh* mesh);
        void (*releaseMesh)(Mesh* mesh);
        void (*retainMaterial)(MaterialData* material);
        void (*releaseMaterial)(MaterialData* material);
    };

    static_assert(sizeof(StoreGlobals) == 80, "NativeStore.StoreGlobals mirrors this layout");

    // Mirrored. What one command carries to a kernel and back, which saves a struct and a function pointer per command.
    // The Bind lines in the command files say which field holds what.
    struct CommandArgs
    {
        Float4 rotation;
        Float3 position;
        uint32_t flags;
        Float3 vector;
        float scalar;
        Float3 vector2;
        float scalar2;
        void* pointer;
        void* pointer2;
        void* pointer3;
        uint64_t size;
        float scalar3;
        uint32_t mode;
        uint32_t enabled;
        uint32_t value;
        uint32_t value2;
        uint32_t reserved0;
    };

    static_assert(sizeof(CommandArgs) == 120, "NativeStore.CommandArgs mirrors this layout");

    // A Burst function pointer: command is the plugin API's id within the category and object the handle it addresses.
    using CommandKernel = void (*)(uint32_t command, void* object, CommandArgs* args, StoreGlobals* globals);

    // Mirrored. Which query a QueryInput asks for.
    enum QueryKind : uint32_t
    {
        kQueryRay = 0,
        kQueryRayAll = 1,
        kQueryRayAgainstShape = 2,
        kQueryOverlapAny = 3,
        kQueryOverlapAll = 4,
        kQueryOverlapBroadphase = 5,
        kQueryCast = 6,
        kQueryCastAll = 7,
        kQueryBodyCast = 8,
        kQueryBodyCastAll = 9,
        kQueryClosestPoint = 10,
        kQueryPenetration = 11,
        kQueryCastAgainstShape = 12,
    };

    // Mirrored. A query's input. What a kind doesn't use stays zero. A geometry reads as a ShapeData's does: the type and the
    // extents, in a pose that is the shape's own. The collector is the plugin API's callback for that kind.
    struct QueryInput
    {
        Float4 rotation;
        Float3 position;
        uint32_t geometryType;

        Float3 extents;
        int32_t layerMask;

        Float3 direction;
        float maxDistance;

        uint32_t filterOptions; // Physics::QueryFilterFlags
        uint32_t keepInitialOverlaps;

        // The second shape's pose in a penetration test.
        Float4 rotation2;
        Float3 position2;
        uint32_t reserved0;

        Shape* shape;
        Shape* shape2;
        Body* body;
        void* collector;
        void* collectorContext;
    };

    static_assert(sizeof(QueryInput) == 144, "NativeStore.QueryInput mirrors this layout");

    // Mirrored. A query hit; the layout equals Physics::QueryHit.
    struct RayCastHit
    {
        Float3 point;
        Float3 normal;
        uint32_t faceId;
        float distance;
        float uv[2];
        uint64_t engineObjectId;
    };

    static_assert(sizeof(RayCastHit) == 48, "NativeStore.RayCastHit mirrors this layout");

    // Mirrored. A query's output. The hit also carries a closest point in point, and a penetration's direction in normal.
    struct QueryOutput
    {
        RayCastHit hit;
        void* shapeUserData;
    };

    static_assert(sizeof(QueryOutput) == 56, "NativeStore.QueryOutput mirrors this layout");

    enum QueryWorldBuild : int32_t
    {
        // The build failed and nothing is published, so queries report no hit.
        kQueryWorldFailed = 0,
        kQueryWorldBuilt = 1,
        // The main thread is running a job it picked up while waiting, where a build can't run. What was published stays.
        kQueryWorldDeferred = 2,
    };

    // Mirrored. The managed core's entry points, called on the thread that issued the command. Null until attached.
    struct ManagedCallbacks
    {
        void (*simulate)(WorldData* world, float deltaTime);
        // Brings the world's query structures up to date with the store and returns a QueryWorldBuild. Called with
        // queryLock held exclusively, on the main thread only: the build needs Allocator.Temp, job scheduling and write
        // access to native containers, and a job worker has none of them.
        int32_t (*buildQueryWorld)(WorldData* world);
        // Returns non-zero and fills output when a query of that kind found something; a collector gets its hits as they
        // come. Called with queryLock held shared, on any thread. This is a Burst function, so no thread pays a Mono
        // transition once it has compiled; until then it is a stub that runs the same code in Mono.
        int32_t (*query)(uint32_t kind, void* queryWorld, const QueryInput* input, QueryOutput* output);
        // Frees everything managed holds for a world that is going away.
        void (*destroyWorld)(WorldData* world);
        // Burst functions like rayCast, called on any thread with g_MeshLock held shared.
        void (*cookMesh)(const CookInput* input, CookOutput* output);
        int32_t (*validateMesh)(const uint8_t* stream, int64_t size, int32_t convex);
        CommandKernel bodyCommand;
        CommandKernel shapeCommand;
        CommandKernel worldCommand;
        CommandKernel jointCommand;
    };

    static_assert(sizeof(ManagedCallbacks) == 80, "NativeStore.ManagedCallbacks mirrors this layout");

    // Mirrored up to references. A cooked mesh, and the handle Unity holds. data is null while the mesh waits for managed
    // code (see Mesh.cpp) or if finishing it failed. Managed works out a convex mesh's moments on first use and keeps them
    // here, on the main thread. Unity holds one reference and each shape using the mesh another, because a mesh goes away
    // with its Mesh asset while colliders can still use it.
    struct Mesh
    {
        MeshData* data = nullptr;
        MeshMoments moments{};
        uint32_t momentsKnown = 0;
        std::atomic<uint32_t> references{1};
    };

    // Mirrored. The cold half of a body, and the handle Unity holds.
    struct Body
    {
        World* world;
        Shape** shapes; // shapeCapacity entries, also what BodyData::shapes points at
        void* userData;
        uint32_t index; // into WorldData::bodies
        uint32_t shapeCount;
        uint32_t shapeCapacity;
        int32_t collisionDetectionMode;
        uint32_t positionIterations;
        uint32_t velocityIterations;
    };

    static_assert(sizeof(Body) == 48, "NativeStore.Body mirrors this layout");

    // Mirrored. A shape, and the handle Unity holds. A shape can exist without a body, for use as query geometry.
    struct Shape
    {
        ShapeData data;
        Body* body;
        // What Unity set, returned verbatim. Only the plugin reads it, because bit field layout belongs to the compiler;
        // ShapeData holds the fields the step reads.
        Physics::FilterData filter;
        float contactOffset;
        // Physics::ShapePairEventFlags, which the contact report reads, and which the filter holds in a bit field otherwise.
        uint32_t messages;
        // One past the shape's entry in the query world last published, so that removing the shape can hide it there.
        uint32_t querySlot;
        uint32_t reserved0;
    };

    static_assert(sizeof(Shape) == 136, "NativeStore.Shape mirrors this layout");

    // The cold half of a world, and the handle Unity holds. Kernels get it as a WorldData, the first field.
    struct World
    {
        WorldData data{};
        void* userData = nullptr;
        std::shared_mutex queryLock;
    };

    extern StoreGlobals g_Globals;
    extern ManagedCallbacks g_Managed;

    // Held shared around a call to the cook functions, and exclusively to change them, so detaching waits for running cooks.
    extern std::shared_mutex g_MeshLock;

    // The thread managed attached on, set before any callback can run.
    extern std::thread::id g_MainThread;

    // Brings every world's query world up to date with its store. Main thread only.
    void SyncAllQueryWorlds();

    // Every world, for managed to walk when it attaches or detaches.
    extern std::vector<World*> g_Worlds;
}
