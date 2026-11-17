// Native physics scene storage, retained across managed domain reloads.
// Mirrored structures must match NativeStore.cs; attach verifies sizes and layouts before use.
// SDK handles point to stable records. Dense BodyData rows may move on swap-removal.
// Mutate the store on the main thread outside Simulate. Queries use snapshots under shared queryLock;
// publication takes the lock exclusively. Mesh cooking uses private data and g_MeshLock protects callbacks.

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
        // Only shapes' poses changed since the collider was built, so managed can rebuild a compound from its children.
        kBodyCollidersMoved = 1 << 12,
    };

    // Mirrors Unity.Transforms.LocalTransform, which holds a body's pose in the store: on the body's entity, or in the
    // body record while the body has no entity. The scale is always 1.
    struct BodyPose
    {
        Float3 position;
        float scale;
        Float4 rotation;
    };

    static_assert(sizeof(BodyPose) == 32, "Unity.Transforms.LocalTransform has this layout");

    // Mirrored prefix of a rigid body's dense row. Managed fields follow it.
    // Use BodyRow to walk rows spaced by StoreGlobals::bodyDataSize.
    struct BodyData
    {
        Body* owner;
        Shape* const* shapes;  // shapeCount entries, owned by the Body
        uint32_t flags; // BodyStateFlags
        uint32_t shapeCount;

        // Pointer to managed pose storage. Only the main thread refreshes it after entity structural changes.
        // Read it directly only where Body.cpp permits.
        const BodyPose* pose;
    };

    static_assert(sizeof(BodyData) == 32, "NativeStore.BodyData starts with this layout");

    // Mirrored material settings, with one reference from Unity and one from each attached shape.
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

        void* mesh; // the Mesh or HeightField of the geometry, when it has one, and the shape holds a reference to it
        MaterialData* material; // never null, and the shape holds a reference
        void* userData;
        Shape* owner;
    };

    static_assert(sizeof(ShapeData) == 96, "NativeStore.ShapeData mirrors this layout");

    // Mirrored header of a relocatable cooked mesh, followed by vertexCount Float3 values and triangleCount index
    // triples.
    // Vertices are unscaled; planar convex meshes are thickened. Convex hulls may have no triangle data.
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

    // Mirrored height field handle and unscaled samples. Main thread only.
    // Unity and each attached shape hold a reference.
    struct HeightField
    {
        int16_t* heights;      // resolution * resolution samples
        uint8_t* cells;        // (resolution - 1) squared, non-zero where the surface is solid, or null when all of it is
        uint32_t resolution;   // samples per side
        uint32_t holeCount;    // cells that aren't solid
        int32_t lowest;        // bounds of the samples, which an update may only widen, as in PhysX
        int32_t highest;
        uint32_t references;
        uint32_t reserved0;
        Shape* firstShape;     // the shapes using this field, linked by Shape::nextOnField
    };

    static_assert(sizeof(HeightField) == 48, "NativeStore.HeightField mirrors this layout");

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

        // The WheelColliders' vehicles, indexed by the id Unity holds, with null for a free id. VehicleKernel.cs writes the
        // records, each one block on the store's heap, and the world frees them with itself.
        void** vehicles;
        uint32_t vehicleCount;
        uint32_t vehicleCapacity;

        // Set to zero while a dynamic tree refit is pending, otherwise one.
        // Clear under exclusive queryLock. Queries holding the shared lock help pending work, then wait for
        // completion.
        uint32_t treeReady;
        uint32_t reserved0;
        void* refit;
    };

    static_assert(sizeof(WorldData) == 216, "NativeStore.WorldData mirrors this layout");

    // WorldData::queryDirty value indicating that only the dynamic tree refit remains before queries can read the
    // world.
    constexpr uint32_t kQueryTreePending = 3;

    // Waits while *word holds expected, for at most the given time. It may return early, so callers check again.
    void WaitWhileWordIs(int32_t* word, int32_t expected, int32_t timeoutMicroseconds);

    // Mirrored. What every world shares, and what a kernel needs from the plugin, which reaches it as function pointers
    // because a Burst function gets everything through its arguments.
    struct StoreGlobals
    {
        // Bumped when any material changes: colliders embed their materials, so all of them are rebuilt.
        uint32_t materialVersion;
        // Where the EngineObjectId sits inside the object a userData pointer refers to.
        uint32_t engineObjectIdOffset;

        // The sizes of managed's records in the domain that first attached. Managed keeps them here because the records
        // outlive a domain reload.
        uint32_t jointDataSize;
        uint32_t bodyDataSize;
        uint32_t bodySize;
        // Bumped by managed whenever the ignored pairs change, so that what was worked out from them can be kept.
        uint32_t ignoreVersion;

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

    static_assert(sizeof(StoreGlobals) == 88, "NativeStore.StoreGlobals mirrors this layout");

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
        // Update query structures and return a QueryWorldBuild result.
        // Main thread only, with exclusive queryLock; requires allocation, job scheduling and container write
        // access.
        int32_t (*buildQueryWorld)(WorldData* world);
        // Execute a query with shared queryLock on any thread. Return nonzero and fill output on a hit;
        // collectors receive hits incrementally.
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
        // Gets the world's data as the handle, because a vehicle command addresses its vehicle by id.
        CommandKernel vehicleCommand;
        // Does the work that is left of the refit in WorldData::refit, on any thread, and returns when the refit has
        // finished or when its last pieces are running on other threads. A Burst function, like query.
        void (*helpRefit)(void* refit);
    };

    static_assert(sizeof(ManagedCallbacks) == 96, "NativeStore.ManagedCallbacks mirrors this layout");

    // Cooked mesh handle, mirrored through references. data is null until managed creation succeeds.
    // Cache convex moments on first main-thread use. Unity and attached shapes each hold a reference.
    struct Mesh
    {
        MeshData* data = nullptr;
        MeshMoments moments{};
        uint32_t momentsKnown = 0;
        std::atomic<uint32_t> references{1};
    };

    // The start of the cold half of a body, mirrored, and the handle Unity holds. Managed allocates it and keeps the
    // body's components after these fields until a world takes them.
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

    static_assert(sizeof(Body) == 48, "NativeStore.Body starts with this layout");

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
        // Managed only: the shape's collision group record plus one (GroupTable.cs). The plugin never reads it.
        uint32_t groupRecord;
        Shape* nextOnField; // the next shape using the same height field
    };

    static_assert(sizeof(Shape) == 144, "NativeStore.Shape mirrors this layout");

    // The cold half of a world, and the handle Unity holds. Kernels get it as a WorldData, the first field.
    struct World
    {
        WorldData data{};
        void* userData = nullptr;
        std::shared_mutex queryLock;
    };

    extern StoreGlobals g_Globals;

    inline const BodyData* BodyRow(const WorldData& world, uint32_t index)
    {
        return reinterpret_cast<const BodyData*>(reinterpret_cast<const std::byte*>(world.bodies) + index * g_Globals.bodyDataSize);
    }
    extern ManagedCallbacks g_Managed;

    // Held shared around a call to the cook functions, and exclusively to change them, so detaching waits for running cooks.
    extern std::shared_mutex g_MeshLock;

    // The thread managed attached on, set before any callback can run.
    extern std::thread::id g_MainThread;

    // World currently being read back after Simulate, or null. Pose pointers remain current throughout this
    // window.
    // Main-thread GetPose reads them directly until CloseReadBack clears the world.
    extern std::atomic<World*> g_ReadBackWorld;

    // End direct pose read-back on commands other than GetFlags, GetIsSleeping and GetPose.
    // This closes the window before scripts can change entity storage.
    inline void CloseReadBack()
    {
        if (g_ReadBackWorld.load(std::memory_order_relaxed) != nullptr)
        {
            g_ReadBackWorld.store(nullptr, std::memory_order_relaxed);
        }
    }

    // Brings every world's query world up to date with its store. Main thread only.
    void SyncAllQueryWorlds();

    // Every world, for managed to walk when it attaches or detaches.
    extern std::vector<World*> g_Worlds;
}
