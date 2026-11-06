// The native store: everything the backend knows about the physics scene, held in plain memory that outlives a managed
// domain reload. The structs marked "mirrored" are read and written directly by managed code and Burst jobs through
// NativeStore.cs, so their layout is a contract: only 4-byte fields and 8-byte pointers, padding spelled out, and a size
// and offset check here and in NativeStore.Validate. RetooledPhysics_StructSize reports the sizes, and managed attach
// refuses to run on a mismatch.
//
// SDK handles are stable pointers to the cold records (Body*, Shape*, Joint*, MaterialData*, World*). Hot body state
// lives in a dense swap-remove BodyData array per world, so jobs walk it linearly. A BodyData moves when another body
// is destroyed; a Body never does.
//
// Threads: store mutation happens on the main thread outside Simulate. Queries can arrive from worker threads, and they
// read only the query world, a snapshot managed builds on the main thread and holds nothing of the live store. Replacing
// that snapshot takes World::queryLock exclusively, and queries hold it shared.

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <shared_mutex>
#include <thread>
#include <unordered_set>
#include <vector>

#include "PhysicsFilterData.h"

namespace Retooled
{
    struct Body;
    struct Joint;
    struct Shape;
    struct World;

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
        kBodyPublicFlagsMask = kBodyGravity | kBodyKinematic | kBodySimulated,

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

        Body* owner;
        Shape* const* shapes;  // shapeCount entries, owned by the Body
        void* collider;        // Unity.Physics blob, owned by managed; null means no collider or a rebuild is due
    };

    static_assert(sizeof(BodyData) == 328, "NativeStore.BodyData mirrors this layout");
    static_assert(offsetof(BodyData, force) == 144);
    static_assert(offsetof(BodyData, targetRotation) == 272);
    static_assert(offsetof(BodyData, owner) == 304);

    // Mirrored. Friction and bounciness of a PhysicsMaterial; combine modes hold Physics::MaterialCombineMode.
    struct MaterialData
    {
        float dynamicFriction;
        float staticFriction;
        float bounciness;
        uint32_t frictionCombine;
        uint32_t bounceCombine;
        uint32_t reserved0;
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

        void* mesh; // the cooked mesh or height field of the geometry, when it has one
        const MaterialData* material;
        void* userData;
        Shape* owner;
    };

    static_assert(sizeof(ShapeData) == 96, "NativeStore.ShapeData mirrors this layout");

    // Mirrored. One physics scene.
    struct WorldData
    {
        BodyData* bodies;
        uint32_t bodyCount;
        uint32_t bodyCapacity;

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
        // What managed hands the ray cast function, null while no built world can be read. Guarded by queryLock.
        void* queryWorld;
    };

    static_assert(sizeof(WorldData) == 72, "NativeStore.WorldData mirrors this layout");

    // Mirrored. What every world shares.
    struct StoreGlobals
    {
        // Bumped when any material changes: colliders embed their materials, so all of them are rebuilt.
        uint32_t materialVersion;
        // Where the EngineObjectId sits inside the object a userData pointer refers to.
        uint32_t engineObjectIdOffset;
    };

    static_assert(sizeof(StoreGlobals) == 8, "NativeStore.StoreGlobals mirrors this layout");

    // Mirrored. Input of the managed ray cast.
    struct RayCastInput
    {
        Float3 origin;
        Float3 direction;
        float maxDistance;
        int32_t layerMask;
        uint32_t filterOptions; // Physics::QueryFilterFlags
    };

    static_assert(sizeof(RayCastInput) == 36, "NativeStore.RayCastInput mirrors this layout");

    // Mirrored. Output of the managed ray cast; the layout equals Physics::QueryHit.
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
        // Returns non-zero and fills hit when the ray hits something. Called with queryLock held shared, on any thread.
        // This is a Burst function, so no thread pays a Mono transition once it has compiled; until then it is a stub
        // that runs the same code in Mono.
        int32_t (*rayCast)(void* queryWorld, const RayCastInput* input, RayCastHit* hit);
        // Frees everything managed holds for a world that is going away.
        void (*destroyWorld)(WorldData* world);
    };

    static_assert(sizeof(ManagedCallbacks) == 32, "NativeStore.ManagedCallbacks mirrors this layout");

    // The cold half of a body, and the handle Unity holds.
    struct Body
    {
        World* world = nullptr;
        uint32_t index = 0; // into WorldData::bodies
        void* userData = nullptr;
        std::vector<Shape*> shapes;
        int32_t collisionDetectionMode = 0;
        uint32_t positionIterations = 0;
        uint32_t velocityIterations = 0;
    };

    // A shape, and the handle Unity holds. A shape can exist without a body, for use as query geometry.
    struct Shape
    {
        ShapeData data{};
        Body* body = nullptr;
        Physics::FilterData filter; // what Unity set, returned verbatim; ShapeData holds the fields the step reads
        float contactOffset = 0.0f;
    };

    // A joint record. Joint.cpp owns its meaning.
    struct Joint
    {
        void* userData = nullptr;
        Body* body0 = nullptr;
        Body* body1 = nullptr;
        int32_t type = 0;
        int32_t flags = 0;
    };

    // The cold half of a world, and the handle Unity holds.
    struct World
    {
        WorldData data{};
        void* userData = nullptr;
        std::shared_mutex queryLock;
    };

    // The body's hot row. It stays valid until a body in the same world is created or destroyed.
    inline BodyData& Hot(Body* body)
    {
        return body->world->data.bodies[body->index];
    }

    extern StoreGlobals g_Globals;
    extern ManagedCallbacks g_Managed;

    // The thread managed attached on, set before any callback can run.
    extern std::thread::id g_MainThread;

    // Brings every world's query world up to date with its store. Main thread only.
    void SyncAllQueryWorlds();

    // Every world, for managed to walk when it attaches or detaches.
    extern std::vector<World*> g_Worlds;

    // Which collision pairs Physics.IgnoreCollision switched off, as (low id << 16 | high id).
    extern std::unordered_set<uint32_t> g_IgnoredPairs;

    inline void MarkQueryDirty(World* world)
    {
        std::atomic_ref<uint32_t>(world->data.queryDirty).store(1, std::memory_order_release);
    }

    inline void MarkPoseChanged(Body* body, uint32_t flags)
    {
        MarkQueryDirty(body->world);
        if ((flags & kBodyStatic) != 0)
        {
            body->world->data.staticsVersion++;
        }
    }

    inline void MarkColliderDirty(Body* body)
    {
        Hot(body).flags |= kBodyColliderDirty;
        MarkPoseChanged(body, Hot(body).flags);
    }
}
