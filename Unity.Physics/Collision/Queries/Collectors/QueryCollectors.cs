using Unity.Burst;
using Unity.Collections;
using Unity.Entities;
using Unity.Mathematics;
using Unity.Physics.Collision.Colliders.Building;
using Unity.Physics.Store.Bodies;
using Unity.Physics.Store.Queries;
using Unity.Physics.Store.World;

namespace Unity.Physics.Collision.Queries.Collectors
{
    internal static unsafe class QueryCollectors
    {
        /// <summary>
        /// Order closest hits by distance, then higher body serial, then lower collider key.
        /// Higher serial matches PhysX's preference for later-created bodies; the leaf tie-break is deterministic.
        /// </summary>
        internal static bool Precedes(float fraction, uint serial, ColliderKey key, float bestFraction, uint bestSerial, ColliderKey bestKey)
        {
            if (fraction != bestFraction)
            {
                return fraction < bestFraction;
            }

            if (serial != bestSerial)
            {
                return serial > bestSerial;
            }

            return key.Value < bestKey.Value;
        }

        /// <summary>
        /// Return the next representable float above fraction so that a closest-hit collector also receives
        /// equal-distance hits.
        /// </summary>
        internal static float JustAbove(float fraction)
        {
            if (fraction > 0.0f)
            {
                return math.asfloat(math.asint(fraction) + 1);
            }

            if (fraction < 0.0f)
            {
                return math.asfloat(math.asint(fraction) - 1);
            }

            // The smallest normal float, rather than a denormal one, which a thread that flushes denormals to zero
            // would read as zero. Precedes still rejects a hit between zero and this value, because it is farther.
            return 1.17549435e-38f;
        }

        // Physics.Raycast uses an infinite distance by default. Unity Physics cannot represent that distance, so use a
        // finite limit.
        internal const float MaxQueryLength = 1e7f;

        /// <summary>
        /// Determine the polygon's hit side from winding, since Unity Physics turns hit normals toward the query.
        /// </summary>
        internal static bool IsBackface(Collider* root, ColliderKey key, RigidTransform worldFromRoot, float3 direction)
        {
            if (!Collider.GetLeafCollider(out ChildCollider leaf, root, key, worldFromRoot))
            {
                return false;
            }

            Collider* collider = leaf.Collider;
            if (collider->Type != ColliderType.Triangle && collider->Type != ColliderType.Quad)
            {
                return false;
            }

            return math.dot(math.rotate(leaf.TransformFromChild.rot, FrontOf(root, key, collider)), direction) > 0.0f;
        }

        /// <summary>
        /// Determines whether a point lies behind the polygon identified by a collider key. Returns false for geometry
        /// other than polygons.
        /// </summary>
        internal static bool IsBehind(Collider* root, ColliderKey key, float3 point)
        {
            if (!Collider.GetLeafCollider(out ChildCollider leaf, root, key, RigidTransform.identity))
            {
                return false;
            }

            Collider* collider = leaf.Collider;
            if (collider->Type != ColliderType.Triangle && collider->Type != ColliderType.Quad)
            {
                return false;
            }

            BlobArray.Accessor<float3> vertices = ((PolygonCollider*)collider)->Vertices;
            float3 inLeaf = math.transform(math.inverse(leaf.TransformFromChild), point);
            return math.dot(FrontOf(root, key, collider), inLeaf - vertices[0]) < 0.0f;
        }

        /// <summary>
        /// Return the leaf polygon's front direction. Reverse terrain winding so that its normal points above the
        /// surface.
        /// </summary>
        private static float3 FrontOf(Collider* root, ColliderKey key, Collider* polygon)
        {
            BlobArray.Accessor<float3> vertices = ((PolygonCollider*)polygon)->Vertices;
            float3 normal = math.cross(vertices[1] - vertices[0], vertices[2] - vertices[0]);
            return IsTerrainLeaf(root, key) ? -normal : normal;
        }

        private static bool IsTerrainLeaf(Collider* root, ColliderKey key)
        {
            if (root->Type == ColliderType.Compound)
            {
                root->GetChild(ref key, out ChildCollider child);
                return child.Collider->Type == ColliderType.Terrain;
            }

            return root->Type == ColliderType.Terrain;
        }
    }
}
