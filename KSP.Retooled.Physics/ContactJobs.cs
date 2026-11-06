using Unity.Burst;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Physics;

namespace KSP.Retooled.PhysicsBackend
{
    /// <summary>
    /// Runs between the narrow phase and the solver over every manifold. Physics.IgnoreCollision acts here, because a manifold
    /// names the leaves that touched, so the pairs of a compound can be told apart, which a filter on body pairs can't do.
    /// </summary>
    [BurstCompile]
    internal unsafe struct ReadContactsJob : IContactsJob
    {
        [NativeDisableUnsafePtrRestriction]
        public NativeStore.BodyData* Store;

        [NativeDisableUnsafePtrRestriction]
        public NativeStore.StoreGlobals* Globals;

        [NativeDisableUnsafePtrRestriction]
        public RigidBody* Bodies;

        public void Execute(ref ModifiableContactHeader header, ref ModifiableContactPoint contact)
        {
            if (contact.Index != 0)
            {
                return;
            }

            NativeStore.ShapeData* shapeA = ShapeOf(Bodies[header.BodyIndexA], header.ColliderKeyA);
            NativeStore.ShapeData* shapeB = ShapeOf(Bodies[header.BodyIndexB], header.ColliderKeyB);
            if (IgnoreTable.Contains(Globals, shapeA->ignorePairId, shapeB->ignorePairId))
            {
                header.JacobianFlags |= JacobianFlags.Disabled;
            }
        }

        /// <summary>The shape behind a key: a compound child knows it, and any other collider stands for one shape.</summary>
        private NativeStore.ShapeData* ShapeOf(RigidBody body, ColliderKey key)
        {
            NativeStore.BodyData* row = Store + body.Entity.Index;
            if (body.Collider.Value.Type != ColliderType.Compound)
            {
                return row->shapes[ColliderBuilder.FirstSupported(row)];
            }

            body.Collider.Value.GetChild(ref key, out ChildCollider child);
            return row->shapes[child.Entity.Index];
        }
    }
}
