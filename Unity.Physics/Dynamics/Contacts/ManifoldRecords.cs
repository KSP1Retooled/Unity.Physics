using Unity.Mathematics;
using Unity.Physics.Store.Shapes;

namespace Unity.Physics.Dynamics.Contacts
{
    internal static class ManifoldRecords
    {
        /// <summary>
        /// Contact manifold retained after filtering, with shapes ordered by body row and statics assigned to body B.
        /// Keep original collider keys to match solver events. Several manifolds may belong to one shape pair.
        /// </summary>
        internal unsafe struct ManifoldRecord
        {
            public ShapeData* ShapeA;
            public ShapeData* ShapeB;
            public int RowA;
            public int RowB;

            // The positions of ShapeA and ShapeB in their bodies' lists of shapes. The contact report sorts the shape pairs of
            // a body pair by these, because the shapes' addresses differ from run to run.
            public int ShapeIndexA;
            public int ShapeIndexB;

            public BodyIndexPair BuiltPair;
            public ColliderKeyPair BuiltKeys;

            public int FirstPoint;
            public int PointCount;

            // Points from B toward A. A reported contact normal points from the other shape toward the shape receiving the
            // event.
            public float3 Normal;
            public float3 LinearVelocityA;
            public float3 AngularVelocityA;
            public float3 LinearVelocityB;
            public float3 AngularVelocityB;

            // The original normal from the narrow phase, as used by the solver's jacobian.
            public float3 BuiltNormal;

            // The cached impulse from the previous step and the final impulse for the current step. Apply
            // WarmStart.WarmStartJob.Fraction of the cached impulse during the warm start.
            public float WarmImpulse;
            public float Impulse;
            public float ArticulationImpulse;
            public bool Touching;
            public bool Trigger;
        }

        /// <summary>
        /// Stores a contact point of a <see cref="ManifoldRecord"/> in the form produced by the narrow phase.
        /// </summary>
        internal struct ManifoldPoint
        {
            public float3 Position;
            public float Distance;
        }
    }
}
