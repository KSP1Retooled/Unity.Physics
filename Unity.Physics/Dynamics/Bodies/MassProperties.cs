using Unity.Physics.Collision.Colliders.Building;
using Unity.Mathematics;
using Unity.Physics.Store.Bodies;
using Unity.Physics.Store.Shapes;
using Unity.Physics.Store.World;
using Unity.Physics.Stepping;
using Unity.Profiling;

namespace Unity.Physics.Dynamics.Bodies
{
    internal static unsafe class MassProperties
    {
        private static readonly ProfilerMarker ClosestPointOnBoundsMarker = new ProfilerMarker(ProfilerCategory.Physics, "Physics.Bodies.ClosestPointOnBounds");
        private static readonly ProfilerMarker RecomputeMassPropertiesMarker = new ProfilerMarker(ProfilerCategory.Physics, "Physics.Bodies.RecomputeMass");

        private const uint OverrideCenterOfMass = 1;
        private const uint OverrideInertiaTensor = 2;

        internal static float InverseOrZero(float value)
        {
            return value > 0.0f ? 1.0f / value : 0.0f;
        }

        internal static void SetInertia(BodyData* hot, float3 inertia)
        {
            hot->bodyMass->Inertia = inertia;
            hot->mass->InverseInertia = new float3(InverseOrZero(inertia.x), InverseOrZero(inertia.y), InverseOrZero(inertia.z));
        }

        internal static void WorldInertiaTensor(BodyData* hot, float* matrix)
        {
            quaternion rotation = math.mul(hot->pose->Rotation, hot->mass->Transform.rot);
            float3x3 axes = new float3x3(rotation);
            float3 moments = hot->bodyMass->Inertia;
            for (int row = 0; row < 3; row++)
            {
                for (int column = 0; column < 3; column++)
                {
                    float sum = 0.0f;
                    for (int axis = 0; axis < 3; axis++)
                    {
                        sum += axes[axis][row] * moments[axis] * axes[axis][column];
                    }

                    matrix[row * 3 + column] = sum;
                }
            }
        }

        /// <summary>
        /// Diagonalize a symmetric 3 by 3 matrix using cyclic Jacobi rotations.
        /// Store eigenvectors in the columns of axes. Both arrays use row-major order.
        /// </summary>
        private static void Diagonalize(double* a, double* axes)
        {
            for (int i = 0; i < 9; i++)
            {
                axes[i] = i % 4 == 0 ? 1.0 : 0.0;
            }

            for (int sweep = 0; sweep < 32; sweep++)
            {
                double offDiagonal = math.abs(a[1]) + math.abs(a[2]) + math.abs(a[5]);
                if (offDiagonal < 1e-14 * (math.abs(a[0]) + math.abs(a[4]) + math.abs(a[8])))
                {
                    return;
                }

                for (int p = 0; p < 2; p++)
                {
                    for (int q = p + 1; q < 3; q++)
                    {
                        double pq = a[p * 3 + q];
                        if (pq == 0.0)
                        {
                            continue;
                        }

                        double theta = (a[q * 3 + q] - a[p * 3 + p]) / (2.0 * pq);
                        double t = (theta >= 0.0 ? 1.0 : -1.0) / (math.abs(theta) + math.sqrt(theta * theta + 1.0));
                        double c = 1.0 / math.sqrt(t * t + 1.0);
                        double s = t * c;

                        for (int k = 0; k < 3; k++)
                        {
                            double akp = a[k * 3 + p];
                            double akq = a[k * 3 + q];
                            a[k * 3 + p] = c * akp - s * akq;
                            a[k * 3 + q] = s * akp + c * akq;
                        }

                        for (int k = 0; k < 3; k++)
                        {
                            double apk = a[p * 3 + k];
                            double aqk = a[q * 3 + k];
                            a[p * 3 + k] = c * apk - s * aqk;
                            a[q * 3 + k] = s * apk + c * aqk;
                        }

                        for (int k = 0; k < 3; k++)
                        {
                            double vkp = axes[k * 3 + p];
                            double vkq = axes[k * 3 + q];
                            axes[k * 3 + p] = c * vkp - s * vkq;
                            axes[k * 3 + q] = s * vkp + c * vkq;
                        }
                    }
                }
            }
        }

        private static quaternion QuaternionFromAxes(double* m)
        {
            double trace = m[0] + m[4] + m[8];
            double x, y, z, w;
            if (trace > 0.0)
            {
                double s = math.sqrt(trace + 1.0) * 2.0;
                w = 0.25 * s;
                x = (m[7] - m[5]) / s;
                y = (m[2] - m[6]) / s;
                z = (m[3] - m[1]) / s;
            }
            else if (m[0] > m[4] && m[0] > m[8])
            {
                double s = math.sqrt(1.0 + m[0] - m[4] - m[8]) * 2.0;
                w = (m[7] - m[5]) / s;
                x = 0.25 * s;
                y = (m[1] + m[3]) / s;
                z = (m[2] + m[6]) / s;
            }
            else if (m[4] > m[8])
            {
                double s = math.sqrt(1.0 + m[4] - m[0] - m[8]) * 2.0;
                w = (m[2] - m[6]) / s;
                x = (m[1] + m[3]) / s;
                y = 0.25 * s;
                z = (m[5] + m[7]) / s;
            }
            else
            {
                double s = math.sqrt(1.0 + m[8] - m[0] - m[4]) * 2.0;
                w = (m[3] - m[1]) / s;
                x = (m[2] + m[6]) / s;
                y = (m[5] + m[7]) / s;
                z = 0.25 * s;
            }

            return new quaternion((float)x, (float)y, (float)z, (float)w);
        }

        private static float3 ShapeCenterInBody(ShapeData* shape, float3 localCenter)
        {
            return shape->localPosition + math.rotate(shape->localRotation, localCenter);
        }

        /// <summary>
        /// Distribute mass among non-trigger shapes by volume, following PhysX updateMassAndInertia.
        /// </summary>
        internal static void RecomputeMassProperties(Body* body, BodyData* hot, CommandArgs* a)
        {
            using Unity.Profiling.ProfilerMarker.AutoScope span = RecomputeMassPropertiesMarker.Auto();

            bool overrideCenter = (a->flags & OverrideCenterOfMass) != 0;
            bool overrideInertia = (a->flags & OverrideInertiaTensor) != 0;

            float totalVolume = 0.0f;
            float3 weightedCenter = float3.zero;
            for (uint i = 0; i < body->shapeCount; i++)
            {
                ShapeData* shape = body->shapes[i];

                // PhysX cannot calculate mass properties for a triangle mesh and rejects the calculation for the whole
                // body. A moving body with a triangle mesh must be kinematic.
                if (shape->geometryType == StoreConstants.GeometryTriangleMesh)
                {
                    totalVolume = 0.0f;
                    break;
                }

                if (TakesMass(shape, out float volume, out float3 localCenter, out _))
                {
                    totalVolume += volume;
                    weightedCenter += ShapeCenterInBody(shape, localCenter) * volume;
                }
            }

            if (totalVolume <= 0.0f)
            {
                hot->mass->Transform.pos = overrideCenter ? a->vector : float3.zero;
                SetInertia(hot, overrideInertia ? a->vector2 : new float3(1.0f));
                BodyKinematics.SetInertiaRotation(*hot, overrideInertia ? a->rotation : quaternion.identity);
                return;
            }

            float3 center = overrideCenter ? a->vector : weightedCenter / totalVolume;
            hot->mass->Transform.pos = center;
            if (overrideInertia)
            {
                SetInertia(hot, a->vector2);
                BodyKinematics.SetInertiaRotation(*hot, a->rotation);
                return;
            }

            double* tensor = stackalloc double[9];
            double* principalAxes = stackalloc double[9];
            for (int i = 0; i < 9; i++)
            {
                tensor[i] = 0.0;
            }

            for (uint i = 0; i < body->shapeCount; i++)
            {
                ShapeData* shape = body->shapes[i];
                if (TakesMass(shape, out float volume, out float3 localCenter, out float3x3 inertiaPerUnitMass))
                {
                    AddToTensor(tensor, shape, hot->bodyMass->Mass * (double)volume / totalVolume, ShapeCenterInBody(shape, localCenter) - center, inertiaPerUnitMass);
                }
            }

            Diagonalize(tensor, principalAxes);

            // The calculated eigenvectors may form a left-handed basis. A quaternion can represent only a proper
            // rotation.
            double determinant = principalAxes[0] * (principalAxes[4] * principalAxes[8] - principalAxes[5] * principalAxes[7])
                - principalAxes[1] * (principalAxes[3] * principalAxes[8] - principalAxes[5] * principalAxes[6])
                + principalAxes[2] * (principalAxes[3] * principalAxes[7] - principalAxes[4] * principalAxes[6]);
            if (determinant < 0.0)
            {
                principalAxes[2] = -principalAxes[2];
                principalAxes[5] = -principalAxes[5];
                principalAxes[8] = -principalAxes[8];
            }

            SetInertia(hot, new float3((float)tensor[0], (float)tensor[4], (float)tensor[8]));
            BodyKinematics.SetInertiaRotation(*hot, QuaternionFromAxes(principalAxes));
        }

        private static bool TakesMass(ShapeData* shape, out float volume, out float3 center, out float3x3 inertiaPerUnitMass)
        {
            bool isTrigger = (shape->flags & StoreConstants.ShapeFlagTrigger) != 0;
            bool hasMass = ShapeMassProperties.MassProperties(shape, out volume, out center, out inertiaPerUnitMass);
            return !isTrigger && hasMass;
        }

        /// <summary>
        /// Add a shape's rotated inertia and parallel axis contribution to the body's inertia tensor.
        /// </summary>
        private static void AddToTensor(double* tensor, ShapeData* shape, double mass, float3 offset, float3x3 inertiaPerUnitMass)
        {
            float3x3 axes = new float3x3(shape->localRotation);
            double offsetSquared = math.dot(offset, offset);
            for (int row = 0; row < 3; row++)
            {
                for (int column = 0; column < 3; column++)
                {
                    double value = (row == column ? offsetSquared : 0.0) - (double)offset[row] * offset[column];
                    for (int first = 0; first < 3; first++)
                    {
                        for (int second = 0; second < 3; second++)
                        {
                            value += (double)axes[first][row] * inertiaPerUnitMass[first][second] * axes[second][column];
                        }
                    }

                    tensor[row * 3 + column] += mass * value;
                }
            }
        }

        internal static void ClosestPointOnBounds(Body* body, BodyData* hot, CommandArgs* a)
        {
            using Unity.Profiling.ProfilerMarker.AutoScope span = ClosestPointOnBoundsMarker.Auto();

            float3 position = a->vector;
            float3 closest = position;
            if (body->shapeCount != 0)
            {
                float3 lower = new float3(float.PositiveInfinity);
                float3 upper = new float3(float.NegativeInfinity);
                for (uint i = 0; i < body->shapeCount; i++)
                {
                    ShapeData* shape = body->shapes[i];
                    ShapeMassProperties.WorldBounds(shape, shape->localPosition, shape->localRotation, hot->pose->Position, hot->pose->Rotation, out float3 shapeLower, out float3 shapeUpper);
                    lower = math.min(lower, shapeLower);
                    upper = math.max(upper, shapeUpper);
                }

                closest = math.clamp(position, lower, upper);
            }

            a->vector = closest;
            a->scalar = math.lengthsq(closest - position);
        }
    }
}
