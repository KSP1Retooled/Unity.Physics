using Unity.Mathematics;

namespace KSP.Retooled.Physics
{
    /// <summary>
    /// Provides a shape's geometry for mass calculations and bounds calculations.
    /// </summary>
    internal static unsafe class ShapeMath
    {
        /// <summary>
        /// Return the capsule's straight length. The API's height field stores half this length, like
        /// PxCapsuleGeometry::halfHeight.
        /// </summary>
        internal static float CapsuleCylinderLength(NativeStore.ShapeData* shape)
        {
            return 2.0f * shape->extents.y;
        }

        internal static bool IsMeshGeometry(uint geometryType)
        {
            return geometryType == NativeStore.GeometryConvexMesh || geometryType == NativeStore.GeometryTriangleMesh;
        }

        /// <summary>
        /// Calculate volume, centre of mass and inertia per unit mass in the shape's frame.
        /// Return false for skipped or zero-volume shapes; their inertia output is undefined.
        /// </summary>
        internal static bool MassProperties(NativeStore.ShapeData* shape, out float volume, out float3 center, out float3x3 inertiaPerUnitMass)
        {
            volume = 0.0f;
            center = float3.zero;
            inertiaPerUnitMass = default;
            switch (shape->geometryType)
            {
                case NativeStore.GeometryBox:
                {
                    float3 h = shape->extents;
                    volume = 8.0f * h.x * h.y * h.z;
                    inertiaPerUnitMass = float3x3.Scale(
                        (h.y * h.y + h.z * h.z) / 3.0f,
                        (h.x * h.x + h.z * h.z) / 3.0f,
                        (h.x * h.x + h.y * h.y) / 3.0f);
                    break;
                }
                case NativeStore.GeometrySphere:
                {
                    float r = shape->extents.x;
                    volume = 4.0f / 3.0f * math.PI * r * r * r;
                    inertiaPerUnitMass = float3x3.Scale(0.4f * r * r);
                    break;
                }
                case NativeStore.GeometryCapsule:
                {
                    // The capsule's axis is local X, as in PhysX. It consists of a cylinder of length h between two
                    // hemispheres.
                    float r = shape->extents.x;
                    float h = CapsuleCylinderLength(shape);
                    float cylinder = math.PI * r * r * h;
                    float spheres = 4.0f / 3.0f * math.PI * r * r * r;
                    volume = cylinder + spheres;
                    if (volume <= 0.0f)
                    {
                        return false;
                    }

                    float axial = cylinder * r * r / 2.0f + spheres * 0.4f * r * r;
                    float transverse = cylinder * (h * h / 12.0f + r * r / 4.0f)
                        + spheres * (0.4f * r * r + h * h / 4.0f + 3.0f * h * r / 8.0f);
                    inertiaPerUnitMass = float3x3.Scale(axial / volume, transverse / volume, transverse / volume);
                    break;
                }
                case NativeStore.GeometryConvexMesh:
                    return ConvexMassProperties(shape, out volume, out center, out inertiaPerUnitMass);
                default:
                    return false;
            }

            return volume > 0.0f;
        }

        /// <summary>
        /// Compute scaled convex mass properties from cached mesh covariance.
        /// Scale covariance entry (i, j) by s_i*s_j, then use I = trace(C)*identity - C.
        /// </summary>
        private static bool ConvexMassProperties(NativeStore.ShapeData* shape, out float volume, out float3 center, out float3x3 inertiaPerUnitMass)
        {
            volume = 0.0f;
            center = float3.zero;
            inertiaPerUnitMass = default;

            NativeStore.Mesh* mesh = (NativeStore.Mesh*)shape->mesh;
            if (mesh == null || mesh->data == null || mesh->data->vertexCount == 0)
            {
                return false;
            }

            if (mesh->momentsKnown == 0)
            {
                ColliderBuilder.ConvexMoments(mesh->data, &mesh->moments);
                mesh->momentsKnown = 1;
            }

            if (mesh->moments.volume <= 0.0f)
            {
                return false;
            }

            float3 s = shape->extents;
            volume = mesh->moments.volume * math.abs(s.x * s.y * s.z);
            center = mesh->moments.centroid * s;

            float3 diagonal = mesh->moments.covarianceDiagonal * s * s;
            float3 offDiagonal = mesh->moments.covarianceOffDiagonal;
            float xy = offDiagonal.x * s.x * s.y;
            float xz = offDiagonal.y * s.x * s.z;
            float yz = offDiagonal.z * s.y * s.z;
            inertiaPerUnitMass = new float3x3(
                diagonal.y + diagonal.z, -xy, -xz,
                -xy, diagonal.x + diagonal.z, -yz,
                -xz, -yz, diagonal.x + diagonal.y);
            return volume > 0.0f;
        }

        /// <summary>
        /// Returns the half extents of an axis-aligned box that encloses the rotated box with the given half extents.
        /// </summary>
        private static float3 RotatedHalfExtents(quaternion rotation, float3 half)
        {
            float3 x = math.rotate(rotation, new float3(half.x, 0.0f, 0.0f));
            float3 y = math.rotate(rotation, new float3(0.0f, half.y, 0.0f));
            float3 z = math.rotate(rotation, new float3(0.0f, 0.0f, half.z));
            return math.abs(x) + math.abs(y) + math.abs(z);
        }

        /// <summary>
        /// Calculates the shape's axis-aligned bounds in world space from the given local pose and body pose.
        /// </summary>
        internal static void WorldBounds(
            NativeStore.ShapeData* shape,
            float3 localPosition,
            quaternion localRotation,
            float3 bodyPosition,
            quaternion bodyRotation,
            out float3 lower,
            out float3 upper)
        {
            float3 center = bodyPosition + math.rotate(bodyRotation, localPosition);
            quaternion rotation = math.mul(bodyRotation, localRotation);

            float3 half = float3.zero;
            switch (shape->geometryType)
            {
                case NativeStore.GeometryBox:
                    half = RotatedHalfExtents(rotation, shape->extents);
                    break;
                case NativeStore.GeometrySphere:
                    half = new float3(shape->extents.x);
                    break;
                case NativeStore.GeometryCapsule:
                {
                    float3 axis = math.rotate(rotation, new float3(CapsuleCylinderLength(shape) / 2.0f, 0.0f, 0.0f));
                    half = math.abs(axis) + shape->extents.x;
                    break;
                }
                case NativeStore.GeometryConvexMesh:
                case NativeStore.GeometryTriangleMesh:
                {
                    NativeStore.MeshData* mesh = NativeStore.MeshOf(shape);
                    if (mesh == null)
                    {
                        break;
                    }

                    float3 s = shape->extents;
                    center += math.rotate(rotation, (mesh->lower + mesh->upper) * 0.5f * s);
                    half = RotatedHalfExtents(rotation, (mesh->upper - mesh->lower) * 0.5f * math.abs(s));
                    break;
                }
                case NativeStore.GeometryHeightField:
                {
                    NativeStore.HeightField* field = (NativeStore.HeightField*)shape->mesh;
                    if (field == null)
                    {
                        break;
                    }

                    float3 s = shape->extents;
                    float3 top = new float3(field->resolution - 1, field->highest, field->resolution - 1) * s;
                    float3 bottom = new float3(0.0f, field->lowest * s.y, 0.0f);
                    center += math.rotate(rotation, (bottom + top) * 0.5f);
                    half = RotatedHalfExtents(rotation, (top - bottom) * 0.5f);
                    break;
                }
            }

            lower = center - half;
            upper = center + half;
        }
    }
}
