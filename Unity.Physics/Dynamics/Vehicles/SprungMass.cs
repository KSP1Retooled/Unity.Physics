using Unity.Mathematics;
using Unity.Physics.Store.Bodies;
using Unity.Physics.Store.Shapes;
using Unity.Physics.Store.World;
using Unity.Physics.Stepping;
using Unity.Physics.Store.Vehicles;

namespace Unity.Physics.Dynamics.Vehicles
{
    internal static unsafe class SprungMass
    {
        /// <summary>
        /// Distribute sprung mass to balance the body in the ground plane, minimizing the sum of squared wheel
        /// masses as PhysX does.
        /// Wheel offsets are relative to the centre of mass in body axes.
        /// </summary>
        internal static void SprungMasses(VehicleData* vehicle, BodyData* hot, uint* ids, float3* offsets, int count)
        {
            float inverseMass = hot->mass->InverseMass;
            float mass = inverseMass > 0.0f ? 1.0f / inverseMass : 0.0f;
            WheelSlot* wheels = VehicleData.WheelsOf(vehicle);
            for (int i = 0; i < count; i++)
            {
                if (ids[i] < vehicle->wheelCount)
                {
                    wheels[ids[i]].suspension.sprungMass = AtWheel(offsets, count, i, mass);
                }
            }
        }

        private static float AtWheel(float3* offsets, int count, int index, float mass)
        {
            if (count == 1)
            {
                return mass;
            }

            // Solve rows*m = (mass, 0, 0) with minimum squared mass using rows^T*(rows*rows^T)^-1.
            // Only the first column of the inverse is needed.
            double3x3 gram = default;
            for (int i = 0; i < count; i++)
            {
                double3 row = new double3(1.0, offsets[i].x, offsets[i].z);
                gram += new double3x3(row * row.x, row * row.y, row * row.z);
            }

            double determinant = math.determinant(gram);
            double3 own = new double3(1.0, offsets[index].x, offsets[index].z);
            if (math.abs(determinant) > 1e-9 * math.abs(gram.c0.x * gram.c1.y * gram.c2.z))
            {
                double3 column = math.inverse(gram).c0;
                return (float)(mass * math.dot(own, column));
            }

            // The wheels lie on a line, so only the moment along that line can be balanced.
            double2 axis = LineOf(offsets, count);
            if (math.lengthsq(axis) == 0.0)
            {
                return mass / count;
            }

            double2x2 line = default;
            for (int i = 0; i < count; i++)
            {
                double2 row = new double2(1.0, math.dot(axis, new double2(offsets[i].x, offsets[i].z)));
                line += new double2x2(row * row.x, row * row.y);
            }

            double2 ownOnLine = new double2(1.0, math.dot(axis, new double2(own.y, own.z)));
            return math.abs(math.determinant(line)) > 1e-12 ? (float)(mass * math.dot(ownOnLine, math.inverse(line).c0)) : mass / count;
        }

        private static double2 LineOf(float3* offsets, int count)
        {
            for (int i = 1; i < count; i++)
            {
                double2 along = new double2(offsets[i].x - offsets[0].x, offsets[i].z - offsets[0].z);
                if (math.lengthsq(along) > 1e-12)
                {
                    return math.normalize(along);
                }
            }

            return double2.zero;
        }
    }
}
