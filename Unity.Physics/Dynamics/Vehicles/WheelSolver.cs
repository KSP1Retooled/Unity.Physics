using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Entities;
using Unity.Jobs;
using Unity.Mathematics;
using static Unity.Physics.Math;
using PhysicsJoint = Unity.Physics.Joint;
using Unity.Physics.Store.Bodies;
using Unity.Physics.Store.Memory;
using Unity.Physics.Store.Shapes;
using Unity.Physics.Store.Vehicles;
using Unity.Physics.Store.World;
using Unity.Physics.Stepping;
using Unity.Physics.Collision.Colliders.Building;
using Unity.Physics.Collision.Filter.Groups;
using Unity.Physics.Collision.Queries.Collectors;
using Unity.Physics.Dynamics.Bodies;

namespace Unity.Physics.Dynamics.Vehicles
{
    internal static unsafe partial class WheelSolver
    {
        /// <summary>
        /// Save pre-force vehicle velocities for suspension damping so that this step's gravity is not counted as
        /// compression.
        /// </summary>
        internal static void RecordStartVelocities(WorldData* world)
        {
            for (uint v = 0; v < world->vehicleCount; v++)
            {
                VehicleData* vehicle = world->vehicles[v];
                if (vehicle != null && vehicle->body != null)
                {
                    BodyData* hot = world->bodies + vehicle->body->index;
                    vehicle->startLinear = (*hot).Velocity()->Linear;
                    vehicle->startAngular = BodyKinematics.WorldAngularVelocity(*hot);
                }
            }
        }
    }
}
