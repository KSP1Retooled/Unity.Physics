using Unity.Mathematics;
using Unity.Physics.Store.Bodies;

namespace Unity.Physics.Store.Articulations
{
    internal unsafe struct ArticulationLink
    {
        public ArticulationData* Tree;
        public ArticulationLink* Parent;
        public Body* Body;
        public int Index;
        public uint Type;
        public int Axes;
        public RigidTransform ParentAnchor;
        public RigidTransform ChildAnchor;
        public float3 Position;
        public float3 Velocity;
        public float3 Acceleration;
        public float3 Force;
        public float3 DriveForce;
        public float Friction;
        public float MaxVelocity;

        public ArticulationAxis AngularX;
        public ArticulationAxis AngularY;
        public ArticulationAxis AngularZ;
        public ArticulationAxis LinearX;
        public ArticulationAxis LinearY;
        public ArticulationAxis LinearZ;

        public float3 FreeLinearVelocity;
        public float3 FreeAngularVelocity;
        public float3 StepStartVelocity;

        public int DofCount => math.countbits(Axes);
        public bool Prismatic => Type == 1;

        public static ArticulationAxis* Axis(ArticulationLink* link, int axis)
        {
            return &link->AngularX + axis + (link->Prismatic ? 3 : 0);
        }

        public static ArticulationAxis* CommandAxis(ArticulationLink* link, uint axis)
        {
            if (axis > 6 || axis == 3)
            {
                return null;
            }

            return &link->AngularX + (axis < 3 ? axis : axis - 1);
        }

        public void UpdateAxes()
        {
            int mask = 0;
            if (Parent != null)
            {
                switch (Type)
                {
                    case 1:
                        mask = LinearX.Lock != 0 ? 1 : LinearY.Lock != 0 ? 2 : LinearZ.Lock != 0 ? 4 : 0;
                        break;
                    case 2:
                        mask = AngularX.Lock != 0 ? 1 : 0;
                        break;
                    case 3:
                        mask = (AngularX.Lock != 0 ? 1 : 0) | (AngularY.Lock != 0 ? 2 : 0) | (AngularZ.Lock != 0 ? 4 : 0);
                        break;
                }
            }

            Axes = mask;
            for (int i = 0; i < 3; i++)
            {
                if ((mask & (1 << i)) == 0)
                {
                    Position[i] = Velocity[i] = Acceleration[i] = Force[i] = DriveForce[i] = 0f;
                }
            }
        }
    }
}
