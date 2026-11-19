using Unity.Mathematics;
using Unity.Numerics.Kernels;

namespace Unity.Physics.Dynamics.Articulations
{
    internal unsafe struct ArticulationFrame
    {
        public double3 Offset;
        public double6 Bias;
        public double3x3 Inertia;
        public double Mass;
        public double6x6 Factor;
        public double6x6 RootInverse;
        public double6 S0;
        public double6 S1;
        public double6 S2;
        public double6 U0;
        public double6 U1;
        public double6 U2;
        public double3x3 Inverse;
        public double6 Load;
        public double6 AppliedLoad;
        public double6 Motion;
        public double3 RightHandSide;
        public double3 Result;
        public double3 Drive;
        public double3 Friction;
        public double3 UnitMass;
        public double3 Compliance;
        public double3 DriveLimit;
        public double3 LimitSpeed;
        public int DrivenAxes;
        public int BoundAxes;

        public double6 S(int axis) => axis == 0 ? S0 : axis == 1 ? S1 : S2;

        public double6 JointMotion(double3 rates) => rates.x * S0 + rates.y * S1 + rates.z * S2;
        public double3 Project(double6 wrench) => new double3(double6.Dot(S0, wrench), double6.Dot(S1, wrench), double6.Dot(S2, wrench));
        public double3 ProjectInertia(double6 motion) => new double3(double6.Dot(U0, motion), double6.Dot(U1, motion), double6.Dot(U2, motion));
        public double6 FromJointForces(double3 forces) => forces.x * U0 + forces.y * U1 + forces.z * U2;

        public double6 ApplyInertia(double6 motion) => ArticulationKinematics.Spatial(
            Mass * ArticulationKinematics.Linear(motion), math.mul(Inertia, ArticulationKinematics.Angular(motion)));
    }
}
