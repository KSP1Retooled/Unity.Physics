namespace Unity.Physics.Store.Articulations
{
    internal struct ArticulationAxis
    {
        public uint Lock;
        public uint DriveType;
        public float Lower;
        public float Upper;
        public float Stiffness;
        public float Damping;
        public float ForceLimit;
        public float Target;
        public float TargetVelocity;

        public bool Limited => Lock == 1;
    }
}
