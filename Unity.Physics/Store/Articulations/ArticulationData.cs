using Unity.Physics.Store.World;

namespace Unity.Physics.Store.Articulations
{
    internal unsafe struct ArticulationData
    {
        public const uint DriveLimitsAsForces = 1;
        public const uint ImmovableRoot = 2;
        public const uint Simulated = 4;

        public ArticulationData* Next;
        public WorldData* World;
        public void* UserData;
        public ArticulationLink** Links;
        public int Count;
        public int Capacity;
        public uint Flags;
        public bool Asleep;
        public float WakeCounter;
        public float SleepThreshold;
        public uint PositionIterations;
        public uint VelocityIterations;

        public bool FixedBase => (Flags & ImmovableRoot) != 0;
    }
}
