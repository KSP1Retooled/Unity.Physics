using System.Runtime.InteropServices;
using Unity.Mathematics;
using Unity.Physics.Store.Shapes;

namespace Unity.Physics.Store.Vehicles
{
    /// <summary>
    /// Stores the settings that Unity supplied for one wheel and the results of the query beneath that wheel during
    /// the last step.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    internal unsafe struct WheelSlot
    {
        public void* userData;

        // The shape recorded by the last hit. When the shape is destroyed, its recorded hit can be cleared too.
        public Shape* hitShape;
        public WheelQueryResult hit;
        public WheelSuspension suspension;
        public WheelBody wheel;

        // The wheel's centre at rest and the point where its force is applied. Both are offsets from the body's
        // centre of mass, expressed along the body's axes.
        public float3 centreOffset;
        public float3 forceOffset;

        // The plugin API's FilterData, stored as four words. The upper half of the last word contains the ignore
        // id.
        public uint collidesWith;
        public uint belongsTo;
        public uint filterFlags;
        public uint filterIds;

        public float rotationSpeed;
        public float motorTorque;
        public float brakeTorque;
        public float steerAngle;
        public float rotationAngle;

        // The height of the wheel above its rest position. It ranges from minus the maximum droop to the maximum
        // compression.
        public float jounce;
        public uint enabled;
        public uint grounded;
    }
}
