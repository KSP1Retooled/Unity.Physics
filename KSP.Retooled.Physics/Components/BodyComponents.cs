using Unity.Entities;
using Unity.Mathematics;

namespace KSP.Retooled.Components.Physics;

// A rigid body's entity uses components from Unity.Physics and Unity.Transforms for the Rigidbody data that they can
// represent. The components below store the remaining data. All vectors are in world space unless their names specify
// local space.

/// <summary>
/// Stores the mass and inertia returned by a Rigidbody. Inverting PhysicsMass would not recover these values exactly.
/// They cannot be recovered at all when the inverse values are zero.
/// </summary>
public struct BodyMass : IComponentData
{
    public float Mass;

    /// <summary>Stores the principal moments of inertia in the frame specified by PhysicsMass.Transform.</summary>
    public float3 Inertia;
}

/// <summary>Defines the limits on the velocity of a rigid body.</summary>
public struct BodyLimits : IComponentData
{
    public float MaxLinearVelocity;
    public float MaxAngularVelocity;
    public float MaxDepenetrationVelocity;
}

/// <summary>Tracks how close a rigid body is to falling asleep.</summary>
public struct BodySleep : IComponentData
{
    public float WakeCounter;
    public float Threshold;

    /// <summary>Stores the sum of linear velocities since the body last had enough energy to stay awake.</summary>
    public float3 LinearSum;

    /// <summary>
    /// Stores the sum of angular velocities in the inertia frame, over the same period as <see cref="LinearSum"/>.
    /// </summary>
    public float3 AngularSum;
}

/// <summary>
/// Accumulates forces, torques, and changes in velocity applied to a rigid body between steps. The accumulated values
/// are cleared during the step. Each ForceMode has its own fields because each mode requires a different calculation.
/// </summary>
public struct BodyForces : IComponentData
{
    /// <summary>Stores ForceMode.Force in newtons. During the step, it is scaled by dt / mass.</summary>
    public float3 Force;

    /// <summary>Stores ForceMode.Force in newton metres. During the step, it is scaled by dt / inertia.</summary>
    public float3 Torque;

    /// <summary>Stores ForceMode.Acceleration in m/s². During the step, it is scaled by dt.</summary>
    public float3 Acceleration;

    /// <summary>Stores ForceMode.Acceleration in rad/s². During the step, it is scaled by dt.</summary>
    public float3 AngularAcceleration;

    /// <summary>Stores ForceMode.Impulse in newton seconds. During the step, it is scaled by 1 / mass.</summary>
    public float3 Impulse;

    /// <summary>
    /// Stores ForceMode.Impulse in newton metre seconds. During the step, it is scaled by 1 / inertia.
    /// </summary>
    public float3 AngularImpulse;

    /// <summary>Stores ForceMode.VelocityChange in m/s.</summary>
    public float3 VelocityChange;

    /// <summary>Stores ForceMode.VelocityChange in rad/s.</summary>
    public float3 AngularVelocityChange;
}

/// <summary>Stores the pose that a kinematic rigid body should reach by the next step.</summary>
public struct BodyTarget : IComponentData
{
    public quaternion Rotation;
    public float3 Position;
}
