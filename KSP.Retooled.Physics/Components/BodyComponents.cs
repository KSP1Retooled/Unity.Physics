using Unity.Entities;
using Unity.Mathematics;

namespace KSP.Retooled.Components.Physics;

// A rigid body's entity holds Unity.Physics' and Unity.Transforms' components where they say what a Rigidbody needs. These
// hold what they can't. All vectors are world space unless named local.

/// <summary>
/// The mass and inertia that a Rigidbody hands back, which inverting PhysicsMass wouldn't return exactly, and which can't
/// be recovered from a body that has inverses of zero.
/// </summary>
public struct BodyMass : IComponentData
{
    public float Mass;

    /// <summary>Principal moments of inertia, in the frame that PhysicsMass.Transform gives.</summary>
    public float3 Inertia;
}

/// <summary>The most motion a rigid body may have.</summary>
public struct BodyLimits : IComponentData
{
    public float MaxLinearVelocity;
    public float MaxAngularVelocity;
    public float MaxDepenetrationVelocity;
}

/// <summary>How close a rigid body is to going to sleep.</summary>
public struct BodySleep : IComponentData
{
    public float WakeCounter;
    public float Threshold;

    /// <summary>The velocity summed since the body last showed enough energy to stay awake.</summary>
    public float3 LinearSum;

    /// <summary>As <see cref="LinearSum"/>, in the inertia frame.</summary>
    public float3 AngularSum;
}

/// <summary>
/// What has been applied to a rigid body since the last step, which zeroes it. Each ForceMode integrates differently, so
/// each has its own fields.
/// </summary>
public struct BodyForces : IComponentData
{
    /// <summary>ForceMode.Force, in newtons. The step scales it by dt / mass.</summary>
    public float3 Force;

    /// <summary>ForceMode.Force, in newton metres. The step scales it by dt / inertia.</summary>
    public float3 Torque;

    /// <summary>ForceMode.Acceleration, in m/s². The step scales it by dt.</summary>
    public float3 Acceleration;

    /// <summary>ForceMode.Acceleration, in rad/s². The step scales it by dt.</summary>
    public float3 AngularAcceleration;

    /// <summary>ForceMode.Impulse, in newton seconds. The step scales it by 1 / mass.</summary>
    public float3 Impulse;

    /// <summary>ForceMode.Impulse, in newton metre seconds. The step scales it by 1 / inertia.</summary>
    public float3 AngularImpulse;

    /// <summary>ForceMode.VelocityChange, in m/s.</summary>
    public float3 VelocityChange;

    /// <summary>ForceMode.VelocityChange, in rad/s.</summary>
    public float3 AngularVelocityChange;
}

/// <summary>Where a kinematic rigid body has been told to move by the next step.</summary>
public struct BodyTarget : IComponentData
{
    public quaternion Rotation;
    public float3 Position;
}
