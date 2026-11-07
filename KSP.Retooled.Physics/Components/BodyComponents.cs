using Unity.Entities;
using Unity.Mathematics;

namespace KSP.Retooled.Components.Physics;

// All vectors are world space unless named local. Velocities are those of the centre of mass, angular in radians per
// second.

/// <summary>Where a rigid body is.</summary>
public struct BodyPose : IComponentData
{
    public quaternion Rotation;
    public float3 Position;
}

/// <summary>How fast a rigid body is moving.</summary>
public struct BodyVelocity : IComponentData
{
    public float3 Linear;
    public float3 Angular;
}

/// <summary>A rigid body's mass distribution. A body that can't be moved has inverses of zero.</summary>
public struct BodyMass : IComponentData
{
    public float Mass;
    public float InverseMass;

    /// <summary>Principal moments of inertia, in the frame <see cref="LocalCenterOfMass"/> and <see cref="InertiaRotation"/> give.</summary>
    public float3 Inertia;

    public float3 InverseInertia;
    public quaternion InertiaRotation;
    public float3 LocalCenterOfMass;
}

/// <summary>What a rigid body's motion loses each step, and the most it may have.</summary>
public struct BodyLimits : IComponentData
{
    public float LinearDamping;
    public float AngularDamping;
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
