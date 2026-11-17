namespace KSP.Retooled.Physics.Commands
{
    /// <summary>
    /// Joint command IDs shared with Joint.cpp. Attach verifies the count.
    /// </summary>
    internal enum JointCommand : uint
    {
        Invalid = 0,
        SetFlags,
        GetFlags,
        GetUserData,
        SetActorLocalPose,
        GetActorLocalPose,
        SetBreakForce,
        GetIsConstraintValid,
        GetForces,
        SetInvMassAndInertiaScale,
        SetActors,
        GetActors,
        WakeUpActors,
        Set6DofLimit,
        Get6DofLimit,
        Set6DofSpringLimit,
        Set6DofAxisLock,
        Set6DofAxisMotorConfiguration,
        Set6DofMotorTargetPosition,
        Set6DofMotorTargetRotation,
        Set6DofMotorTargetLinearVelocity,
        Set6DofMotorTargetAngularVelocity,
        Set6DofProjectionTolerance,
        SetDistanceLimit,
        SetDistanceSpringLimit,
        SetDistanceErrorTolerance,
        SetHingeMotorConfiguration,
        SetHingeMotorTargetVelocity,
        SetHingeLimit,
        SetHingeLimitEnabled,
        Count,
    }
}
