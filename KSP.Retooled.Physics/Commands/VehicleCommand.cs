namespace KSP.Retooled.Physics.Commands
{
    /// <summary>
    /// Vehicle command IDs shared with Vehicle.cpp. Attach verifies the count.
    /// </summary>
    internal enum VehicleCommand : uint
    {
        CreateVehicle,
        DestroyVehicle,
        SetVehicleWheelEnabled,
        GetFirstDisabledWheelFromVehicle,
        SetVehicleHasImplicitSprungMasses,
        GetVehicleHasImplicitSprungMasses,
        GetVehicleWheelQueryResult,
        SetVehicleWheelFilterData,
        GetVehicleWheelFilterData,
        GetVehicleWheelCount,
        GetVehicleWheelsUserData,
        ResetVehicleQueryResultsForShape,
        SetVehicleLongitudinalSpeedSubSteps,
        SetVehicleWheelComOffset,
        GetVehicleWheelComOffset,
        SetVehicleSuspensionExpansionLimitEnabled,
        GetVehicleSuspensionExpansionLimitEnabled,
        SetVehicleWheelSuspensionData,
        GetVehicleWheelSuspensionData,
        SetVehicleWheelData,
        GetVehicleWheelData,
        GetVehicleWheelLocalPose,
        SetVehicleWheelRotationSpeed,
        GetVehicleWheelRotationSpeed,
        SetVehicleWheelMotorTorque,
        GetVehicleWheelMotorTorque,
        SetVehicleWheelBrakeTorque,
        GetVehicleWheelBrakeTorque,
        SetVehicleWheelSteerAngle,
        GetVehicleWheelSteerAngle,
        CalculateVehicleSprungMasses,
        Count,
    }
}
