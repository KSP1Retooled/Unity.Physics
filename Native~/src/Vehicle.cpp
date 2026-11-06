// PhysX extension commands: vehicles, character controllers, immediate mode. The backend does not advertise any of
// these features, so Unity is not expected to send them, and the slots keep the logging stubs Plugin.cpp installed.
// The development trace build advertises vehicles and records what Unity sends.

#include "Commands.h"

namespace Retooled
{
#ifdef RETOOLED_VEHICLE_TRACE
    void BindExtensionCommands(FunctionBinding* table)
    {
        BindVehicleTrace(table);
    }
#else
    void BindExtensionCommands(FunctionBinding*)
    {
    }
#endif
}
