cmake_minimum_required(VERSION 3.24)

# The game ships x86_64 only, and a fresh macOS configure on an arm64 host would otherwise build for arm64.
if(APPLE)
    set(CMAKE_OSX_ARCHITECTURES "x86_64" CACHE STRING "" FORCE)
endif()

project(RetooledPhysics CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Unity's plugin headers are licensed for Unity-dependent projects and change with each editor,
# so they are read from the editor that builds the project instead of being copied here.
set(UNITY_PLUGIN_API "" CACHE PATH "Editor/Data/PluginAPI of the project's Unity editor")
if(NOT EXISTS "${UNITY_PLUGIN_API}/IUnityPhysicsBackendSystem.h")
    message(FATAL_ERROR "UNITY_PLUGIN_API must point at the editor's PluginAPI folder")
endif()

option(RETOOLED_VEHICLE_TRACE "Advertise vehicles and record their commands, for development only" OFF)

add_library(RetooledPhysics SHARED
    src/Plugin.cpp
    src/Sdk.cpp
    src/World.cpp
    src/Body.cpp
    src/Shape.cpp
    src/Mesh.cpp
    src/Joint.cpp
    src/Query.cpp
    src/Vehicle.cpp
    src/VehicleTrace.cpp
    src/Wait.cpp
)
target_include_directories(RetooledPhysics PRIVATE "${UNITY_PLUGIN_API}" "${UNITY_PLUGIN_API}/PhysicsBackend")
set_target_properties(RetooledPhysics PROPERTIES PREFIX "" CXX_VISIBILITY_PRESET hidden)

find_package(Threads REQUIRED)
target_link_libraries(RetooledPhysics PRIVATE Threads::Threads)

# The vehicle header reaches a sibling through its path inside Unity's source tree, which the plugin API does not have.
set(ENGINE_PATHS "${CMAKE_CURRENT_BINARY_DIR}/EnginePaths")
file(WRITE "${ENGINE_PATHS}/Modules/Physics/CommandLayer/PhysicsFilterData.h"
    "#include \"${UNITY_PLUGIN_API}/PhysicsBackend/PhysicsFilterData.h\"\n")
file(WRITE "${ENGINE_PATHS}/Modules/Physics/CommandLayer/PhysicsPose.h"
    "#include \"${UNITY_PLUGIN_API}/PhysicsBackend/PhysicsPose.h\"\n")
target_include_directories(RetooledPhysics PRIVATE "${ENGINE_PATHS}")

if(RETOOLED_VEHICLE_TRACE)
    target_compile_definitions(RetooledPhysics PRIVATE RETOOLED_VEHICLE_TRACE)
endif()

option(RETOOLED_COMMAND_TRACE "Count the commands of each category and id, for development only" OFF)
if(RETOOLED_COMMAND_TRACE)
    target_compile_definitions(RetooledPhysics PRIVATE RETOOLED_COMMAND_TRACE)
endif()
