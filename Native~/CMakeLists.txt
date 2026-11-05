cmake_minimum_required(VERSION 3.24)
project(RetooledPhysics CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Unity's plugin headers are licensed for Unity-dependent projects and change with each editor,
# so they are read from the editor that builds the project instead of being copied here.
set(UNITY_PLUGIN_API "" CACHE PATH "Editor/Data/PluginAPI of the project's Unity editor")
if(NOT EXISTS "${UNITY_PLUGIN_API}/IUnityPhysicsBackendSystem.h")
    message(FATAL_ERROR "UNITY_PLUGIN_API must point at the editor's PluginAPI folder")
endif()

add_library(RetooledPhysics SHARED
    src/Plugin.cpp
    src/Sdk.cpp
    src/World.cpp
    src/Body.cpp
    src/Shape.cpp
    src/Joint.cpp
    src/Query.cpp
    src/Vehicle.cpp
)
target_include_directories(RetooledPhysics PRIVATE "${UNITY_PLUGIN_API}" "${UNITY_PLUGIN_API}/PhysicsBackend")
set_target_properties(RetooledPhysics PROPERTIES PREFIX "")
