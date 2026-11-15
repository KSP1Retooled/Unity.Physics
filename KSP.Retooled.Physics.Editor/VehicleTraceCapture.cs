using System;
using System.Runtime.InteropServices;
using UnityEngine;

namespace KSP.Retooled.Physics.Editor
{
    /// <summary>
    /// Exercises WheelCollider creation, configuration and destruction using a build from `native build
    /// --vehicle-trace`. Marks each phase in the plugin's vehicle trace, including the addition of other wheels to the
    /// same body. Run it with `unity exec --backend retooled`.
    /// </summary>
    public static class VehicleTraceCapture
    {
        [DllImport("RetooledPhysics", CharSet = CharSet.Ansi)]
        private static extern void RetooledPhysics_TraceMark(string text);

        private static void Steps(int count)
        {
            for (int i = 0; i < count; i++)
            {
                UnityEngine.Physics.Simulate(0.02f);
            }
        }

        public static void Run()
        {
            if (!BackendInfo.IsRetooled)
            {
                Debug.LogError("[RetooledPhysics] the vehicle trace needs --backend retooled");
                return;
            }

            try
            {
                RetooledPhysics_TraceMark("start");
            }
            catch (EntryPointNotFoundException)
            {
                Debug.LogError("[RetooledPhysics] this plugin has no trace; run `native build --vehicle-trace`");
                return;
            }

            UnityEngine.Physics.simulationMode = SimulationMode.Script;
            UnityEngine.Physics.gravity = new Vector3(0.0f, -9.81f, 0.0f);
            Capture();
            Debug.Log("[RetooledPhysics] vehicle trace written");
        }

        private static void Capture()
        {
            GameObject ground = new GameObject("ground");
            ground.transform.position = new Vector3(0.0f, -0.5f, 0.0f);
            ground.AddComponent<BoxCollider>().size = new Vector3(100.0f, 1.0f, 100.0f);

            GameObject chassis = new GameObject("chassis");
            chassis.transform.position = new Vector3(0.0f, 1.0f, 0.0f);
            Rigidbody body = chassis.AddComponent<Rigidbody>();
            body.mass = 100.0f;
            chassis.AddComponent<BoxCollider>();
            UnityEngine.Physics.SyncTransforms();
            Steps(1);

            RetooledPhysics_TraceMark("add first WheelCollider");
            GameObject firstHolder = new GameObject("wheel-a");
            firstHolder.transform.SetParent(chassis.transform, false);
            firstHolder.transform.localPosition = new Vector3(1.0f, -0.5f, 0.0f);
            WheelCollider first = firstHolder.AddComponent<WheelCollider>();

            RetooledPhysics_TraceMark("step with one default wheel");
            Steps(2);

            RetooledPhysics_TraceMark("configure: radius, mass, damping");
            first.radius = 0.4f;
            first.mass = 15.0f;
            first.wheelDampingRate = 0.5f;

            RetooledPhysics_TraceMark("configure: suspension distance, spring, center, force point");
            first.suspensionDistance = 0.25f;
            first.suspensionSpring = new JointSpring { spring = 20000.0f, damper = 3000.0f, targetPosition = 0.4f };
            first.center = new Vector3(0.0f, 0.1f, 0.0f);
            first.forceAppPointDistance = 0.2f;

            RetooledPhysics_TraceMark("configure: friction curves (stock sets stiffness 0)");
            WheelFrictionCurve curve = new WheelFrictionCurve { stiffness = 0.0f };
            first.forwardFriction = curve;
            first.sidewaysFriction = curve;

            RetooledPhysics_TraceMark("configure: sprung mass, substeps");
            first.sprungMass = 50.0f;
            first.ConfigureVehicleSubsteps(1000.0f, 1, 1);
            first.ResetSprungMasses();

            RetooledPhysics_TraceMark("drive: motor, brake, steer");
            first.motorTorque = 1e-5f;
            first.brakeTorque = 3.0f;
            first.steerAngle = 15.0f;
            Steps(2);

            RetooledPhysics_TraceMark("read: ground hit, rpm, pose, sprung mass, grounded");
            first.GetGroundHit(out WheelHit hit);
            first.GetWorldPose(out Vector3 position, out Quaternion rotation);
            Debug.Log($"[RetooledPhysics] wheel {first.rpm} {first.isGrounded} {first.sprungMass} {hit.force} {position} {rotation}");

            RetooledPhysics_TraceMark("change body mass and center of mass");
            body.mass = 200.0f;
            body.centerOfMass = new Vector3(0.0f, -0.25f, 0.0f);
            Steps(1);

            RetooledPhysics_TraceMark("change layer of the wheel");
            firstHolder.layer = 26;
            Steps(1);

            RetooledPhysics_TraceMark("IgnoreCollision of the wheel against the chassis collider");
            UnityEngine.Physics.IgnoreCollision(first, chassis.GetComponent<BoxCollider>());
            Steps(1);

            RetooledPhysics_TraceMark("add second WheelCollider");
            GameObject secondHolder = new GameObject("wheel-b");
            secondHolder.transform.SetParent(chassis.transform, false);
            secondHolder.transform.localPosition = new Vector3(-1.0f, -0.5f, 0.0f);
            WheelCollider second = secondHolder.AddComponent<WheelCollider>();
            Steps(1);

            RetooledPhysics_TraceMark("add third WheelCollider, past the slots the vehicle has");
            GameObject thirdHolder = new GameObject("wheel-c");
            thirdHolder.transform.SetParent(chassis.transform, false);
            thirdHolder.transform.localPosition = new Vector3(0.0f, -0.5f, 1.0f);
            WheelCollider third = thirdHolder.AddComponent<WheelCollider>();
            Steps(1);

            RetooledPhysics_TraceMark("disable and enable the first wheel");
            first.enabled = false;
            Steps(1);
            first.enabled = true;
            Steps(1);

            RetooledPhysics_TraceMark("destroy the second wheel");
            UnityEngine.Object.DestroyImmediate(second);
            Steps(1);

            RetooledPhysics_TraceMark("destroy the third wheel");
            UnityEngine.Object.DestroyImmediate(third);
            Steps(1);

            RetooledPhysics_TraceMark("destroy the ground under the wheel");
            UnityEngine.Object.DestroyImmediate(ground);
            Steps(1);

            RetooledPhysics_TraceMark("destroy the first wheel");
            UnityEngine.Object.DestroyImmediate(first);
            Steps(1);

            RetooledPhysics_TraceMark("add a wheel again, then destroy the body");
            WheelCollider again = firstHolder.AddComponent<WheelCollider>();
            Steps(1);
            UnityEngine.Object.DestroyImmediate(chassis);
            Steps(1);

            RetooledPhysics_TraceMark("end");
            _ = again;
        }
    }
}
