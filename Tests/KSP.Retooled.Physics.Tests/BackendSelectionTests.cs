using System;
using KSP.Retooled.Systems;
using NUnit.Framework;
using Unity.Entities;

namespace KSP.Retooled.PhysicsBackend.Tests
{
    /// <summary>The backend the run selected, and the part of Unity Physics the game must not run.</summary>
    public class BackendSelectionTests
    {
        [Test]
        public void TheCurrentBackendIsOneWeShip()
        {
            string name = UnityEngine.Physics.GetCurrentIntegrationInfo().name;

            Assert.That(BackendInfo.IsRetooled, Is.EqualTo(name == "RetooledPhysics"), name);
        }

        [Test]
        public void TheDefaultWorldHasNoUnityPhysicsSystems()
        {
            foreach (Type system in GameWorldBootstrap.GameSystems())
            {
                Assert.That(system.Assembly.GetName().Name, Does.Not.StartWith("Unity.Physics"), system.FullName);
            }

            Assert.That(GameWorldBootstrap.GameSystems(), Does.Contain(typeof(SimulationSystemGroup)));
        }
    }
}
