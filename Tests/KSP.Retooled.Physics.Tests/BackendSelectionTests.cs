using NUnit.Framework;

namespace KSP.Retooled.Physics.Tests
{
    /// <summary>The backend the run selected.</summary>
    public class BackendSelectionTests
    {
        [Test]
        public void TheCurrentBackendIsOneWeShip()
        {
            string name = UnityEngine.Physics.GetCurrentIntegrationInfo().name;

            Assert.That(BackendInfo.IsRetooled, Is.EqualTo(name == "RetooledPhysics"), name);
        }

        /// <summary>
        /// Unity falls back to its Null backend, which simulates nothing, when it can't find the backend that the project
        /// asks for, and it only logs a message. That happens, for example, when the plugin isn't marked to load on
        /// startup, or in an editor older than 6000.7 that selects the backend before it loads the plugins. The test
        /// above passes with the Null backend too, because the Null backend is neither ours nor named RetooledPhysics.
        /// </summary>
        [Test]
        public void TheRunDidNotFallBackToTheNullBackend()
        {
            UnityEngine.IntegrationInfo info = UnityEngine.Physics.GetCurrentIntegrationInfo();

            Assert.That(info.isFallback, Is.False, $"Unity selected its fallback backend '{info.name}' (id 0x{info.id:X8})");
        }
    }
}
