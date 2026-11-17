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

        // Reject Unity's Null fallback, which can pass the backend-name check while simulating nothing.
        [Test]
        public void TheRunDidNotFallBackToTheNullBackend()
        {
            UnityEngine.IntegrationInfo info = UnityEngine.Physics.GetCurrentIntegrationInfo();

            Assert.That(info.isFallback, Is.False, $"Unity selected its fallback backend '{info.name}' (id 0x{info.id:X8})");
        }
    }
}
