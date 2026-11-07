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

    }
}
