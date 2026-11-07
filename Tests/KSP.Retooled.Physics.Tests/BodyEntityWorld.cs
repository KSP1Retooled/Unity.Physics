using KSP.Retooled.Systems.Physics;
using NUnit.Framework;
using Unity.Entities;

namespace KSP.Retooled.Physics.Tests
{
    /// <summary>
    /// Gives rigid bodies an Entities world for every test in this namespace. Edit mode never starts Entities, and bodies
    /// live on entities in the game, so without this the tests would miss the path the game takes.
    /// </summary>
    [SetUpFixture]
    public class BodyEntityWorld
    {
        internal static World World { get; private set; }

        [OneTimeSetUp]
        public void CreateWorld()
        {
            World = new World("Rigid Bodies");
            World.GetOrCreateSystemManaged<BodyBindingSystem>();
        }

        [OneTimeTearDown]
        public void DisposeWorld()
        {
            World.Dispose();
            World = null;
        }
    }
}
