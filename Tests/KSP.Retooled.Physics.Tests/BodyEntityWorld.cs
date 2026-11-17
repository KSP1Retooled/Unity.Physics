using System;
using System.Collections.Generic;
using NUnit.Framework;
using Unity.Entities;

namespace KSP.Retooled.Physics.Tests
{
    /// <summary>
    /// Create the Entities world and stepping systems required by edit-mode backend tests.
    /// </summary>
    [SetUpFixture]
    public class BodyEntityWorld
    {
        internal static World World { get; private set; }

        [OneTimeSetUp]
        public void CreateWorld()
        {
            World = new World("Rigid Bodies");
            DefaultWorldInitialization.AddSystemsToRootLevelSystemGroups(World, PhysicsSystems());
        }

        [OneTimeTearDown]
        public void DisposeWorld()
        {
            World.Dispose();
            World = null;
        }

        private static List<Type> PhysicsSystems()
        {
            List<Type> systems = new List<Type>();
            foreach (Type system in DefaultWorldInitialization.GetAllSystems(WorldSystemFilterFlags.Default))
            {
                string assembly = system.Assembly.GetName().Name;
                if (assembly.StartsWith("Unity.Physics", StringComparison.Ordinal)
                    || assembly == "Unity.Entities"
                    || system.Assembly == typeof(BackendHost).Assembly)
                {
                    systems.Add(system);
                }
            }

            return systems;
        }
    }
}
