using System;
using System.Collections.Generic;
using KSP.Retooled.Systems.Physics;
using NUnit.Framework;
using Unity.Entities;

namespace KSP.Retooled.Physics.Tests
{
    /// <summary>
    /// Creates an Entities world and the systems that step rigid bodies for all tests in this namespace. Edit mode does
    /// not start Entities automatically, and stepping requires the world's physics system group.
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
                    || system.Namespace == typeof(BodyBindingSystem).Namespace)
                {
                    systems.Add(system);
                }
            }

            return systems;
        }
    }
}
