using System;
using System.Collections.Generic;
using KSP.Retooled.Systems.Physics;
using NUnit.Framework;
using Unity.Entities;

namespace KSP.Retooled.Physics.Tests
{
    /// <summary>
    /// Gives rigid bodies an Entities world for every test in this namespace, with the systems that step them. Edit mode
    /// never starts Entities, and only a world's physics system group can step.
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
