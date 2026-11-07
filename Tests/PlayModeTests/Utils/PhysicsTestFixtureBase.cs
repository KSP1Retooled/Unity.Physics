using System.Collections.Generic;
using NUnit.Framework;

namespace Unity.Physics.Tests.Utils
{
    /// <summary>
    /// Destroys Unity objects registered during a test, in reverse order of registration.
    /// Fixtures that already derive from another base class can own one of these and call <see cref="Run"/> from their own [TearDown].
    /// </summary>
    sealed class TestCleanup
    {
        readonly List<UnityEngine.Object> m_Objects = new List<UnityEngine.Object>();

        public T Add<T>(T obj) where T : UnityEngine.Object
        {
            m_Objects.Add(obj);
            return obj;
        }

        public void Run()
        {
            for (int i = m_Objects.Count - 1; i >= 0; i--)
            {
                if (m_Objects[i])
                {
                    UnityEngine.Object.DestroyImmediate(m_Objects[i]);
                }
            }
            m_Objects.Clear();
        }
    }

    /// <summary>
    /// Base fixture that destroys the Unity objects registered with <see cref="AddForTearDown{T}"/> after every test, including failed ones.
    /// </summary>
    abstract class PhysicsTestFixtureBase
    {
        readonly TestCleanup m_Cleanup = new TestCleanup();

        protected T AddForTearDown<T>(T obj) where T : UnityEngine.Object => m_Cleanup.Add(obj);

        [TearDown]
        public void TearDownBase() => m_Cleanup.Run();
    }
}
