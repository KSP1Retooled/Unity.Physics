using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.LowLevel;

namespace KSP.Retooled.PhysicsBackend.Tests
{
    /// <summary>
    /// Records, at the end of each frame, where a transform and its body are. It joins the player loop rather than being
    /// a component, because play mode does not attach behaviours from the editor assembly that tests compile into.
    /// </summary>
    public sealed class InterpolationSampler : IDisposable
    {
        public struct Sample
        {
            public float Time;
            public float FixedTime;
            public float Transform;
            public float Body;
        }

        public readonly List<Sample> Samples = new List<Sample>();

        private readonly Rigidbody body;
        private readonly PlayerLoopSystem previousLoop;

        public InterpolationSampler(Rigidbody body)
        {
            this.body = body;
            previousLoop = PlayerLoop.GetCurrentPlayerLoop();

            PlayerLoopSystem loop = previousLoop;
            PlayerLoopSystem system = new PlayerLoopSystem { type = typeof(InterpolationSampler), updateDelegate = Record };
            loop.subSystemList = loop.subSystemList.Append(system).ToArray();
            PlayerLoop.SetPlayerLoop(loop);
        }

        public void Dispose()
        {
            PlayerLoop.SetPlayerLoop(previousLoop);
        }

        private void Record()
        {
            if (body == null)
            {
                return;
            }

            Samples.Add(new Sample
            {
                Time = Time.time,
                FixedTime = Time.fixedTime,
                Transform = body.transform.position.x,
                Body = body.position.x,
            });
        }
    }
}
