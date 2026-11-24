using System;
using System.IO;
using NUnit.Framework;
using UnityEngine;

namespace KSP.Retooled.Physics.Tests.Tools
{
    public class ReplayFlightRecording
    {
        [Test, Explicit("Set RETOOLED_REPLAY_FILE to a flight recording.")]
        public void Replay()
        {
            string path = Environment.GetEnvironmentVariable("RETOOLED_REPLAY_FILE");
            Assert.That(path, Is.Not.Null.And.Not.Empty);
            string stop = Environment.GetEnvironmentVariable("RETOOLED_REPLAY_STOP");
            long stopAt = string.IsNullOrEmpty(stop) ? long.MaxValue : long.Parse(stop);
            using var summary = new StringWriter();
            try
            {
                Assert.That(FlightReplay.Run(path, stopAt, summary), Is.GreaterThan(0));
            }
            finally
            {
                Debug.Log(summary.ToString());
            }
        }
    }
}
