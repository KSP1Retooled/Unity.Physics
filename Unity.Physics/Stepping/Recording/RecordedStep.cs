using System.Collections.Generic;

namespace Unity.Physics.Stepping.Recording
{
    internal sealed class RecordedStep
    {
        internal long Number;
        internal float DeltaTime;
        internal uint ContactStep;
        internal byte[] Input;
        internal RecordedBody[] Bodies;
        internal readonly HashSet<byte[]> Geometry = new HashSet<byte[]>();
        internal string Trigger;
    }
}
