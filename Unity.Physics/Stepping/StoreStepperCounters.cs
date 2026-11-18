namespace Unity.Physics.Stepping
{
    internal struct StoreStepperCounters
    {
        /// <summary>Bodies whose solved pose or velocity had to be discarded since this world was created.</summary>
        internal int NonFiniteBodies;
    }
}
