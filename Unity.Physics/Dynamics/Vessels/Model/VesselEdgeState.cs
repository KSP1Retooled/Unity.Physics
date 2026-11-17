namespace Unity.Physics.Dynamics.Vessels.Model
{
    /// <summary>The state of the edge joining a part to its parent in a vessel's part tree.</summary>
    internal enum VesselEdgeState : byte
    {
        /// <summary>
        /// Part rigidly joined to its parent's section. Omit solver rows and recover edge loads by inverse
        /// dynamics.
        /// </summary>
        Locked,

        /// <summary>
        /// Uses three hard linear rows and three angular rows with stiff, damped springs: one twist and two swings.
        /// </summary>
        Yielding,

        /// <summary>
        /// A servo or hinge with five hard rows. The twist row uses a spring and damper with
        /// <see cref="VesselEdge.DriveTarget"/> as the target angle.
        /// </summary>
        FreeAxis,

        /// <summary>
        /// The edge is broken, so the part and its descendants can move independently of its former parent.
        /// </summary>
        Broken,
    }
}
