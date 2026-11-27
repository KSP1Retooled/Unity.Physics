using Unity.Burst;

namespace Unity.Physics.Configuration
{
    /// <summary>
    /// Stiffness caps on the hard rows of vessel links: loop closures and yielding edges. A hard row's stiffness comes from
    /// the reduced mass of the two sections it joins (about 1.7e9 kN/m for two 88 t sections). When a loop spans several
    /// sections, its hard rows fight each other over misfits well under a millimetre, and the force is that stiffness
    /// times the misfit: hundreds of meganewtons, enough to keep a resting ship's joints far over their break strength.
    /// A cap bounds that force. A capped row keeps the hard damping ratio at its own, lower frequency.
    /// </summary>
    internal static class VesselRowSwitches
    {
        /// <summary>
        /// The default linear cap, in kN/m. A misfit of 1 mm then costs at most 250 kN. On the 713-part Pegasus, this
        /// keeps every resting joint under half its break force; a cap of 1e6 let light-part chains creep past it.
        /// </summary>
        internal const double DefaultLinearCap = 2.5e5;

        /// <summary>The default angular cap, in kN m per radian.</summary>
        internal const double DefaultAngularCap = 2.5e5;

        private struct LinearCapKey { }

        private struct AngularCapKey { }

        private static readonly SharedStatic<double> LinearCapShared = SharedStatic<double>.GetOrCreate<LinearCapKey>();
        private static readonly SharedStatic<double> AngularCapShared = SharedStatic<double>.GetOrCreate<AngularCapKey>();

        /// <summary>
        /// The largest stiffness, in kN/m, of a link's hard linear row. Setting 0 restores the default; setting a
        /// negative value turns the cap off, and the getter then returns 0.
        /// </summary>
        internal static double LinearCap
        {
            get => Resolve(LinearCapShared.Data, DefaultLinearCap);
            set => LinearCapShared.Data = value;
        }

        /// <summary>
        /// The largest stiffness, in kN m per radian, of a link's hard angular row. Setting 0 restores the default; setting
        /// a negative value turns the cap off, and the getter then returns 0.
        /// </summary>
        internal static double AngularCap
        {
            get => Resolve(AngularCapShared.Data, DefaultAngularCap);
            set => AngularCapShared.Data = value;
        }

        private static double Resolve(double stored, double fallback) => stored > 0.0 ? stored : stored < 0.0 ? 0.0 : fallback;
    }
}
