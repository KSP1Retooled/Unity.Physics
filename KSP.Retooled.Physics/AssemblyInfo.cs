using System.Runtime.CompilerServices;

// The project's tests and build steps reach the backend's store and kernels directly.
[assembly: InternalsVisibleTo("Assembly-CSharp-Editor")]
[assembly: InternalsVisibleTo("KSP.Retooled.Physics.Editor")]
[assembly: InternalsVisibleTo("KSP.Retooled.Physics.Tests")]
