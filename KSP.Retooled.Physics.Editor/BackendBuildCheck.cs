using System.IO;
using System.Linq;
using UnityEditor;
using UnityEditor.Build;
using UnityEditor.Build.Reporting;
using UnityEngine;

namespace KSP.Retooled.Physics.Editor
{
    /// <summary>
    /// Check that builds using the Unity Physics backend include a plugin configured to load at startup.
    /// </summary>
    internal sealed class BackendBuildCheck : IPreprocessBuildWithReport, IPostprocessBuildWithReport
    {
        public int callbackOrder => 0;

        public void OnPreprocessBuild(BuildReport report)
        {
            if (!BackendInfo.IsRetooled)
            {
                return;
            }

            BuildTarget target = report.summary.platform;
            PluginImporter plugin = PluginImporter.GetAllImporters()
                .FirstOrDefault(importer => Path.GetFileNameWithoutExtension(importer.assetPath) == "RetooledPhysics");
            if (plugin == null)
            {
                throw new BuildFailedException("the build selects RetooledPhysics, but the plugin is missing: run `RetooledCli native build`");
            }

            if (!plugin.isPreloaded || !plugin.GetCompatibleWithPlatform(target))
            {
                throw new BuildFailedException($"{plugin.assetPath} must load at startup on {target}: run `RetooledCli native build`");
            }
        }

        public void OnPostprocessBuild(BuildReport report)
        {
            if (!BackendInfo.IsRetooled)
            {
                return;
            }

            string playerFolder = Path.GetDirectoryName(report.summary.outputPath);
            bool shipped = Directory.EnumerateFiles(playerFolder, "RetooledPhysics.*", SearchOption.AllDirectories)
                .Any(file => Path.GetExtension(file) is ".dll" or ".so" or ".dylib");
            if (!shipped)
            {
                throw new BuildFailedException("the player selects RetooledPhysics, but the build has no such plugin");
            }

            Debug.Log("[RetooledPhysics] the player selects this backend and ships its plugin");
        }
    }
}
