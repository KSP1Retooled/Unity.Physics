using UnityEditor;
using UnityEngine;

namespace KSP.Retooled.PhysicsBackend.Editor
{
    /// <summary>
    /// The import settings of the RetooledPhysics plugin, which `RetooledCli native build` installs. It must be preloaded,
    /// or it would load after Unity has picked its physics backend, and only the host's editor and player use it.
    /// </summary>
    public static class NativePluginImporter
    {
        private const string Folder = "Assets/Plugins/x86_64/";

        public static void Configure()
        {
            string extension = Application.platform switch
            {
                RuntimePlatform.WindowsEditor => ".dll",
                RuntimePlatform.OSXEditor => ".dylib",
                _ => ".so",
            };

            string path = Folder + "RetooledPhysics" + extension;
            AssetDatabase.ImportAsset(path);
            PluginImporter importer = AssetImporter.GetAtPath(path) as PluginImporter;
            if (importer == null)
            {
                Debug.LogError($"[RetooledPhysics] {path} did not import as a plugin");
                return;
            }

            importer.isPreloaded = true;
            importer.SetCompatibleWithAnyPlatform(false);
            importer.SetCompatibleWithEditor(true);
            importer.SetEditorData("CPU", "x86_64");
            importer.SetEditorData("OS", EditorOperatingSystem());

            BuildTarget player = PlayerTarget();
            foreach (BuildTarget target in new[] { BuildTarget.StandaloneWindows64, BuildTarget.StandaloneOSX, BuildTarget.StandaloneLinux64 })
            {
                importer.SetCompatibleWithPlatform(target, target == player);
                importer.SetPlatformData(target, "CPU", target == player ? "x86_64" : "None");
            }

            importer.SaveAndReimport();
        }

        private static string EditorOperatingSystem()
        {
            return Application.platform switch
            {
                RuntimePlatform.WindowsEditor => "Windows",
                RuntimePlatform.OSXEditor => "OSX",
                _ => "Linux",
            };
        }

        private static BuildTarget PlayerTarget()
        {
            return Application.platform switch
            {
                RuntimePlatform.WindowsEditor => BuildTarget.StandaloneWindows64,
                RuntimePlatform.OSXEditor => BuildTarget.StandaloneOSX,
                _ => BuildTarget.StandaloneLinux64,
            };
        }
    }
}
