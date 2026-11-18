using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;
using Unity.Burst;
using Unity.Collections;
using UnityEditor;
using UnityEditor.TestTools.TestRunner.Api;
using UnityEngine;

namespace KSP.Retooled.Physics.EditorService
{
    /// <summary>
    /// Run test requests in a persistent editor when RETOOLED_TEST_SERVICE names a folder.
    /// Read request.txt and write done.txt. Request fields are id, results, filter, category and repeatable
    /// env=NAME=VALUE.
    /// Import changed scripts before each run and reload when environment settings change.
    /// Restore omitted environment settings to their startup values. quit.txt closes the editor.
    /// Use normal Burst safety checks unless a request sets RETOOLED_TEST_SAFETY=1.
    /// </summary>
    [InitializeOnLoad]
    internal static class RetooledTestService
    {
        private const string StageKey = "RetooledTestService.Stage";
        private const string AppliedKey = "RetooledTestService.Applied";
        private const string OriginalPrefix = "RetooledTestService.Original.";
        private const string Unset = "\u0001";

        private static readonly string Folder = Environment.GetEnvironmentVariable("RETOOLED_TEST_SERVICE");
        private static double nextPoll;
        private static NativeLeakDetectionMode? previousLeakMode;

        static RetooledTestService()
        {
            // Import workers inherit the service environment but must not claim requests or replace the
            // editor's pid.txt.
            if (string.IsNullOrEmpty(Folder) || AssetDatabase.IsAssetImportWorkerProcess())
            {
                return;
            }

            Directory.CreateDirectory(Folder);
            File.WriteAllText(Path.Combine(Folder, "pid.txt"), Process.GetCurrentProcess().Id.ToString(CultureInfo.InvariantCulture));

            // The test framework calls registered callbacks after a domain reload only if they are registered again
            // when the scripts load, which this static constructor does.
            ScriptableObject.CreateInstance<TestRunnerApi>().RegisterCallbacks(new Callbacks());
            EditorApplication.update += Poll;
        }

        private static string ActivePath => Path.Combine(Folder, "active.txt");

        private static void Poll()
        {
            if (EditorApplication.timeSinceStartup < nextPoll)
            {
                return;
            }

            nextPoll = EditorApplication.timeSinceStartup + 0.2;
            if (EditorApplication.isCompiling || EditorApplication.isUpdating)
            {
                return;
            }

            string quit = Path.Combine(Folder, "quit.txt");
            if (File.Exists(quit))
            {
                File.Delete(quit);
                EditorApplication.Exit(0);
                return;
            }

            string stage = SessionState.GetString(StageKey, "");
            if (stage == "running")
            {
                return;
            }

            if (!File.Exists(ActivePath))
            {
                string request = Path.Combine(Folder, "request.txt");
                if (!File.Exists(request))
                {
                    return;
                }

                File.Move(request, ActivePath);
                stage = "";
            }

            Request current = Request.Read(ActivePath);
            if (stage == "")
            {
                string applied = ApplyEnvironment(current.Environment);
                bool changed = applied != SessionState.GetString(AppliedKey, "");
                SessionState.SetString(AppliedKey, applied);
                SessionState.SetString(StageKey, "refreshed");
                AssetDatabase.Refresh();
                if (changed)
                {
                    EditorUtility.RequestScriptReload();
                }

                // A domain reload that the refresh asked for starts on a later editor tick. Waiting a second lets it
                // start before the run does; after the reload this method continues at the "refreshed" stage.
                nextPoll = EditorApplication.timeSinceStartup + 1.0;
                return;
            }

            if (EditorUtility.scriptCompilationFailed)
            {
                Finish(current, "compile failed; search the editor log for \"error CS\"\n");
                return;
            }

            SessionState.SetString(StageKey, "running");
            bool forceSafety = current.Environment.TryGetValue("RETOOLED_TEST_SAFETY", out string safety) && safety == "1";
            BurstCompilerOptions options = BurstCompiler.Options;
            try
            {
                SetBurstSafety(options, forceSafety);
            }
            catch (Exception exception)
            {
                UnityEngine.Debug.LogException(exception);
                Finish(current, "could not configure Burst safety checks\n");
                return;
            }

            if (forceSafety)
            {
                previousLeakMode = NativeLeakDetection.Mode;
                NativeLeakDetection.Mode = NativeLeakDetectionMode.EnabledWithStackTrace;
                UnityEngine.Debug.Log($"[RetooledTestService] Burst compilation {options.EnableBurstCompilation}, forced safety {options.ForceEnableBurstSafetyChecks}, leak detection {NativeLeakDetection.Mode}");
            }
            var filter = new Filter
            {
                testMode = TestMode.EditMode,
                groupNames = current.Filters,
                categoryNames = current.Categories,
            };
            ScriptableObject.CreateInstance<TestRunnerApi>().Execute(new ExecutionSettings(filter));
        }

        private static void SetBurstSafety(BurstCompilerOptions options, bool forceSafety)
        {
            if (options.EnableBurstSafetyChecks && options.ForceEnableBurstSafetyChecks == forceSafety)
            {
                return;
            }

            // Burst's editor listener writes the settings to shared EditorPrefs. Bypass that listener so a
            // stopped test run cannot change what another editor reads. The setters still recompile Burst code.
            PropertyInfo optionsChanged = typeof(BurstCompilerOptions).GetProperty(
                "OptionsChanged", BindingFlags.Instance | BindingFlags.NonPublic)
                ?? throw new MissingMemberException(typeof(BurstCompilerOptions).FullName, "OptionsChanged");
            object listener = optionsChanged.GetValue(options);
            optionsChanged.SetValue(options, null);
            try
            {
                options.ForceEnableBurstSafetyChecks = forceSafety;
                options.EnableBurstSafetyChecks = true;
            }
            finally
            {
                optionsChanged.SetValue(options, listener);
            }
        }

        /// <summary>
        /// Apply request environment settings and restore omitted settings to startup values.
        /// Return a sorted representation for detecting changes.
        /// </summary>
        private static string ApplyEnvironment(SortedDictionary<string, string> wanted)
        {
            foreach (string line in SessionState.GetString(AppliedKey, "").Split('\n'))
            {
                int equals = line.IndexOf('=');
                if (equals > 0 && !wanted.ContainsKey(line.Substring(0, equals)))
                {
                    string name = line.Substring(0, equals);
                    string original = SessionState.GetString(OriginalPrefix + name, Unset);
                    Environment.SetEnvironmentVariable(name, original == Unset ? null : original);
                }
            }

            foreach (KeyValuePair<string, string> pair in wanted)
            {
                if (SessionState.GetString(OriginalPrefix + pair.Key, null) == null)
                {
                    SessionState.SetString(OriginalPrefix + pair.Key, Environment.GetEnvironmentVariable(pair.Key) ?? Unset);
                }

                Environment.SetEnvironmentVariable(pair.Key, pair.Value);
            }

            return string.Join("\n", wanted.Select(pair => pair.Key + "=" + pair.Value));
        }

        /// <summary>Writes done.txt with the request's id and the summary, and makes the service ready for the next request.</summary>
        private static void Finish(Request request, string summary)
        {
            if (previousLeakMode is { } leakMode)
            {
                NativeLeakDetection.Mode = leakMode;
                previousLeakMode = null;
            }

            string done = Path.Combine(Folder, "done.txt");
            string temporary = done + ".tmp";
            File.WriteAllText(temporary, "id=" + request.Id + "\n" + summary);
            File.Delete(done);
            File.Move(temporary, done);
            File.Delete(ActivePath);
            SessionState.SetString(StageKey, "");
        }

        private sealed class Request
        {
            public string Id = "";
            public string Results = "";
            public string[] Filters;
            public string[] Categories;
            public readonly SortedDictionary<string, string> Environment = new SortedDictionary<string, string>(StringComparer.Ordinal);

            public static Request Read(string path)
            {
                var request = new Request();
                foreach (string line in File.ReadAllLines(path))
                {
                    int equals = line.IndexOf('=');
                    if (equals <= 0)
                    {
                        continue;
                    }

                    string key = line.Substring(0, equals);
                    string value = line.Substring(equals + 1);
                    switch (key)
                    {
                        case "id":
                            request.Id = value;
                            break;
                        case "results":
                            request.Results = value;
                            break;
                        case "filter":
                            request.Filters = Split(value);
                            break;
                        case "category":
                            request.Categories = Split(value);
                            break;
                        case "env":
                            int inner = value.IndexOf('=');
                            if (inner > 0)
                            {
                                request.Environment[value.Substring(0, inner)] = value.Substring(inner + 1);
                            }

                            break;
                    }
                }

                return request;
            }

            private static string[] Split(string value)
            {
                string[] parts = value.Split(new[] { ';' }, StringSplitOptions.RemoveEmptyEntries);
                return parts.Length == 0 ? null : parts;
            }
        }

        private sealed class Callbacks : ICallbacks
        {
            public void RunStarted(ITestAdaptor testsToRun)
            {
            }

            public void RunFinished(ITestResultAdaptor result)
            {
                if (!File.Exists(ActivePath) || SessionState.GetString(StageKey, "") != "running")
                {
                    return;
                }

                Request request = Request.Read(ActivePath);
                if (request.Results.Length > 0)
                {
                    TestRunnerApi.SaveResultToFile(result, request.Results);
                }

                // The same lines that Tests/run-suite.sh prints, so that scripts that read one can read the other.
                var summary = new StringBuilder();
                int total = result.PassCount + result.FailCount + result.SkipCount + result.InconclusiveCount;
                summary.Append($"total=\"{total}\" passed=\"{result.PassCount}\" failed=\"{result.FailCount}\"\n");
                summary.Append($"tests: {(int)result.Duration} s\n");
                foreach (string name in FailedTests(result).Take(20))
                {
                    summary.Append($"fullname=\"{name}\"\n");
                }

                Finish(request, summary.ToString());
            }

            public void TestStarted(ITestAdaptor test)
            {
            }

            public void TestFinished(ITestResultAdaptor result)
            {
            }

            private static IEnumerable<string> FailedTests(ITestResultAdaptor result)
            {
                if (!result.HasChildren)
                {
                    if (result.TestStatus == TestStatus.Failed)
                    {
                        yield return result.FullName;
                    }

                    yield break;
                }

                foreach (ITestResultAdaptor child in result.Children)
                {
                    foreach (string name in FailedTests(child))
                    {
                        yield return name;
                    }
                }
            }
        }
    }
}
