using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using UnityEditor;
using UnityEditor.TestTools.TestRunner.Api;
using UnityEngine;

namespace KSP.Retooled.Physics.EditorService
{
    /// <summary>
    /// Runs test requests in an editor that stays open, so that each run pays for the editor's start-up only once.
    ///
    /// The service does nothing unless the editor was started with the environment variable RETOOLED_TEST_SERVICE set
    /// to a folder. Tests/editor-test.sh starts the editor, writes request.txt into that folder, and waits for
    /// done.txt. A request holds lines of the form key=value:
    /// <list type="bullet">
    /// <item>id: the name of the request, which done.txt repeats.</item>
    /// <item>results: the path of the results file to write.</item>
    /// <item>filter: test filters separated by ';', with the same meaning as Unity's -testFilter argument.</item>
    /// <item>category: categories separated by ';', with the same meaning as Unity's -testCategory argument.</item>
    /// <item>env: an environment variable to set for this run, as NAME=VALUE. The line may repeat.</item>
    /// </list>
    /// Before each run the service imports changed scripts, so that the run tests the current source. When the set of
    /// environment variables differs from the last run, the service also reloads the scripts, so that settings that the
    /// code reads at start-up see the new values. A variable that the last run set and this run doesn't set goes back
    /// to the value that it had when the editor started. A file named quit.txt in the folder closes the editor.
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

        static RetooledTestService()
        {
            // The editor's asset import workers are Unity processes of their own. They inherit the editor's environment
            // and load the same scripts, so without the second check a worker would write its own process id into
            // pid.txt, and could take a request or a quit.txt that was meant for the editor.
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
            var filter = new Filter
            {
                testMode = TestMode.EditMode,
                groupNames = current.Filters,
                categoryNames = current.Categories,
            };
            ScriptableObject.CreateInstance<TestRunnerApi>().Execute(new ExecutionSettings(filter));
        }

        /// <summary>
        /// Sets the request's environment variables, and puts back those that an earlier request set and this one
        /// doesn't. Returns the variables as one sorted string, so that the caller can tell whether the set changed.
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
