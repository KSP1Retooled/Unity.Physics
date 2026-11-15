#!/bin/sh
# Runs the EditMode tests of this package in a Unity project, in batch mode, and prints how long each stage took.
#
# usage: Tests/run-suite.sh <project folder> <run name> [full|quick|retooled] [extra Unity arguments...]
#
#   full      All tests, with the same filter that the agents' runsub.sh scripts use. This is the default.
#   quick     The tests in the KSP.Retooled.Physics.Tests namespace without the "Slow" category. This leaves out
#             the upstream tests in the Unity.Physics.Tests and Unity.Numerics namespaces, and it leaves out
#             StepThreadTests.StepsKeepWorkingAcrossMonoAndBurstJobs.
#             Use it in the inner loop while you change the backend, and run "full" before you merge.
#   retooled  The tests in the KSP.Retooled.Physics.Tests namespace, including the "Slow" category.
#
# Set the solver path with the usual environment variables, for example: RETOOLED_VESSEL=1 Tests/run-suite.sh ...
#
# Unity exits with code 2 when some tests fail. The script prints the failed tests after the summary.
#
# The results go to <project>/Logs/<run name>-results.xml and the Unity log goes to <project>/Logs/<run name>.log.
# Never run two Unity instances on the same project at the same time.

set -u

if [ $# -lt 2 ]; then
    sed -n '2,19p' "$0"
    exit 2
fi

project=$1
name=$2
shift 2
mode=full
case "${1:-}" in
    full|quick|retooled)
        mode=$1
        shift
        ;;
esac

unity=${UNITY_EXE:-"/c/Program Files/Unity/Hub/Editor/6000.7.0b9/Editor/Unity.exe"}
logs="$project/Logs"
results="$logs/$name-results.xml"
log="$logs/$name.log"
mkdir -p "$logs"
rm -f "$results"

case "$mode" in
    full)
        set -- -testFilter "!KSP.Retooled.Physics.Tests.PlayModeLifecycleTests" "$@"
        ;;
    # Use only exclusions here. A filter that includes a fixture by name also runs the [Explicit] tests in it,
    # such as VesselSolverBench, which takes half an hour.
    quick)
        set -- -testFilter "!KSP.Retooled.Physics.Tests.PlayModeLifecycleTests;!^Unity.Physics.Tests.;!^Unity.Numerics." -testCategory "!Slow" "$@"
        ;;
    retooled)
        set -- -testFilter "!KSP.Retooled.Physics.Tests.PlayModeLifecycleTests;!^Unity.Physics.Tests.;!^Unity.Numerics." "$@"
        ;;
esac

start=$(date +%s)
"$unity" -batchmode -nographics -projectPath "$project" -runTests -testPlatform EditMode \
    -testResults "$results" -logFile "$log" "$@" >/dev/null 2>&1
code=$?
end=$(date +%s)

echo "$name: $mode, exit code $code, $((end - start)) s from start to exit"
if [ ! -f "$results" ]; then
    echo "no results file; read $log"
    exit 1
fi

grep -o 'total="[0-9]*" passed="[0-9]*" failed="[0-9]*"' "$results" | head -1
grep -o 'duration="[0-9.]*"' "$results" | head -1 | sed 's/duration="\([0-9]*\).*"/tests: \1 s/'
grep -o 'Loading completed in [0-9.]* seconds' "$log" | head -1 | sed 's/Loading completed in/editor start-up and project load:/'
grep -o 'script compilation time: [0-9.]*s' "$log" | head -1
grep -c '^Domain Reload Profiling' "$log" | sed 's/$/ domain reloads/'
grep -o '<test-case [^>]*result="Failed"[^>]*>' "$results" | grep -o 'fullname="[^"]*"' | head -20
