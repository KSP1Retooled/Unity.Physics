#!/bin/sh
# Run package EditMode tests in a Unity project and report stage timings.
#
# Usage:
#   Tests/run-suite.sh <project folder> <run name> [full|quick|retooled] [extra Unity arguments...]
#
# Selection:
#   full      All tests except PlayModeLifecycleTests (default).
#   quick     Retooled tests excluding the Slow category.
#   retooled  Retooled tests including the Slow category.
#   quick and retooled also exclude PlayModeLifecycleTests and upstream suites.
#
# Configure solver paths through environment variables.
# Unity returns exit code 2 for failed tests; the script lists failures.
#
# Output:
#   Logs/<run name>-results.xml   Test results.
#   Logs/<run name>.log           Unity log.
# Do not run two Unity instances on the same project.

set -u

if [ $# -lt 2 ]; then
    sed -n '2,19p' "$0"
    exit 2
fi

here=$(cd "$(dirname "$0")" && pwd)
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

# Exclude the upstream fixtures by name so new Retooled fixtures are selected automatically.
# NUnit keeps explicit tests out of runs that use exclusions.
upstream=$(sed -e '/^#/d' -e '/^$/d' -e 's/[.]/\\./g' "$here/upstream-tests.txt" | paste -sd '|' -)
retooled_filter="!KSP.Retooled.Physics.Tests.Host.PlayModeLifecycleTests;!^(?:$upstream)(?:\\.|\\(|$)"

case "$mode" in
    full)
        set -- -testFilter "!KSP.Retooled.Physics.Tests.Host.PlayModeLifecycleTests" "$@"
        ;;
    quick)
        set -- -testFilter "$retooled_filter" -testCategory "!Slow" "$@"
        ;;
    retooled)
        set -- -testFilter "$retooled_filter" "$@"
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
