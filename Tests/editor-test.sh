#!/bin/sh
# Run EditMode tests in a persistent editor, retaining Burst compilation work.
#
# Usage:
#   Tests/editor-test.sh <project folder> <run name> full|quick|retooled|filter=<filters> [category=<categories>] [NAME=VALUE...]
#   Tests/editor-test.sh <project folder> stop
#
# Selection:
#   full, quick and retooled use the run-suite.sh test selections.
#   filter=<filters> accepts semicolon-separated regular expressions.
#   Use $ for an exact suffix, for example filter=VesselForestBench.Measure$.
#   category=<categories> accepts semicolon-separated test categories.
#
# Environment:
#   NAME=VALUE applies to this run only. Changes trigger a script reload.
#   Omitted variables return to the values from editor startup.
#   EDITOR_ARGS adds editor startup arguments; stop the editor to change them.
#
# The first run installs Tests/EditorService~ and starts the editor.
# Later runs import changed scripts before testing.
# Do not open another Unity instance or use run-suite.sh on this project.
#
# Output:
#   Logs/<run name>-results.xml   Test results.
#   Logs/editor-service.log      Editor log.
#   TestService                  Service requests and state.

set -u

usage() {
    sed -n '2,26p' "$0"
    exit 2
}

[ $# -ge 2 ] || usage

here=$(cd "$(dirname "$0")" && pwd)
project=$(cd "$1" && pwd)
name=$2
shift 2
unity=${UNITY_EXE:-"/c/Program Files/Unity/Hub/Editor/6000.7.0b9/Editor/Unity.exe"}
service="$project/TestService"
mkdir -p "$service" "$project/Logs"

project_windows=$(cygpath -w "$project")

# Print the editor PID for this project, excluding import workers. Return nothing if absent.
find_editor() {
    powershell.exe -NoProfile -Command "\$p = '$project_windows'; Get-CimInstance Win32_Process -Filter 'Name=''Unity.exe''' |
        Where-Object { \$_.CommandLine -like ('*-projectPath ' + [char]34 + \$p + [char]34 + '*') -and \$_.CommandLine -notlike '*AssetImportWorker*' } |
        Select-Object -First 1 -ExpandProperty ProcessId" | tr -d '\r\n'
}

# Check pid.txt, falling back to a command-line search and updating stale PIDs.
running() {
    if [ -f "$service/pid.txt" ] && tasklist //FI "PID eq $(cat "$service/pid.txt")" //NH 2>/dev/null | grep -qi "unity"; then
        return 0
    fi

    found=$(find_editor)
    [ -n "$found" ] || return 1
    echo "$found" > "$service/pid.txt"
    return 0
}

if [ "$name" = stop ]; then
    if running; then
        : > "$service/quit.txt"
        waited=0
        while running && [ $waited -lt 120 ]; do
            sleep 1
            waited=$((waited + 1))
        done
    fi
    rm -f "$service"/*.txt "$service"/*.tmp
    echo "the editor for $project is stopped"
    exit 0
fi

[ $# -ge 1 ] || usage
mode=$1
shift
category=""
case "$mode" in
    full)
        filter='!KSP.Retooled.Physics.Tests.PlayModeLifecycleTests'
        ;;
    quick)
        filter='!KSP.Retooled.Physics.Tests.PlayModeLifecycleTests;!^Unity.Physics.Tests.;!^Unity.Numerics.'
        category='!Slow'
        ;;
    retooled)
        filter='!KSP.Retooled.Physics.Tests.PlayModeLifecycleTests;!^Unity.Physics.Tests.;!^Unity.Numerics.'
        ;;
    filter=*)
        filter=${mode#filter=}
        ;;
    *)
        usage
        ;;
esac

environment=""
for argument in "$@"; do
    case "$argument" in
        category=*)
            category=${argument#category=}
            ;;
        *=*)
            environment="${environment}env=$argument
"
            ;;
        *)
            usage
            ;;
    esac
done

# Install or update the service in the project. The editor compiles it like any other script.
installed="$project/Assets/RetooledTestService"
mkdir -p "$installed"
for file in "$here/EditorService~"/*; do
    if ! cmp -s "$file" "$installed/$(basename "$file")"; then
        cp "$file" "$installed/"
    fi
done

if ! running; then
    rm -f "$service"/*.txt "$service"/*.tmp
    arguments="-batchmode -nographics -projectPath \"$project_windows\" -logFile \"$project_windows\\Logs\\editor-service.log\" ${EDITOR_ARGS:-}"
    pid=$(RETOOLED_TEST_SERVICE=$(cygpath -w "$service") powershell.exe -NoProfile -Command \
        "(Start-Process -FilePath '$(cygpath -w "$unity")' -ArgumentList '$arguments' -WindowStyle Hidden -PassThru).Id" | tr -d '\r')
    echo "$pid" > "$service/pid.txt"
    echo "started the editor for $project (process $pid)"
fi

if [ -f "$service/request.txt" ] || [ -f "$service/active.txt" ]; then
    echo "the editor is busy with another run; wait for it, or stop the editor"
    exit 1
fi

id="$name-$(date +%s)"
results="$project/Logs/$name-results.xml"
rm -f "$results" "$service/done.txt"
{
    echo "id=$id"
    echo "results=$(cygpath -w "$results")"
    echo "filter=$filter"
    echo "category=$category"
    printf '%s' "$environment"
} > "$service/request.tmp"
mv "$service/request.tmp" "$service/request.txt"

start=$(date +%s)
while ! grep -qx "id=$id" "$service/done.txt" 2>/dev/null; do
    if ! running; then
        echo "the editor exited before the run finished; read $project/Logs/editor-service.log"
        rm -f "$service/request.txt" "$service/active.txt"
        exit 1
    fi
    sleep 2
done

end=$(date +%s)
echo "$name: $mode, $((end - start)) s from request to result"
grep -v '^id=' "$service/done.txt"
