#!/bin/sh
# Runs EditMode tests in an editor that stays open between runs, so that a run doesn't pay for the editor's start-up
# (30 to 60 s) every time, and the editor's Burst compiler keeps its work between runs.
#
# usage: Tests/editor-test.sh <project folder> <run name> full|quick|retooled|filter=<filters> [category=<categories>] [NAME=VALUE...]
#        Tests/editor-test.sh <project folder> stop
#
#   full, quick and retooled pick the same tests as in Tests/run-suite.sh. filter=<filters> takes test filters
#   separated by ';', as Unity's -testFilter argument does, for example
#   'filter=KSP.Retooled.Physics.Tests.VesselForestBench.Measure$'. Each filter is a regular expression that can match
#   any part of a test's full name, so without the "$" that filter also runs VesselForestBench.MeasureStages.
#   category=<categories> works like -testCategory.
#   NAME=VALUE sets an environment variable for this run only, for example RETOOLED_VESSEL=0. When the set of variables
#   differs from the last run, the editor reloads its scripts before the run, so that settings read at start-up see the
#   new values.
#
# The first run installs the service from Tests/EditorService~ into <project>/Assets/RetooledTestService and starts the
# editor in batch mode. Each later run reuses that editor: the editor imports and compiles the changed scripts, then
# runs the tests. EDITOR_ARGS adds arguments to the editor's command line when this script starts it, for example
# EDITOR_ARGS=-burst-disable-compilation; stop the editor to change them.
#
# Stop the editor with "stop" when you are done with work for now. Never run Tests/run-suite.sh or another Unity
# instance on a project while its editor is open.
#
# The results go to <project>/Logs/<run name>-results.xml, the editor's log to <project>/Logs/editor-service.log, and
# the service's files to <project>/TestService.

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

# Prints the process id of the editor that has this project open, found by its command line, or nothing. An asset
# import worker also has the project on its command line, so the search leaves workers out. The search takes a few
# seconds, so running() uses it only when the process in pid.txt doesn't answer.
find_editor() {
    powershell.exe -NoProfile -Command "\$p = '$project_windows'; Get-CimInstance Win32_Process -Filter 'Name=''Unity.exe''' |
        Where-Object { \$_.CommandLine -like ('*-projectPath ' + [char]34 + \$p + [char]34 + '*') -and \$_.CommandLine -notlike '*AssetImportWorker*' } |
        Select-Object -First 1 -ExpandProperty ProcessId" | tr -d '\r\n'
}

# Succeeds when the project's editor is running. The service writes the editor's process id into pid.txt when its
# scripts load. If that process is gone, the editor may still be running under another id, for example when pid.txt
# is stale; the command-line search then finds it, and pid.txt is corrected.
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
