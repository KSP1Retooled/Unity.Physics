#!/bin/sh
# Times the whole-step bench (VesselForestBench.Measure) in two projects, the base and a change, in alternating rounds,
# and prints a summary of the two.
#
# usage: Tests/bench-ab.sh <base project> <change project> <output folder> [rounds] [crafts] [scenes] [threads]
#
#   rounds   the number of rounds, 4 by default. Each round runs the bench once in each project.
#   crafts   RETOOLED_BENCH_CRAFT, by default tu160,synthetic4500,tu160x4,tu160:128x16
#   scenes   RETOOLED_BENCH_SCENE, by default floating,ground
#   threads  RETOOLED_BENCH_THREADS, by default 8,1
#
#   AB_STEPS (default 60) and AB_BLOCKS (default 2) set RETOOLED_BENCH_STEPS and RETOOLED_BENCH_BLOCKS, and AB_PATHS
#   (default compiled) sets RETOOLED_BENCH_PATHS. AB_KEEP=1 leaves both editors open at the end.
#
# The script first stops both projects' editors and starts them again, at the same time, with a short test, so that
# neither editor has run anything else before it is timed. Both editors stay open for the whole run. Each run of the
# bench builds each scene once and measures it with every number of threads in turn (RETOOLED_BENCH_SHARE=1), because
# building a scene of 10,000 bodies takes much longer than measuring 60 steps of it. Every run in one project has the
# same environment variables, so that the editor doesn't reload its scripts between runs. The rounds alternate which
# project goes first (base and change, then change and base), so that a slow drift of the machine's speed affects both
# sides alike.
#
# Each run's table goes to <output folder>/<side>-<round>.txt, where side is base or change. The summary at the end
# (Tests/bench-ab-summary.py) gives, per scene and number of threads, the minimum and the median of the runs' medians
# on each side, and the CPU load that other processes put on the machine during each run.
#
# The filter ends in "$" because Unity's test filters are regular expressions: without it,
# VesselForestBench.Measure also matches VesselForestBench.MeasureStages, which then runs after it and takes longer
# than Measure does.

set -u

usage() {
    sed -n '2,29p' "$0"
    exit 2
}

[ $# -ge 3 ] || usage

here=$(cd "$(dirname "$0")" && pwd)
tester=${EDITOR_TEST:-"$here/editor-test.sh"}
base=$1
change=$2
out=$3
rounds=${4:-4}
crafts=${5:-tu160,synthetic4500,tu160x4,tu160:128x16}
scenes=${6:-floating,ground}
threads=${7:-8,1}
mkdir -p "$out"

# The same variables for every run, the short first test included.
set -- RETOOLED_BENCH_PATHS="${AB_PATHS:-compiled}" RETOOLED_BENCH_THREADS="$threads" RETOOLED_BENCH_CRAFT="$crafts" \
    RETOOLED_BENCH_SCENE="$scenes" RETOOLED_BENCH_STEPS="${AB_STEPS:-60}" RETOOLED_BENCH_BLOCKS="${AB_BLOCKS:-2}" \
    RETOOLED_BENCH_SHARE=1

started=$(date +%s)
echo "stopping both editors"
sh "$tester" "$base" stop
sh "$tester" "$change" stop

# Start both editors at once; nothing is timed yet.
echo "starting both editors"
sh "$tester" "$base" ab-start 'filter=KSP.Retooled.Physics.Tests.BackendSelectionTests.TheCurrentBackendIsOneWeShip$' "$@" \
    > "$out/start-base.log" 2>&1 &
first=$!
sh "$tester" "$change" ab-start 'filter=KSP.Retooled.Physics.Tests.BackendSelectionTests.TheCurrentBackendIsOneWeShip$' "$@" \
    > "$out/start-change.log" 2>&1
wait $first
cat "$out/start-base.log" "$out/start-change.log"
echo "both editors are ready after $(( $(date +%s) - started )) s"

run() {
    side=$1
    round=$2
    shift 2
    if [ "$side" = base ]; then project=$base; else project=$change; fi
    sh "$tester" "$project" "ab-$side-$round" 'filter=KSP.Retooled.Physics.Tests.VesselForestBench.Measure$' "$@" || exit 1
    if ! grep -q 'passed="1"' "$project/Logs/ab-$side-$round-results.xml"; then
        echo "the bench failed in $project; read $project/Logs/ab-$side-$round-results.xml"
        exit 1
    fi

    # Keep only the last table in the file.
    awk '/^\[VesselForestBench\] whole/ { buffer = "" } { buffer = buffer $0 "\n" } END { printf "%s", buffer }' \
        "$project/Logs/VesselForestBench.txt" > "$out/$side-$round.txt"
}

round=1
while [ $round -le "$rounds" ]; do
    if [ $((round % 2)) -eq 1 ]; then
        run base $round "$@"
        run change $round "$@"
    else
        run change $round "$@"
        run base $round "$@"
    fi
    round=$((round + 1))
done

if [ "${AB_KEEP:-0}" != 1 ]; then
    sh "$tester" "$base" stop
    sh "$tester" "$change" stop
fi

echo "the A/B took $(( $(date +%s) - started )) s"
python "$here/bench-ab-summary.py" "$out"
