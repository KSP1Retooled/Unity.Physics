#!/bin/sh
# Compare whole-step benchmarks between base and changed Unity projects.
#
# Usage:
#   Tests/bench-ab.sh <base project> <change project> <output folder> [rounds] [crafts] [scenes] [threads]
#
# Arguments:
#   rounds   Alternating measurement rounds (default 4).
#   crafts   Default: tu160,synthetic4500,tu160x4,tu160:128x16.
#   scenes   Default: floating,ground.
#   threads  Default: 8,1.
#
# Environment:
#   AB_STEPS   Steps per measurement block (default 60).
#   AB_BLOCKS  Blocks per configuration (default 2).
#   AB_PATHS   Solver paths (default compiled).
#   AB_KEEP=1  Leave both editors open after the benchmark.
#
# Restart both editors before measuring and reuse them for all rounds.
# Alternate which project runs first to reduce bias from changing machine load.
# Reuse each scene across thread counts, with a fresh warm-up for each count.
#
# Output:
#   <output folder>/<side>-<round>.txt, where side is base or change.
#   The summary compares minimum and median timings and competing CPU load.
#
# Test filters end in $ so that Measure does not also match MeasureStages.
# Stop other editor instances before running against these projects.
#

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
