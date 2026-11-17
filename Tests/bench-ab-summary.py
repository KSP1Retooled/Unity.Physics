"""Summarizes the tables that Tests/bench-ab.sh saves, one file per side and round (base-1.txt, change-1.txt, ...).

usage: python Tests/bench-ab-summary.py <output folder>

For each craft, scene and number of threads, and for each side, the summary gives the minimum step time over all runs,
the median of the runs' medians, every run's median, and the spread of the runs' medians (the largest minus the
smallest, over their median). It also gives the CPU load that other processes put on the machine: at the start of each
run, and during the measured steps of each scene. Files from the older A/B scripts, which name the side "new" and add
the number of threads to the name (new-1-8.txt), are read too.
"""
import glob
import os
import re
import statistics
import sys

SIDES = ("base", "change")


def side_of(path):
    side = os.path.basename(path).split("-")[0]
    return "change" if side == "new" else side


def number(pattern, text):
    match = re.search(pattern, text)
    return float(match.group(1)) if match else None


def main(folder):
    results = {}
    start_loads = {}
    durations = {}
    set_up = {}
    for path in sorted(glob.glob(os.path.join(folder, "*.txt"))):
        side = side_of(path)
        if side not in SIDES:
            continue

        for line in open(path, encoding="utf-8"):
            load = number(r"CPU load at start ([\d.]+)%", line)
            if load is not None:
                start_loads.setdefault(side, []).append(load)
                continue

            took = number(r"The bench took ([\d.]+) s", line)
            if took is not None:
                durations.setdefault(side, []).append(took)
                continue

            cells = [c.strip() for c in line.split("|")]
            if len(cells) < 5 or not re.match(r"[\d.]+ / [\d.]+$", cells[4]):
                continue

            low, median = (float(x) for x in cells[4].split(" / "))
            key = (cells[0], cells[2], cells[3])
            # The set-up column is the last one, and only newer benches have it.
            last = cells[-1] if len(cells) > 6 else ""
            loads = [float(x) for x in re.findall(r"other processes' load ([\d.]+)%", last)]
            results.setdefault(key, {}).setdefault(side, []).append((low, median, loads[-1] if loads else None))
            times = set_up.setdefault(key, {}).setdefault(side, {})
            for name in ("build", "first step", "warm-up", "measured"):
                value = number(name + r" ([\d.]+) ms", last)
                if value is not None:
                    times.setdefault(name, []).append(value)

    for side in SIDES:
        if side in start_loads:
            print(f"CPU load at the start of each run, {side}: {', '.join(f'{v:.1f}%' for v in start_loads[side])}")
        if side in durations:
            print(f"seconds that each run of the bench took, {side}: {', '.join(f'{v:.0f}' for v in durations[side])}")

    print()
    print("craft | scene | threads | base min / median of medians (spread) | change min / median of medians (spread) | "
          "change of the median | runs' medians, base; change | other processes' load in the measured steps, base; change")
    for key, sides in results.items():
        cells = []
        medians = {}
        runs_text = []
        loads_text = []
        for side in SIDES:
            runs = sides.get(side, [])
            if not runs:
                cells.append("-")
                continue
            run_medians = [r[1] for r in runs]
            medians[side] = statistics.median(run_medians)
            spread = (max(run_medians) - min(run_medians)) / medians[side] * 100.0
            cells.append(f"{min(r[0] for r in runs):.2f} / {medians[side]:.2f} ({spread:.0f}%)")
            runs_text.append(" ".join(f"{m:.2f}" for m in run_medians))
            loads_text.append(" ".join(f"{r[2]:.0f}%" for r in runs if r[2] is not None))
        change = ""
        if len(medians) == 2:
            change = f"{medians['change'] - medians['base']:+.2f} ({(medians['change'] / medians['base'] - 1.0) * 100.0:+.0f}%)"
        print(f"{key[0]} | {key[1]} | {key[2]} | {cells[0]} | {cells[1]} | {change} | "
              f"{'; '.join(runs_text)} | {'; '.join(loads_text)}")

    if any(times for sides in set_up.values() for times in sides.values()):
        print()
        print("set-up, median ms over the runs: craft | scene | threads | base build, first step, warm-up, measured | "
              "change build, first step, warm-up, measured")
        for key, sides in set_up.items():
            cells = []
            for side in SIDES:
                times = sides.get(side, {})
                cells.append(", ".join(f"{statistics.median(times[name]):.0f}" if name in times else "-"
                                       for name in ("build", "first step", "warm-up", "measured")))
            print(f"{key[0]} | {key[1]} | {key[2]} | {cells[0]} | {cells[1]}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    main(sys.argv[1])
