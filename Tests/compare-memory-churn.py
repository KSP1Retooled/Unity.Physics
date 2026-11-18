"""Report the first differing step and the first differing body for each seed."""

import argparse
from pathlib import Path


def read_steps(path):
    steps = {}
    seed = None
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        fields = line.split()
        if fields[0] == "seed":
            seed = int(fields[1])
        elif fields[0] == "body":
            bodies[fields[1]] = fields[2:]
        elif fields[1] == "loads":
            key = seed, int(fields[0])
            hashes, bodies = [line], {}
            steps[key] = hashes, bodies
        elif fields[1] == "velocities":
            hashes.append(line)
        else:
            raise ValueError(f"Unexpected trace record in {path}: {line}")
    return steps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("caches_on", type=Path)
    parser.add_argument("caches_off", type=Path)
    args = parser.parse_args()
    on, off = read_steps(args.caches_on), read_steps(args.caches_off)
    expected = {(seed, step) for seed in range(50) for step in range(100)}
    if on.keys() != expected or off.keys() != expected:
        print(f"Incomplete traces. Expected {len(expected)} steps per run, got {len(on)} on and {len(off)} off.")
        return 2

    differing = set()
    for (seed, step), (on_hashes, on_bodies) in sorted(on.items()):
        off_hashes, off_bodies = off[seed, step]
        if seed in differing or (on_hashes == off_hashes and on_bodies == off_bodies):
            continue

        differing.add(seed)
        first_body = next(
            (name for name in dict.fromkeys([*on_bodies, *off_bodies]) if on_bodies.get(name) != off_bodies.get(name)),
            None,
        )
        print(f"Seed {seed}, step {step}, first differing body: {first_body or 'no body motion differs at this step'}")
        for on_hash, off_hash in zip(on_hashes, off_hashes):
            if on_hash != off_hash:
                print(f"  on  {on_hash}")
                print(f"  off {off_hash}")
        if first_body is not None:
            print(f"  on  {on_bodies.get(first_body)}")
            print(f"  off {off_bodies.get(first_body)}")

    print(f"{50 - len(differing)} identical seeds and {len(differing)} differing seeds across {len(on)} steps.")
    return int(bool(differing))


if __name__ == "__main__":
    raise SystemExit(main())
