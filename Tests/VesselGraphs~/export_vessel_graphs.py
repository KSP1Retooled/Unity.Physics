"""Writes the vessel graphs the section solver's tests and bench read.

Each graph is one vessel as the solver sees it: every physics body (a part with physical significance plus the
physicsless parts folded into it) with its mass, inertia, position and its edge to its parent, and the closures
(autostruts and manual struts) between bodies. The four reference craft come from the census tooling of the vessel
solver study, which reads craft files and part configs; the synthetic rocket comes from that study's generator.

Usage (numpy and scipy required; run with -I so nothing in the current folder shadows a module):

    python -I export_vessel_graphs.py <census_one folder> <experiments folder> <output folder>

The census folder holds census_hook.py, and its install paths decide where part configs are read from. The craft paths
are listed in CRAFT below.

Format, gzip-compressed text, one record per line, whitespace separated:

    vessel <name>
    bodies <count>
    <parent> <kind> <mass t> <Ixx> <Iyy> <Izz> <x> <y> <z> <joint x> <joint y> <joint z> <axis x> <axis y> <axis z> <swing k>
    ...
    closures <count>
    <body a> <body b> <kind>
    ...

Bodies come parents first, the root has parent -1 and kind "root". Kind is one of stack, srf, servo, docking,
decoupler. Inertia is diagonal in the vessel frame (an approximation: the census only estimates each body's size), every
body starts with the identity orientation, and the joint point and axis are in the vessel frame. Swing k is the stock
joint's swing stiffness in kN m/rad. Closure kind is autostrut or manual.
"""
import gzip
import importlib.util
import os
import sys

import numpy as np

CRAFT = {
    "tu160": r"C:\Users\sean-lynch\Downloads\Tupolev Tu-160 White Swan.craft",
    "behemoth": r"C:\Users\sean-lynch\Downloads\'Behemoth' Mining System.craft",
    "epsilon": r"C:\Users\sean-lynch\Downloads\Epsilon.craft",
    "pegasus": r"E:\KSP\ksp1redux\Ships\VAB\606bb484-Starship_Pegasus.craft",
}


def load_census(folder):
    """census_hook with its internal per-body geometry (centre of mass, inertia, size) kept in the result."""
    path = os.path.join(folder, "census_hook.py")
    source = open(path, encoding="utf8").read()
    marker = 'stats["_int"] = dict(bpar=bpar,'
    if marker not in source:
        raise RuntimeError("census_hook.py no longer has the line this exporter extends")
    source = source.replace(marker, 'stats["_int"] = dict(com=com, Ib_own=Ib_own, kg2=kg2, bpar=bpar,')
    sys.path.insert(0, folder)
    spec = importlib.util.spec_from_loader("census_export", loader=None)
    module = importlib.util.module_from_spec(spec)
    module.__file__ = path
    exec(compile(source, path, "exec"), module.__dict__)
    return module


def craft_graph(census, path):
    name, parts, root = census.load_vessel("craft", path, 0)
    stats = census.analyse(name, parts, root, census.get_db(census.MAINKSP), path)
    it = stats["_int"]
    bpar, edge, Mb, bodies = it["bpar"], it["edge"], it["Mb"], it["bodies"]
    P = it["parts"]
    com, Ib_own, kg2 = it["com"], it["Ib_own"], it["kg2"]
    count = len(bodies)

    def mods(b):
        return P[bodies[b]["main"]]["modset"]

    def is_dock(b):
        return "ModuleDockingNode" in mods(b) or (bpar[b] >= 0 and "ModuleDockingNode" in mods(bpar[b]))

    def is_decoupler(b):
        for x in (b, bpar[b]):
            if x >= 0 and ("ModuleDecouple" in mods(x) or "ModuleAnchoredDecoupler" in mods(x)):
                return True
        return False

    rows = []
    for b in range(count):
        if bpar[b] >= b:
            raise RuntimeError(f"{name}: body {b} comes before its parent {bpar[b]}")
        # The census's estimate: an isotropic inertia from the body's size, scaled by max(1 t, m) as the game does,
        # plus the folded physicsless parts' offsets.
        inertia = np.diag(Ib_own[b]) + max(1.0, Mb[b]) * kg2[b]
        if bpar[b] < 0:
            kind, joint, axis, swing = "root", com[b], np.array([0.0, 1.0, 0.0]), 0.0
        else:
            e = edge[b]
            # As the study's lock rule "only servos, docking ports and decouplers live" reads it: a docking port or
            # decoupler counts only on a stack edge without rigid attachment.
            if e["kind"] == "servo":
                kind = "servo"
            elif e["kind"] != "rigid" and e["mode"] == "stack" and is_dock(b):
                kind = "docking"
            elif e["kind"] != "rigid" and e["mode"] == "stack" and is_decoupler(b):
                kind = "decoupler"
            else:
                kind = e["mode"]
            joint, axis, swing = np.asarray(e["J"], float), np.asarray(e["n"], float), float(e["k"])
        rows.append((int(bpar[b]), kind, float(Mb[b]), *inertia, *com[b], *joint, *axis, swing))
    closures = [(int(a), int(b), "autostrut") for a, b, *_ in it["struts"]]
    closures += [(int(a), int(b), "manual") for a, b in it["manual"]]
    return name, rows, closures


def synthetic_graph(folder, count):
    sys.path.insert(0, folder)
    import vessel
    v = vessel.generate(count)
    rows = []
    for i in range(v.n):
        p = int(v.parent[i])
        if p < 0:
            kind, swing = "root", 0.0
        else:
            role = v.role[i]
            if role in ("decoupler", "rdecoupler"):
                kind = "decoupler"
            elif v.attach[i][0] == "stack":
                kind = "stack"
            else:
                kind = "srf"
            s = min(v.size[i], v.size[p]) if v.attach[i][0] == "stack" else min(v.size[i], 1)
            swing = 30000.0 * (s + 1)
        axis = v.axis[i] if np.linalg.norm(v.axis[i]) > 0 else np.array([0.0, 1.0, 0.0])
        rows.append((p, kind, float(v.mass[i]), *v.I[i], *v.pos[i], *v.anchor[i], *axis, swing))
    return f"synthetic {count}", rows, []


def write(path, name, rows, closures):
    with gzip.open(path, "wt", encoding="ascii", newline="\n") as f:
        f.write(f"vessel {name}\n")
        f.write(f"bodies {len(rows)}\n")
        for r in rows:
            f.write(f"{r[0]} {r[1]} " + " ".join(f"{x:.9g}" for x in r[2:]) + "\n")
        f.write(f"closures {len(closures)}\n")
        for a, b, kind in closures:
            f.write(f"{a} {b} {kind}\n")
    kinds = {}
    for r in rows:
        kinds[r[1]] = kinds.get(r[1], 0) + 1
    print(f"{os.path.basename(path)}: {name}, {len(rows)} bodies {kinds}, {len(closures)} closures")


def main():
    census_folder, experiments_folder, out = sys.argv[1:4]
    os.makedirs(out, exist_ok=True)
    census = load_census(census_folder)
    for key, path in CRAFT.items():
        write(os.path.join(out, f"{key}.txt.gz"), *craft_graph(census, path))
    write(os.path.join(out, "synthetic4500.txt.gz"), *synthetic_graph(experiments_folder, 4500))


if __name__ == "__main__":
    main()
