"""Compile the package's assemblies with Unity's own C# compiler, without starting Unity.

A Unity run costs 1 to 3 minutes, and a compile error found that way costs all of it. This script finds the same errors
in seconds. It reuses the compiler arguments that Unity saved for each assembly at its last compile
(<project>/Library/Bee/artifacts/*.dag/<Assembly>.rsp), with the assembly's current source files in place of the saved
list, so that added and removed files count too.

Usage:
    python Tests/compile-check.py <project> [--all] [--warnings] [Assembly ...]

With no assembly named, it checks each package assembly whose sources or compiler arguments changed since the last
check that passed (the first run checks them all), and every package assembly that depends on one of those.
--all checks every package assembly that the project compiles. Assemblies are compiled in dependency order, and an
assembly compiles against the assemblies compiled before it in this run, so a change to Unity.Physics is checked
together with the assemblies that use it.

It runs no IL post-processors and no Burst compilation, so it finds C# errors only: a Burst error (BC....) still needs a
Unity run. The project must have compiled at least once in Unity. The compiled assemblies go to
<project>/Temp/compile-check, which Unity clears when it exits.

Exit status: 0 when every checked assembly compiled, 1 when one failed, 2 when the script couldn't run.
"""
import concurrent.futures
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
import time

PACKAGE = 'io.github.ksp1retooled.physics'


def windows_path(path):
    """Accept /e/KSP/... paths from Git Bash as well as Windows paths."""
    match = re.match(r'^/([a-zA-Z])/(.*)$', path)
    return f'{match.group(1).upper()}:/{match.group(2)}' if match else path


def find_compiler(project):
    version_file = os.path.join(project, 'ProjectSettings', 'ProjectVersion.txt')
    with open(version_file, encoding='utf-8') as f:
        version = re.search(r'm_EditorVersion:\s*(\S+)', f.read()).group(1)
    data = os.environ.get('UNITY_DATA') or f'C:/Program Files/Unity/Hub/Editor/{version}/Editor/Data'
    dotnet = os.path.join(data, 'DotNetSdk', 'dotnet.exe')
    compilers = sorted(glob.glob(os.path.join(data, 'DotNetSdk', 'sdk', '*', 'Roslyn', 'bincore', 'csc.dll')))
    if not os.path.exists(dotnet) or not compilers:
        sys.exit(f'compile-check: no compiler under {data}; set UNITY_DATA to the editor\'s Data folder')
    return dotnet, compilers[-1]


def find_assemblies(project):
    """Map each package assembly that the project compiles to its definition folder and its folders from .asmref."""
    root = os.path.join(project, 'Packages', PACKAGE)
    if not os.path.isdir(root):
        sys.exit(f'compile-check: {root} doesn\'t exist; link the package into the project first')
    definitions, references = {}, {}
    for folder, dirs, files in os.walk(root, followlinks=True):
        dirs[:] = [d for d in dirs if not d.endswith('~') and not d.startswith('.')]
        for name in files:
            path = os.path.join(folder, name)
            if name.endswith('.asmdef'):
                with open(path, encoding='utf-8-sig') as f:
                    definitions[json.load(f)['name']] = folder
            elif name.endswith('.asmref'):
                with open(path, encoding='utf-8-sig') as f:
                    references.setdefault(json.load(f)['reference'], []).append(folder)
    return root, definitions, references


def find_response(project, assembly):
    candidates = glob.glob(os.path.join(project, 'Library', 'Bee', 'artifacts', '*.dag', assembly + '.rsp'))
    return max(candidates, key=os.path.getmtime) if candidates else None


def sources(project, folders):
    """The .cs files under the given folders, leaving out folders that belong to another assembly."""
    found = []
    for top in folders:
        for folder, dirs, files in os.walk(top, followlinks=True):
            dirs[:] = [d for d in dirs if not d.endswith('~') and not d.startswith('.')
                       and not glob.glob(os.path.join(folder, d, '*.asmdef'))
                       and not glob.glob(os.path.join(folder, d, '*.asmref'))]
            found += [os.path.join(folder, f) for f in files if f.endswith('.cs')]
    return sorted(os.path.relpath(f, project).replace('\\', '/') for f in found)


def read_response(path):
    with open(path, encoding='utf-8-sig') as f:
        lines = [line.rstrip('\r\n') for line in f if line.strip()]
    options, listed = [], []
    for line in lines:
        bare = line.strip().strip('"')
        if bare.endswith('.cs') and not line.lstrip().startswith('-'):
            listed.append(bare)
        elif not re.match(r'^-(out|refout|pdb|doc):', line.strip()):
            options.append(line)
    return options, listed


def dependencies(options, known):
    deps = set()
    for line in options:
        match = re.match(r'^-r(?:eference)?:"?.*?[/\\]([^/\\"]+?)(?:\.ref)?\.dll"?$', line.strip())
        if match and match.group(1) in known:
            deps.add(match.group(1))
    return deps


def main():
    args = [a for a in sys.argv[1:]]
    if not args or args[0].startswith('-'):
        print(__doc__)
        return 2
    project = os.path.abspath(windows_path(args.pop(0)))
    check_all = '--all' in args
    show_warnings = '--warnings' in args
    named = [a for a in args if not a.startswith('--')]

    dotnet, compiler = find_compiler(project)
    root, definitions, references = find_assemblies(project)
    plans = {}
    for assembly, folder in definitions.items():
        response = find_response(project, assembly)
        if response is None:
            continue
        options, listed = read_response(response)
        current = sources(project, [folder] + references.get(assembly, []))
        # File times can't say what changed: a checkout touches files without changing them, and Unity doesn't
        # rewrite an assembly whose output stays the same. So the script compares a hash of the sources and arguments
        # with the hash that it recorded after the last check that passed.
        digest = hashlib.sha1('\n'.join(options).encode())
        for f in current:
            digest.update(f.encode())
            with open(os.path.join(project, f), 'rb') as source:
                digest.update(source.read())
        stamp = os.path.join(project, 'Temp', 'compile-check', assembly + '.hash')
        previous = open(stamp).read() if os.path.exists(stamp) else ''
        plans[assembly] = dict(options=options, sources=current, stale=previous != digest.hexdigest(),
                               hash=digest.hexdigest(), stamp=stamp, response=response)
    for assembly in named:
        if assembly not in plans:
            sys.exit(f'compile-check: the project doesn\'t compile {assembly}, or it isn\'t in the package')
    deps = {a: dependencies(p['options'], plans) for a, p in plans.items()}

    if named:
        chosen = set(named)
    elif check_all:
        chosen = set(plans)
    else:
        chosen = {a for a, p in plans.items() if p['stale']}
    # Everything that depends on a chosen assembly compiles against the new one.
    grew = True
    while grew:
        grew = False
        for assembly in plans:
            if assembly not in chosen and deps[assembly] & chosen:
                chosen.add(assembly)
                grew = True
    if not chosen:
        print('compile-check: nothing changed since the last check that passed; --all checks everything')
        return 0

    output = os.path.join(project, 'Temp', 'compile-check')
    os.makedirs(output, exist_ok=True)
    done, failed, results = set(), set(), {}

    def compile_one(assembly):
        plan = plans[assembly]
        options = []
        for line in plan['options']:
            match = re.match(r'^-r(?:eference)?:"?.*?[/\\]([^/\\"]+?)(?:\.ref)?\.dll"?$', line.strip())
            if match and match.group(1) in done:
                line = f'-r:"{os.path.join(output, match.group(1) + ".dll")}"'
            options.append(line)
        response = os.path.join(output, assembly + '.rsp')
        with open(response, 'w', encoding='utf-8') as f:
            f.write('\n'.join(options) + '\n')
            f.write(f'-out:"{os.path.join(output, assembly + ".dll")}"\n')
            f.write('\n'.join(f'"{s}"' for s in plan['sources']) + '\n')
        start = time.time()
        run = subprocess.run([dotnet, 'exec', compiler, '-nologo', '@' + response], cwd=project,
                             capture_output=True, text=True, encoding='utf-8', errors='replace')
        lines = (run.stdout + run.stderr).splitlines()
        errors = sorted(set(l for l in lines if ': error ' in l))
        warnings = sorted(set(l for l in lines if ': warning ' in l))
        return assembly, run.returncode, errors, warnings, time.time() - start

    start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as pool:
        pending = set(chosen)
        running = {}
        while pending or running:
            # running maps futures to assembly names, so the names are its values.
            ready = [a for a in pending if not (deps[a] & (pending | set(running.values())))]
            for assembly in ready:
                pending.discard(assembly)
                if deps[assembly] & failed:
                    failed.add(assembly)
                    results[assembly] = (None, ['skipped: an assembly it depends on failed'], [], 0.0)
                    continue
                running[pool.submit(compile_one, assembly)] = assembly
            if not running:
                continue
            finished, _ = concurrent.futures.wait(running, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in finished:
                assembly, code, errors, warnings, seconds = future.result()
                del running[future]
                results[assembly] = (code, errors, warnings, seconds)
                if code != 0 or errors:
                    failed.add(assembly)
                else:
                    done.add(assembly)
                    with open(plans[assembly]['stamp'], 'w') as f:
                        f.write(plans[assembly]['hash'])

    for assembly in sorted(results, key=lambda a: (a in failed, a)):
        code, errors, warnings, seconds = results[assembly]
        state = 'FAILED' if assembly in failed else 'ok'
        print(f'{state:6} {assembly}: {len(errors)} errors, {len(warnings)} warnings, {seconds:.1f} s')
        for line in errors:
            print('   ' + line)
        if show_warnings:
            for line in warnings:
                print('   ' + line)
    print(f'compile-check: {len(done)} compiled, {len(failed)} failed, {time.time() - start:.1f} s')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
