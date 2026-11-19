#!/usr/bin/env python3
"""Check that the native plug-in links only to what Unity's own player links to.

The plug-in loads into the Unity editor and into players on Windows, Linux and macOS. A library that the player does
not load, or a newer version of one that it does load, would make the plug-in fail to load on machines that run the
game. This script reads the import tables of Windows PE, Linux ELF and macOS Mach-O binaries with the standard library
only, so it gives the same answer on every CI runner and on a developer machine.

    check-links.py inspect FILE
        Print what FILE links to.
    check-links.py baseline --unity-version V --windows DLL --linux SO --macos DYLIB --output JSON
        Record what Unity's players link to. Run it on the UnityPlayer binaries of the editor that the project uses.
    check-links.py check FILE --baseline JSON
        Fail when FILE links to a library that the player does not link to, needs a newer symbol version than the
        player needs, or needs a newer macOS than the player needs.
"""

import argparse
import json
import re
import struct
import sys


def read_c_string(data, offset):
    end = data.index(b"\0", offset)
    return data[offset:end].decode("utf-8")


# Windows PE


def pe_imports(data):
    pe = struct.unpack_from("<I", data, 0x3C)[0]
    if data[pe:pe + 4] != b"PE\0\0":
        raise ValueError("not a PE file")
    sections_count, optional_size = struct.unpack_from("<2xH12xH", data, pe + 4)
    optional = pe + 24
    magic = struct.unpack_from("<H", data, optional)[0]
    if magic != 0x20B:
        raise ValueError("only 64-bit PE files are supported")
    image_base = struct.unpack_from("<Q", data, optional + 24)[0]
    directories = optional + 112
    sections = []
    for i in range(sections_count):
        header = optional + optional_size + i * 40
        virtual_size, virtual_address, raw_size, raw_pointer = struct.unpack_from("<IIII", data, header + 8)
        sections.append((virtual_address, max(virtual_size, raw_size), raw_pointer))

    def file_offset(rva):
        for address, size, pointer in sections:
            if address <= rva < address + size:
                return rva - address + pointer
        raise ValueError(f"RVA {rva:#x} is outside every section")

    names = []
    # Directory 1 holds the ordinary imports and directory 13 the delay-loaded ones. Both load a DLL.
    for index, entry_size, name_field in ((1, 20, 12), (13, 32, 4)):
        rva, size = struct.unpack_from("<II", data, directories + index * 8)
        if rva == 0 or size == 0:
            continue
        offset = file_offset(rva)
        while True:
            entry = data[offset:offset + entry_size]
            if entry == bytes(entry_size):
                break
            name_rva = struct.unpack_from("<I", entry, name_field)[0]
            if index == 13 and struct.unpack_from("<I", entry, 0)[0] & 1 == 0:
                # An old-style delay descriptor stores addresses, not offsets from the image base.
                name_rva -= image_base
            names.append(read_c_string(data, file_offset(name_rva)))
            offset += entry_size
    return {"format": "pe", "libraries": sorted({name.lower() for name in names})}


# Linux ELF


def elf_imports(data):
    if data[4] != 2 or data[5] != 1:
        raise ValueError("only 64-bit little-endian ELF files are supported")
    section_offset = struct.unpack_from("<Q", data, 0x28)[0]
    section_size, section_count = struct.unpack_from("<HH", data, 0x3A)
    sections = []
    for i in range(section_count):
        header = section_offset + i * section_size
        kind = struct.unpack_from("<I", data, header + 4)[0]
        offset, size, link = struct.unpack_from("<QQI", data, header + 0x18)
        sections.append((kind, offset, size, link))

    libraries = []
    versions = {}
    for kind, offset, size, link in sections:
        strings = sections[link][1]
        if kind == 6:  # SHT_DYNAMIC
            for entry in range(offset, offset + size, 16):
                tag, value = struct.unpack_from("<qQ", data, entry)
                if tag == 0:
                    break
                if tag == 1:  # DT_NEEDED
                    libraries.append(read_c_string(data, strings + value))
        elif kind == 0x6FFFFFFE:  # SHT_GNU_verneed
            need = offset
            while True:
                _, count, file_name, aux, following = struct.unpack_from("<HHIII", data, need)
                library = read_c_string(data, strings + file_name)
                entry = need + aux
                for _ in range(count):
                    _, _, _, name, entry_next = struct.unpack_from("<IHHII", data, entry)
                    versions.setdefault(library, set()).add(read_c_string(data, strings + name))
                    entry += entry_next
                if following == 0:
                    break
                need += following
    return {
        "format": "elf",
        "libraries": sorted(set(libraries)),
        "versions": {library: sorted(names, key=version_key) for library, names in sorted(versions.items())},
    }


# macOS Mach-O

LOAD_COMMANDS = {0xC, 0x80000018, 0x8000001F, 0x20, 0x80000023}
CPU_X86_64 = 0x01000007


def macho_imports(data):
    magic = struct.unpack_from(">I", data, 0)[0]
    if magic in (0xCAFEBABE, 0xCAFEBABF):
        # A universal binary. The game ships x86_64, so the x86_64 slice is the one that has to load.
        count = struct.unpack_from(">I", data, 4)[0]
        wide = magic == 0xCAFEBABF
        for i in range(count):
            if wide:
                cpu, _, offset, size = struct.unpack_from(">iiQQ", data, 8 + i * 32)
            else:
                cpu, _, offset, size = struct.unpack_from(">iiII", data, 8 + i * 20)
            if cpu == CPU_X86_64:
                return macho_imports(data[offset:offset + size])
        raise ValueError("the universal binary has no x86_64 slice")
    if struct.unpack_from("<I", data, 0)[0] != 0xFEEDFACF:
        raise ValueError("only 64-bit Mach-O files are supported")
    cpu, _, _, command_count = struct.unpack_from("<iiII", data, 4)
    if cpu != CPU_X86_64:
        raise ValueError("the Mach-O file is not x86_64")
    libraries = []
    minimum = None
    command = 32
    for _ in range(command_count):
        kind, size = struct.unpack_from("<II", data, command)
        if kind in LOAD_COMMANDS:
            name = struct.unpack_from("<I", data, command + 8)[0]
            libraries.append(read_c_string(data, command + name))
        elif kind == 0x32:  # LC_BUILD_VERSION
            minimum = struct.unpack_from("<I", data, command + 12)[0]
        elif kind == 0x24:  # LC_VERSION_MIN_MACOSX
            minimum = struct.unpack_from("<I", data, command + 8)[0]
        command += size
    result = {"format": "macho", "libraries": sorted(set(libraries))}
    if minimum is not None:
        result["minimum_macos"] = f"{minimum >> 16}.{(minimum >> 8) & 0xFF}.{minimum & 0xFF}"
    return result


def inspect(path):
    with open(path, "rb") as file:
        data = file.read()
    if data[:2] == b"MZ":
        return pe_imports(data)
    if data[:4] == b"\x7fELF":
        return elf_imports(data)
    return macho_imports(data)


# Versions such as GLIBC_2.34, GLIBCXX_3.4.30 or CXXABI_1.3.13 compare within their family, the part before the
# last underscore.


def version_family(name):
    match = re.fullmatch(r"(.*)_(\d+(?:\.\d+)*)", name)
    return (match.group(1), tuple(int(part) for part in match.group(2).split("."))) if match else (name, ())


def version_key(name):
    return version_family(name)


def newest_versions(versions):
    newest = {}
    for names in versions.values():
        for name in names:
            family, number = version_family(name)
            if family not in newest or number > newest[family]:
                newest[family] = number
    return newest


def dotted(numbers):
    return ".".join(str(number) for number in numbers)


def check(path, baseline):
    found = inspect(path)
    expected = {"pe": "windows", "elf": "linux", "macho": "macos"}[found["format"]]
    allowed = baseline[expected]
    problems = []

    allowed_libraries = {name.lower() if expected == "windows" else name for name in allowed["libraries"]}
    for library in found["libraries"]:
        if library not in allowed_libraries:
            problems.append(f"links {library}, which Unity's {expected} player does not link")

    if expected == "linux":
        unity_newest = newest_versions(allowed["versions"])
        for family, number in sorted(newest_versions(found["versions"]).items()):
            if family not in unity_newest:
                problems.append(f"needs {family}_{dotted(number)}, a version family that Unity's player never needs")
            elif number > unity_newest[family]:
                problems.append(f"needs {family}_{dotted(number)}, newer than the {family}_"
                                f"{dotted(unity_newest[family])} that Unity's player needs")

    if expected == "macos":
        ours = found.get("minimum_macos")
        unity = allowed.get("minimum_macos")
        if ours is None:
            problems.append("records no minimum macOS version")
        elif unity is not None and tuple(map(int, ours.split("."))) > tuple(map(int, unity.split("."))):
            problems.append(f"needs macOS {ours}, newer than the macOS {unity} that Unity's player needs")

    print(json.dumps(found, indent=2))
    for problem in problems:
        print(f"error: {path} {problem}", file=sys.stderr)
    if not problems:
        print(f"{path} links only to what Unity {baseline['unity_version']} links to on {expected}.")
    return 1 if problems else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    inspect_command = commands.add_parser("inspect")
    inspect_command.add_argument("file")
    baseline_command = commands.add_parser("baseline")
    baseline_command.add_argument("--unity-version", required=True)
    baseline_command.add_argument("--windows", required=True)
    baseline_command.add_argument("--linux", required=True)
    baseline_command.add_argument("--macos", required=True)
    baseline_command.add_argument("--output", required=True)
    check_command = commands.add_parser("check")
    check_command.add_argument("file")
    check_command.add_argument("--baseline", required=True)
    arguments = parser.parse_args()

    if arguments.command == "inspect":
        print(json.dumps(inspect(arguments.file), indent=2))
        return 0
    if arguments.command == "baseline":
        baseline = {
            "unity_version": arguments.unity_version,
            "windows": inspect(arguments.windows),
            "linux": inspect(arguments.linux),
            "macos": inspect(arguments.macos),
        }
        with open(arguments.output, "w", newline="\n") as file:
            json.dump(baseline, file, indent=2)
            file.write("\n")
        return 0
    with open(arguments.baseline) as file:
        baseline = json.load(file)
    return check(arguments.file, baseline)


if __name__ == "__main__":
    sys.exit(main())
