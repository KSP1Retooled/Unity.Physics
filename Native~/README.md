# RetooledPhysics native plug-in

This folder holds the C++ plug-in that Unity loads as its physics backend. Unity ignores folders whose names end in
`~`, so nothing here is imported into a project.

## Built binaries

The `Native plug-in` workflow (`.github/workflows/native.yml`) runs whenever a push changes the plug-in's source. It
builds the plug-in for Windows, Linux and macOS, all for x86_64, and commits the three binaries to `Plugins/x86_64`
in this folder on the branch that changed. The `.meta` files next to them make Unity preload each binary in the
editor and player of its own platform.

## Releases

Publishing a GitHub release runs the `Publish to UPM` workflow (`.github/workflows/publish-upm.yml`). It copies
`Plugins` from this folder to the package root, where Unity loads it, packs the package, attaches the `.tgz` to the
release, and adds the release to the registry at `https://ksp1retooled.github.io/upm/`. The workflow refuses a
release when:

- the tag is not `v` followed by the `version` in `package.json`, such as `v6.7.0`;
- the binaries are older than the plug-in's source, because the tag was made before the native workflow committed
  them.

A project that installs the package from the registry gets the plug-in from the package. Remove any copy of
`RetooledPhysics` from the project's `Assets/Plugins`, or Unity will find two plug-ins with the same name. A project
that uses this repository through a `file:` path does not get the plug-in from the package, so it still needs the
copy in `Assets/Plugins`.

## Building locally

The plug-in needs the plug-in headers of the Unity editor that the project uses. Unity ships them only inside the
editor, so pass that editor's `Editor/Data/PluginAPI` folder to CMake:

```
cmake -S Native~ -B build -DCMAKE_BUILD_TYPE=Release -DUNITY_PLUGIN_API="<editor>/Editor/Data/PluginAPI"
cmake --build build --config Release
```

## What the plug-in may link to

The plug-in loads into the Unity editor and into players, so it may only link to libraries that Unity's own player
links to. Otherwise it would fail to load on a machine that runs the game. `CMakeLists.txt` sets this up on each
platform:

- On Windows, the plug-in links the Visual C++ runtime, `msvcp140.dll`, which Unity's player also links.
- On Linux, Unity's player does not link `libstdc++`, so the plug-in carries its own copy of it. It may need glibc
  2.34 at most, the version that the player needs.
- On macOS, the plug-in targets macOS 12, the version that the player targets.

`check-links.py` checks a built binary against `unity-links.json`, which records what Unity's Windows, Linux and macOS
players link to. The workflow runs it on every build:

```
python Native~/check-links.py check build/Release/RetooledPhysics.dll --baseline Native~/unity-links.json
```

When the project moves to another Unity editor, record that editor's players again. The workflow downloads the
plug-in headers of the editor that `unity-links.json` names:

```
python Native~/check-links.py baseline --unity-version "<version> (<changeset>)" \
    --windows "<editor>/Editor/Data/PlaybackEngines/windowsstandalonesupport/Variations/win64_player_nondevelopment_mono/UnityPlayer.dll" \
    --linux "<editor>/Editor/Data/PlaybackEngines/LinuxStandaloneSupport/Variations/linux64_player_nondevelopment_mono/UnityPlayer.so" \
    --macos "<editor>/Editor/Data/PlaybackEngines/MacStandaloneSupport/Variations/macos_x64_player_nondevelopment_mono/UnityPlayer.app/Contents/Frameworks/UnityPlayer.dylib" \
    --output Native~/unity-links.json
```
