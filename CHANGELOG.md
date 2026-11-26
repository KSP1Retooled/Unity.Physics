---
uid: changelog
---

# Changelog

## 0.1.5
A part that a fast impact destroys no longer stops its vessel before the game destroys it. The game gives each such
collider its crash speed with `CrashContacts.SetCrashSpeed`. A contact that begins faster than that speed pushes with a
limited impulse for its first step, enough to stop the collider's own mass and no more. The collision report is
unchanged. A contact that the game leaves alone pushes without a limit from its second step.

## 0.1.4
Player builds compile the package again. BackendHost imported the namespace of the native world layout only in
the editor, so every player build failed with five CS0246 errors.

## 0.1.3
Player builds no longer run the package's build check. The check looked only at the first plug-in named RetooledPhysics
and failed the build when that copy didn't load at startup on the target, so a project's editor-only copy in
`Assets/Plugins/x86_64` failed every player build.

## 0.1.2
Projects reference the KSP.Retooled.Physics assemblies automatically again, as in 0.1.0, because game code outside an
assembly definition uses them.

The package's native plug-ins now load in players only. The editor chooses its physics backend before it registers
registry packages, so it never loaded them. A project copies the plug-in for its editor's platform into
`Assets/Plugins/x86_64` and enables it for the editor only.

## 0.1.1
Projects no longer reference the KSP.Retooled.Physics assemblies automatically. Their namespace, KSP.Retooled.Physics,
hid UnityEngine.Physics from code in KSP.Retooled namespaces, so calls such as Physics.Raycast stopped compiling. The
package still runs without being referenced. Code that uses its types must reference the assemblies by name.

## 0.1.0
This is the initial package version.
