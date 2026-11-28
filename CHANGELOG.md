---
uid: changelog
---

# Changelog

## 0.1.6
DLC rotors spin and stay attached. A drive whose spring is negligible next to its damper, as on KSP's rotors, now
builds a velocity motor. Before, it aimed a position motor at a target so far away that the rotor stayed at 0 rad/s.
Slerp drives get the same fix. A hinge's two locked angles are now one cone row, which no longer adds energy to a
fast rotor, and a locked part's offset turns by the same rotation as its orientation.

The vessel solver uses a free-axis edge for each DLC hinge, rotation servo, rotor, or piston that joins two vessel
trees. A servo that cannot become an edge still uses boundary rows. On a 500-part ship with 60 servos, a step takes
1.4 ms instead of 4.4 ms. Servo edges keep their full stiffness: a hinge that holds a loaded arm keeps its anchors
1.1 µm apart.

The rows that link two sections of a vessel are capped at 2.5e5 kN/m and 2.5e5 kN·m/rad (`VesselRowSwitches`). Hard
rows fought misfits well under a millimetre with hundreds of meganewtons, which overloaded resting joints and flung
light parts in crashes. Contacts between two parts of one vessel start each step cold, so the deep contacts left
after a split no longer fling light parts.

A contact whose crash impulse cap has expired no longer sends its vessel to non-finite speeds. The solver ignored
that the saved bound was gone, and a landed helicopter reached about 1e15 m/s.

The backend has a flight recorder, which keeps the last 300 steps in memory and saves them after a velocity jump or
on request (`FlightRecorderSwitches`). It is off unless `RETOOLED_FLIGHT_RECORDER=1` is set. The test tools replay
the recordings step by step.

The backend no longer reads `RETOOLED_VESSEL`, `RETOOLED_FOREST`, `RETOOLED_VESSEL_FUSED`,
`RETOOLED_STRUT_LOADS`, or `RETOOLED_DORMANT_ROWS`. It drops the code that let those variables bypass vessel
compilation or forest solving, disable fused solving for all islands, use the tree method for strut loads, or
keep dormant rows in the vessel boundary. With the default settings, the backend already behaved as it does now.
Craft with fewer parts than the configured minimum still use ordinary Unity Physics joints.

The flight recorder writes version 14. The replay tool requires that version.

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
