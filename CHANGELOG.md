---
uid: changelog
---

# Changelog

## 0.1.1
Projects no longer reference the KSP.Retooled.Physics assemblies automatically. Their namespace, KSP.Retooled.Physics,
hid UnityEngine.Physics from code in KSP.Retooled namespaces, so calls such as Physics.Raycast stopped compiling. The
package still runs without being referenced. Code that uses its types must reference the assemblies by name.

## 0.1.0
This is the initial package version.
