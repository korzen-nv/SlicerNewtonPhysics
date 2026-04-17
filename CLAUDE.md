# CLAUDE.md — SlicerNewton

Working context for future Claude Code sessions on this project.
Read this before making changes.

## What this is

A 3D Slicer extension that embeds NVIDIA Newton physics inside the
Slicer Qt/VTK application. One scripted module (`NewtonPhysics`) that
does two things today:

1. **Rigid demo** — falling sphere as a smoke test that Newton runs
   headlessly inside Slicer and updates a `vtkMRMLModelNode` through
   a `vtkMRMLLinearTransformNode`.
2. **Soft body from a segmentation** — voxelize a
   `vtkMRMLSegmentationNode` segment into an XPBD particle-spring
   grid, extract its marching-cubes surface, skin the surface to the
   particles with K-nearest-neighbour weights, and stream particle
   displacements to the surface vertices each step.

The user is NVIDIA / focused on soft-body organ deformation from
medical segmentations; long-term target is voxel-native / MPM.

## File layout

```
SlicerNewton/
  CMakeLists.txt                              # extension metadata
  License.txt
  README.md                                   # user-facing; keep in sync
  CLAUDE.md                                   # this file
  NewtonPhysics/
    CMakeLists.txt                            # slicerMacroBuildScriptedModule
    NewtonPhysics.py                          # 4 classes; ~1100 lines
    Resources/Icons/NewtonPhysics.png         # placeholder, replaceable
    Testing/
      CMakeLists.txt
      Python/CMakeLists.txt
```

## The four classes

Scripted Slicer module pattern:

- `NewtonPhysics(ScriptedLoadableModule)` — metadata only.
- `NewtonPhysicsWidget(ScriptedLoadableModuleWidget)` — Qt/ctk UI in
  `setup()`. Holds `self.logic`. All UI state stays here; no physics.
- `NewtonPhysicsLogic(ScriptedLoadableModuleLogic)` — Newton model,
  solver, state, timer. Owns MRML nodes for rendering.
- `NewtonPhysicsTest(ScriptedLoadableModuleTest)` — unit tests,
  runnable from the **Reload & Test** panel.

Edit boundary: Widget → Logic only. Logic never touches Widget.

## Mode dispatch

`NewtonPhysicsLogic` holds one simulation at a time, tagged by
`self._mode`:

- `MODE_NONE` — nothing built.
- `MODE_RIGID` — sphere demo.
- `MODE_SOFTBODY` — particle-spring grid from a segmentation.

`stepOnce()` calls `_doOneStep()` (one `solver.step`) plus
`_pushCurrentModeToMrml()`, which branches on mode to either
`_pushBodyTransformToMrml` (rigid) or `_pushSoftBodyToMrml` (soft).

`_onTimerTick()` runs `_doOneStep()` `self._substeps` times per tick
and then pushes to MRML once — so render is decoupled from sim rate.

## Units

Slicer is in **mm** (RAS); Newton is in **m** (SI). Conversion via
`SCENE_SCALE_MM = 1000.0`. Applied at ingress (positions into
`add_particle`) and egress (positions into `vtkPoints`).

Time: UI is ms, solver is s. `_applyTiming(fps, dt_ms, substeps)`
sets `self._timer_interval_ms`, `self._dt_s`, `self._substeps`.

Axes align: Slicer +Z (superior) = Newton +Z (up). Gravity is
Newton's default (−9.81 m/s² along +Z).

## Newton API gotchas (hard-won, do not forget)

**These patterns were verified by reading the installed Newton source
at `C:/Users/korze/AppData/Local/slicer.org/3D Slicer 5.10.0/lib/Python/Lib/site-packages/newton/`.**

1. **Particle radius matters a lot.** Newton's
   `builder.default_particle_radius` is ~0.1 m. On a 6 mm voxel grid,
   every particle overlaps every neighbour → self-collision
   explodes the body on step 1. Always set
   `builder.default_particle_radius = spacing_m * 0.49` before adding
   particles, and pass `radius=` to each `add_particle`.

2. **Pinning a particle needs BOTH `mass=0` AND
   `flags=~ParticleFlags.ACTIVE`.** Newton's own `add_cloth_grid`
   does this. `mass=0` alone is less stable. `~ACTIVE` alone is not
   enough either — the mass-based integrator can still do arithmetic
   with non-zero inverse mass leftovers.

3. **Disabling particle self-collision needs BOTH
   `model.particle_max_radius = 0.0` AND `model.particle_grid = None`
   after finalize.** The solver has three code paths:
   - Line ~273 `if model.particle_count > 1 and model.particle_grid
     is not None` → builds `particle_grid` with
     `search_radius = particle_max_radius * 2 + particle_cohesion`.
     If grid exists but max_radius=0, Warp rejects search_radius=0.
   - Line ~378 `if model.particle_max_radius > 0.0 and
     model.particle_count > 1` → runs self-collision kernel.
   - Line ~380 inside that: `assert model.particle_grid is not None`.
   So nulling only the grid asserts, zeroing only max_radius raises a
   Warp error. Null both.

4. **`add_spring(i, j, ke, kd, control)`** — auto-computes rest
   length from particle positions at call time. No stiffness
   compliance, only `ke` (N/m). `control=0.0` is correct for passive
   springs.

5. **`model.collide(state, contacts)` + `solver.step(...)`** — call
   both every step. `collide` populates `contacts.soft_contact_*`
   used by the particle-vs-shape kernel; it's independent of
   `particle_grid`.

6. **State swap pattern** — after every `solver.step`,
   `state_0, state_1 = state_1, state_0`. The solver writes to
   `state_1` and reads from `state_0`.

7. **Warp array conversion** — `state.particle_q.numpy()` returns an
   `(N, 3)` float32. `state.body_q.numpy()` is `(N, 7)`: px py pz
   qx qy qz qw.

## Soft-body pipeline (the complex bit)

`setupSoftBodyFromSegment(...)` does, in order:

1. `_exportSegmentLabelmap(segNode, segmentId)` — exports the chosen
   segment to a temporary `vtkMRMLLabelMapVolumeNode`, deep-copies
   its `vtkImageData` and `IJKToRAS` matrix, then removes the temp.
2. `_particleGridFromLabelmap(imagedata, ijk_to_ras, spacing_mm)` —
   computes a stride (integer) from target spacing / original
   spacing, block-reduces the boolean labelmap via strided reshape +
   `any`, converts block-centre IJK to RAS then to metres. Returns
   `(positions_m, grid_ijk, spacing_m_iso)`.
3. `_pinMask(positions_m, spacing_m, pin_top)` — marks particles
   within `0.6 * spacing_m` of the max world z.
4. `ModelBuilder` → `add_particle` per voxel centre with correct
   mass, radius, flags. Springs via `_enumerateGridEdges` when not
   disabled.
5. `builder.finalize()`; optionally null `particle_grid` and zero
   `particle_max_radius`.
6. `SolverXPBD(model, iterations=...)`, state, control, contacts.
7. `_extractSurfaceMesh(imagedata, ijk_to_ras)` — marching cubes
   (`vtkDiscreteFlyingEdges3D` with `vtkDiscreteMarchingCubes`
   fallback) → IJK→RAS transform → decimation to ≤20k verts →
   recompute normals.
8. `_computeSkinWeights(surface_pts_mm, particle_pts_mm, K=4)` —
   scipy `cKDTree` K-nearest, inverse-distance weights. Falls back
   to a chunked numpy brute-force if scipy is missing.
9. Stores `_particlesRestM`, `_softSurfaceRestMm`, `_skinIdxs`,
   `_skinWeights`.
10. Creates `_softSurfaceNode` (`vtkMRMLModelNode` holding the
    surface polydata) and `_particleVizNode` (point-glyph model, off
    by default).

Per step:
```
disp_m = particles_m - rest_particles_m
surface_disp_mm = (weights[..., None] * disp_m[idxs]).sum(axis=1) * 1000
surface = surface_rest + surface_disp
slicer.util.arrayFromModelPoints(node)[:] = surface
```

Particle viz node is updated from `particles_m * 1000` unconditionally
(cost is trivial).

## Parallel viewer

`_createAuxViewer(name)` → `None` | `ViewerRerun(server=True)` |
`ViewerViser()`. `_logToAuxViewer()` calls
`begin_frame / log_state / log_contacts / end_frame`. `ViewerGL`
(pyglet) is deliberately not exposed — fights Qt event loop.

## Quick-load file & persistence

`ctk.ctkPathLineEdit` with `settingKey =
"NewtonPhysics/SegmentationFileHistory"` (MRU dropdown, persisted by
ctk). Current path persisted manually via `qt.QSettings()` under
`NewtonPhysics/SegmentationLastPath`, restored in `setup()`, saved
after successful load.

`loadSegmentationFromFile(path)` — tries `loadLabelVolume` first,
falls back to `loadVolume` + manual cast to
`vtkMRMLLabelMapVolumeNode` (AbdomenAtlas `combined_labels.nii.gz`
has voxel types the strict labelmap loader rejects). Casts float
voxel types to `VTK_UNSIGNED_SHORT` via `vtkImageCast`.

## Environment

- **OS**: Windows 11, bash via Git Bash.
- **Slicer install**:
  `C:\Users\korze\AppData\Local\slicer.org\3D Slicer 5.10.0\`
  - Python: `bin/PythonSlicer.exe` (3.12.10)
  - Newton installed into:
    `lib/Python/Lib/site-packages/newton/`
- **Newton working copy**: `G:/warp/newton` (Apache-2.0). Editable
  install option: `-e G:/warp/newton[sim,importers]`.
- **GPU**: user has CUDA-capable NVIDIA GPU; default device `cuda:0`.

## Build / run cheatsheet

Syntax-check the module with Slicer's Python:

```bash
"C:/Users/korze/AppData/Local/slicer.org/3D Slicer 5.10.0/bin/PythonSlicer.exe" \
    -c "import ast; ast.parse(open('E:/Slicer3D/SlicerNewton/NewtonPhysics/NewtonPhysics.py').read())"
```

Reload in Slicer: click **Reload** in the Reload & Test panel (enable
Developer mode in Application Settings if missing). No CMake rebuild
needed while iterating on `.py`.

Full CMake build (only needed to package the extension for
distribution):

```bash
cmake -S E:/Slicer3D/SlicerNewton -B E:/Slicer3D/SlicerNewton-build ^
      -DSlicer_DIR="C:/Users/korze/AppData/Local/slicer.org/3D Slicer 5.10.0/lib/Slicer-5.10"
cmake --build E:/Slicer3D/SlicerNewton-build --config Release
cpack --config E:/Slicer3D/SlicerNewton-build/CPackConfig.cmake
```

## Tests

`NewtonPhysicsTest` has: logic instantiation, Newton-detection tuple
shape, grid-edge enumeration for 2×2×2 cube (expected 12/24/28
edges for 6/18/26 connectivity), pin-mask correctness.

Reload & Test in Slicer runs them; no separate runner.

## Follow-ups parked but not done

- Shape-matching / volume constraint to prevent collapse.
- Mouse grab / drag with `vtkMRMLMarkupsNode` picking.
- Second segment as pinned region.
- MPM solver (`SolverImplicitMPM`) for voxel-native deformation.
- Swap placeholder icon at `Resources/Icons/NewtonPhysics.png`.

## Don'ts

- Don't try to embed Newton's `ViewerGL` — pyglet owns the event
  loop and will hang Slicer.
- Don't use `slicer.util.loadLabelVolume` as the only load path for
  NIfTI segmentations — it's too strict.
- Don't revert to `mass=0`-only pinning; always use the flags too.
- Don't set `model.particle_grid = None` without also zeroing
  `model.particle_max_radius`, and vice versa.
- Don't skip the per-particle `radius=` in `add_particle` — relying
  on the default is the #1 cause of explosion.
