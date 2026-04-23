# SlicerNewton

Run [NVIDIA Newton](https://github.com/newton-physics/newton) physics
simulations inside 3D Slicer. Scope of the `NewtonPhysics` module:

- Install Newton into Slicer's embedded Python on first use.
- **Rigid demo** — falling sphere onto a ground plane, driven through a
  `vtkMRMLLinearTransformNode`.
- **Soft body from segmentation** — pick any `vtkMRMLSegmentationNode`
  segment, voxelize it at a user-chosen spacing into an XPBD
  particle-spring grid, extract the segment's surface with marching
  cubes, skin the surface to the particles, and update the surface
  vertices every step. Quick-load any multi-label NIfTI / NRRD file
  directly from disk.
- Optional parallel viewer (Rerun or Viser) alongside Slicer's 3D view.

Planned next:

- Shape-matching / volume constraint to prevent gravity-driven collapse
  in unpinned soft bodies.
- User-driven grab/drag on the deforming mesh.
- Second segment as a pinned region (attached anatomy) instead of
  "top layer".
- MPM / voxel-native deformation via `SolverImplicitMPM`.

---

## Requirements

- 3D Slicer **5.10.0** (tested at
  `C:\Users\korze\AppData\Local\slicer.org\3D Slicer 5.10.0`). Slicer 5.10
  ships Python 3.12, which Newton requires (`>=3.10, <3.14`).
- NVIDIA GPU (Maxwell or newer) with driver 545+ for CUDA 12. CPU
  fallback works but is 10–100× slower.
- ~1–2 GB free disk for Newton + Warp + CUDA runtime bits.

## Loading the module

1. Drag-and-drop `NewtonPhysics` folder to the application screen.
2. Click `OK` (to proceed with loading Python-scripted modules in that folder)
3. Click `Yes` (to load it immediately and make the module always load automatically)

If this does not work for any reason then add the module's folder manually to additional moudule paths:

1. **Edit → Application Settings → Modules**.
2. Under **Additional module paths**, add
   `E:/Slicer3D/SlicerNewton/NewtonPhysics` (folder containing
   `NewtonPhysics.py`). Restart when prompted.
3. The module appears permanently under **Simulation → Newton Physics**.

## Installing Newton

1. Go to `Newton Physics` module
2. Click 3**Install Newton into Slicer's Python** button. Takes a few minutes on first run.

## Iteration workflow

- Edit `NewtonPhysics.py` in any editor.
- In Slicer, the module panel exposes a **Reload & Test** section at the
  top (enable **Edit → Application Settings → Developer mode** if you
  don't see it). Click **Reload** to pick up edits without restarting.
- The Python console (`View → Python Console` / `Ctrl+3`) shows all
  `print` output, tracebacks, and Warp kernel compile messages.

## Run the rigid demo

1. **Simulation settings**: pick **Device** (`cuda:0` if available).
   Defaults — `60 fps`, `1 ms` timestep, `8` substeps, `20` iterations.
2. **Rigid demo → Solver** (XPBD or MuJoCo) → **Setup demo scene**.
3. **Playback → Play**. The sphere drops onto the ground plane.

## Run a soft body on a segmentation

1. Load a volume and make / load a segmentation. Any
   `vtkMRMLSegmentationNode` with at least one segment works.
2. **Soft body from segmentation**:
   - **Quick-load file** — paste or browse to a `.nii`, `.nii.gz`,
     `.nrrd`, `.seg.nrrd`, `.mha`, `.mhd` path (multi-label or
     single-segment). Click **Load**. Each distinct label value becomes
     a separate segment. The path is remembered across sessions.
   - **Segment** — pick the segmentation node and segment to simulate.
   - **Particle spacing** (mm) — block size for the voxel-particle
     grid. Default 6 mm.
   - **Connectivity** — 6 / 18 / 26 grid neighbours for distance
     constraints. 18 is a good default; 26 is stiffer.
   - **Spring stiffness ke** (N/m) — default 10. Range 0–100.
   - **Spring damping kd** — default 0.
   - **Density** (kg/m³) — used to compute per-particle mass from
     voxel volume. Default 1000 (water).
   - **Pin top layer** — freezes the topmost slab of particles (mass=0
     + `~ParticleFlags.ACTIVE`) so the body hangs under gravity. Off
     adds a ground plane instead.
   - **Disable springs** — smoke-test mode: only gravity and pins
     act; no inter-particle cohesion. Useful for verifying the voxel
     → particle → render pipeline.
   - **Particle self-collision** — off by default. See "Stability
     notes" below.
   - **Show ☑ Surface / ☐ Particles** — toggle the deformable mesh
     and the particle point glyphs live.
3. Click **Prepare soft body**. Status shows particle and spring
   counts.
4. **Playback → Play**.

### Rendering strategy

Newton simulates a particle-spring grid; Slicer renders a triangle
surface. The two are linked by a one-time skinning step at setup:
each surface vertex is bound to its `K=4` nearest particles with
inverse-distance weights. Per frame, the displacement of each particle
(relative to its rest position) is blended to the surface vertices,
added to their rest positions, and pushed into the polydata via
`slicer.util.arrayFromModelPoints`. No topology change — just point
updates.

The separate **Particles** model node uses
`vtkVertexGlyphFilter` + `SetRepresentation(VTK_POINTS)` with a 5 px
point size.

## Parallel viewer (optional)

**Simulation settings → Parallel viewer** streams the simulation to a
second display alongside Slicer's 3D view:

- `none` (default) — Slicer only.
- `rerun` — opens a Rerun desktop window. Requires
  `slicer.util.pip_install('rerun-sdk')`.
- `viser` — opens a browser tab. Requires
  `slicer.util.pip_install('viser')`.

Newton's pyglet-based `ViewerGL` is deliberately not offered — it
wants the main event loop and fights Slicer's Qt loop.

## Units

| Quantity | Slicer side | Newton side |
|---|---|---|
| Length | mm (RAS) | m (SI) |
| Time | ms (UI) | s (solver dt) |
| Mass | — | kg |
| Density | — | kg/m³ |
| Stiffness ke | — | N/m |
| Damping kd | — | N·s/m |
| Gravity | — | −9.81 m/s² along +Z |

`NewtonPhysicsLogic.SCENE_SCALE_MM = 1000.0` handles mm↔m conversion
at both ingress (positions into Newton) and egress (positions into
`vtkPoints`). World axes align (Slicer RAS +Z = Newton +Z up), so
gravity matches "superior → inferior" for a supine patient.

## Stability notes

Newton's XPBD solver has two particle-collision code paths, both
gated by different things:

- **Particle-vs-shape** (incl. ground): gated by `model.shape_count`,
  uses `model.particle_radius` (per-particle). Always works when a
  ground plane is added.
- **Particle-vs-particle** (self-collision): gated by
  `model.particle_max_radius > 0.0` AND existence of
  `model.particle_grid`.

To disable self-collision safely, the module sets **both**
`model.particle_max_radius = 0.0` AND `model.particle_grid = None`
after `builder.finalize()`. Setting just one is broken: an assert
fires if the grid is nulled alone, and Warp rejects a zero search
radius if only the radius is zeroed.

Pinning uses the Newton cloth-grid pattern: `mass = 0` AND
`flags = ~ParticleFlags.ACTIVE`. Setting only mass to 0 is less
stable.

### Defaults that work for a dense-mesh soft body

- Timestep 1 ms, substeps 8, iterations 20, render FPS 60 (so physics
  runs at ~real time: 1 ms × 8 = 8 ms ≈ 60 fps).
- Stiffness 10 N/m, damping 0, density 1000 kg/m³, spacing 6 mm,
  connectivity 18.
- Pin top ✓, self-collision ✗ while tuning.

### Smoke-test recipe (verify the pipeline, no physics cohesion)

Disable springs ✓, Pin top ✓, Self-collision ✗, Show Particles ✓ →
**Prepare** → **Play**. Unpinned particles fall under gravity; top
layer stays frozen; the surface drifts downward via the skinning.

### Troubleshooting

- **Body explodes on first step** — particle self-collision with an
  oversized default radius. Fixed by setting
  `builder.default_particle_radius = spacing_m * 0.49` and passing
  the radius to each `add_particle`. If you change spacing
  drastically and see this again, verify radius is still sub-spacing.
- **`assert model.particle_grid is not None`** — self-collision was
  disabled by nulling only the grid. Fixed now (also zeros
  `particle_max_radius`). If you see it again, check the solver
  version against the pattern above.
- **`Hash grid cell width must be positive, got 0.0`** —
  self-collision disabled by zeroing only `particle_max_radius`
  (grid still gets built). Fixed now (also nulls the grid).
- **Nothing moves on Play** — usually an exception in
  `_onTimerTick` / `_doOneStep`. Check the Python Console.
- **Ground far from organ** — segmentations live in RAS mm; Newton
  ground plane is at `z = 0` m (Slicer z = 0). If your organ is at
  z ≈ 1800 mm, pin the top or expect a long fall.
- If Newton installation fails, it can be installed from a terminal:

```bash
"C:/Users/korze/AppData/Local/slicer.org/3D Slicer 5.10.0/bin/PythonSlicer.exe" \
    -m pip install -e G:/warp/newton[sim,importers]
```

## For developers

### Newton installation

Default **Source** is `newton[sim,importers]` (pulls from PyPI).
For an editable local install of your working copy, change it to e.g.
   `-e G:/warp/newton[sim,importers]`.

### Build as an installable extension

```bash
cmake -S E:/Slicer3D/SlicerNewton -B E:/Slicer3D/SlicerNewton-build ^
      -DSlicer_DIR="C:/Users/korze/AppData/Local/slicer.org/3D Slicer 5.10.0/lib/Slicer-5.10"
cmake --build E:/Slicer3D/SlicerNewton-build --config Release
cpack --config E:/Slicer3D/SlicerNewton-build/CPackConfig.cmake
```

Install the resulting archive via **Extensions Manager → Install from file**.
