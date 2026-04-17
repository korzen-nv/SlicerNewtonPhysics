import ctk
import numpy as np
import qt
import slicer
import vtk
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleTest,
    ScriptedLoadableModuleWidget,
)


class NewtonPhysics(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        parent.title = "Newton Physics"
        parent.categories = ["Simulation"]
        parent.dependencies = []
        parent.contributors = ["Piotr Korzeniowski (NVIDIA)"]
        parent.helpText = (
            "Run NVIDIA Newton physics simulations inside Slicer. "
            "Rigid-body demo (falling sphere) and XPBD soft-body simulation "
            "driven directly from a segmentation (voxel-particle grid + "
            "surface skinning)."
        )
        parent.acknowledgementText = (
            "Built on Newton (https://github.com/newton-physics/newton) and NVIDIA Warp."
        )


class NewtonPhysicsWidget(ScriptedLoadableModuleWidget):
    SEG_LAST_PATH_KEY = "NewtonPhysics/SegmentationLastPath"

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)

        self.logic = NewtonPhysicsLogic()

        # --- Installation section -------------------------------------------
        installBox = ctk.ctkCollapsibleButton()
        installBox.text = "Installation"
        self.layout.addWidget(installBox)
        installForm = qt.QFormLayout(installBox)

        self.installStatusLabel = qt.QLabel()
        installForm.addRow("Newton:", self.installStatusLabel)

        self.sourceEdit = qt.QLineEdit("newton[sim,importers]")
        self.sourceEdit.toolTip = (
            "Passed verbatim to slicer.util.pip_install(). Default installs "
            "Newton from PyPI. For editable local install use e.g. "
            "'-e G:/warp/newton[sim,importers]'."
        )
        installForm.addRow("Source:", self.sourceEdit)

        self.installButton = qt.QPushButton("Install Newton into Slicer's Python")
        installForm.addRow(self.installButton)

        # --- Shared simulation settings -------------------------------------
        simBox = ctk.ctkCollapsibleButton()
        simBox.text = "Simulation settings"
        self.layout.addWidget(simBox)
        simForm = qt.QFormLayout(simBox)

        self.deviceCombo = qt.QComboBox()
        self.deviceCombo.addItems(["cuda:0", "cpu"])
        simForm.addRow("Device:", self.deviceCombo)

        self.fpsSpin = qt.QSpinBox()
        self.fpsSpin.setRange(1, 240)
        self.fpsSpin.setValue(60)
        self.fpsSpin.toolTip = "Render / timer tick rate. Affects UI refresh, not sim rate."
        simForm.addRow("Render FPS:", self.fpsSpin)

        self.timestepSpin = qt.QDoubleSpinBox()
        self.timestepSpin.setRange(0.01, 100.0)
        self.timestepSpin.setDecimals(3)
        self.timestepSpin.setSingleStep(0.1)
        self.timestepSpin.setValue(1.0)
        self.timestepSpin.setSuffix(" ms")
        self.timestepSpin.toolTip = (
            "dt passed to solver.step(). Smaller = more stable, slower real-time. "
            "Real-time match: dt × substeps = 1000/fps ms."
        )
        simForm.addRow("Timestep:", self.timestepSpin)

        self.substepsSpin = qt.QSpinBox()
        self.substepsSpin.setRange(1, 64)
        self.substepsSpin.setValue(8)
        self.substepsSpin.toolTip = (
            "Number of solver.step() calls per timer tick. Raise for stability."
        )
        simForm.addRow("Substeps per tick:", self.substepsSpin)

        self.iterationsSpin = qt.QSpinBox()
        self.iterationsSpin.setRange(1, 32)
        self.iterationsSpin.setValue(4)
        self.iterationsSpin.toolTip = (
            "XPBD constraint-solve iterations per step. Raise for stiffer / tighter constraints."
        )
        simForm.addRow("XPBD iterations:", self.iterationsSpin)

        self.auxViewerCombo = qt.QComboBox()
        self.auxViewerCombo.addItems(["none", "rerun", "viser"])
        self.auxViewerCombo.toolTip = (
            "Optional second viewer running alongside Slicer's 3D view. "
            "Requires rerun-sdk or viser to be installed."
        )
        simForm.addRow("Parallel viewer:", self.auxViewerCombo)

        # --- Rigid demo -----------------------------------------------------
        rigidBox = ctk.ctkCollapsibleButton()
        rigidBox.text = "Rigid demo (falling sphere)"
        self.layout.addWidget(rigidBox)
        rigidForm = qt.QFormLayout(rigidBox)

        self.solverCombo = qt.QComboBox()
        self.solverCombo.addItems(["XPBD", "MuJoCo"])
        rigidForm.addRow("Solver:", self.solverCombo)

        self.setupButton = qt.QPushButton("Setup demo scene")
        rigidForm.addRow(self.setupButton)

        # --- Soft body from segmentation ------------------------------------
        sbBox = ctk.ctkCollapsibleButton()
        sbBox.text = "Soft body from segmentation"
        self.layout.addWidget(sbBox)
        sbForm = qt.QFormLayout(sbBox)

        quickLoadRow = qt.QHBoxLayout()
        self.segFilePath = ctk.ctkPathLineEdit()
        self.segFilePath.filters = ctk.ctkPathLineEdit.Files
        self.segFilePath.nameFilters = [
            "Labelmaps / segmentations (*.nii *.nii.gz *.nrrd *.seg.nrrd *.mha *.mhd)",
            "All files (*)",
        ]
        self.segFilePath.settingKey = "NewtonPhysics/SegmentationFileHistory"
        self.segFilePath.toolTip = (
            "Path to a multi-label NIfTI / NRRD / MHA volume or a .seg.nrrd file. "
            "Loaded as a labelmap (each distinct value becomes a segment) and "
            "auto-selected in the Segment picker below. The last used path and "
            "a short history are remembered across Slicer sessions."
        )
        lastPath = qt.QSettings().value(self.SEG_LAST_PATH_KEY, "")
        if lastPath:
            self.segFilePath.currentPath = lastPath
        quickLoadRow.addWidget(self.segFilePath, 1)
        self.loadSegButton = qt.QPushButton("Load")
        self.loadSegButton.setMaximumWidth(80)
        quickLoadRow.addWidget(self.loadSegButton)
        sbForm.addRow("Quick-load file:", quickLoadRow)

        self.segSelector = slicer.qMRMLSegmentSelectorWidget()
        self.segSelector.setMRMLScene(slicer.mrmlScene)
        sbForm.addRow("Segment:", self.segSelector)

        self.spacingSpin = qt.QDoubleSpinBox()
        self.spacingSpin.setRange(1.0, 40.0)
        self.spacingSpin.setSingleStep(0.5)
        self.spacingSpin.setValue(6.0)
        self.spacingSpin.setSuffix(" mm")
        sbForm.addRow("Particle spacing:", self.spacingSpin)

        self.connCombo = qt.QComboBox()
        self.connCombo.addItems(["6", "18", "26"])
        self.connCombo.setCurrentText("18")
        sbForm.addRow("Connectivity:", self.connCombo)

        self.stiffnessSpin = qt.QDoubleSpinBox()
        self.stiffnessSpin.setRange(0.0, 100.0)
        self.stiffnessSpin.setDecimals(1)
        self.stiffnessSpin.setValue(10.0)
        self.stiffnessSpin.setSuffix(" N/m")
        sbForm.addRow("Spring stiffness ke:", self.stiffnessSpin)

        self.dampingSpin = qt.QDoubleSpinBox()
        self.dampingSpin.setRange(0.0, 1000.0)
        self.dampingSpin.setValue(0.0)
        sbForm.addRow("Spring damping kd:", self.dampingSpin)

        self.densitySpin = qt.QDoubleSpinBox()
        self.densitySpin.setRange(100.0, 10000.0)
        self.densitySpin.setValue(1000.0)
        self.densitySpin.setSuffix(" kg/m³")
        sbForm.addRow("Density:", self.densitySpin)

        self.pinTopCheck = qt.QCheckBox("Pin top layer (body hangs under gravity)")
        self.pinTopCheck.setChecked(True)
        sbForm.addRow(self.pinTopCheck)

        self.disableSpringsCheck = qt.QCheckBox(
            "Disable springs (smoke test: gravity + pins only, no cohesion)"
        )
        sbForm.addRow(self.disableSpringsCheck)

        self.selfCollideCheck = qt.QCheckBox("Particle self-collision")
        self.selfCollideCheck.setChecked(False)
        self.selfCollideCheck.toolTip = (
            "When off, particles pass through each other. Keep on for normal "
            "soft bodies. Turn off to isolate spring/gravity behaviour or if "
            "self-collision is driving instability."
        )
        sbForm.addRow(self.selfCollideCheck)

        displayRow = qt.QHBoxLayout()
        self.showSurfaceCheck = qt.QCheckBox("Surface")
        self.showSurfaceCheck.setChecked(True)
        self.showParticlesCheck = qt.QCheckBox("Particles")
        displayRow.addWidget(qt.QLabel("Show:"))
        displayRow.addWidget(self.showSurfaceCheck)
        displayRow.addWidget(self.showParticlesCheck)
        displayRow.addStretch(1)
        sbForm.addRow(displayRow)

        self.prepareSoftButton = qt.QPushButton("Prepare soft body")
        sbForm.addRow(self.prepareSoftButton)

        # --- Playback -------------------------------------------------------
        playBox = ctk.ctkCollapsibleButton()
        playBox.text = "Playback"
        self.layout.addWidget(playBox)
        playForm = qt.QFormLayout(playBox)

        playRow = qt.QHBoxLayout()
        self.playButton = qt.QPushButton("Play")
        self.pauseButton = qt.QPushButton("Pause")
        self.stepButton = qt.QPushButton("Step")
        self.resetButton = qt.QPushButton("Reset")
        for b in (self.playButton, self.pauseButton, self.stepButton, self.resetButton):
            playRow.addWidget(b)
        playForm.addRow(playRow)

        self.statusLabel = qt.QLabel("Not initialized.")
        playForm.addRow("Status:", self.statusLabel)

        self.layout.addStretch(1)

        # --- Connections ----------------------------------------------------
        self.installButton.connect("clicked(bool)", self.onInstall)
        self.setupButton.connect("clicked(bool)", self.onSetupScene)
        self.loadSegButton.connect("clicked(bool)", self.onLoadSegmentation)
        self.prepareSoftButton.connect("clicked(bool)", self.onPrepareSoftBody)
        self.playButton.connect("clicked(bool)", self.onPlay)
        self.pauseButton.connect("clicked(bool)", self.onPause)
        self.stepButton.connect("clicked(bool)", self.onStep)
        self.resetButton.connect("clicked(bool)", self.onReset)
        self.showSurfaceCheck.connect("toggled(bool)", self._onShowSurface)
        self.showParticlesCheck.connect("toggled(bool)", self._onShowParticles)

        self._refreshInstallStatus()
        self._refreshSimButtons(sceneReady=False, playing=False)

    def cleanup(self):
        if self.logic:
            self.logic.stopTimer()

    # ------------------------------------------------------------------ UI --
    def _refreshInstallStatus(self):
        ok, version = self.logic.checkNewton()
        if ok:
            self.installStatusLabel.text = f"installed ({version})"
            self.installButton.enabled = False
        else:
            self.installStatusLabel.text = "not installed"
            self.installButton.enabled = True
        self.setupButton.enabled = ok
        self.prepareSoftButton.enabled = ok

    def _refreshSimButtons(self, sceneReady, playing):
        self.playButton.enabled = sceneReady and not playing
        self.pauseButton.enabled = sceneReady and playing
        self.stepButton.enabled = sceneReady and not playing
        self.resetButton.enabled = sceneReady

    # -------------------------------------------------------------- actions -
    def onInstall(self):
        source = self.sourceEdit.text.strip()
        with slicer.util.tryWithErrorDisplay("Newton install failed", waitCursor=True):
            self.logic.installNewton(source)
        self._refreshInstallStatus()

    def onSetupScene(self):
        with slicer.util.tryWithErrorDisplay("Failed to build scene", waitCursor=True):
            self.logic.setupFallingSphere(
                device=self.deviceCombo.currentText,
                solverName=self.solverCombo.currentText,
                fps=self.fpsSpin.value,
                dt_ms=self.timestepSpin.value,
                substeps=self.substepsSpin.value,
                iterations=self.iterationsSpin.value,
                auxViewer=self.auxViewerCombo.currentText,
            )
        self.statusLabel.text = "Scene ready. Press Play."
        self._refreshSimButtons(sceneReady=True, playing=False)

    def onLoadSegmentation(self):
        path = self.segFilePath.currentPath
        if not path:
            slicer.util.errorDisplay("Pick a file first.")
            return
        with slicer.util.tryWithErrorDisplay("Failed to load segmentation", waitCursor=True):
            segNode = self.logic.loadSegmentationFromFile(path)
        if segNode is None:
            return
        qt.QSettings().setValue(self.SEG_LAST_PATH_KEY, path)
        self.segFilePath.addCurrentPathToHistory()
        self.segSelector.setCurrentNode(segNode)
        segIds = vtk.vtkStringArray()
        segNode.GetSegmentation().GetSegmentIDs(segIds)
        if segIds.GetNumberOfValues() > 0:
            self.segSelector.setCurrentSegmentID(segIds.GetValue(0))
        self.statusLabel.text = (
            f"Loaded '{segNode.GetName()}' with "
            f"{segIds.GetNumberOfValues()} segment(s)."
        )

    def onPrepareSoftBody(self):
        segNode = self.segSelector.currentNode()
        segmentId = self.segSelector.currentSegmentID()
        if not segNode or not segmentId:
            slicer.util.errorDisplay("Select a segmentation and a segment first.")
            return
        with slicer.util.tryWithErrorDisplay("Soft body build failed", waitCursor=True):
            self.logic.setupSoftBodyFromSegment(
                segNode=segNode,
                segmentId=segmentId,
                device=self.deviceCombo.currentText,
                fps=self.fpsSpin.value,
                dt_ms=self.timestepSpin.value,
                substeps=self.substepsSpin.value,
                iterations=self.iterationsSpin.value,
                auxViewer=self.auxViewerCombo.currentText,
                spacing_mm=self.spacingSpin.value,
                connectivity=int(self.connCombo.currentText),
                stiffness_ke=self.stiffnessSpin.value,
                damping_kd=self.dampingSpin.value,
                density=self.densitySpin.value,
                pin_top=self.pinTopCheck.isChecked(),
                disable_springs=self.disableSpringsCheck.isChecked(),
                self_collide=self.selfCollideCheck.isChecked(),
            )
        self.logic.setSurfaceVisible(self.showSurfaceCheck.isChecked())
        self.logic.setParticlesVisible(self.showParticlesCheck.isChecked())
        self.statusLabel.text = (
            f"Soft body ready ({self.logic.particleCount()} particles, "
            f"{self.logic.springCount()} springs). Press Play."
        )
        self._refreshSimButtons(sceneReady=True, playing=False)

    def _onShowSurface(self, on):
        self.logic.setSurfaceVisible(on)

    def _onShowParticles(self, on):
        self.logic.setParticlesVisible(on)

    def onPlay(self):
        self.logic.play(onStatus=self._onStatus)
        self._refreshSimButtons(sceneReady=True, playing=True)

    def onPause(self):
        self.logic.pause()
        self._refreshSimButtons(sceneReady=True, playing=False)

    def onStep(self):
        self.logic.stepOnce()
        self._onStatus(self.logic.statusText())

    def onReset(self):
        self.logic.reset()
        self.statusLabel.text = "Scene reset."
        self._refreshSimButtons(sceneReady=False, playing=False)

    def _onStatus(self, text):
        self.statusLabel.text = text


class NewtonPhysicsLogic(ScriptedLoadableModuleLogic):
    SPHERE_RADIUS_M = 0.2
    SPHERE_START_M = (0.0, 0.0, 2.0)
    GROUND_HALF_EXTENT_M = 2.0
    SCENE_SCALE_MM = 1000.0  # Newton uses meters; Slicer scene is in mm.

    MODE_NONE = "none"
    MODE_RIGID = "rigid"
    MODE_SOFTBODY = "softbody"

    SOFTBODY_MAX_PARTICLES = 80000
    SOFTBODY_SKIN_K = 4
    SURFACE_VERT_CAP = 20000

    def __init__(self):
        ScriptedLoadableModuleLogic.__init__(self)
        self._reset_runtime()

    def _reset_runtime(self):
        self._model = None
        self._solver = None
        self._state_0 = None
        self._state_1 = None
        self._control = None
        self._contacts = None
        self._dt_s = 1.0 / 60.0
        self._substeps = 1
        self._timer_interval_ms = 1000 // 60
        self._frame = 0
        self._timer = None
        self._statusCb = None
        self._auxViewer = None
        self._mode = self.MODE_NONE
        self._spring_count = 0

        self._sphereModelNode = None
        self._sphereTransformNode = None
        self._groundModelNode = None

        self._softSurfaceNode = None
        self._softSurfaceRestMm = None
        self._particleVizNode = None
        self._particlesRestM = None
        self._skinIdxs = None
        self._skinWeights = None

    # ------------------------------------------------------------- install --
    @staticmethod
    def checkNewton():
        try:
            import newton  # noqa: F401

            version = getattr(newton, "__version__", "unknown")
            return True, version
        except Exception:
            return False, None

    @staticmethod
    def installNewton(source="newton[sim,importers]"):
        slicer.util.pip_install(source)

    @staticmethod
    def loadSegmentationFromFile(path):
        import os

        if path.lower().endswith(".seg.nrrd"):
            node = slicer.util.loadSegmentation(path)
            if node is None:
                raise RuntimeError(f"Could not load segmentation: {path}")
            return node

        labelmap = None
        try:
            labelmap = slicer.util.loadLabelVolume(path, properties={"show": False})
        except Exception:
            labelmap = None
        cleanup_scalar = None
        if labelmap is None:
            scalarNode = slicer.util.loadVolume(path, properties={"show": False})
            if scalarNode is None:
                raise RuntimeError(f"Could not load volume: {path}")
            cleanup_scalar = scalarNode
            labelmap = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLLabelMapVolumeNode", scalarNode.GetName() + "_label"
            )
            imd = scalarNode.GetImageData()
            scalarType = imd.GetScalarType() if imd is not None else None
            if scalarType in (vtk.VTK_FLOAT, vtk.VTK_DOUBLE):
                cast = vtk.vtkImageCast()
                cast.SetInputData(imd)
                cast.SetOutputScalarTypeToUnsignedShort()
                cast.Update()
                labelmap.SetAndObserveImageData(cast.GetOutput())
            else:
                labelmap.SetAndObserveImageData(imd)
            ijkToRas = vtk.vtkMatrix4x4()
            scalarNode.GetIJKToRASMatrix(ijkToRas)
            labelmap.SetIJKToRASMatrix(ijkToRas)

        base = os.path.basename(path)
        for ext in (".nii.gz", ".nii", ".nrrd", ".mha", ".mhd"):
            if base.lower().endswith(ext):
                base = base[: -len(ext)]
                break
        segNode = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLSegmentationNode", f"{base}_segmentation"
        )
        segNode.CreateDefaultDisplayNodes()
        ok = slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(
            labelmap, segNode
        )
        slicer.mrmlScene.RemoveNode(labelmap)
        if cleanup_scalar is not None:
            slicer.mrmlScene.RemoveNode(cleanup_scalar)
        if not ok:
            slicer.mrmlScene.RemoveNode(segNode)
            raise RuntimeError(f"ImportLabelmapToSegmentationNode failed: {path}")
        return segNode

    # ---------------------------------------------------------- rigid demo --
    def setupFallingSphere(
        self,
        device,
        solverName,
        fps,
        dt_ms,
        substeps,
        iterations,
        auxViewer="none",
    ):
        import newton
        import warp as wp

        self.reset()

        wp.set_device(device)
        self._applyTiming(fps, dt_ms, substeps)

        builder = newton.ModelBuilder()
        builder.add_ground_plane()

        body_id = builder.add_body(
            xform=wp.transform(p=self.SPHERE_START_M, q=(0.0, 0.0, 0.0, 1.0))
        )
        builder.add_shape_sphere(body_id, radius=self.SPHERE_RADIUS_M)

        self._model = builder.finalize()
        if solverName == "MuJoCo":
            self._solver = newton.solvers.SolverMuJoCo(self._model)
        else:
            self._solver = newton.solvers.SolverXPBD(self._model, iterations=int(iterations))

        self._state_0 = self._model.state()
        self._state_1 = self._model.state()
        self._control = self._model.control()
        self._contacts = self._model.contacts()
        self._frame = 0

        self._auxViewer = self._createAuxViewer(auxViewer)
        if self._auxViewer is not None:
            self._auxViewer.set_model(self._model)

        self._buildRigidMrmlNodes()
        self._mode = self.MODE_RIGID
        self._pushBodyTransformToMrml()
        self._logToAuxViewer()

    # ------------------------------------------------------------- soft body
    def setupSoftBodyFromSegment(
        self,
        segNode,
        segmentId,
        device,
        fps,
        dt_ms,
        substeps,
        iterations,
        auxViewer,
        spacing_mm,
        connectivity,
        stiffness_ke,
        damping_kd,
        density,
        pin_top,
        disable_springs=False,
        self_collide=True,
    ):
        import newton
        import warp as wp

        self.reset()
        wp.set_device(device)
        self._applyTiming(fps, dt_ms, substeps)

        imagedata, ijk_to_ras = self._exportSegmentLabelmap(segNode, segmentId)
        positions_m, grid_ijk, spacing_m = self._particleGridFromLabelmap(
            imagedata, ijk_to_ras, spacing_mm
        )
        P = positions_m.shape[0]
        if P == 0:
            raise RuntimeError("Segment has no voxels at the chosen spacing.")
        if P > self.SOFTBODY_MAX_PARTICLES:
            raise RuntimeError(
                f"{P} particles exceeds the limit of {self.SOFTBODY_MAX_PARTICLES}. "
                f"Increase particle spacing."
            )

        pinned_mask = self._pinMask(positions_m, spacing_m, pin_top)
        particle_mass = float(density) * (spacing_m ** 3)
        masses = np.where(pinned_mask, 0.0, particle_mass).astype(np.float32)

        builder = newton.ModelBuilder()
        particle_radius = max(spacing_m * 0.49, 1e-4)
        builder.default_particle_radius = particle_radius
        if not pin_top:
            builder.add_ground_plane()

        from newton import ParticleFlags

        active_flags = int(ParticleFlags.ACTIVE)
        pinned_flags = active_flags & ~int(ParticleFlags.ACTIVE)
        for i in range(P):
            p = positions_m[i]
            flags = pinned_flags if pinned_mask[i] else active_flags
            builder.add_particle(
                pos=(float(p[0]), float(p[1]), float(p[2])),
                vel=(0.0, 0.0, 0.0),
                mass=float(masses[i]),
                radius=particle_radius,
                flags=flags,
            )

        if disable_springs:
            self._spring_count = 0
        else:
            edges = self._enumerateGridEdges(grid_ijk, connectivity)
            for i, j in edges:
                builder.add_spring(
                    int(i), int(j),
                    ke=float(stiffness_ke), kd=float(damping_kd), control=0.0,
                )
            self._spring_count = len(edges)

        self._model = builder.finalize()
        if not self_collide:
            self._model.particle_max_radius = 0.0
            self._model.particle_grid = None
        self._solver = newton.solvers.SolverXPBD(self._model, iterations=int(iterations))

        self._state_0 = self._model.state()
        self._state_1 = self._model.state()
        self._control = self._model.control()
        self._contacts = self._model.contacts()
        self._frame = 0

        self._auxViewer = self._createAuxViewer(auxViewer)
        if self._auxViewer is not None:
            self._auxViewer.set_model(self._model)

        surfacePolyData, surface_pts_mm = self._extractSurfaceMesh(imagedata, ijk_to_ras)
        particle_pts_mm = positions_m * self.SCENE_SCALE_MM
        idxs, weights = self._computeSkinWeights(
            surface_pts_mm, particle_pts_mm, self.SOFTBODY_SKIN_K
        )

        self._particlesRestM = positions_m.astype(np.float32)
        self._softSurfaceRestMm = surface_pts_mm.astype(np.float32)
        self._skinIdxs = idxs
        self._skinWeights = weights.astype(np.float32)

        self._softSurfaceNode = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLModelNode", "NewtonSoftBody"
        )
        self._softSurfaceNode.SetAndObservePolyData(surfacePolyData)
        self._softSurfaceNode.CreateDefaultDisplayNodes()
        disp = self._softSurfaceNode.GetDisplayNode()
        disp.SetColor(0.92, 0.55, 0.45)
        disp.SetOpacity(0.9)

        self._buildParticleVizNode(particle_pts_mm)

        self._mode = self.MODE_SOFTBODY
        self._pushSoftBodyToMrml()
        self._logToAuxViewer()

    def _buildParticleVizNode(self, positions_mm):
        pts = vtk.vtkPoints()
        pts.SetNumberOfPoints(positions_mm.shape[0])
        for i, p in enumerate(positions_mm):
            pts.SetPoint(i, float(p[0]), float(p[1]), float(p[2]))
        pd = vtk.vtkPolyData()
        pd.SetPoints(pts)
        vgf = vtk.vtkVertexGlyphFilter()
        vgf.SetInputData(pd)
        vgf.Update()

        self._particleVizNode = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLModelNode", "NewtonParticles"
        )
        self._particleVizNode.SetAndObservePolyData(vgf.GetOutput())
        self._particleVizNode.CreateDefaultDisplayNodes()
        d = self._particleVizNode.GetDisplayNode()
        d.SetRepresentation(0)  # 0 = VTK_POINTS
        d.SetPointSize(5.0)
        d.SetColor(0.2, 0.85, 0.25)
        d.SetVisibility(False)

    def setSurfaceVisible(self, on):
        if self._softSurfaceNode is not None:
            self._softSurfaceNode.GetDisplayNode().SetVisibility(bool(on))

    def setParticlesVisible(self, on):
        if self._particleVizNode is not None:
            self._particleVizNode.GetDisplayNode().SetVisibility(bool(on))

    # ---------------------------------------------------- aux viewer plumbing
    @staticmethod
    def _createAuxViewer(name):
        name = (name or "none").lower()
        if name == "none":
            return None
        if name == "rerun":
            from newton.viewer import ViewerRerun

            return ViewerRerun(server=True)
        if name == "viser":
            from newton.viewer import ViewerViser

            return ViewerViser()
        raise ValueError(f"Unknown parallel viewer: {name!r}")

    def _logToAuxViewer(self):
        if self._auxViewer is None or self._state_0 is None:
            return
        self._auxViewer.begin_frame(self._frame * self._dt_s)
        self._auxViewer.log_state(self._state_0)
        if self._contacts is not None:
            self._auxViewer.log_contacts(self._contacts, self._state_0)
        self._auxViewer.end_frame()

    # ---------------------------------------------------- rigid MRML plumbing
    def _buildRigidMrmlNodes(self):
        sphereSrc = vtk.vtkSphereSource()
        sphereSrc.SetRadius(self.SPHERE_RADIUS_M * self.SCENE_SCALE_MM)
        sphereSrc.SetThetaResolution(32)
        sphereSrc.SetPhiResolution(24)
        sphereSrc.Update()

        self._sphereModelNode = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLModelNode", "NewtonSphere"
        )
        self._sphereModelNode.SetAndObservePolyData(sphereSrc.GetOutput())
        self._sphereModelNode.CreateDefaultDisplayNodes()
        self._sphereModelNode.GetDisplayNode().SetColor(0.9, 0.3, 0.2)

        self._sphereTransformNode = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLLinearTransformNode", "NewtonSphereTransform"
        )
        self._sphereModelNode.SetAndObserveTransformNodeID(
            self._sphereTransformNode.GetID()
        )

        ext = self.GROUND_HALF_EXTENT_M * self.SCENE_SCALE_MM
        plane = vtk.vtkPlaneSource()
        plane.SetOrigin(-ext, -ext, 0.0)
        plane.SetPoint1(ext, -ext, 0.0)
        plane.SetPoint2(-ext, ext, 0.0)
        plane.Update()

        self._groundModelNode = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLModelNode", "NewtonGround"
        )
        self._groundModelNode.SetAndObservePolyData(plane.GetOutput())
        self._groundModelNode.CreateDefaultDisplayNodes()
        self._groundModelNode.GetDisplayNode().SetColor(0.6, 0.6, 0.6)
        self._groundModelNode.GetDisplayNode().SetOpacity(0.5)

    def _pushBodyTransformToMrml(self):
        if self._sphereTransformNode is None:
            return
        body_q = self._state_0.body_q.numpy()
        if body_q.shape[0] == 0:
            return
        px, py, pz, qx, qy, qz, qw = (float(v) for v in body_q[0])
        s = self.SCENE_SCALE_MM

        matrix = vtk.vtkMatrix4x4()
        xx, yy, zz = qx * qx, qy * qy, qz * qz
        xy, xz, yz = qx * qy, qx * qz, qy * qz
        wx, wy, wz = qw * qx, qw * qy, qw * qz
        matrix.SetElement(0, 0, 1 - 2 * (yy + zz))
        matrix.SetElement(0, 1, 2 * (xy - wz))
        matrix.SetElement(0, 2, 2 * (xz + wy))
        matrix.SetElement(0, 3, px * s)
        matrix.SetElement(1, 0, 2 * (xy + wz))
        matrix.SetElement(1, 1, 1 - 2 * (xx + zz))
        matrix.SetElement(1, 2, 2 * (yz - wx))
        matrix.SetElement(1, 3, py * s)
        matrix.SetElement(2, 0, 2 * (xz - wy))
        matrix.SetElement(2, 1, 2 * (yz + wx))
        matrix.SetElement(2, 2, 1 - 2 * (xx + yy))
        matrix.SetElement(2, 3, pz * s)
        self._sphereTransformNode.SetMatrixTransformToParent(matrix)

    # ---------------------------------------- soft-body segmentation helpers
    @staticmethod
    def _exportSegmentLabelmap(segNode, segmentId):
        segLogic = slicer.modules.segmentations.logic()
        lm = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLLabelMapVolumeNode", "__NewtonSoftBodyLM_tmp"
        )
        ids = vtk.vtkStringArray()
        ids.InsertNextValue(segmentId)
        try:
            ok = segLogic.ExportSegmentsToLabelmapNode(segNode, ids, lm)
            if not ok:
                raise RuntimeError("ExportSegmentsToLabelmapNode failed.")
            imd = vtk.vtkImageData()
            imd.DeepCopy(lm.GetImageData())
            ijk_to_ras = vtk.vtkMatrix4x4()
            lm.GetIJKToRASMatrix(ijk_to_ras)
            return imd, ijk_to_ras
        finally:
            slicer.mrmlScene.RemoveNode(lm)

    def _particleGridFromLabelmap(self, imagedata, ijk_to_ras, spacing_mm):
        from vtk.util import numpy_support as vns

        dims = imagedata.GetDimensions()  # (I, J, K)
        scalars = imagedata.GetPointData().GetScalars()
        arr = vns.vtk_to_numpy(scalars).reshape(dims[2], dims[1], dims[0]) != 0

        orig_spacing = imagedata.GetSpacing()  # (i, j, k) in mm
        stride = tuple(max(1, int(round(spacing_mm / s))) for s in orig_spacing)
        si, sj, sk = stride
        K, J, I = arr.shape
        Kt, Jt, It = (K // sk) * sk, (J // sj) * sj, (I // si) * si
        if Kt == 0 or Jt == 0 or It == 0:
            raise RuntimeError(
                "Particle spacing is larger than the segment extent."
            )
        blocks = arr[:Kt, :Jt, :It].reshape(
            Kt // sk, sk, Jt // sj, sj, It // si, si
        )
        occupied = blocks.any(axis=(1, 3, 5))  # (Kb, Jb, Ib)
        kb, jb, ib = np.nonzero(occupied)
        if ib.size == 0:
            empty = np.empty((0, 3), dtype=np.float32)
            return empty, empty.astype(np.int32), 0.0

        i_c = ib * si + si / 2.0
        j_c = jb * sj + sj / 2.0
        k_c = kb * sk + sk / 2.0
        N = ib.size

        M = np.array(
            [[ijk_to_ras.GetElement(r, c) for c in range(4)] for r in range(4)]
        )
        ijk_h = np.stack([i_c, j_c, k_c, np.ones(N)], axis=0)
        ras_mm = (M @ ijk_h)[:3].T
        pos_m = ras_mm / self.SCENE_SCALE_MM

        grid_ijk = np.stack([ib, jb, kb], axis=1).astype(np.int32)
        spacing_m_iso = float(
            np.mean([orig_spacing[0] * si, orig_spacing[1] * sj, orig_spacing[2] * sk])
            / self.SCENE_SCALE_MM
        )
        return pos_m.astype(np.float32), grid_ijk, spacing_m_iso

    @staticmethod
    def _enumerateGridEdges(grid_ijk, connectivity):
        coord_to_idx = {tuple(c): i for i, c in enumerate(grid_ijk.tolist())}
        offsets6 = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
        offsets18 = offsets6 + [
            (1, 1, 0), (1, -1, 0), (1, 0, 1), (1, 0, -1),
            (0, 1, 1), (0, 1, -1),
        ]
        offsets26 = offsets18 + [
            (1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1),
        ]
        if connectivity == 6:
            offsets = offsets6
        elif connectivity == 18:
            offsets = offsets18
        else:
            offsets = offsets26

        edges = []
        for i, c in enumerate(grid_ijk.tolist()):
            for di, dj, dk in offsets:
                key = (c[0] + di, c[1] + dj, c[2] + dk)
                j = coord_to_idx.get(key)
                if j is not None:
                    edges.append((i, j))
        return edges

    @staticmethod
    def _pinMask(positions_m, spacing_m, pin_top):
        if not pin_top or positions_m.shape[0] == 0:
            return np.zeros(positions_m.shape[0], dtype=bool)
        z = positions_m[:, 2]
        threshold = float(z.max()) - max(spacing_m, 1e-4) * 0.6
        return z >= threshold

    def _extractSurfaceMesh(self, imagedata, ijk_to_ras):
        try:
            dmc = vtk.vtkDiscreteFlyingEdges3D()
        except AttributeError:
            dmc = vtk.vtkDiscreteMarchingCubes()
        dmc.SetInputData(imagedata)
        dmc.GenerateValues(1, 1, 1)
        dmc.ComputeNormalsOn()
        dmc.Update()

        transform = vtk.vtkTransform()
        transform.SetMatrix(ijk_to_ras)
        tf = vtk.vtkTransformPolyDataFilter()
        tf.SetInputData(dmc.GetOutput())
        tf.SetTransform(transform)
        tf.Update()
        mesh = tf.GetOutput()

        nPts = mesh.GetNumberOfPoints()
        if nPts > self.SURFACE_VERT_CAP:
            deci = vtk.vtkDecimatePro()
            deci.SetInputData(mesh)
            deci.SetTargetReduction(1.0 - float(self.SURFACE_VERT_CAP) / nPts)
            deci.PreserveTopologyOn()
            deci.Update()
            mesh = deci.GetOutput()

        normals = vtk.vtkPolyDataNormals()
        normals.SetInputData(mesh)
        normals.ComputePointNormalsOn()
        normals.ConsistencyOn()
        normals.SplittingOff()
        normals.Update()
        mesh = normals.GetOutput()

        from vtk.util import numpy_support as vns

        pts = vns.vtk_to_numpy(mesh.GetPoints().GetData()).copy()
        return mesh, pts

    def _computeSkinWeights(self, surface_pts_mm, particle_pts_mm, k):
        k = min(k, particle_pts_mm.shape[0])
        try:
            from scipy.spatial import cKDTree

            tree = cKDTree(particle_pts_mm)
            dists, idxs = tree.query(surface_pts_mm, k=k)
            if k == 1:
                dists = dists[:, None]
                idxs = idxs[:, None]
        except ImportError:
            idxs, dists = self._knnBruteforce(surface_pts_mm, particle_pts_mm, k)

        eps = 1e-6
        w = 1.0 / (dists + eps)
        w = w / w.sum(axis=1, keepdims=True)
        return idxs.astype(np.int64), w

    @staticmethod
    def _knnBruteforce(surface_pts_mm, particle_pts_mm, k, batch=1024):
        V = surface_pts_mm.shape[0]
        idxs = np.empty((V, k), dtype=np.int64)
        dists = np.empty((V, k), dtype=np.float32)
        for start in range(0, V, batch):
            end = min(start + batch, V)
            diffs = surface_pts_mm[start:end, None, :] - particle_pts_mm[None, :, :]
            d2 = np.einsum("ijk,ijk->ij", diffs, diffs)
            part = np.argpartition(d2, k - 1, axis=1)[:, :k]
            rows = np.arange(end - start)[:, None]
            d2k = d2[rows, part]
            order = np.argsort(d2k, axis=1)
            idxs[start:end] = np.take_along_axis(part, order, axis=1)
            dists[start:end] = np.sqrt(np.take_along_axis(d2k, order, axis=1))
        return idxs, dists

    def _pushSoftBodyToMrml(self):
        if self._state_0 is None:
            return
        particles_m = self._state_0.particle_q.numpy().astype(np.float32)
        particles_mm = particles_m * self.SCENE_SCALE_MM

        if self._particleVizNode is not None:
            parr = slicer.util.arrayFromModelPoints(self._particleVizNode)
            parr[:] = particles_mm
            slicer.util.arrayFromModelPointsModified(self._particleVizNode)

        if self._softSurfaceNode is not None and self._particlesRestM is not None:
            disp_mm = (particles_m - self._particlesRestM) * self.SCENE_SCALE_MM
            gathered = disp_mm[self._skinIdxs]
            surface_disp_mm = (self._skinWeights[..., None] * gathered).sum(axis=1)
            current_mm = self._softSurfaceRestMm + surface_disp_mm.astype(np.float32)
            arr = slicer.util.arrayFromModelPoints(self._softSurfaceNode)
            arr[:] = current_mm
            slicer.util.arrayFromModelPointsModified(self._softSurfaceNode)

    # ---------------------------------------------------------------- loop --
    def _applyTiming(self, fps, dt_ms, substeps):
        self._timer_interval_ms = max(1, int(round(1000.0 / float(fps))))
        self._dt_s = max(1e-6, float(dt_ms) * 1e-3)
        self._substeps = max(1, int(substeps))

    def _doOneStep(self):
        self._state_0.clear_forces()
        self._model.collide(self._state_0, self._contacts)
        self._solver.step(
            self._state_0, self._state_1, self._control, self._contacts, self._dt_s
        )
        self._state_0, self._state_1 = self._state_1, self._state_0
        self._frame += 1

    def _pushCurrentModeToMrml(self):
        if self._mode == self.MODE_RIGID:
            self._pushBodyTransformToMrml()
        elif self._mode == self.MODE_SOFTBODY:
            self._pushSoftBodyToMrml()

    def stepOnce(self):
        if self._solver is None:
            return
        self._doOneStep()
        self._pushCurrentModeToMrml()
        self._logToAuxViewer()

    def play(self, onStatus=None):
        self._statusCb = onStatus
        if self._timer is None:
            self._timer = qt.QTimer()
            self._timer.timeout.connect(self._onTimerTick)
        self._timer.start(self._timer_interval_ms)

    def pause(self):
        self.stopTimer()

    def stopTimer(self):
        if self._timer is not None:
            self._timer.stop()

    def _onTimerTick(self):
        if self._solver is None:
            return
        for _ in range(self._substeps):
            self._doOneStep()
        self._pushCurrentModeToMrml()
        self._logToAuxViewer()
        if self._statusCb:
            self._statusCb(self.statusText())

    def statusText(self):
        if self._state_0 is None or self._mode == self.MODE_NONE:
            return "Not initialized."
        if self._mode == self.MODE_RIGID:
            body_q = self._state_0.body_q.numpy()
            if body_q.shape[0] == 0:
                return f"frame={self._frame}"
            return f"frame={self._frame}  z={float(body_q[0][2]):+.3f} m"
        if self._mode == self.MODE_SOFTBODY:
            q = self._state_0.particle_q.numpy()
            return (
                f"frame={self._frame}  P={q.shape[0]}  "
                f"<z>={float(q[:, 2].mean()):+.3f} m"
            )
        return f"frame={self._frame}"

    def particleCount(self):
        if self._state_0 is None:
            return 0
        return int(self._state_0.particle_q.shape[0])

    def springCount(self):
        return self._spring_count

    # --------------------------------------------------------------- reset --
    def reset(self):
        self.stopTimer()
        if self._auxViewer is not None:
            try:
                self._auxViewer.close()
            except Exception:
                pass
        for node in (
            self._sphereModelNode,
            self._sphereTransformNode,
            self._groundModelNode,
            self._softSurfaceNode,
            self._particleVizNode,
        ):
            if node is not None:
                slicer.mrmlScene.RemoveNode(node)
        self._reset_runtime()


class NewtonPhysicsTest(ScriptedLoadableModuleTest):
    def setUp(self):
        slicer.mrmlScene.Clear(0)

    def runTest(self):
        self.setUp()
        self.test_LogicInstantiates()
        self.test_CheckNewtonReportsStatus()
        self.test_GridEdgesEnumeration()
        self.test_PinMask()

    def test_LogicInstantiates(self):
        self.delayDisplay("Instantiate logic")
        logic = NewtonPhysicsLogic()
        self.assertIsNotNone(logic)
        self.delayDisplay("OK")

    def test_CheckNewtonReportsStatus(self):
        self.delayDisplay("checkNewton() returns (bool, version|None)")
        ok, version = NewtonPhysicsLogic.checkNewton()
        self.assertIsInstance(ok, bool)
        if ok:
            self.assertIsNotNone(version)
        self.delayDisplay("OK")

    def test_GridEdgesEnumeration(self):
        self.delayDisplay("Grid edges: 2x2x2 cube, 6/18/26 connectivity")
        grid = np.array(
            [(i, j, k) for i in range(2) for j in range(2) for k in range(2)],
            dtype=np.int32,
        )
        for conn, expected in [(6, 12), (18, 24), (26, 28)]:
            edges = NewtonPhysicsLogic._enumerateGridEdges(grid, conn)
            self.assertEqual(
                len(edges), expected,
                f"conn={conn}: got {len(edges)} edges, expected {expected}",
            )
        self.delayDisplay("OK")

    def test_PinMask(self):
        self.delayDisplay("Pin mask picks top-layer particles")
        pos = np.array(
            [[0, 0, 0], [0, 0, 0.1], [0, 0, 0.1], [1, 0, 0.1]],
            dtype=np.float32,
        )
        mask = NewtonPhysicsLogic._pinMask(pos, 0.1, pin_top=True)
        self.assertTrue(np.array_equal(mask, [False, True, True, True]))
        mask_off = NewtonPhysicsLogic._pinMask(pos, 0.1, pin_top=False)
        self.assertTrue(np.array_equal(mask_off, [False, False, False, False]))
        self.delayDisplay("OK")
