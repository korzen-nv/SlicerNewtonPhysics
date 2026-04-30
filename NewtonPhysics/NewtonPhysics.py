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
            "Rigid-body demo (falling sphere), legacy XPBD spring soft bodies, "
            "and hexahedral shape-matching soft bodies with GPU marching-cubes "
            "rendering and cell deletion."
        )
        parent.acknowledgementText = (
            "Built on Newton (https://github.com/newton-physics/newton) and NVIDIA Warp."
        )


class NewtonPhysicsWidget(ScriptedLoadableModuleWidget):
    SEG_LAST_PATH_KEY = "NewtonPhysics/SegmentationLastPath"

    ABDOMEN_ATLAS_NAMES = {
        1: "aorta",
        2: "gall_bladder",
        3: "kidney_left",
        4: "kidney_right",
        5: "liver",
        6: "pancreas",
        7: "postcava",
        8: "spleen",
        9: "stomach",
        10: "adrenal_gland_left",
        11: "adrenal_gland_right",
        12: "bladder",
        13: "celiac_trunk",
        14: "colon",
        15: "duodenum",
        16: "esophagus",
        17: "femur_left",
        18: "femur_right",
        19: "hepatic_vessel",
        20: "intestine",
        21: "lung_left",
        22: "lung_right",
        23: "portal_vein_and_splenic_vein",
        24: "prostate",
        25: "rectum",
    }

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

        self.cuttingInstallStatusLabel = qt.QLabel()
        installForm.addRow("Newton cutting:", self.cuttingInstallStatusLabel)

        self.cuttingSourceEdit = qt.QLineEdit("--no-deps -e G:/warp/warp-cutting")
        self.cuttingSourceEdit.toolTip = (
            "Passed verbatim to slicer.util.pip_install(). Default installs "
            "the local warp-cutting package in editable mode without standalone "
            "viewer dependencies such as imgui."
        )
        installForm.addRow("Cutting source:", self.cuttingSourceEdit)

        self.installCuttingButton = qt.QPushButton(
            "Install newton-cutting into Slicer's Python"
        )
        installForm.addRow(self.installCuttingButton)

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

        self.segNodeSelector = slicer.qMRMLNodeComboBox()
        self.segNodeSelector.nodeTypes = ["vtkMRMLSegmentationNode"]
        self.segNodeSelector.selectNodeUponCreation = True
        self.segNodeSelector.addEnabled = False
        self.segNodeSelector.removeEnabled = False
        self.segNodeSelector.noneEnabled = True
        self.segNodeSelector.showHidden = False
        self.segNodeSelector.setMRMLScene(slicer.mrmlScene)
        sbForm.addRow("Segmentation:", self.segNodeSelector)

        self.segTable = qt.QTableWidget()
        self.segTable.setColumnCount(5)
        self.segTable.setHorizontalHeaderLabels(
            ["Sim", "Lock", "Segment", "Color", "Stiffness ke (N/m)"]
        )
        header = self.segTable.horizontalHeader()
        header.setSectionResizeMode(0, qt.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, qt.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, qt.QHeaderView.Stretch)
        header.setSectionResizeMode(3, qt.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, qt.QHeaderView.ResizeToContents)
        self.segTable.verticalHeader().setVisible(False)
        self.segTable.setSelectionMode(qt.QAbstractItemView.NoSelection)
        self.segTable.setEditTriggers(qt.QAbstractItemView.NoEditTriggers)
        self.segTable.setMinimumHeight(180)
        self.segTable.toolTip = (
            "Per-segment controls. Sim = include in simulation. Lock = freeze all "
            "particles in this segment (mass=0, e.g. bones). Color = surface mesh "
            "color. Stiffness is used by the legacy spring solver."
        )
        sbForm.addRow(self.segTable)

        self.renameAtlasButton = qt.QPushButton("Apply AbdomenAtlas names")
        self.renameAtlasButton.toolTip = (
            "Rename segments in the current segmentation using AbdomenAtlas "
            "label values (1=aorta, 2=gall_bladder, ...). Label value is parsed "
            "from the current segment name; falls back to row order."
        )
        sbForm.addRow(self.renameAtlasButton)

        self.spacingSpin = qt.QDoubleSpinBox()
        self.spacingSpin.setRange(1.0, 40.0)
        self.spacingSpin.setSingleStep(0.5)
        self.spacingSpin.setValue(6.0)
        self.spacingSpin.setSuffix(" mm")
        sbForm.addRow("Particle spacing:", self.spacingSpin)

        self.softSolverCombo = qt.QComboBox()
        self.softSolverCombo.addItems(["Hex shape matching", "Legacy springs"])
        self.softSolverCombo.setCurrentText("Hex shape matching")
        self.softSolverCombo.toolTip = (
            "Hex shape matching uses the new corner-node hexahedral solver. "
            "Legacy springs keeps the previous voxel-centre XPBD path."
        )
        sbForm.addRow("Soft-body solver:", self.softSolverCombo)

        self.connCombo = qt.QComboBox()
        self.connCombo.addItems(["6", "18", "26"])
        self.connCombo.setCurrentText("18")
        sbForm.addRow("Connectivity:", self.connCombo)

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
        self.pinTopCheck.setChecked(False)
        sbForm.addRow(self.pinTopCheck)

        self.disableSpringsCheck = qt.QCheckBox(
            "Disable springs (smoke test: gravity + pins only, no cohesion)"
        )
        sbForm.addRow(self.disableSpringsCheck)

        self.selfCollideCheck = qt.QCheckBox("Particle self-collision")
        self.selfCollideCheck.setChecked(True)
        self.selfCollideCheck.toolTip = (
            "When off, particles pass through each other. Keep on for normal "
            "soft bodies. Turn off to isolate spring/gravity behaviour or if "
            "self-collision is driving instability."
        )
        sbForm.addRow(self.selfCollideCheck)

        self.radiusFactorSpin = qt.QDoubleSpinBox()
        self.radiusFactorSpin.setRange(0.05, 0.5)
        self.radiusFactorSpin.setSingleStep(0.05)
        self.radiusFactorSpin.setDecimals(2)
        self.radiusFactorSpin.setValue(0.35)
        self.radiusFactorSpin.toolTip = (
            "Particle collision radius as a fraction of particle spacing. "
            "0.5 = neighbours just touch at rest (explodes under any compression). "
            "0.3–0.4 is usually stable with self-collision on. "
            "Also used for ground-plane contact."
        )
        sbForm.addRow("Particle radius (× spacing):", self.radiusFactorSpin)

        self.shapeStiffnessSpin = qt.QDoubleSpinBox()
        self.shapeStiffnessSpin.setRange(0.0, 1.0)
        self.shapeStiffnessSpin.setSingleStep(0.05)
        self.shapeStiffnessSpin.setDecimals(2)
        self.shapeStiffnessSpin.setValue(0.75)
        self.shapeStiffnessSpin.toolTip = (
            "Local shape-matching stiffness for the hexahedral solver."
        )
        sbForm.addRow("Shape stiffness:", self.shapeStiffnessSpin)

        self.shapePassesSpin = qt.QSpinBox()
        self.shapePassesSpin.setRange(1, 8)
        self.shapePassesSpin.setValue(1)
        self.shapePassesSpin.toolTip = (
            "Shape-matching projection passes per solver iteration."
        )
        sbForm.addRow("Shape passes:", self.shapePassesSpin)

        cutRow = qt.QHBoxLayout()
        self.armCuttingCheck = qt.QCheckBox("Arm cutting")
        self.armCuttingCheck.toolTip = (
            "When enabled in hex mode, left-click in a 3D view deletes cells "
            "along the camera ray."
        )
        cutRow.addWidget(self.armCuttingCheck)
        cutRow.addStretch(1)
        sbForm.addRow(cutRow)

        self.cutRadiusSpin = qt.QDoubleSpinBox()
        self.cutRadiusSpin.setRange(0.1, 50.0)
        self.cutRadiusSpin.setSingleStep(0.5)
        self.cutRadiusSpin.setValue(3.0)
        self.cutRadiusSpin.setSuffix(" mm")
        sbForm.addRow("Cut radius:", self.cutRadiusSpin)

        self.cutDepthSpin = qt.QDoubleSpinBox()
        self.cutDepthSpin.setRange(1.0, 500.0)
        self.cutDepthSpin.setSingleStep(5.0)
        self.cutDepthSpin.setValue(60.0)
        self.cutDepthSpin.setSuffix(" mm")
        sbForm.addRow("Cut depth:", self.cutDepthSpin)

        displayRow = qt.QHBoxLayout()
        self.showSurfaceCheck = qt.QCheckBox("Surface")
        self.showSurfaceCheck.setChecked(True)
        self.showParticlesCheck = qt.QCheckBox("Particles")
        displayRow.addWidget(qt.QLabel("Show:"))
        displayRow.addWidget(self.showSurfaceCheck)
        displayRow.addWidget(self.showParticlesCheck)
        displayRow.addStretch(1)
        sbForm.addRow(displayRow)

        self.prepareSoftButton = qt.QPushButton("Init Sim")
        sbForm.addRow(self.prepareSoftButton)

        # --- Soft body from tetrahedral model -------------------------------
        tetBox = ctk.ctkCollapsibleButton()
        tetBox.text = "Soft body from tetrahedral model"
        self.layout.addWidget(tetBox)
        tetForm = qt.QFormLayout(tetBox)

        self.tetModelSelector = slicer.qMRMLNodeComboBox()
        self.tetModelSelector.nodeTypes = ["vtkMRMLModelNode"]
        self.tetModelSelector.selectNodeUponCreation = True
        self.tetModelSelector.addEnabled = False
        self.tetModelSelector.removeEnabled = False
        self.tetModelSelector.noneEnabled = True
        self.tetModelSelector.showHidden = False
        self.tetModelSelector.setMRMLScene(slicer.mrmlScene)
        self.tetModelSelector.toolTip = (
            "SegmentMesher volumetric model with a vtkUnstructuredGrid and "
            "cell-data scalar array named 'labels'."
        )
        tetForm.addRow("Tetrahedral model:", self.tetModelSelector)

        self.tetSolverCombo = qt.QComboBox()
        self.tetSolverCombo.addItems(["Tet FEM XPBD", "Tet FEM VBD"])
        self.tetSolverCombo.setCurrentText("Tet FEM XPBD")
        tetForm.addRow("Tet solver:", self.tetSolverCombo)

        self.tetLabelTable = qt.QTableWidget()
        self.tetLabelTable.setColumnCount(8)
        self.tetLabelTable.setHorizontalHeaderLabels(
            ["Sim", "Lock", "Label", "Name", "Color", "k_mu", "k_lambda", "k_damp"]
        )
        tetHeader = self.tetLabelTable.horizontalHeader()
        tetHeader.setSectionResizeMode(0, qt.QHeaderView.ResizeToContents)
        tetHeader.setSectionResizeMode(1, qt.QHeaderView.ResizeToContents)
        tetHeader.setSectionResizeMode(2, qt.QHeaderView.ResizeToContents)
        tetHeader.setSectionResizeMode(3, qt.QHeaderView.Stretch)
        tetHeader.setSectionResizeMode(4, qt.QHeaderView.ResizeToContents)
        tetHeader.setSectionResizeMode(5, qt.QHeaderView.ResizeToContents)
        tetHeader.setSectionResizeMode(6, qt.QHeaderView.ResizeToContents)
        tetHeader.setSectionResizeMode(7, qt.QHeaderView.ResizeToContents)
        self.tetLabelTable.verticalHeader().setVisible(False)
        self.tetLabelTable.setSelectionMode(qt.QAbstractItemView.NoSelection)
        self.tetLabelTable.setEditTriggers(qt.QAbstractItemView.NoEditTriggers)
        self.tetLabelTable.setMinimumHeight(160)
        self.tetLabelTable.toolTip = (
            "Per-label controls for nonzero cell labels. Sim includes the label "
            "in the deforming mesh. Lock freezes all nodes incident to that label. "
            "Colors come from Slicer's Labels color table."
        )
        tetForm.addRow(self.tetLabelTable)

        self.prepareTetButton = qt.QPushButton("Init Tet FEM")
        tetForm.addRow(self.prepareTetButton)

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
        self.installCuttingButton.connect("clicked(bool)", self.onInstallCutting)
        self.setupButton.connect("clicked(bool)", self.onSetupScene)
        self.loadSegButton.connect("clicked(bool)", self.onLoadSegmentation)
        self.prepareSoftButton.connect("clicked(bool)", self.onPrepareSoftBody)
        self.prepareTetButton.connect("clicked(bool)", self.onPrepareTetBody)
        self.playButton.connect("clicked(bool)", self.onPlay)
        self.pauseButton.connect("clicked(bool)", self.onPause)
        self.stepButton.connect("clicked(bool)", self.onStep)
        self.resetButton.connect("clicked(bool)", self.onReset)
        self.showSurfaceCheck.connect("toggled(bool)", self._onShowSurface)
        self.showParticlesCheck.connect("toggled(bool)", self._onShowParticles)
        self.segNodeSelector.connect(
            "currentNodeChanged(vtkMRMLNode*)", self._onSegmentationChanged
        )
        self.tetModelSelector.connect(
            "currentNodeChanged(vtkMRMLNode*)", self._onTetModelChanged
        )
        self.renameAtlasButton.connect("clicked(bool)", self.onApplyAbdomenAtlasNames)
        self.softSolverCombo.connect("currentIndexChanged(int)", self._onSoftSolverChanged)
        self.armCuttingCheck.connect("toggled(bool)", self._onCuttingSettingsChanged)
        self.cutRadiusSpin.connect("valueChanged(double)", self._onCuttingSettingsChanged)
        self.cutDepthSpin.connect("valueChanged(double)", self._onCuttingSettingsChanged)

        self._refreshInstallStatus()
        self._onSoftSolverChanged()
        self._refreshSimButtons(sceneReady=False, playing=False)

    def cleanup(self):
        if self.logic:
            self.logic.configureCutting(False)
            self.logic.stopTimer()

    # ------------------------------------------------------------------ UI --
    def _refreshInstallStatus(self):
        ok, version = self.logic.checkNewton()
        cutting_ok, cutting_version = self.logic.checkNewtonCutting()
        if ok:
            self.installStatusLabel.text = f"installed ({version})"
            self.installButton.enabled = False
        else:
            self.installStatusLabel.text = "not installed"
            self.installButton.enabled = True
        if cutting_ok:
            self.cuttingInstallStatusLabel.text = f"installed ({cutting_version})"
            self.installCuttingButton.enabled = False
        else:
            self.cuttingInstallStatusLabel.text = "not installed"
            self.installCuttingButton.enabled = True
        self.setupButton.enabled = ok
        self.prepareSoftButton.enabled = ok and (
            self.softSolverCombo.currentText != "Hex shape matching" or cutting_ok
        )
        self.prepareTetButton.enabled = ok

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

    def onInstallCutting(self):
        source = self.cuttingSourceEdit.text.strip()
        with slicer.util.tryWithErrorDisplay("newton-cutting install failed", waitCursor=True):
            self.logic.installNewtonCutting(source)
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
        self.segNodeSelector.setCurrentNode(segNode)
        segIds = vtk.vtkStringArray()
        segNode.GetSegmentation().GetSegmentIDs(segIds)
        self.statusLabel.text = (
            f"Loaded '{segNode.GetName()}' with "
            f"{segIds.GetNumberOfValues()} segment(s)."
        )

    def _onSegmentationChanged(self, node):
        self.segTable.setRowCount(0)
        if node is None:
            return
        segIds = vtk.vtkStringArray()
        node.GetSegmentation().GetSegmentIDs(segIds)
        for i in range(segIds.GetNumberOfValues()):
            segId = segIds.GetValue(i)
            segment = node.GetSegmentation().GetSegment(segId)
            name = segment.GetName() if segment else segId
            color = segment.GetColor() if segment else (0.92, 0.55, 0.45)
            self._appendSegmentRow(segId, name, color)

    @staticmethod
    def _makeCheckCell(checked):
        chk = qt.QCheckBox()
        chk.setChecked(bool(checked))
        cell = qt.QWidget()
        lay = qt.QHBoxLayout(cell)
        lay.addWidget(chk)
        lay.setAlignment(qt.Qt.AlignCenter)
        lay.setContentsMargins(0, 0, 0, 0)
        return cell, chk

    def _appendSegmentRow(self, segId, name, color):
        row = self.segTable.rowCount
        self.segTable.insertRow(row)

        simCell, _ = self._makeCheckCell(checked=True)
        self.segTable.setCellWidget(row, 0, simCell)

        lockCell, _ = self._makeCheckCell(checked=False)
        self.segTable.setCellWidget(row, 1, lockCell)

        nameItem = qt.QTableWidgetItem(name)
        nameItem.setData(qt.Qt.UserRole, segId)
        self.segTable.setItem(row, 2, nameItem)

        colorBtn = ctk.ctkColorPickerButton()
        colorBtn.displayColorName = False
        colorBtn.color = qt.QColor.fromRgbF(
            float(color[0]), float(color[1]), float(color[2])
        )
        colorBtn.connect(
            "colorChanged(QColor)",
            lambda c, sid=segId: self.logic.setBodyColor(
                sid, (c.redF(), c.greenF(), c.blueF())
            ),
        )
        self.segTable.setCellWidget(row, 3, colorBtn)

        stiffSpin = qt.QDoubleSpinBox()
        stiffSpin.setRange(0.0, 10000.0)
        stiffSpin.setDecimals(1)
        stiffSpin.setValue(10.0)
        stiffSpin.setSuffix(" N/m")
        self.segTable.setCellWidget(row, 4, stiffSpin)

    def _collectCheckedSegments(self):
        segments = []
        for row in range(self.segTable.rowCount):
            simCell = self.segTable.cellWidget(row, 0)
            simChk = simCell.findChild(qt.QCheckBox) if simCell else None
            if simChk is None or not simChk.isChecked():
                continue
            lockCell = self.segTable.cellWidget(row, 1)
            lockChk = lockCell.findChild(qt.QCheckBox) if lockCell else None
            nameItem = self.segTable.item(row, 2)
            colorBtn = self.segTable.cellWidget(row, 3)
            stiffSpin = self.segTable.cellWidget(row, 4)
            c = colorBtn.color
            segments.append({
                "id": nameItem.data(qt.Qt.UserRole),
                "name": nameItem.text(),
                "locked": bool(lockChk and lockChk.isChecked()),
                "color": (c.redF(), c.greenF(), c.blueF()),
                "stiffness_ke": float(stiffSpin.value),
            })
        return segments

    def _onTetModelChanged(self, node):
        self.tetLabelTable.setRowCount(0)
        if node is None:
            return
        try:
            labels = self.logic.discoverTetModelLabels(node)
        except Exception as exc:
            self.statusLabel.text = f"Tet labels unavailable: {exc}"
            return
        for label_info in labels:
            self._appendTetLabelRow(label_info)
        self.statusLabel.text = (
            f"Found {len(labels)} nonzero label(s) in '{node.GetName()}'."
        )

    def _appendTetLabelRow(self, label_info):
        row = self.tetLabelTable.rowCount
        self.tetLabelTable.insertRow(row)

        simCell, _ = self._makeCheckCell(checked=True)
        self.tetLabelTable.setCellWidget(row, 0, simCell)

        lockCell, _ = self._makeCheckCell(checked=False)
        self.tetLabelTable.setCellWidget(row, 1, lockCell)

        label = int(label_info["label"])
        labelItem = qt.QTableWidgetItem(str(label))
        labelItem.setData(qt.Qt.UserRole, label)
        self.tetLabelTable.setItem(row, 2, labelItem)

        nameItem = qt.QTableWidgetItem(str(label_info.get("name", f"Label {label}")))
        self.tetLabelTable.setItem(row, 3, nameItem)

        color = label_info.get("color", (0.92, 0.55, 0.45))
        colorBtn = ctk.ctkColorPickerButton()
        colorBtn.displayColorName = False
        colorBtn.color = qt.QColor.fromRgbF(
            float(color[0]), float(color[1]), float(color[2])
        )
        colorBtn.enabled = False
        colorBtn.toolTip = "Displayed color from Slicer's Labels color table."
        self.tetLabelTable.setCellWidget(row, 4, colorBtn)

        muSpin = qt.QDoubleSpinBox()
        muSpin.setRange(0.0, 1.0e8)
        muSpin.setDecimals(1)
        muSpin.setSingleStep(1000.0)
        muSpin.setValue(1.0e5)
        muSpin.setSuffix(" Pa")
        self.tetLabelTable.setCellWidget(row, 5, muSpin)

        lambdaSpin = qt.QDoubleSpinBox()
        lambdaSpin.setRange(0.0, 1.0e8)
        lambdaSpin.setDecimals(1)
        lambdaSpin.setSingleStep(1000.0)
        lambdaSpin.setValue(1.0e5)
        lambdaSpin.setSuffix(" Pa")
        self.tetLabelTable.setCellWidget(row, 6, lambdaSpin)

        dampSpin = qt.QDoubleSpinBox()
        dampSpin.setRange(0.0, 1000.0)
        dampSpin.setDecimals(6)
        dampSpin.setSingleStep(0.001)
        dampSpin.setValue(0.001)
        self.tetLabelTable.setCellWidget(row, 7, dampSpin)

    def _collectTetLabelSettings(self):
        settings = {}
        for row in range(self.tetLabelTable.rowCount):
            simCell = self.tetLabelTable.cellWidget(row, 0)
            simChk = simCell.findChild(qt.QCheckBox) if simCell else None
            lockCell = self.tetLabelTable.cellWidget(row, 1)
            lockChk = lockCell.findChild(qt.QCheckBox) if lockCell else None
            labelItem = self.tetLabelTable.item(row, 2)
            nameItem = self.tetLabelTable.item(row, 3)
            colorBtn = self.tetLabelTable.cellWidget(row, 4)
            muSpin = self.tetLabelTable.cellWidget(row, 5)
            lambdaSpin = self.tetLabelTable.cellWidget(row, 6)
            dampSpin = self.tetLabelTable.cellWidget(row, 7)
            if labelItem is None:
                continue
            label = int(labelItem.data(qt.Qt.UserRole))
            c = colorBtn.color
            settings[label] = {
                "sim": bool(simChk and simChk.isChecked()),
                "locked": bool(lockChk and lockChk.isChecked()),
                "name": nameItem.text() if nameItem is not None else f"Label {label}",
                "color": (c.redF(), c.greenF(), c.blueF()),
                "k_mu": float(muSpin.value),
                "k_lambda": float(lambdaSpin.value),
                "k_damp": float(dampSpin.value),
            }
        return settings

    @staticmethod
    def _parseLabelValue(name, fallback):
        import re
        m = re.search(r"\d+", name or "")
        if m:
            try:
                return int(m.group(0))
            except ValueError:
                pass
        return fallback

    def onApplyAbdomenAtlasNames(self):
        node = self.segNodeSelector.currentNode()
        if node is None:
            slicer.util.errorDisplay("Select a segmentation first.")
            return
        renamed = 0
        segmentation = node.GetSegmentation()
        for row in range(self.segTable.rowCount):
            nameItem = self.segTable.item(row, 2)
            if nameItem is None:
                continue
            segId = nameItem.data(qt.Qt.UserRole)
            label = self._parseLabelValue(nameItem.text(), fallback=row + 1)
            newName = self.ABDOMEN_ATLAS_NAMES.get(label)
            if newName is None:
                continue
            segment = segmentation.GetSegment(segId)
            if segment is None:
                continue
            segment.SetName(newName)
            nameItem.setText(newName)
            renamed += 1
        self.statusLabel.text = (
            f"Renamed {renamed} segment(s) using AbdomenAtlas names."
        )

    def onPrepareSoftBody(self):
        segNode = self.segNodeSelector.currentNode()
        if segNode is None:
            slicer.util.errorDisplay("Select a segmentation first.")
            return
        segments = self._collectCheckedSegments()
        if not segments:
            slicer.util.errorDisplay("Check at least one segment to simulate.")
            return
        with slicer.util.tryWithErrorDisplay("Soft body build failed", waitCursor=True):
            if self.softSolverCombo.currentText == "Hex shape matching":
                self.logic.setupHexShapeMatchingFromSegments(
                    segNode=segNode,
                    segments=segments,
                    device=self.deviceCombo.currentText,
                    fps=self.fpsSpin.value,
                    dt_ms=self.timestepSpin.value,
                    substeps=self.substepsSpin.value,
                    iterations=self.iterationsSpin.value,
                    auxViewer=self.auxViewerCombo.currentText,
                    spacing_mm=self.spacingSpin.value,
                    density=self.densitySpin.value,
                    pin_top=self.pinTopCheck.isChecked(),
                    self_collide=self.selfCollideCheck.isChecked(),
                    particle_radius_factor=self.radiusFactorSpin.value,
                    shape_matching_stiffness=self.shapeStiffnessSpin.value,
                    shape_matching_passes=self.shapePassesSpin.value,
                )
            else:
                self.logic.setupSoftBodyFromSegments(
                    segNode=segNode,
                    segments=segments,
                    device=self.deviceCombo.currentText,
                    fps=self.fpsSpin.value,
                    dt_ms=self.timestepSpin.value,
                    substeps=self.substepsSpin.value,
                    iterations=self.iterationsSpin.value,
                    auxViewer=self.auxViewerCombo.currentText,
                    spacing_mm=self.spacingSpin.value,
                    connectivity=int(self.connCombo.currentText),
                    damping_kd=self.dampingSpin.value,
                    density=self.densitySpin.value,
                    pin_top=self.pinTopCheck.isChecked(),
                    disable_springs=self.disableSpringsCheck.isChecked(),
                    self_collide=self.selfCollideCheck.isChecked(),
                    particle_radius_factor=self.radiusFactorSpin.value,
                )
            self.logic.configureCutting(
                self.armCuttingCheck.isChecked(),
                radius_mm=self.cutRadiusSpin.value,
                depth_mm=self.cutDepthSpin.value,
                onStatus=self._onStatus,
            )
        self.logic.setSurfaceVisible(self.showSurfaceCheck.isChecked())
        self.logic.setParticlesVisible(self.showParticlesCheck.isChecked())
        self.statusLabel.text = (
            f"Soft body ready ({self.softSolverCombo.currentText}, "
            f"{self.logic.particleCount()} particles, "
            f"{self.logic.springCount()} springs). Press Play."
        )
        self._refreshSimButtons(sceneReady=True, playing=False)

    def onPrepareTetBody(self):
        modelNode = self.tetModelSelector.currentNode()
        if modelNode is None:
            slicer.util.errorDisplay("Select a tetrahedral model first.")
            return
        labelSettings = self._collectTetLabelSettings()
        if not labelSettings:
            slicer.util.errorDisplay(
                "The selected model has no nonzero cell labels named 'labels'."
            )
            return
        if not any(v["sim"] or v["locked"] for v in labelSettings.values()):
            slicer.util.errorDisplay("Enable Sim or Lock for at least one label.")
            return
        with slicer.util.tryWithErrorDisplay("Tet FEM build failed", waitCursor=True):
            self.logic.setupTetFEMFromModel(
                modelNode=modelNode,
                labelSettings=labelSettings,
                solverName=self.tetSolverCombo.currentText,
                device=self.deviceCombo.currentText,
                fps=self.fpsSpin.value,
                dt_ms=self.timestepSpin.value,
                substeps=self.substepsSpin.value,
                iterations=self.iterationsSpin.value,
                auxViewer=self.auxViewerCombo.currentText,
                density=self.densitySpin.value,
                pin_top=self.pinTopCheck.isChecked(),
                self_collide=self.selfCollideCheck.isChecked(),
                particle_radius_factor=self.radiusFactorSpin.value,
            )
            self.logic.configureCutting(False)
        if not self.showSurfaceCheck.isChecked():
            was_blocked = self.showSurfaceCheck.blockSignals(True)
            self.showSurfaceCheck.setChecked(True)
            self.showSurfaceCheck.blockSignals(was_blocked)
        self.logic.setSurfaceVisible(True)
        self.logic.setParticlesVisible(self.showParticlesCheck.isChecked())
        self.statusLabel.text = (
            f"Tet FEM ready ({self.tetSolverCombo.currentText}, "
            f"{self.logic.particleCount()} nodes, "
            f"{self.logic.tetCount()} tets, "
            f"{self.logic.tetFreeCount()} free, ground contact on). Press Play."
        )
        self._refreshSimButtons(sceneReady=True, playing=False)

    def _onSoftSolverChanged(self, *args):
        is_hex = self.softSolverCombo.currentText == "Hex shape matching"
        self.connCombo.enabled = not is_hex
        self.dampingSpin.enabled = not is_hex
        self.disableSpringsCheck.enabled = not is_hex
        self.shapeStiffnessSpin.enabled = is_hex
        self.shapePassesSpin.enabled = is_hex
        self.armCuttingCheck.enabled = is_hex
        self.cutRadiusSpin.enabled = is_hex
        self.cutDepthSpin.enabled = is_hex
        self._refreshInstallStatus()

    def _onCuttingSettingsChanged(self, *args):
        self.logic.configureCutting(
            self.armCuttingCheck.isChecked(),
            radius_mm=self.cutRadiusSpin.value,
            depth_mm=self.cutDepthSpin.value,
            onStatus=self._onStatus,
        )

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
    MODE_HEX = "hex"
    MODE_TET = "tet"

    SOFTBODY_MAX_PARTICLES = 80000
    HEX_MAX_ELEMENTS = 500000
    TET_MAX_ELEMENTS = 500000
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

        self._softBodies = []
        self._particlesRestM = None

        self._hexPg = None
        self._hexClusters = None
        self._hexDeleteState = None
        self._hexMcTables = None
        self._hexMcBuffers = None
        self._hexMcFactor = 0.0
        self._hexSurfaceNode = None
        self._hexParticleVizNode = None
        self._hexTriCount = 0
        self._hexRenderedTopologyRevision = None
        self._hexPicker = None
        self._hexDevice = None

        self._tetOutputNode = None
        self._tetSurfaceNode = None
        self._tetParticleVizNode = None
        self._tetRestPointsM = None
        self._tetIndices = None
        self._tetLabels = None
        self._tetSurfaceTriangles = None
        self._tetSolverName = ""
        self._tetSelectedLabelCount = 0
        self._tetFixedCount = 0
        self._tetFreeCount = 0
        self._tetSelfCollideEnabled = False
        self._tetCollisionEnabled = False

        self._cuttingEnabled = False
        self._cutRadiusM = 0.003
        self._cutDepthM = 0.06
        self._cutStatusCb = None
        self._cuttingObserverTags = []

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
    def checkNewtonCutting():
        try:
            import newton_cutting  # noqa: F401

            version = getattr(newton_cutting, "__version__", "unknown")
            return True, version
        except Exception:
            return False, None

    @staticmethod
    def installNewton(source="newton[sim,importers]"):
        slicer.util.pip_install(source)

    @staticmethod
    def installNewtonCutting(source="--no-deps -e G:/warp/warp-cutting"):
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
    def setupSoftBodyFromSegments(
        self,
        segNode,
        segments,
        device,
        fps,
        dt_ms,
        substeps,
        iterations,
        auxViewer,
        spacing_mm,
        connectivity,
        damping_kd,
        density,
        pin_top,
        disable_springs=False,
        self_collide=True,
        particle_radius_factor=0.35,
    ):
        import newton
        import warp as wp
        from newton import ParticleFlags

        self.reset()
        if not segments:
            raise RuntimeError("No segments selected.")
        wp.set_device(device)
        self._applyTiming(fps, dt_ms, substeps)

        bodies = []
        for seg in segments:
            imagedata, ijk_to_ras = self._exportSegmentLabelmap(segNode, seg["id"])
            positions_m, grid_ijk, spacing_m = self._particleGridFromLabelmap(
                imagedata, ijk_to_ras, spacing_mm
            )
            if positions_m.shape[0] == 0:
                raise RuntimeError(
                    f"Segment '{seg['name']}' has no voxels at the chosen spacing."
                )
            locked = bool(seg.get("locked", False))
            if locked:
                pinned_mask = np.ones(positions_m.shape[0], dtype=bool)
            else:
                pinned_mask = self._pinMask(positions_m, spacing_m, pin_top)
            bodies.append({
                "segment_id": seg["id"],
                "name": seg["name"],
                "stiffness_ke": float(seg["stiffness_ke"]),
                "color": tuple(seg.get("color", (0.92, 0.55, 0.45))),
                "locked": locked,
                "imagedata": imagedata,
                "ijk_to_ras": ijk_to_ras,
                "positions_m": positions_m,
                "grid_ijk": grid_ijk,
                "spacing_m": spacing_m,
                "pinned_mask": pinned_mask,
            })

        total_particles = sum(b["positions_m"].shape[0] for b in bodies)
        if total_particles > self.SOFTBODY_MAX_PARTICLES:
            raise RuntimeError(
                f"{total_particles} particles exceeds the limit of "
                f"{self.SOFTBODY_MAX_PARTICLES}. Increase particle spacing "
                f"or uncheck segments."
            )

        builder = newton.ModelBuilder()
        min_spacing_m = min(b["spacing_m"] for b in bodies)
        radius_factor = float(particle_radius_factor)
        builder.default_particle_radius = max(min_spacing_m * radius_factor, 1e-4)
        if not pin_top:
            global_min_z = min(float(b["positions_m"][:, 2].min()) for b in bodies)
            builder.add_ground_plane(height=global_min_z - 2.0 * min_spacing_m)

        active_flags = int(ParticleFlags.ACTIVE)
        pinned_flags = active_flags & ~int(ParticleFlags.ACTIVE)

        total_spring_count = 0
        for body in bodies:
            positions_m = body["positions_m"]
            pinned_mask = body["pinned_mask"]
            spacing_m = body["spacing_m"]
            particle_mass = float(density) * (spacing_m ** 3)
            masses = np.where(pinned_mask, 0.0, particle_mass).astype(np.float32)
            radius_body = max(spacing_m * radius_factor, 1e-4)

            body["particle_offset"] = builder.particle_count
            body["particle_count"] = positions_m.shape[0]

            for i in range(positions_m.shape[0]):
                p = positions_m[i]
                flags = pinned_flags if pinned_mask[i] else active_flags
                builder.add_particle(
                    pos=(float(p[0]), float(p[1]), float(p[2])),
                    vel=(0.0, 0.0, 0.0),
                    mass=float(masses[i]),
                    radius=radius_body,
                    flags=flags,
                )

            if not disable_springs:
                edges = self._enumerateGridEdges(body["grid_ijk"], connectivity)
                offset = body["particle_offset"]
                ke = body["stiffness_ke"]
                for i, j in edges:
                    builder.add_spring(
                        int(i) + offset, int(j) + offset,
                        ke=float(ke), kd=float(damping_kd), control=0.0,
                    )
                total_spring_count += len(edges)

        self._spring_count = total_spring_count
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

        global_rest_m = np.zeros((total_particles, 3), dtype=np.float32)
        body_records = []
        for body in bodies:
            positions_m = body["positions_m"]
            offset = body["particle_offset"]
            count = body["particle_count"]
            global_rest_m[offset:offset + count] = positions_m

            surfacePolyData, surface_pts_mm = self._extractSurfaceMesh(
                body["imagedata"], body["ijk_to_ras"]
            )
            particle_pts_mm = positions_m * self.SCENE_SCALE_MM
            idxs, weights = self._computeSkinWeights(
                surface_pts_mm, particle_pts_mm, self.SOFTBODY_SKIN_K
            )

            surface_node = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLModelNode", f"NewtonSoftBody_{body['name']}"
            )
            surface_node.SetAndObservePolyData(surfacePolyData)
            surface_node.CreateDefaultDisplayNodes()
            disp = surface_node.GetDisplayNode()
            disp.SetScalarVisibility(False)
            disp.SetColor(*body["color"])
            disp.SetOpacity(0.9)

            particle_viz_node = self._buildParticleVizNode(
                particle_pts_mm, name=f"NewtonParticles_{body['name']}"
            )

            normals_filter = vtk.vtkPolyDataNormals()
            normals_filter.SetInputData(surfacePolyData)
            normals_filter.ComputePointNormalsOn()
            normals_filter.ConsistencyOn()
            normals_filter.SplittingOff()
            normals_filter.Update()

            body_records.append({
                "segment_id": body["segment_id"],
                "name": body["name"],
                "offset": offset,
                "count": count,
                "surface_node": surface_node,
                "surface_rest_mm": surface_pts_mm.astype(np.float32),
                "particle_viz_node": particle_viz_node,
                "skin_idxs": idxs,
                "skin_weights": weights.astype(np.float32),
                "normals_filter": normals_filter,
            })

        self._particlesRestM = global_rest_m
        self._softBodies = body_records

        self._mode = self.MODE_SOFTBODY
        self._pushSoftBodyToMrml()
        self._logToAuxViewer()

    def setupHexShapeMatchingFromSegments(
        self,
        segNode,
        segments,
        device,
        fps,
        dt_ms,
        substeps,
        iterations,
        auxViewer,
        spacing_mm,
        density,
        pin_top,
        self_collide=True,
        particle_radius_factor=0.35,
        shape_matching_stiffness=0.75,
        shape_matching_passes=1,
    ):
        import warp as wp
        from newton_cutting.corner_delete import make_corner_deletion_state
        from newton_cutting.corner_grid import (
            build_corner_grid,
            build_corner_shape_matching_clusters,
        )
        from newton_cutting.corner_solver import (
            SHAPE_MATCHING_SOLVE_SCATTER,
            SolverCornerShapeMatching,
        )
        from newton_cutting.kernels.corner_render import HoverPicker
        from newton_cutting.kernels.marching_cubes import (
            allocate_mc_buffers,
            bake_vertex_uv3,
            upload_mc_tables,
        )

        self.reset()
        if not segments:
            raise RuntimeError("No segments selected.")

        wp.set_device(device)
        self._applyTiming(fps, dt_ms, substeps)

        atlas, locked_material_ids = self._hexAtlasFromSegments(
            segNode=segNode,
            segments=segments,
            spacing_mm=spacing_mm,
            density=density,
        )
        active_cells = int(np.count_nonzero(atlas.labels))
        if active_cells == 0:
            raise RuntimeError("Selected segments have no occupied cells at the chosen spacing.")
        if active_cells > self.HEX_MAX_ELEMENTS:
            raise RuntimeError(
                f"{active_cells} hex cells exceeds the limit of {self.HEX_MAX_ELEMENTS}. "
                "Increase particle spacing or uncheck segments."
            )

        spacing_m = float(atlas.voxel_size)
        radius = max(spacing_m * float(particle_radius_factor), 1.0e-5)
        pg = build_corner_grid(
            atlas,
            edge_ke=0.0,
            edge_kd=0.0,
            face_ke=0.0,
            face_kd=0.0,
            body_ke=0.0,
            body_kd=0.0,
            origin=atlas.origin,
            particle_radius=radius,
            device=device,
            kinematic_bones=True,
        )
        if pg.model.particle_count > self.HEX_MAX_ELEMENTS:
            raise RuntimeError(
                f"{pg.model.particle_count} hex nodes exceeds the limit of "
                f"{self.HEX_MAX_ELEMENTS}. Increase particle spacing."
            )

        clusters = build_corner_shape_matching_clusters(pg)
        delete_state = make_corner_deletion_state(pg.model, pg.aux)

        locked_nodes = []
        if locked_material_ids:
            cell_material = pg.aux.cell_material.numpy()
            locked_cells = np.nonzero(np.isin(cell_material, np.asarray(locked_material_ids, dtype=np.int32)))[0]
            if locked_cells.size > 0:
                locked_nodes.append(pg.aux.cell_nodes_host[locked_cells].reshape(-1))
        if pin_top:
            q = pg.model.particle_q.numpy()
            top = float(q[:, 2].max()) - max(spacing_m, 1.0e-6) * 0.6
            locked_nodes.append(np.nonzero(q[:, 2] >= top)[0].astype(np.int32))
        if locked_nodes:
            nodes = np.unique(np.concatenate(locked_nodes).astype(np.int32, copy=False))
            if nodes.size > 0:
                delete_state.lock_nodes(nodes)

        ground_height = 0.0
        enable_ground = not bool(pin_top)
        if enable_ground:
            q = pg.model.particle_q.numpy()
            ground_height = float(q[:, 2].min()) - 2.0 * spacing_m

        solver = SolverCornerShapeMatching(
            pg.model,
            clusters,
            iterations=int(iterations),
            enable_springs=False,
            enable_shape_matching=True,
            enable_self_collisions=bool(self_collide),
            enable_ground_plane=enable_ground,
            shape_matching_stiffness=float(shape_matching_stiffness),
            shape_matching_passes=int(shape_matching_passes),
            shape_matching_mode=SHAPE_MATCHING_SOLVE_SCATTER,
            ground_height=ground_height,
        )

        self._model = pg.model
        self._solver = solver
        self._state_0 = self._model.state()
        self._state_1 = self._model.state()
        self._control = None
        self._contacts = None
        self._frame = 0
        self._spring_count = 0
        self._hexPg = pg
        self._hexClusters = clusters
        self._hexDeleteState = delete_state
        self._hexDevice = pg.model.device
        self._hexMcTables = upload_mc_tables(device=pg.model.device)
        self._hexMcBuffers = allocate_mc_buffers(
            pg.aux.grid_shape,
            pg.aux.num_cells,
            device=pg.model.device,
        )
        self._hexMcFactor = spacing_m * 0.5
        bake_vertex_uv3(
            self._hexMcBuffers,
            pg.aux.cell_grid_xyz,
            pg.aux.grid_shape,
            device=pg.model.device,
        )
        self._hexPicker = HoverPicker(pg.aux.num_cells, pg.model.device)

        self._auxViewer = self._createAuxViewer(auxViewer)
        if self._auxViewer is not None:
            self._auxViewer.set_model(self._model)

        self._buildHexMrmlNodes(segments)
        self._mode = self.MODE_HEX
        self._pushHexToMrml(force_topology=True)
        self._logToAuxViewer()

    # ------------------------------------------------------- tetrahedral FEM
    def setupTetFEMFromModel(
        self,
        modelNode,
        labelSettings,
        solverName,
        device,
        fps,
        dt_ms,
        substeps,
        iterations,
        auxViewer,
        density,
        pin_top,
        self_collide=True,
        particle_radius_factor=0.35,
    ):
        import newton
        import warp as wp
        from newton import ParticleFlags

        self.reset()
        if modelNode is None:
            raise RuntimeError("No tetrahedral model selected.")

        ug = self._modelNodeUnstructuredGrid(modelNode)
        tet_data = self._extractTetFEMDataFromUnstructuredGrid(
            ug,
            labelSettings,
            density=float(density),
            pin_top=bool(pin_top),
        )
        tet_count = int(tet_data["tets"].shape[0])
        if tet_count > self.TET_MAX_ELEMENTS:
            raise RuntimeError(
                f"{tet_count} tetrahedra exceeds the limit of {self.TET_MAX_ELEMENTS}. "
                "Uncheck labels or simplify the SegmentMesher output."
            )

        wp.set_device(device)
        self._applyTiming(fps, dt_ms, substeps)

        points_m = tet_data["points_m"]
        tets = tet_data["tets"]
        materials = tet_data["materials"]
        volumes_m3 = tet_data["volumes_m3"]
        fixed_mask = tet_data["fixed_mask"]
        char_spacing_m = max(float(tet_data["characteristic_spacing_m"]), 1.0e-5)
        radius = max(char_spacing_m * float(particle_radius_factor), 1.0e-5)

        builder = newton.ModelBuilder()
        builder.default_particle_radius = radius
        builder.add_ground_plane(height=float(points_m[:, 2].min()) - 2.0 * char_spacing_m)

        active_flags = int(ParticleFlags.ACTIVE)
        pinned_flags = active_flags & ~int(ParticleFlags.ACTIVE)
        for i, p in enumerate(points_m):
            flags = pinned_flags if fixed_mask[i] else active_flags
            builder.add_particle(
                pos=wp.vec3(float(p[0]), float(p[1]), float(p[2])),
                vel=wp.vec3(0.0, 0.0, 0.0),
                mass=0.0,
                radius=radius,
                flags=flags,
            )

        for tet_id, tet in enumerate(tets):
            k_mu, k_lambda, k_damp = (float(v) for v in materials[tet_id])
            volume = builder.add_tetrahedron(
                int(tet[0]),
                int(tet[1]),
                int(tet[2]),
                int(tet[3]),
                k_mu=k_mu,
                k_lambda=k_lambda,
                k_damp=k_damp,
            )
            if volume <= 0.0:
                volume = float(volumes_m3[tet_id])
            nodal_mass = float(density) * float(volume) / 4.0
            for node_id in tet:
                builder.particle_mass[int(node_id)] += nodal_mass

        for node_id in np.nonzero(fixed_mask)[0].tolist():
            builder.particle_mass[int(node_id)] = 0.0
            builder.particle_flags[int(node_id)] = pinned_flags

        use_vbd = "VBD" in str(solverName).upper()
        if use_vbd:
            builder.color()

        self._model = builder.finalize(device=device)
        # Tet v1 only supports contact against scene shapes (the auto ground plane).
        # Particle self-collision / organ-organ contact is deliberately disabled:
        # dense tet vertices create very large contact sets and can stall Slicer's UI.
        self._model.particle_max_radius = 0.0
        self._model.particle_grid = None
        self._tetSelfCollideEnabled = False
        self._tetCollisionEnabled = True

        if use_vbd:
            self._solver = newton.solvers.SolverVBD(
                self._model,
                iterations=int(iterations),
                particle_enable_self_contact=False,
                particle_self_contact_radius=radius,
                particle_self_contact_margin=max(radius * 1.5, radius + 1.0e-6),
                particle_enable_tile_solve=False,
            )
        else:
            self._solver = newton.solvers.SolverXPBD(
                self._model,
                iterations=int(iterations),
            )

        self._state_0 = self._model.state()
        self._state_1 = self._model.state()
        self._control = self._model.control()
        self._contacts = self._model.contacts()
        self._frame = 0
        self._spring_count = 0

        self._tetOutputNode = self._buildTetOutputNode(modelNode, tet_data)
        self._tetSurfaceNode = self._buildTetSurfaceNode(modelNode, tet_data)
        self._tetParticleVizNode = self._buildParticleVizNode(
            points_m * self.SCENE_SCALE_MM,
            name=f"NewtonTetNodes_{modelNode.GetName()}",
        )
        self._tetRestPointsM = points_m.astype(np.float32, copy=True)
        self._tetIndices = tets.astype(np.int32, copy=True)
        self._tetLabels = tet_data["labels"].astype(np.int32, copy=True)
        self._tetSurfaceTriangles = tet_data["surface_triangles"].astype(np.int32, copy=True)
        self._tetSolverName = str(solverName)
        self._tetSelectedLabelCount = len(tet_data["selected_labels"])
        self._tetFixedCount = int(np.count_nonzero(fixed_mask))
        self._tetFreeCount = int(points_m.shape[0] - self._tetFixedCount)

        self._softBodies = [{
            "segment_id": "__tet__",
            "name": "Tet FEM",
            "surface_node": self._tetSurfaceNode,
            "volume_node": self._tetOutputNode,
            "particle_viz_node": self._tetParticleVizNode,
        }]

        self._auxViewer = self._createAuxViewer(auxViewer)
        if self._auxViewer is not None:
            self._auxViewer.set_model(self._model)

        self._mode = self.MODE_TET
        self._pushTetToMrml()
        self._logToAuxViewer()

    @staticmethod
    def _modelNodeUnstructuredGrid(modelNode):
        ug = None
        if modelNode is not None:
            try:
                ug = modelNode.GetUnstructuredGrid()
            except AttributeError:
                ug = None
            if ug is None:
                try:
                    mesh = modelNode.GetMesh()
                except AttributeError:
                    mesh = None
                if mesh is not None and mesh.IsA("vtkUnstructuredGrid"):
                    ug = mesh
        if ug is None or not ug.IsA("vtkUnstructuredGrid"):
            raise RuntimeError(
                "Tet FEM input must be a vtkMRMLModelNode containing a vtkUnstructuredGrid."
            )
        return ug

    @staticmethod
    def _requireTetLabelArray(ug):
        if ug is None or not ug.IsA("vtkUnstructuredGrid"):
            raise RuntimeError("Tet FEM input must be a vtkUnstructuredGrid.")
        cell_data = ug.GetCellData()
        labels = cell_data.GetArray("labels") if cell_data is not None else None
        if labels is None:
            raise RuntimeError("Tet FEM input requires a cell-data scalar array named 'labels'.")
        if labels.GetNumberOfTuples() < ug.GetNumberOfCells():
            raise RuntimeError("The 'labels' array has fewer tuples than mesh cells.")
        return labels

    @classmethod
    def _discoverTetLabelsInUnstructuredGrid(cls, ug):
        from vtk.util import numpy_support as vns

        labels_vtk = cls._requireTetLabelArray(ug)
        labels = vns.vtk_to_numpy(labels_vtk)[: ug.GetNumberOfCells()]
        labels = np.asarray(labels).astype(np.int64, copy=False)
        return [int(v) for v in np.unique(labels) if int(v) != 0]

    def discoverTetModelLabels(self, modelNode):
        ug = self._modelNodeUnstructuredGrid(modelNode)
        labels = self._discoverTetLabelsInUnstructuredGrid(ug)
        color_node = self._labelsColorNode()
        return [
            self._labelInfoFromColorNode(label, color_node)
            for label in labels
        ]

    @classmethod
    def _extractTetFEMDataFromUnstructuredGrid(
        cls,
        ug,
        labelSettings,
        density,
        pin_top,
    ):
        from vtk.util import numpy_support as vns

        labels_vtk = cls._requireTetLabelArray(ug)
        raw_labels = vns.vtk_to_numpy(labels_vtk)[: ug.GetNumberOfCells()]
        settings = cls._normaliseTetLabelSettings(labelSettings)

        points_vtk = ug.GetPoints()
        if points_vtk is None or points_vtk.GetNumberOfPoints() == 0:
            raise RuntimeError("Tet FEM input has no points.")

        point_map = {}
        source_point_ids = []
        points_mm = []
        tets = []
        labels = []
        materials = []
        volumes_m3 = []
        source_cell_ids = []

        id_list = vtk.vtkIdList()
        for cell_id in range(ug.GetNumberOfCells()):
            cell_type = ug.GetCellType(cell_id)
            if cell_type != vtk.VTK_TETRA:
                raise RuntimeError(
                    f"Tet FEM only accepts linear VTK_TETRA cells. "
                    f"Cell {cell_id} has VTK type {cell_type}."
                )
            label = int(raw_labels[cell_id])
            if label == 0:
                continue
            setting = settings.get(label, cls._defaultTetLabelSetting(label))
            if not (setting["sim"] or setting["locked"]):
                continue

            ug.GetCellPoints(cell_id, id_list)
            if id_list.GetNumberOfIds() != 4:
                raise RuntimeError(f"Cell {cell_id} is not a 4-node tetrahedron.")
            old_ids = [int(id_list.GetId(i)) for i in range(4)]
            cell_points_mm = np.asarray(
                [points_vtk.GetPoint(pid) for pid in old_ids],
                dtype=np.float64,
            )
            volume = cls._tetVolumeM3(cell_points_mm / cls.SCENE_SCALE_MM)
            eps = cls._tetVolumeEpsM3(cell_points_mm / cls.SCENE_SCALE_MM)
            if abs(volume) <= eps:
                raise RuntimeError(
                    f"Tetrahedron cell {cell_id} has near-zero volume ({volume:.3e} m^3)."
                )

            compact = []
            for old_id in old_ids:
                new_id = point_map.get(old_id)
                if new_id is None:
                    new_id = len(points_mm)
                    point_map[old_id] = new_id
                    source_point_ids.append(old_id)
                    points_mm.append(points_vtk.GetPoint(old_id))
                compact.append(new_id)

            if volume < 0.0:
                compact[1], compact[2] = compact[2], compact[1]
                volume = -volume

            tets.append(compact)
            labels.append(label)
            materials.append([setting["k_mu"], setting["k_lambda"], setting["k_damp"]])
            volumes_m3.append(volume)
            source_cell_ids.append(cell_id)

        if not tets:
            raise RuntimeError("No nonzero tetrahedra are enabled for Tet FEM.")

        points_mm = np.asarray(points_mm, dtype=np.float32)
        points_m = (points_mm / cls.SCENE_SCALE_MM).astype(np.float32)
        tets = np.asarray(tets, dtype=np.int32)
        labels = np.asarray(labels, dtype=np.int32)
        materials = np.asarray(materials, dtype=np.float32)
        volumes_m3 = np.asarray(volumes_m3, dtype=np.float32)
        char_spacing_m = cls._tetCharacteristicSpacingM(points_m, tets)

        locked_labels = {
            int(label)
            for label in np.unique(labels)
            if settings.get(int(label), cls._defaultTetLabelSetting(int(label)))["locked"]
        }
        fixed_mask = np.zeros(points_m.shape[0], dtype=bool)
        if locked_labels:
            for tet, label in zip(tets, labels):
                if int(label) in locked_labels:
                    fixed_mask[tet] = True
        elif pin_top:
            fixed_mask = cls._pinMask(points_m, char_spacing_m, pin_top=True)

        selected_labels = sorted(int(v) for v in np.unique(labels))
        surface_triangles, surface_labels = cls._surfaceTrianglesAndLabelsForTets(tets, labels)
        return {
            "points_mm": points_mm,
            "points_m": points_m,
            "tets": tets,
            "labels": labels,
            "materials": materials,
            "volumes_m3": volumes_m3,
            "fixed_mask": fixed_mask,
            "selected_labels": selected_labels,
            "locked_labels": sorted(locked_labels),
            "source_point_ids": np.asarray(source_point_ids, dtype=np.int64),
            "source_cell_ids": np.asarray(source_cell_ids, dtype=np.int64),
            "surface_triangles": surface_triangles,
            "surface_labels": surface_labels,
            "characteristic_spacing_m": float(char_spacing_m),
            "density": float(density),
        }

    @staticmethod
    def _normaliseTetLabelSettings(labelSettings):
        settings = {}
        for key, raw in (labelSettings or {}).items():
            label = int(key)
            raw = raw or {}
            default = NewtonPhysicsLogic._defaultTetLabelSetting(label)
            settings[label] = {
                "sim": bool(raw.get("sim", default["sim"])),
                "locked": bool(raw.get("locked", default["locked"])),
                "name": str(raw.get("name", default["name"])),
                "color": tuple(float(v) for v in raw.get("color", default["color"])),
                "k_mu": float(raw.get("k_mu", default["k_mu"])),
                "k_lambda": float(raw.get("k_lambda", default["k_lambda"])),
                "k_damp": float(raw.get("k_damp", default["k_damp"])),
            }
        return settings

    @staticmethod
    def _defaultTetLabelSetting(label):
        return {
            "sim": int(label) != 0,
            "locked": False,
            "name": f"Label {int(label)}",
            "color": NewtonPhysicsLogic._fallbackLabelColor(label),
            "k_mu": 1.0e5,
            "k_lambda": 1.0e5,
            "k_damp": 0.001,
        }

    @staticmethod
    def _fallbackLabelColor(label):
        palette = [
            (0.89, 0.25, 0.20),
            (0.20, 0.53, 0.91),
            (0.18, 0.65, 0.36),
            (0.94, 0.68, 0.18),
            (0.58, 0.35, 0.78),
            (0.10, 0.68, 0.73),
            (0.85, 0.38, 0.60),
            (0.55, 0.55, 0.55),
        ]
        return palette[(max(int(label), 1) - 1) % len(palette)]

    @classmethod
    def _labelInfoFromColorNode(cls, label, color_node):
        name = f"Label {int(label)}"
        color = cls._fallbackLabelColor(label)
        if color_node is not None:
            try:
                candidate = color_node.GetColorName(int(label))
                if candidate:
                    name = str(candidate)
            except Exception:
                pass
            rgba = [0.0, 0.0, 0.0, 0.0]
            try:
                ok = color_node.GetColor(int(label), rgba)
                if ok is None or ok:
                    color = (float(rgba[0]), float(rgba[1]), float(rgba[2]))
            except Exception:
                pass
        return {"label": int(label), "name": name, "color": color}

    @staticmethod
    def _tetVolumeM3(points_m):
        p0, p1, p2, p3 = np.asarray(points_m, dtype=np.float64).reshape(4, 3)
        return float(np.linalg.det(np.column_stack((p1 - p0, p2 - p0, p3 - p0))) / 6.0)

    @staticmethod
    def _tetVolumeEpsM3(points_m):
        pts = np.asarray(points_m, dtype=np.float64).reshape(4, 3)
        pairs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
        max_edge = max(float(np.linalg.norm(pts[i] - pts[j])) for i, j in pairs)
        return max(1.0e-18, (max_edge ** 3) * 1.0e-12)

    @staticmethod
    def _tetCharacteristicSpacingM(points_m, tets):
        if points_m.size == 0 or tets.size == 0:
            return 1.0e-3
        pairs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
        lengths = []
        for i, j in pairs:
            lengths.append(np.linalg.norm(points_m[tets[:, i]] - points_m[tets[:, j]], axis=1))
        lengths = np.concatenate(lengths)
        lengths = lengths[np.isfinite(lengths) & (lengths > 0.0)]
        if lengths.size:
            return max(float(np.median(lengths)), 1.0e-5)
        bounds = points_m.max(axis=0) - points_m.min(axis=0)
        return max(float(np.linalg.norm(bounds)), 1.0e-5)

    @staticmethod
    def _surfaceTrianglesForTets(tets):
        triangles, _ = NewtonPhysicsLogic._surfaceTrianglesAndLabelsForTets(
            tets,
            np.zeros(np.asarray(tets, dtype=np.int32).reshape(-1, 4).shape[0], dtype=np.int32),
        )
        return triangles

    @staticmethod
    def _surfaceTrianglesAndLabelsForTets(tets, labels):
        tets = np.asarray(tets, dtype=np.int32).reshape(-1, 4)
        labels = np.asarray(labels, dtype=np.int32).reshape(-1)
        if labels.shape[0] != tets.shape[0]:
            raise RuntimeError("Surface extraction requires one label per tetrahedron.")
        if tets.shape[0] == 0:
            return np.empty((0, 3), dtype=np.int32), np.empty((0,), dtype=np.int32)
        face_idx = np.asarray(
            [
                [0, 2, 1],
                [1, 2, 3],
                [0, 1, 3],
                [0, 3, 2],
            ],
            dtype=np.int32,
        )
        all_faces = tets[:, face_idx].reshape(-1, 3)
        all_labels = np.repeat(labels, 4)
        sorted_faces = np.sort(all_faces, axis=1)
        _, inverse, counts = np.unique(
            sorted_faces,
            axis=0,
            return_inverse=True,
            return_counts=True,
        )
        boundary_mask = counts[inverse] == 1
        return (
            all_faces[boundary_mask].astype(np.int32, copy=False),
            all_labels[boundary_mask].astype(np.int32, copy=False),
        )

    @staticmethod
    def _addTetCollisionSurface(builder, tets):
        surface_tris = NewtonPhysicsLogic._surfaceTrianglesForTets(tets)
        start_tri = len(builder.tri_indices)
        for tri in surface_tris:
            builder.add_triangle(
                int(tri[0]),
                int(tri[1]),
                int(tri[2]),
                tri_ke=0.0,
                tri_ka=0.0,
                tri_kd=0.0,
                tri_drag=0.0,
                tri_lift=0.0,
            )
        if len(builder.tri_indices) <= start_tri:
            return
        try:
            from newton._src.utils.mesh import MeshAdjacency
        except Exception:
            return
        adj = MeshAdjacency(builder.tri_indices[start_tri:])
        for edge in adj.edges.values():
            builder.add_edge(
                int(edge.o0),
                int(edge.o1),
                int(edge.v0),
                int(edge.v1),
                None,
                edge_ke=0.0,
                edge_kd=0.0,
            )

    def _buildTetOutputNode(self, sourceNode, tet_data):
        output_grid = self._buildTetOutputGrid(tet_data)
        name = f"NewtonTet_{sourceNode.GetName()}"
        node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", name)
        self._setModelNodeMesh(node, output_grid)
        node.CreateDefaultDisplayNodes()
        self._configureTetDisplayNode(node)
        if node.GetDisplayNode() is not None:
            node.GetDisplayNode().SetVisibility(True)
            node.GetDisplayNode().SetOpacity(0.15)
        return node

    def _buildTetSurfaceNode(self, sourceNode, tet_data):
        surface = self._buildTetSurfacePolyData(tet_data)
        name = f"NewtonTetSurface_{sourceNode.GetName()}"
        node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", name)
        node.SetAndObservePolyData(surface)
        node.CreateDefaultDisplayNodes()
        self._configureTetDisplayNode(node)
        if node.GetDisplayNode() is not None:
            node.GetDisplayNode().SetVisibility(True)
        return node

    @staticmethod
    def _buildTetOutputGrid(tet_data):
        from vtk.util import numpy_support as vns

        points = vtk.vtkPoints()
        points.SetData(
            vns.numpy_to_vtk(
                np.ascontiguousarray(tet_data["points_mm"], dtype=np.float32),
                deep=True,
            )
        )
        grid = vtk.vtkUnstructuredGrid()
        grid.SetPoints(points)
        ids = vtk.vtkIdList()
        ids.SetNumberOfIds(4)
        for tet in tet_data["tets"]:
            for i in range(4):
                ids.SetId(i, int(tet[i]))
            grid.InsertNextCell(vtk.VTK_TETRA, ids)

        label_array = vns.numpy_to_vtk(
            np.ascontiguousarray(tet_data["labels"], dtype=np.int32),
            deep=True,
            array_type=vtk.VTK_INT,
        )
        label_array.SetName("labels")
        grid.GetCellData().AddArray(label_array)
        grid.GetCellData().SetScalars(label_array)
        grid.GetCellData().SetActiveScalars("labels")
        return grid

    @staticmethod
    def _buildTetSurfacePolyData(tet_data):
        from vtk.util import numpy_support as vns

        points = vtk.vtkPoints()
        points.SetData(
            vns.numpy_to_vtk(
                np.ascontiguousarray(tet_data["points_mm"], dtype=np.float32),
                deep=True,
            )
        )
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)

        tris = np.ascontiguousarray(tet_data["surface_triangles"], dtype=np.int64)
        cells = vtk.vtkCellArray()
        if tris.size > 0:
            offsets = np.arange(tris.shape[0] + 1, dtype=np.int64) * 3
            connectivity = np.ascontiguousarray(tris.reshape(-1), dtype=np.int64)
            try:
                cells.SetData(
                    vns.numpy_to_vtkIdTypeArray(offsets, deep=True),
                    vns.numpy_to_vtkIdTypeArray(connectivity, deep=True),
                )
            except AttributeError:
                for tri in tris.tolist():
                    cells.InsertNextCell(3)
                    cells.InsertCellPoint(int(tri[0]))
                    cells.InsertCellPoint(int(tri[1]))
                    cells.InsertCellPoint(int(tri[2]))
        polydata.SetPolys(cells)

        label_array = vns.numpy_to_vtk(
            np.ascontiguousarray(tet_data["surface_labels"], dtype=np.int32),
            deep=True,
            array_type=vtk.VTK_INT,
        )
        label_array.SetName("labels")
        polydata.GetCellData().AddArray(label_array)
        polydata.GetCellData().SetScalars(label_array)
        polydata.GetCellData().SetActiveScalars("labels")
        polydata.BuildCells()
        return polydata

    @staticmethod
    def _setModelNodeMesh(modelNode, mesh):
        if hasattr(modelNode, "SetAndObserveMesh"):
            modelNode.SetAndObserveMesh(mesh)
            return
        if mesh.IsA("vtkPolyData") and hasattr(modelNode, "SetAndObservePolyData"):
            modelNode.SetAndObservePolyData(mesh)
            return
        raise RuntimeError("This Slicer build cannot attach a mesh to vtkMRMLModelNode.")

    def _configureTetDisplayNode(self, outputNode):
        disp = outputNode.GetDisplayNode()
        if disp is None:
            return
        disp.SetEdgeVisibility(True)
        disp.SetClipping(True)
        disp.ScalarVisibilityOn()
        disp.SetActiveScalarName("labels")
        disp.SetActiveAttributeLocation(vtk.vtkAssignAttribute.CELL_DATA)
        if hasattr(disp, "SetVisibility2D"):
            disp.SetVisibility2D(True)
        else:
            disp.SetSliceIntersectionVisibility(True)
        disp.SetSliceIntersectionOpacity(0.5)
        disp.SetOpacity(0.9)
        color_node = self._labelsColorNode()
        if color_node is not None:
            disp.SetAndObserveColorNodeID(color_node.GetID())
            try:
                disp.SetScalarRangeFlag(slicer.vtkMRMLDisplayNode.UseColorNodeScalarRange)
            except Exception:
                pass

    @staticmethod
    def _labelsColorNode():
        scene = slicer.mrmlScene
        for i in range(scene.GetNumberOfNodesByClass("vtkMRMLColorTableNode")):
            node = scene.GetNthNodeByClass(i, "vtkMRMLColorTableNode")
            if node is not None and node.GetName() == "Labels":
                return node
        try:
            node = slicer.util.getNode("Labels")
            if node is not None and node.IsA("vtkMRMLColorTableNode"):
                return node
        except Exception:
            pass
        return None

    def _pushTetToMrml(self):
        if self._state_0 is None or self._tetOutputNode is None:
            return
        particles_mm = self._state_0.particle_q.numpy().astype(np.float32) * self.SCENE_SCALE_MM
        arr = slicer.util.arrayFromModelPoints(self._tetOutputNode)
        if arr.shape == particles_mm.shape:
            arr[:] = particles_mm
            self._markModelPointsModified(self._tetOutputNode)

        if self._tetSurfaceNode is not None:
            sarr = slicer.util.arrayFromModelPoints(self._tetSurfaceNode)
            if sarr.shape == particles_mm.shape:
                sarr[:] = particles_mm
                self._markModelPointsModified(self._tetSurfaceNode)

        if self._tetParticleVizNode is not None and self._modelNodeVisible(self._tetParticleVizNode):
            parr = slicer.util.arrayFromModelPoints(self._tetParticleVizNode)
            if parr.shape == particles_mm.shape:
                parr[:] = particles_mm
                self._markModelPointsModified(self._tetParticleVizNode)

    @staticmethod
    def _modelNodeVisible(modelNode):
        if modelNode is None:
            return False
        display = modelNode.GetDisplayNode()
        return bool(display is not None and display.GetVisibility())

    @staticmethod
    def _markModelPointsModified(modelNode):
        if modelNode is None:
            return
        mesh = modelNode.GetMesh()
        if mesh is not None:
            points = mesh.GetPoints()
            if points is not None:
                data = points.GetData()
                if data is not None:
                    data.Modified()
                points.Modified()
            mesh.Modified()
        display = modelNode.GetDisplayNode()
        if display is not None:
            display.Modified()
        modelNode.Modified()

    def _buildParticleVizNode(self, positions_mm, name="NewtonParticles"):
        from vtk.util import numpy_support as vns

        pts = vtk.vtkPoints()
        pts.SetData(vns.numpy_to_vtk(np.ascontiguousarray(positions_mm, dtype=np.float32), deep=True))
        pd = vtk.vtkPolyData()
        pd.SetPoints(pts)
        vgf = vtk.vtkVertexGlyphFilter()
        vgf.SetInputData(pd)
        vgf.Update()

        node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", name)
        node.SetAndObservePolyData(vgf.GetOutput())
        node.CreateDefaultDisplayNodes()
        d = node.GetDisplayNode()
        d.SetRepresentation(0)  # 0 = VTK_POINTS
        d.SetPointSize(5.0)
        d.SetColor(0.2, 0.85, 0.25)
        d.SetVisibility(False)
        return node

    def _buildHexMrmlNodes(self, segments):
        from vtk.util import numpy_support as vns

        if self._hexPg is None or self._hexMcBuffers is None:
            return

        n_vertices = int(self._hexMcBuffers.vertex_pos.shape[0])
        points_np = np.zeros((n_vertices, 3), dtype=np.float32)
        pts = vtk.vtkPoints()
        pts.SetData(vns.numpy_to_vtk(points_np, deep=True))
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(pts)
        polydata.SetPolys(vtk.vtkCellArray())

        self._hexSurfaceNode = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLModelNode", "NewtonHexSurface"
        )
        self._hexSurfaceNode.SetAndObservePolyData(polydata)
        self._hexSurfaceNode.CreateDefaultDisplayNodes()
        disp = self._hexSurfaceNode.GetDisplayNode()
        disp.SetScalarVisibility(False)
        if segments:
            disp.SetColor(*tuple(segments[0].get("color", (0.92, 0.55, 0.45))))
        disp.SetOpacity(0.9)

        node_positions_mm = self._hexPg.model.particle_q.numpy().astype(np.float32) * self.SCENE_SCALE_MM
        self._hexParticleVizNode = self._buildParticleVizNode(
            node_positions_mm,
            name="NewtonHexNodes",
        )

        self._softBodies = [{
            "segment_id": "__hex__",
            "name": "Hex shape matching",
            "surface_node": self._hexSurfaceNode,
            "particle_viz_node": self._hexParticleVizNode,
        }]

    def _updateHexSurfaceTopology(self, tri_count):
        if self._hexSurfaceNode is None or self._hexMcBuffers is None:
            return
        from vtk.util import numpy_support as vns

        tri_count = int(tri_count)
        polydata = self._hexSurfaceNode.GetPolyData()
        cells = vtk.vtkCellArray()
        if tri_count > 0:
            tris = self._hexMcBuffers.tri_indices.numpy()[:tri_count].astype(
                np.int64,
                copy=True,
            )
            offsets = (np.arange(tri_count + 1, dtype=np.int64) * 3)
            connectivity = np.ascontiguousarray(tris.reshape(-1), dtype=np.int64)
            try:
                cells.SetData(
                    vns.numpy_to_vtkIdTypeArray(offsets, deep=True),
                    vns.numpy_to_vtkIdTypeArray(connectivity, deep=True),
                )
            except AttributeError:
                for tri in tris.tolist():
                    cells.InsertNextCell(3)
                    cells.InsertCellPoint(int(tri[0]))
                    cells.InsertCellPoint(int(tri[1]))
                    cells.InsertCellPoint(int(tri[2]))
        polydata.SetPolys(cells)
        polydata.Modified()
        self._hexTriCount = tri_count

    def _pushHexToMrml(self, force_topology=False):
        if (
            self._state_0 is None
            or self._hexPg is None
            or self._hexMcTables is None
            or self._hexMcBuffers is None
            or self._hexSurfaceNode is None
        ):
            return

        from newton_cutting.kernels.corner_render import update_cell_render_state
        from newton_cutting.kernels.marching_cubes import (
            compute_mc_topology,
            compute_mc_vertex_positions,
        )

        device = self._hexPg.model.device
        if self._hexPicker is not None:
            self._hexPicker.refresh_render_state_and_aabbs(
                self._hexPg.aux,
                self._state_0.particle_q,
            )
        else:
            update_cell_render_state(self._hexPg.aux, self._state_0.particle_q, device=device)

        compute_mc_vertex_positions(
            self._hexMcBuffers,
            self._hexPg.aux.cell_center_q,
            self._hexPg.aux.cell_orientation,
            self._hexMcFactor,
            device=device,
        )

        current_revision = (
            int(self._hexDeleteState.topology_revision)
            if self._hexDeleteState is not None
            else 0
        )
        if force_topology or self._hexRenderedTopologyRevision != current_revision:
            tri_count = compute_mc_topology(
                self._hexMcBuffers,
                self._hexMcTables,
                self._hexPg.aux.cell_render_flags,
                self._hexPg.aux.grid_to_cell,
                device=device,
            )
            self._updateHexSurfaceTopology(tri_count)
            self._hexRenderedTopologyRevision = current_revision

        current_mm = self._hexMcBuffers.vertex_pos.numpy().astype(np.float32) * self.SCENE_SCALE_MM
        arr = slicer.util.arrayFromModelPoints(self._hexSurfaceNode)
        if arr.shape == current_mm.shape:
            arr[:] = current_mm
            slicer.util.arrayFromModelPointsModified(self._hexSurfaceNode)

        if self._hexParticleVizNode is not None:
            particles_mm = self._state_0.particle_q.numpy().astype(np.float32) * self.SCENE_SCALE_MM
            parr = slicer.util.arrayFromModelPoints(self._hexParticleVizNode)
            if parr.shape == particles_mm.shape:
                parr[:] = particles_mm
                slicer.util.arrayFromModelPointsModified(self._hexParticleVizNode)

    def setSurfaceVisible(self, on):
        for body in self._softBodies:
            node = body.get("surface_node")
            if node is not None:
                node.GetDisplayNode().SetVisibility(bool(on))
            volume_node = body.get("volume_node")
            if volume_node is not None:
                volume_node.GetDisplayNode().SetVisibility(bool(on))

    def setParticlesVisible(self, on):
        for body in self._softBodies:
            node = body.get("particle_viz_node")
            if node is not None:
                node.GetDisplayNode().SetVisibility(bool(on))

    def setBodyColor(self, segment_id, rgb):
        for body in self._softBodies:
            if body.get("segment_id") != segment_id:
                continue
            node = body.get("surface_node")
            if node is None:
                return
            node.GetDisplayNode().SetColor(float(rgb[0]), float(rgb[1]), float(rgb[2]))
            return

    # ---------------------------------------------------------- cutting UI --
    def configureCutting(self, enabled, radius_mm=None, depth_mm=None, onStatus=None):
        self._cuttingEnabled = bool(enabled)
        if radius_mm is not None:
            self._cutRadiusM = max(0.0, float(radius_mm) / self.SCENE_SCALE_MM)
        if depth_mm is not None:
            self._cutDepthM = max(0.0, float(depth_mm) / self.SCENE_SCALE_MM)
        if onStatus is not None:
            self._cutStatusCb = onStatus
        if self._cuttingEnabled:
            self._installCuttingObservers()
        else:
            self._removeCuttingObservers()

    def _installCuttingObservers(self):
        if self._cuttingObserverTags:
            return
        lm = slicer.app.layoutManager()
        if lm is None:
            return
        for i in range(lm.threeDViewCount):
            view = lm.threeDWidget(i).threeDView()
            interactor = view.interactor()
            tag = interactor.AddObserver(
                vtk.vtkCommand.LeftButtonPressEvent,
                self._onCutMouseEvent,
                1.0,
            )
            self._cuttingObserverTags.append((interactor, tag))

    def _removeCuttingObservers(self):
        for interactor, tag in self._cuttingObserverTags:
            try:
                interactor.RemoveObserver(tag)
            except Exception:
                pass
        self._cuttingObserverTags = []

    @staticmethod
    def _displayToWorld(renderer, x, y, z):
        renderer.SetDisplayPoint(float(x), float(y), float(z))
        renderer.DisplayToWorld()
        p = renderer.GetWorldPoint()
        w = float(p[3]) if len(p) > 3 else 1.0
        if abs(w) < 1.0e-8:
            w = 1.0
        return np.asarray([float(p[0]) / w, float(p[1]) / w, float(p[2]) / w], dtype=np.float32)

    def _rayFromInteractorEvent(self, interactor):
        rw = interactor.GetRenderWindow()
        renderers = rw.GetRenderers() if rw is not None else None
        renderer = renderers.GetFirstRenderer() if renderers is not None else None
        if renderer is None:
            raise RuntimeError("Could not find a 3D renderer for cutting.")
        x, y = interactor.GetEventPosition()
        p0 = self._displayToWorld(renderer, x, y, 0.0)
        p1 = self._displayToWorld(renderer, x, y, 1.0)
        direction = p1 - p0
        norm = float(np.linalg.norm(direction))
        if norm <= 1.0e-8:
            raise RuntimeError("Could not build a valid cutting ray.")
        return p0, direction / norm

    def _onCutMouseEvent(self, caller, event):
        del event
        if not self._cuttingEnabled or self._mode != self.MODE_HEX:
            return
        try:
            origin_mm, direction = self._rayFromInteractorEvent(caller)
            deleted = self.cutHexByRay(origin_mm, direction)
            if deleted > 0:
                self._pushHexToMrml(force_topology=True)
                if self._cutStatusCb is not None:
                    self._cutStatusCb(self.statusText())
        except Exception as exc:
            slicer.util.errorDisplay(f"Cutting failed: {exc}")

    def cutHexByRay(self, origin_mm, direction):
        if (
            self._mode != self.MODE_HEX
            or self._hexPicker is None
            or self._hexDeleteState is None
            or self._state_0 is None
        ):
            return 0
        direction = np.asarray(direction, dtype=np.float32).reshape(3)
        norm = float(np.linalg.norm(direction))
        if norm <= 1.0e-8:
            return 0
        direction = direction / norm
        origin_m = np.asarray(origin_mm, dtype=np.float32).reshape(3) / self.SCENE_SCALE_MM

        self._hexPicker.refresh_render_state_and_aabbs(
            self._hexPg.aux,
            self._state_0.particle_q,
        )
        hit_device = self._hexPicker.pick_device(
            self._hexPg.aux,
            tuple(float(v) for v in origin_m),
            tuple(float(v) for v in direction),
        )
        hit_cell = int(hit_device.numpy()[0])
        if hit_cell == int(self._hexPicker._NO_HIT_CELL):
            return 0

        deleted_now = self._hexDeleteState.delete_cells_by_ray_segment_from_cell(
            self._state_0.particle_q,
            hit_device,
            direction,
            self._cutDepthM,
            padding=self._cutRadiusM,
        )
        if deleted_now <= 0:
            return 0
        result = self._hexDeleteState.last_deletion_result
        if result is None:
            return int(deleted_now)
        self._solver.update_l0_sleep_after_deletion_device(
            result.deleted_cells_device,
            result.deleted_count_device,
            result.candidate_capacity,
            sync_host=True,
        )
        self._hexRenderedTopologyRevision = None
        return int(deleted_now)

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

    def _hexAtlasFromSegments(self, segNode, segments, spacing_mm, density):
        from vtk.util import numpy_support as vns
        from newton_cutting.io.digimouse import DigimouseAtlas
        from newton_cutting.materials import BACKGROUND, Material, MaterialTable, Phase

        spacing_mm = float(spacing_mm)
        spacing_m = spacing_mm / self.SCENE_SCALE_MM
        exports = []
        occupied_ras_blocks = []

        for material_id, seg in enumerate(segments, start=1):
            imagedata, ijk_to_ras = self._exportSegmentLabelmap(segNode, seg["id"])
            dims = imagedata.GetDimensions()
            scalars = imagedata.GetPointData().GetScalars()
            if scalars is None:
                continue
            labels_kji = vns.vtk_to_numpy(scalars).reshape(
                dims[2], dims[1], dims[0]
            )
            occ_k, occ_j, occ_i = np.nonzero(labels_kji)
            if occ_i.size == 0:
                continue

            M = np.array(
                [[ijk_to_ras.GetElement(r, c) for c in range(4)] for r in range(4)],
                dtype=np.float64,
            )
            ijk_h = np.stack(
                [occ_i, occ_j, occ_k, np.ones(occ_i.size, dtype=np.float64)],
                axis=0,
            )
            occ_ras = (M @ ijk_h)[:3].T
            occupied_ras_blocks.append(occ_ras)
            exports.append({
                "material_id": int(material_id),
                "segment": seg,
                "dims": dims,
                "labels": labels_kji,
                "ras_to_ijk": np.linalg.inv(M),
            })

        if not exports:
            raise RuntimeError("Selected segments have no occupied voxels.")

        occ_ras_all = np.concatenate(occupied_ras_blocks, axis=0)
        center_min = occ_ras_all.min(axis=0) - spacing_mm
        center_max = occ_ras_all.max(axis=0) + spacing_mm
        grid_shape = (
            np.floor((center_max - center_min) / spacing_mm).astype(np.int64) + 1
        )
        grid_shape = np.maximum(grid_shape, 1)
        candidate_count = int(np.prod(grid_shape))
        if candidate_count > self.HEX_MAX_ELEMENTS * 20:
            raise RuntimeError(
                f"Hex sampling grid has {candidate_count} candidate cells. "
                "Increase particle spacing or crop/uncheck segments."
            )

        labels_xyz = np.zeros(tuple(int(v) for v in grid_shape), dtype=np.uint16)
        flat_labels = labels_xyz.reshape(-1)
        coords = np.indices(labels_xyz.shape, dtype=np.float64).reshape(3, -1).T
        centers_ras = center_min[None, :] + coords * spacing_mm
        centers_h = np.concatenate(
            [centers_ras, np.ones((centers_ras.shape[0], 1), dtype=np.float64)],
            axis=1,
        )

        chunk = 250000
        for export in exports:
            dims = export["dims"]
            labels_kji = export["labels"]
            ras_to_ijk = export["ras_to_ijk"]
            material_id = int(export["material_id"])
            for start in range(0, centers_h.shape[0], chunk):
                end = min(start + chunk, centers_h.shape[0])
                cand_ijk = (ras_to_ijk @ centers_h[start:end].T)[:3]
                i_idx = np.round(cand_ijk[0]).astype(np.int64)
                j_idx = np.round(cand_ijk[1]).astype(np.int64)
                k_idx = np.round(cand_ijk[2]).astype(np.int64)
                in_bounds = (
                    (i_idx >= 0) & (i_idx < dims[0])
                    & (j_idx >= 0) & (j_idx < dims[1])
                    & (k_idx >= 0) & (k_idx < dims[2])
                )
                occupied = np.zeros(end - start, dtype=bool)
                occupied[in_bounds] = labels_kji[
                    k_idx[in_bounds], j_idx[in_bounds], i_idx[in_bounds]
                ] != 0
                write_mask = occupied & (flat_labels[start:end] == 0)
                if np.any(write_mask):
                    flat_labels[start:end][write_mask] = material_id

        materials = [BACKGROUND]
        locked_material_ids = []
        class_map = {0: "background"}
        cell_mass = float(density) * (spacing_m ** 3)
        for material_id, seg in enumerate(segments, start=1):
            locked = bool(seg.get("locked", False))
            if locked:
                locked_material_ids.append(material_id)
            name = str(seg.get("name", f"segment_{material_id}"))
            color = tuple(float(v) for v in seg.get("color", (0.92, 0.55, 0.45)))
            materials.append(
                Material(
                    id=material_id,
                    name=name,
                    phase=Phase.RIGID if locked else Phase.SOFT,
                    mass=cell_mass,
                    resistance=80.0,
                    conductivity=0.9,
                    color=color,
                )
            )
            class_map[material_id] = name

        origin_mm = center_min - 0.5 * spacing_mm
        atlas = DigimouseAtlas(
            labels=np.ascontiguousarray(labels_xyz),
            voxel_size=spacing_m,
            materials=MaterialTable(materials),
            origin=tuple(float(v) for v in (origin_mm / self.SCENE_SCALE_MM)),
            metadata={"class_map": class_map},
        )
        return atlas, locked_material_ids

    def _particleGridFromLabelmap(self, imagedata, ijk_to_ras, spacing_mm):
        from vtk.util import numpy_support as vns

        dims = imagedata.GetDimensions()  # (I, J, K)
        scalars = imagedata.GetPointData().GetScalars()
        labels = vns.vtk_to_numpy(scalars).reshape(dims[2], dims[1], dims[0])  # (K, J, I)

        occ_k, occ_j, occ_i = np.nonzero(labels)
        if occ_i.size == 0:
            empty = np.empty((0, 3), dtype=np.float32)
            return empty, empty.astype(np.int32), 0.0

        M = np.array(
            [[ijk_to_ras.GetElement(r, c) for c in range(4)] for r in range(4)]
        )
        ijk_h = np.stack([occ_i, occ_j, occ_k, np.ones(occ_i.size)], axis=0)
        occ_ras = (M @ ijk_h)[:3].T  # (N, 3) mm
        ras_min = occ_ras.min(axis=0)
        ras_max = occ_ras.max(axis=0)

        n_x = max(1, int(np.floor((ras_max[0] - ras_min[0]) / spacing_mm)) + 1)
        n_y = max(1, int(np.floor((ras_max[1] - ras_min[1]) / spacing_mm)) + 1)
        n_z = max(1, int(np.floor((ras_max[2] - ras_min[2]) / spacing_mm)) + 1)

        xs = ras_min[0] + np.arange(n_x) * spacing_mm
        ys = ras_min[1] + np.arange(n_y) * spacing_mm
        zs = ras_min[2] + np.arange(n_z) * spacing_mm
        X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
        flat = X.size
        cand_ras_h = np.stack(
            [X.ravel(), Y.ravel(), Z.ravel(), np.ones(flat)], axis=0
        )

        M_inv = np.linalg.inv(M)
        cand_ijk = (M_inv @ cand_ras_h)[:3]  # (3, flat) continuous IJK
        i_idx = np.round(cand_ijk[0]).astype(np.int64)
        j_idx = np.round(cand_ijk[1]).astype(np.int64)
        k_idx = np.round(cand_ijk[2]).astype(np.int64)
        in_bounds = (
            (i_idx >= 0) & (i_idx < dims[0])
            & (j_idx >= 0) & (j_idx < dims[1])
            & (k_idx >= 0) & (k_idx < dims[2])
        )
        occupied = np.zeros(flat, dtype=bool)
        occupied[in_bounds] = labels[
            k_idx[in_bounds], j_idx[in_bounds], i_idx[in_bounds]
        ] != 0
        if not occupied.any():
            empty = np.empty((0, 3), dtype=np.float32)
            return empty, empty.astype(np.int32), 0.0

        pos_mm = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)[occupied]
        pos_m = (pos_mm / self.SCENE_SCALE_MM).astype(np.float32)

        ix, iy, iz = np.meshgrid(
            np.arange(n_x), np.arange(n_y), np.arange(n_z), indexing="ij"
        )
        grid_ijk = np.stack(
            [ix.ravel(), iy.ravel(), iz.ravel()], axis=1
        )[occupied].astype(np.int32)

        spacing_m_iso = float(spacing_mm) / self.SCENE_SCALE_MM
        return pos_m, grid_ijk, spacing_m_iso

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
        dmc.ComputeNormalsOff()
        dmc.Update()

        transform = vtk.vtkTransform()
        transform.SetMatrix(ijk_to_ras)
        tf = vtk.vtkTransformPolyDataFilter()
        tf.SetInputData(dmc.GetOutput())
        tf.SetTransform(transform)
        tf.Update()
        mesh = tf.GetOutput()

        cleaner = vtk.vtkCleanPolyData()
        cleaner.SetInputData(mesh)
        cleaner.PointMergingOn()
        cleaner.ConvertLinesToPointsOff()
        cleaner.ConvertPolysToLinesOff()
        cleaner.ConvertStripsToPolysOff()
        cleaner.Update()
        mesh = cleaner.GetOutput()

        smoother = vtk.vtkWindowedSincPolyDataFilter()
        smoother.SetInputData(mesh)
        smoother.SetNumberOfIterations(30)
        smoother.SetPassBand(0.01)
        smoother.BoundarySmoothingOff()
        smoother.FeatureEdgeSmoothingOff()
        smoother.NonManifoldSmoothingOn()
        smoother.NormalizeCoordinatesOn()
        smoother.Update()
        mesh = smoother.GetOutput()

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
        if self._state_0 is None or not self._softBodies:
            return
        particles_m = self._state_0.particle_q.numpy().astype(np.float32)
        particles_mm = particles_m * self.SCENE_SCALE_MM
        disp_m_global = particles_m - self._particlesRestM

        for body in self._softBodies:
            off, cnt = body["offset"], body["count"]
            body_particles_mm = particles_mm[off:off + cnt]

            viz = body.get("particle_viz_node")
            if viz is not None:
                parr = slicer.util.arrayFromModelPoints(viz)
                parr[:] = body_particles_mm
                slicer.util.arrayFromModelPointsModified(viz)

            surface_node = body.get("surface_node")
            if surface_node is not None:
                disp_mm = disp_m_global[off:off + cnt] * self.SCENE_SCALE_MM
                gathered = disp_mm[body["skin_idxs"]]
                surface_disp_mm = (body["skin_weights"][..., None] * gathered).sum(axis=1)
                current_mm = body["surface_rest_mm"] + surface_disp_mm.astype(np.float32)
                arr = slicer.util.arrayFromModelPoints(surface_node)
                arr[:] = current_mm
                slicer.util.arrayFromModelPointsModified(surface_node)

                nf = body.get("normals_filter")
                if nf is not None:
                    nf.Update()
                    polydata = surface_node.GetPolyData()
                    target = polydata.GetPointData().GetNormals()
                    src = nf.GetOutput().GetPointData().GetNormals()
                    if target is not None and src is not None:
                        target.DeepCopy(src)
                        target.Modified()

    # ---------------------------------------------------------------- loop --
    def _applyTiming(self, fps, dt_ms, substeps):
        self._timer_interval_ms = max(1, int(round(1000.0 / float(fps))))
        self._dt_s = max(1e-6, float(dt_ms) * 1e-3)
        self._substeps = max(1, int(substeps))

    def _doOneStep(self):
        self._state_0.clear_forces()
        if self._mode == self.MODE_HEX:
            self._solver.step(self._state_0, self._state_1, None, None, self._dt_s)
        else:
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
        elif self._mode == self.MODE_HEX:
            self._pushHexToMrml()
        elif self._mode == self.MODE_TET:
            self._pushTetToMrml()

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
                f"frame={self._frame}  bodies={len(self._softBodies)}  "
                f"P={q.shape[0]}  <z>={float(q[:, 2].mean()):+.3f} m"
            )
        if self._mode == self.MODE_HEX:
            q = self._state_0.particle_q.numpy()
            active_cells = 0
            deleted_total = 0
            if self._hexDeleteState is not None:
                active_cells = self._hexPg.aux.num_cells - int(self._hexDeleteState.deleted_total)
                deleted_total = int(self._hexDeleteState.deleted_total)
            return (
                f"frame={self._frame}  hex nodes={q.shape[0]}  "
                f"cells={active_cells}  deleted={deleted_total}  "
                f"tris={self._hexTriCount}  <z>={float(q[:, 2].mean()):+.3f} m"
            )
        if self._mode == self.MODE_TET:
            q = self._state_0.particle_q.numpy()
            max_disp_mm = 0.0
            if self._tetRestPointsM is not None and self._tetRestPointsM.shape == q.shape:
                disp_m = q - self._tetRestPointsM
                max_disp_mm = float(np.linalg.norm(disp_m, axis=1).max() * self.SCENE_SCALE_MM)
            return (
                f"frame={self._frame}  solver={self._tetSolverName}  "
                f"nodes={q.shape[0]}  tets={self.tetCount()}  "
                f"labels={self._tetSelectedLabelCount}  "
                f"free={self._tetFreeCount}  fixed={self._tetFixedCount}  "
                f"ground=on  self=off  "
                f"max|u|={max_disp_mm:.2f} mm  <z>={float(q[:, 2].mean()):+.3f} m"
            )
        return f"frame={self._frame}"

    def particleCount(self):
        if self._state_0 is None:
            return 0
        return int(self._state_0.particle_q.shape[0])

    def springCount(self):
        return self._spring_count

    def tetCount(self):
        if self._tetIndices is None:
            return 0
        return int(self._tetIndices.shape[0])

    def tetFreeCount(self):
        return int(self._tetFreeCount)

    # --------------------------------------------------------------- reset --
    def reset(self):
        self.stopTimer()
        self._removeCuttingObservers()
        if self._auxViewer is not None:
            try:
                self._auxViewer.close()
            except Exception:
                pass
        nodes = [
            self._sphereModelNode,
            self._sphereTransformNode,
            self._groundModelNode,
        ]
        for body in self._softBodies:
            nodes.append(body.get("surface_node"))
            nodes.append(body.get("volume_node"))
            nodes.append(body.get("particle_viz_node"))
        for node in nodes:
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
        self.test_TetLabelsAndExtraction()
        self.test_TetRejectsMixedCells()
        self.test_TetWindingAndPinning()
        self.test_TetNewtonSmokeXPBD()
        self.test_TetNewtonSmokeVBD()

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

    @staticmethod
    def _makeTetGrid(points, cells, labels, extra_vertex_cell=False):
        pts = vtk.vtkPoints()
        for p in points:
            pts.InsertNextPoint(float(p[0]), float(p[1]), float(p[2]))
        grid = vtk.vtkUnstructuredGrid()
        grid.SetPoints(pts)
        id_list = vtk.vtkIdList()
        for tet in cells:
            id_list.Reset()
            for pid in tet:
                id_list.InsertNextId(int(pid))
            grid.InsertNextCell(vtk.VTK_TETRA, id_list)
        label_values = list(labels)
        if extra_vertex_cell:
            id_list.Reset()
            id_list.InsertNextId(0)
            grid.InsertNextCell(vtk.VTK_VERTEX, id_list)
            label_values.append(1)
        label_array = vtk.vtkIntArray()
        label_array.SetName("labels")
        for label in label_values:
            label_array.InsertNextValue(int(label))
        grid.GetCellData().AddArray(label_array)
        grid.GetCellData().SetScalars(label_array)
        grid.GetCellData().SetActiveScalars("labels")
        return grid

    @staticmethod
    def _tetSettings(**overrides):
        settings = {
            "sim": True,
            "locked": False,
            "name": "soft",
            "color": (0.8, 0.2, 0.1),
            "k_mu": 1000.0,
            "k_lambda": 1000.0,
            "k_damp": 0.0,
        }
        settings.update(overrides)
        return settings

    def test_TetLabelsAndExtraction(self):
        self.delayDisplay("Tet labels: label 0 ignored, nonzero labels discovered")
        points = [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 1.0, 1.0),
        ]
        grid = self._makeTetGrid(
            points,
            [(0, 1, 2, 3), (1, 2, 3, 4), (0, 1, 3, 4)],
            [0, 2, 5],
        )
        self.assertEqual(
            NewtonPhysicsLogic._discoverTetLabelsInUnstructuredGrid(grid),
            [2, 5],
        )
        data = NewtonPhysicsLogic._extractTetFEMDataFromUnstructuredGrid(
            grid,
            {
                2: self._tetSettings(sim=True),
                5: self._tetSettings(sim=False, locked=False),
            },
            density=1000.0,
            pin_top=False,
        )
        self.assertEqual(data["labels"].tolist(), [2])
        self.assertEqual(int(data["tets"].shape[0]), 1)
        self.assertEqual(data["selected_labels"], [2])
        self.delayDisplay("OK")

    def test_TetRejectsMixedCells(self):
        self.delayDisplay("Tet extraction rejects non-tetrahedral cells")
        grid = self._makeTetGrid(
            [
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
            ],
            [(0, 1, 2, 3)],
            [1],
            extra_vertex_cell=True,
        )
        with self.assertRaises(RuntimeError):
            NewtonPhysicsLogic._extractTetFEMDataFromUnstructuredGrid(
                grid,
                {1: self._tetSettings(sim=True)},
                density=1000.0,
                pin_top=False,
            )
        self.delayDisplay("OK")

    def test_TetWindingAndPinning(self):
        self.delayDisplay("Tet extraction corrects negative winding and computes pins")
        points = [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ]
        grid = self._makeTetGrid(points, [(0, 2, 1, 3)], [1])
        locked = NewtonPhysicsLogic._extractTetFEMDataFromUnstructuredGrid(
            grid,
            {1: self._tetSettings(sim=True, locked=True)},
            density=1000.0,
            pin_top=True,
        )
        volume = NewtonPhysicsLogic._tetVolumeM3(
            locked["points_m"][locked["tets"][0]]
        )
        self.assertGreater(volume, 0.0)
        self.assertTrue(np.all(locked["fixed_mask"]))

        top = NewtonPhysicsLogic._extractTetFEMDataFromUnstructuredGrid(
            grid,
            {1: self._tetSettings(sim=True, locked=False)},
            density=1000.0,
            pin_top=True,
        )
        self.assertEqual(int(np.count_nonzero(top["fixed_mask"])), 1)
        fixed_z = top["points_m"][top["fixed_mask"], 2]
        self.assertTrue(np.allclose(fixed_z, top["points_m"][:, 2].max()))
        self.delayDisplay("OK")

    def _runTetNewtonSmoke(self, solverName):
        ok, _ = NewtonPhysicsLogic.checkNewton()
        if not ok:
            self.delayDisplay(f"Skipping {solverName} smoke: Newton is not installed")
            return
        if "VBD" in solverName:
            import newton
            if not hasattr(newton.solvers, "SolverVBD"):
                self.delayDisplay("Skipping VBD smoke: SolverVBD is unavailable")
                return

        grid = self._makeTetGrid(
            [
                (0.0, 0.0, 0.0),
                (20.0, 0.0, 0.0),
                (0.0, 20.0, 0.0),
                (0.0, 0.0, 20.0),
            ],
            [(0, 1, 2, 3)],
            [1],
        )
        node = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLModelNode",
            f"TetSmoke_{solverName.replace(' ', '_')}",
        )
        NewtonPhysicsLogic._setModelNodeMesh(node, grid)
        node.CreateDefaultDisplayNodes()

        logic = NewtonPhysicsLogic()
        logic.setupTetFEMFromModel(
            modelNode=node,
            labelSettings={1: self._tetSettings(sim=True, locked=False)},
            solverName=solverName,
            device="cpu",
            fps=60,
            dt_ms=0.1,
            substeps=1,
            iterations=1,
            auxViewer="none",
            density=1000.0,
            pin_top=True,
            self_collide=False,
            particle_radius_factor=0.1,
        )
        self.assertEqual(logic.tetCount(), 1)
        rest_q = logic._state_0.particle_q.numpy().copy()
        for _ in range(20):
            logic.stepOnce()
        q = logic._state_0.particle_q.numpy()
        self.assertTrue(np.all(np.isfinite(q)))
        self.assertGreater(float(np.linalg.norm(q - rest_q, axis=1).max()), 0.0)
        out_labels = logic._tetOutputNode.GetMesh().GetCellData().GetArray("labels")
        self.assertIsNotNone(out_labels)
        self.assertIsNone(logic._tetOutputNode.GetMesh().GetCellData().GetArray("displayLabels"))
        out_display = logic._tetOutputNode.GetDisplayNode()
        self.assertIsNotNone(out_display.GetColorNode())
        self.assertEqual(out_display.GetColorNode().GetName(), "Labels")
        self.assertTrue(out_display.GetVisibility())
        self.assertIsNotNone(logic._tetSurfaceNode)
        self.assertTrue(logic._tetSurfaceNode.GetDisplayNode().GetVisibility())
        surface_labels = logic._tetSurfaceNode.GetPolyData().GetCellData().GetArray("labels")
        self.assertIsNotNone(surface_labels)
        self.assertIsNone(logic._tetSurfaceNode.GetPolyData().GetCellData().GetArray("displayLabels"))
        out_points = slicer.util.arrayFromModelPoints(logic._tetSurfaceNode)
        np.testing.assert_allclose(out_points, q * logic.SCENE_SCALE_MM, rtol=1.0e-5, atol=1.0e-5)
        self.assertIsNone(logic._model.particle_grid)
        logic.reset()

    def test_TetNewtonSmokeXPBD(self):
        self.delayDisplay("Tet FEM XPBD one-tet smoke")
        self._runTetNewtonSmoke("Tet FEM XPBD")
        self.delayDisplay("OK")

    def test_TetNewtonSmokeVBD(self):
        self.delayDisplay("Tet FEM VBD one-tet smoke")
        self._runTetNewtonSmoke("Tet FEM VBD")
        self.delayDisplay("OK")
