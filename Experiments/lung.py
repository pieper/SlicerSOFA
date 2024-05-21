"""
export DIR=/media/volume/sdb
SOFA_ROOT=${DIR}/Slicer-SOFA-build/SOFA-build \
        ${DIR}/Slicer-superbuild/Slicer-build/Slicer \
        --launcher-additional-settings ${DIR}/Slicer-SOFA-build/inner-build/AdditionalLauncherSettings.ini \
        --additional-module-path ${DIR}/SlicerOpenIGTLink-build/inner-build/lib/Slicer-5.7/qt-loadable-modules \
        --additional-module-path ${DIR}/SlicerOpenIGTLink-build/inner-build/lib/Slicer-5.7/qt-scripted-modules

SOFA_ROOT=${DIR}/Slicer-SOFA-build/SOFA-build \
        ${DIR}/Slicer-superbuild/Slicer-build/Slicer \
        --launcher-additional-settings ${DIR}/Slicer-SOFA-build/inner-build/AdditionalLauncherSettings.ini \
        --launcher-additional-settings ${DIR}/SlicerOpenIGTLink-build/inner-build/AdditionalLauncherSettings.ini

SOFA_ROOT=${DIR}/Slicer-SOFA-build/SOFA-build \
        ${DIR}/Slicer-superbuild/Slicer-build/Slicer \
        --launcher-additional-settings \
            ${DIR}/Slicer-SOFA-build/inner-build/AdditionalLauncherSettings.ini \
        --launch PythonSlicer

"""

import numpy
import os

import slicer
from qt import QObject, QTimer

import Sofa
import SofaRuntime

from stlib3.scene import MainHeader, ContactHeader
from stlib3.solver import DefaultSolver
from stlib3.physics.deformable import ElasticMaterialObject
from stlib3.physics.rigid import Floor
from splib3.numerics import Vec3

vonMisesMode = {
    "none": 0,
    "corotational": 1,
    "fullGreen": 2
}

class ParameterNode(object):
    dt=0.01
    #modelNodeFileName="/home/exouser/right-lung-mesh-low.vtk"
    #modelNodeFileName="/home/exouser/Documents/head-top-2seg.vtk"
    #modelNodeFileName="/home/exouser/Documents/sphere-2seg.vtk"
    modelNodeFileName="/home/exouser/Documents/sphere.vtk"
    # grav and roi are in LPS
    def getGravityVector(self):
        #return [0,0,-10000]
        return [0,10000 * 5,0]
    def getBoundaryROI(self):
        #[ 0, -220, 0, 30, -170, -300],
        #return [0, 0, 0, 0, 0, 0]
        #return [0, -170, 0, 48, -80, -100]
        return [-300, 300, -380, 300, -300, -200]
        #return [-300, 300, 130, 380, -300, 300]
parameterNode = ParameterNode()

# Load the mesh
modelNode = slicer.util.loadNodeFromFile(parameterNode.modelNodeFileName)

# mark the back 25% of the surface nodes as fixed
surfaceFilter = vtk.vtkDataSetSurfaceFilter()
surfaceFilter.SetInputData(modelNode.GetUnstructuredGrid())
surfaceFilter.SetPassThroughPointIds(True)
surfaceFilter.Update()
surfaceMesh = surfaceFilter.GetOutputDataObject(0)

surfacePointIDs = vtk.util.numpy_support.vtk_to_numpy(surfaceMesh.GetPointData().GetArray("vtkOriginalPointIds"))
surfacePoints = vtk.util.numpy_support.vtk_to_numpy(surfaceMesh.GetPoints().GetData())

paSurfacePoints = surfacePoints.transpose()[1]
divisionPlane = paSurfacePoints.min() + 0.25 * (paSurfacePoints.max() - paSurfacePoints.min())
backOfLung = (paSurfacePoints < divisionPlane)

# create a stress array

labelsArray = slicer.util.arrayFromModelCellData(modelNode, "labels")
stressVTKArray = vtk.vtkFloatArray()
stressVTKArray.SetNumberOfValues(labelsArray.shape[0])
stressVTKArray.SetName("VonMisesStress")
modelNode.GetUnstructuredGrid().GetCellData().AddArray(stressVTKArray)

rootNode = Sofa.Core.Node()

MainHeader(rootNode, plugins=[
    "Sofa.Component.IO.Mesh",
    "Sofa.Component.LinearSolver.Direct",
    "Sofa.Component.LinearSolver.Iterative",
    "Sofa.Component.Mapping.Linear",
    "Sofa.Component.Mass",
    "Sofa.Component.ODESolver.Backward",
    "Sofa.Component.Setting",
    "Sofa.Component.SolidMechanics.FEM.Elastic",
    "Sofa.Component.StateContainer",
    "Sofa.Component.Topology.Container.Dynamic",
    "Sofa.Component.Visual",
    "Sofa.GL.Component.Rendering3D",
    "Sofa.Component.AnimationLoop",
    "Sofa.Component.Collision.Detection.Algorithm",
    "Sofa.Component.Collision.Detection.Intersection",
    "Sofa.Component.Collision.Geometry",
    "Sofa.Component.Collision.Response.Contact",
    "Sofa.Component.Constraint.Lagrangian.Solver",
    "Sofa.Component.Constraint.Lagrangian.Correction",
    "Sofa.Component.LinearSystem",
    "Sofa.Component.MechanicalLoad",
    "MultiThreading",
    "Sofa.Component.SolidMechanics.Spring",
    "Sofa.Component.Constraint.Lagrangian.Model",
    "Sofa.Component.Mapping.NonLinear",
    "Sofa.Component.Topology.Container.Constant",
    "Sofa.Component.Topology.Mapping",
    "Sofa.Component.Engine.Select",
    "Sofa.Component.Constraint.Projective",
    "SofaIGTLink"
], dt=parameterNode.dt, gravity=parameterNode.getGravityVector())


meshNode = rootNode.addChild('Mesh')
meshNode.addObject('EulerImplicitSolver', firstOrder=False, rayleighMass=0.1, rayleighStiffness=0.1)
meshNode.addObject('SparseLDLSolver', name="precond", template="CompressedRowSparseMatrixd", parallelInverseProduct=True)
meshNode.addObject('MeshVTKLoader', name="loader", filename=parameterNode.modelNodeFileName)
meshNode.addObject('TetrahedronSetTopologyContainer', name="Container", src="@loader")
meshNode.addObject('TetrahedronSetTopologyModifier', name="Modifier")
meshNode.addObject('MechanicalObject', name="mstate", template="Vec3f")
meshNode.addObject('TetrahedronFEMForceField', name="FEM", youngModulus=1.5, poissonRatio=0.45, method="large", computeVonMisesStress=vonMisesMode['fullGreen'])
meshNode.addObject('MeshMatrixMass', totalMass=1)

fixedSurface = meshNode.addChild('FixedSurface')
fixedSurface.addObject('FixedConstraint', indices=surfacePointIDs[numpy.where(backOfLung)])


Sofa.Simulation.init(rootNode)
Sofa.Simulation.reset(rootNode)
mechanicalState = meshNode.getMechanicalState()
forceField = meshNode.getForceField(0)

browserNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSequenceBrowserNode", "SOFA Simulation")
browserNode.SetPlaybackActive(False)

sequenceNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSequenceNode", f"{modelNode.GetName()}-Sequence")

# Synchronize and set up the sequence browser node
browserNode.AddSynchronizedSequenceNodeID(sequenceNode.GetID())
browserNode.AddProxyNode(modelNode, sequenceNode, False)
browserNode.SetRecording(sequenceNode, True)
browserNode.SetRecordingActive(True)

iteration = 0
iterations = 10
simulating = True
def updateSimulation():
    global iteration, simulating
    Sofa.Simulation.animate(rootNode, rootNode.dt.value)
    meshPointsArray = mechanicalState.position.array()
    modelPointsArray = slicer.util.arrayFromModelPoints(modelNode)
    modelPointsArrayNew = meshPointsArray * numpy.array([-1, -1, 1])
    modelPointsArray[:] = modelPointsArrayNew
    slicer.util.arrayFromModelPointsModified(modelNode)
    stressArray = slicer.util.arrayFromModelCellData(modelNode, "VonMisesStress")
    stressArray[:] = forceField.vonMisesPerElement.array()
    slicer.util.arrayFromModelCellDataModified(modelNode, "VonMisesStress")
    iteration += 1
    simulating = iteration < iterations
    if iteration % 10 == 0:
        print(f"Iteration {iteration}")
    if simulating:
        qt.QTimer.singleShot(10, updateSimulation)
    else:
        print("Simlation stopped")
        browserNode.SetRecordingActive(False)
        browserNode.SetPlaybackActive(True)

#updateSimulation()
