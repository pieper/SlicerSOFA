import math
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

# TODO: this function belongs in slicer.util
def addGridTransformFromArray(narray, name="Grid Transform"):
    """Create a new grid transform node from content of a numpy array and add it to the scene.

    Voxels values are deep-copied, therefore if the numpy array
    is modified after calling this method, voxel values in the volume node will not change.
    :param narray: numpy array containing grid vectors.
    Must be [slices, rows, columns, 3]
    :param name: grid transform node name
    """
    import slicer

    if len(narray.shape) != 4 or narray.shape[3] != 3:
        raise RuntimeError("Need vector volume numpy array for grid transform")
    nodeClassName = "vtkMRMLGridTransformNode"
    gridNode = slicer.mrmlScene.AddNewNodeByClass(nodeClassName, name)
    gridNode.CreateDefaultDisplayNodes()
    displacementGrid = gridNode.GetTransformFromParent().GetDisplacementGrid()
    arrayShape = narray.shape
    displacementGrid.SetDimensions(arrayShape[2], arrayShape[1], arrayShape[0])
    scalarType = vtk.util.numpy_support.get_vtk_array_type(narray.dtype)
    displacementGrid.AllocateScalars(scalarType, 3)
    displacementArray = slicer.util.arrayFromGridTransform(gridNode)
    displacementArray[:] = narray
    slicer.util.arrayFromGridTransformModified(gridNode)
    return gridNode


# Load the mesh
baseDir = "/media/volume/sdb/data/vertebra"
modelFilePath = f"{baseDir}/mesh-MD15011447A-L1-transferred.vtk"
#modelFilePath = f"{baseDir}/MD15011447-L1-low-res-mesh.vtk"
modelNode = slicer.util.loadModel(modelFilePath)

class ParameterNode(object):
    dt=0.01
    #modelNodeFileName="/home/exouser/right-lung-mesh-low.vtk"
    #modelNodeFileName="/home/exouser/Documents/head-top-2seg.vtk"
    #modelNodeFileName="/home/exouser/Documents/sphere-2seg.vtk"
    modelNodeFileName=modelFilePath
    # grav and roi are in LPS
    def getGravityVector(self):
        #return [0,0,-10000]
        #return [0,10000 * 3,0]
        return [0,0,-10000 * 3]
    def getBoundaryROI(self):
        #[ 0, -220, 0, 30, -170, -300],
        #return [0, 0, 0, 0, 0, 0]
        #return [0, -170, 0, 48, -80, -100]
        return [-300, 300, -380, 300, -300, -200]
        #return [-300, 300, 130, 380, -300, 300]
parameterNode = ParameterNode()


try:
    slicer.util.getNode("Cropped*3145mm")
except slicer.util.MRMLNodeNotFoundException:
    #ctVolumePath = slicer.util.loadVolume(f"{baseDir}/Cropped volume-MD15011447-L1-0.3145mm.nrrd")
    ctVolume = slicer.util.loadVolume(f"{baseDir}/Cropped volume-MD15011447-L1-0.3145mm.nrrd")
    segmentationNode = slicer.util.loadSegmentation(f"{baseDir}/Segmentation-MD15011447-L1-0.3145mm.seg.nrrd")

segmentation = segmentationNode.GetSegmentation()
backgroundID = segmentation.GetSegmentIdBySegmentName("Background")
segmentationNode.GetDisplayNode().SetSegmentVisibility(backgroundID, False)

grid = modelNode.GetUnstructuredGrid()

# mark the bottom 10% of the surface nodes as fixed
surfaceFilter = vtk.vtkDataSetSurfaceFilter()
surfaceFilter.SetInputData(grid)
surfaceFilter.SetPassThroughPointIds(True)
surfaceFilter.Update()
surfaceMesh = surfaceFilter.GetOutputDataObject(0)

surfacePointIDs = vtk.util.numpy_support.vtk_to_numpy(surfaceMesh.GetPointData().GetArray("vtkOriginalPointIds"))
surfacePoints = vtk.util.numpy_support.vtk_to_numpy(surfaceMesh.GetPoints().GetData())

paSurfacePoints = surfacePoints.transpose()[2]
divisionPlane = paSurfacePoints.min() + 0.1 * (paSurfacePoints.max() - paSurfacePoints.min())
bottomOfVertebra = (paSurfacePoints < divisionPlane)


# calculate the per-element young's moduli

pointsArray = numpy.array(grid.GetPoints().GetData())
cellsArray = numpy.array(grid.GetCells().GetData())
tetrahedraArray = cellsArray.reshape(-1,5)[:,1:5]
centroidsArray = numpy.mean(pointsArray[tetrahedraArray], axis=1)

ctArray = slicer.util.arrayFromVolume(ctVolume)

rasToIJK = vtk.vtkMatrix4x4()
ctVolume.GetRASToIJKMatrix(rasToIJK)

centroidsIJK = []
for centroid in centroidsArray:
    centroidH = [*centroid, 1.]
    centroidIJK = [0.]*4
    rasToIJK.MultiplyPoint(centroidH, centroidIJK)
    centroidIJK = list(map(math.floor, centroidIJK))[:3]
    centroidsIJK.append(centroidIJK)

ctMax = ctArray.max()
ctMean = ctArray.mean()

youngModulusBase = 1.5
youngModulusArray = numpy.ones(centroidsArray.shape[0]) * youngModulusBase

for elementIndex in range(len(youngModulusArray)):
    centroidIJK = centroidsIJK[elementIndex]
    ctValue = ctArray[centroidIJK[2], centroidIJK[1], centroidIJK[0]]
    if ctValue > ctMean:
        youngModulusArray[elementIndex] *= 3 * (ctValue - ctMean) / (ctMax - ctMean)

youngModulusVTKArray = vtk.vtkFloatArray()
youngModulusVTKArray.SetNumberOfValues(grid.GetNumberOfCells())
youngModulusVTKArray.SetName("YoungsModulus")
modelNode.GetUnstructuredGrid().GetCellData().AddArray(youngModulusVTKArray)

youngArray = slicer.util.arrayFromModelCellData(modelNode, "YoungsModulus")
youngArray[:] = youngModulusArray
slicer.util.arrayFromModelCellDataModified(modelNode, "YoungsModulus")

# create a stress array

stressVTKArray = vtk.vtkFloatArray()
stressVTKArray.SetNumberOfValues(grid.GetNumberOfCells())
stressVTKArray.SetName("VonMisesStress")
modelNode.GetUnstructuredGrid().GetCellData().AddArray(stressVTKArray)

# create displacement array

displacementVTKArray = vtk.vtkFloatArray()
displacementVTKArray.SetNumberOfComponents(3)
displacementVTKArray.SetNumberOfTuples(grid.GetNumberOfPoints())
displacementVTKArray.SetName("Displacement")
modelNode.GetUnstructuredGrid().GetPointData().AddArray(displacementVTKArray)

# do Sofa things

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
meshNode.addObject('TetrahedronFEMForceField', name="FEM", youngModulus=youngModulusArray, poissonRatio=0.45, method="large", computeVonMisesStress=vonMisesMode['fullGreen'])
meshNode.addObject('MeshMatrixMass', totalMass=1)

fixedSurface = meshNode.addChild('FixedSurface')
fixedSurface.addObject('FixedConstraint', indices=surfacePointIDs[numpy.where(bottomOfVertebra)])


Sofa.Simulation.init(rootNode)
Sofa.Simulation.reset(rootNode)
mechanicalState = meshNode.getMechanicalState()
forceField = meshNode.getForceField(0)

# do Sequence things (not working)

browserNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSequenceBrowserNode", "SOFA Simulation")
browserNode.SetPlaybackActive(False)

sequenceNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSequenceNode", f"{modelNode.GetName()}-Sequence")

# Synchronize and set up the sequence browser node
browserNode.AddSynchronizedSequenceNodeID(sequenceNode.GetID())
browserNode.AddProxyNode(modelNode, sequenceNode, False)
browserNode.SetRecording(sequenceNode, True)
browserNode.SetRecordingActive(True)

# Set up probe to calculate grid transform

probeGrid = vtk.vtkImageData()
probeDimension = 10
probeGrid.SetDimensions(probeDimension, probeDimension, probeDimension)
probeGrid.AllocateScalars(vtk.VTK_DOUBLE, 1)
meshBounds = [0]*6
modelNode.GetRASBounds(meshBounds)
probeGrid.SetOrigin(meshBounds[0], meshBounds[2], meshBounds[4])
probeSize = (meshBounds[1] - meshBounds[0], meshBounds[3] - meshBounds[2], meshBounds[5] - meshBounds[4])
probeGrid.SetSpacing(probeSize[0]/probeDimension, probeSize[1]/probeDimension, probeSize[2]/probeDimension)

probeFilter = vtk.vtkProbeFilter()
probeFilter.SetInputData(probeGrid)
probeFilter.SetSourceData(modelNode.GetUnstructuredGrid())
probeFilter.SetPassPointArrays(True)
probeFilter.Update()

probeImage = probeFilter.GetOutputDataObject(0)
probeArray = vtk.util.numpy_support.vtk_to_numpy(probeImage.GetPointData().GetArray("Displacement"))
probeArray = numpy.reshape(probeArray, (probeDimension,probeDimension,probeDimension,3))
displacementGridNode = addGridTransformFromArray(probeArray, name="Displacement")
displacementGrid = displacementGridNode.GetTransformFromParent().GetDisplacementGrid()
# TODO: next two lines should be in ijkToRAS of grid node
displacementGrid.SetOrigin(probeImage.GetOrigin())
displacementGrid.SetSpacing(probeImage.GetSpacing())


# run simulation in steps, updating model and grid transform

iteration = 0
iterations = 1
simulating = True
def updateSimulation():
    global iteration, simulating

    # perform analysis
    Sofa.Simulation.animate(rootNode, rootNode.dt.value)

    # update model from mechanical state
    meshPointsArray = mechanicalState.position.array()
    modelPointsArray = slicer.util.arrayFromModelPoints(modelNode)
    modelPointsArrayNew = meshPointsArray * numpy.array([-1, -1, 1])
    modelPointsArray[:] = modelPointsArrayNew
    slicer.util.arrayFromModelPointsModified(modelNode)
    # update stress from forceField
    stressArray = slicer.util.arrayFromModelCellData(modelNode, "VonMisesStress")
    stressArray[:] = forceField.vonMisesPerElement.array()
    slicer.util.arrayFromModelCellDataModified(modelNode, "VonMisesStress")

    # update grid transform from displacements
    displacementArray = slicer.util.arrayFromModelPointData(modelNode, "Displacement")
    displacementArray[:] = (mechanicalState.position - mechanicalState.rest_position) 
    displacementArray *= numpy.array([-1, -1, 1])
    slicer.util.arrayFromModelPointsModified(modelNode)
    probeFilter.Update()
    probeImage = probeFilter.GetOutputDataObject(0)
    probeVTKArray = probeImage.GetPointData().GetArray("Displacement")
    probeArray = vtk.util.numpy_support.vtk_to_numpy(probeVTKArray)
    probeArrayShape = (probeDimension,probeDimension,probeDimension,3)
    probeArray = probeArray.reshape(probeArrayShape)
    gridArray = slicer.util.arrayFromGridTransform(displacementGridNode)
    gridArray[:] = -1. * probeArray
    slicer.util.arrayFromGridTransformModified(displacementGridNode)

    # decide if we should keep iterating
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

updateSimulation()
