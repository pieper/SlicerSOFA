import math
import numpy
import vtk.util.numpy_support

#targetMeshNodesPerMM = 0.1
targetMeshNodesPerMM = 0.075
#targetMeshNodesPerMM = 0.15
#targetMeshNodesPerMM = 0.3
targetMaxEdgeMM = 1.1 * (1. / targetMeshNodesPerMM)

attachmentRAS = numpy.array([38.49312772285862, 159.75102715306605, -225.53033447265625])
attachmentRadiusMM = 42

# set up input data

ctFilePath = "/opt/data/SlicerLung/EservalRocha/Model case/6 VOL MEDIASTINO.nrrd"
segFilePath = "/opt/data/SlicerLung/EservalRocha/Model case/RightLung.seg.nrrd"

ctNode = slicer.mrmlScene.GetFirstNodeByName("ct")
if not ctNode:
    ctNode = slicer.util.loadVolume(ctFilePath, properties = {"name": "ct"})

segNode = slicer.mrmlScene.GetFirstNodeByName("seg")
if not segNode:
    segNode = slicer.util.loadSegmentation(segFilePath, properties = {"name": "seg"})

meshNode = slicer.mrmlScene.GetFirstNodeByName("mesh")
if not meshNode:
    meshNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode")
    meshNode.SetName("mesh")
    meshNode.CreateDefaultDisplayNodes()

labelNode = slicer.mrmlScene.GetFirstNodeByName("label")
if not labelNode:
    labelNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
    labelNode.SetName("label")

# create mesh points

segBounds = numpy.ndarray(6)
segNode.GetRASBounds(segBounds)
axes = []
for start,end in segBounds.reshape(-1,2):
    nodeCount = math.ceil((end-start) * targetMeshNodesPerMM)
    axes.append(numpy.linspace(start, end, nodeCount))
gridPoints = numpy.transpose(numpy.meshgrid(*axes))
inputPoints = gridPoints.ravel().reshape(-1,3)
homogeneousCoordinates = numpy.ones((inputPoints.shape[0], 1))
inputPoints = numpy.hstack((inputPoints, homogeneousCoordinates))

# run Delaunay on points that are in segmentation

pointSet = vtk.vtkPointSet()
points = vtk.vtkPoints()

segNode.SetReferenceImageGeometryParameterFromVolumeNode(ctNode)
forceToSingleLayer = True
segNode.GetSegmentation().CollapseBinaryLabelmaps(forceToSingleLayer)

slicer.modules.segmentations.logic().ExportVisibleSegmentsToLabelmapNode(segNode, labelNode, ctNode)
labelArray = slicer.util.arrayFromVolume(labelNode)

rasToIJK = vtk.vtkMatrix4x4()
labelNode.GetRASToIJKMatrix(rasToIJK)

def rasPointToIndex(rasToIJK, rasPoint):
    pointCoordinate = numpy.floor(rasToIJK.MultiplyPoint(rasPoint))
    return tuple(reversed(numpy.array(pointCoordinate[:3], dtype="uint32")))

for p in inputPoints:
    pointIndex = rasPointToIndex(rasToIJK, p)
    if labelArray[pointIndex]:
        points.InsertNextPoint(*p[:3])

pointSet.SetPoints(points)

delaunay = vtk.vtkDelaunay3D()
delaunay.SetInputData(pointSet)
delaunay.Update()
meshGrid = delaunay.GetOutputDataObject(0)

meshNode.SetAndObserveMesh(meshGrid)

# Remove elements with centroids outside segmentation

pointsArray = numpy.array(meshGrid.GetPoints().GetData())
cellsArray = numpy.array(meshGrid.GetCells().GetData())
tetrahedraArray = cellsArray.reshape(-1,5)[:,1:5]
centroidsArray = numpy.mean(pointsArray[tetrahedraArray], axis=1)
homogeneousCoordinates = numpy.ones((centroidsArray.shape[0], 1))
centroidsArray = numpy.hstack((centroidsArray, homogeneousCoordinates))

tetrahedraToKeep = vtk.vtkIdList()
for tetraID,centroid in enumerate(centroidsArray):
    centroidIndex = rasPointToIndex(rasToIJK, centroid)
    tetra = meshGrid.GetCell(tetraID)
    bounds = numpy.array(tetra.GetBounds())
    mins = bounds[::2]
    maxes = bounds[1::2]
    maxSize = (maxes-mins).max()
    if labelArray[centroidIndex] and maxSize < targetMaxEdgeMM:
        tetrahedraToKeep.InsertNextId(tetraID)

extractCells = vtk.vtkExtractCells()
extractCells.SetInputData(meshGrid)
extractCells.SetCellList(tetrahedraToKeep)
extractCells.Update()

extractedMeshGrid = extractCells.GetOutputDataObject(0)
meshNode.SetAndObserveMesh(extractedMeshGrid)

# create displacement array

displacementVTKArray = vtk.vtkFloatArray()
displacementVTKArray.SetNumberOfComponents(3)
displacementVTKArray.SetNumberOfTuples(extractedMeshGrid.GetNumberOfPoints())
displacementVTKArray.SetName("Displacement")
meshNode.GetMesh().GetPointData().AddArray(displacementVTKArray)

# create a stress array

numberOfCells = meshNode.GetUnstructuredGrid().GetNumberOfCells()
stressVTKArray = vtk.vtkFloatArray()
stressVTKArray.SetNumberOfValues(numberOfCells)
stressVTKArray.SetName("VonMisesStress")
meshNode.GetMesh().GetCellData().AddArray(stressVTKArray)

# get pointIDs in attachment sphere

extractedPointsArray = numpy.array(extractedMeshGrid.GetPoints().GetData())
attachedPoints = numpy.linalg.norm(pointsArray - attachmentRAS, axis=1) < attachmentRadiusMM
cellsArray = numpy.array(extractedMeshGrid.GetCells().GetData())
extractedTetrahedraArray = cellsArray.reshape(-1,5)[:,1:5]

# Set up probe to calculate grid transform

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


probeGrid = vtk.vtkImageData()
probeDimension = 10
probeGrid.SetDimensions(probeDimension, probeDimension, probeDimension)
probeGrid.AllocateScalars(vtk.VTK_DOUBLE, 1)
meshBounds = [0]*6
meshNode.GetRASBounds(meshBounds)
probeGrid.SetOrigin(meshBounds[0], meshBounds[2], meshBounds[4])
probeSize = (meshBounds[1] - meshBounds[0], meshBounds[3] - meshBounds[2], meshBounds[5] - meshBounds[4])
probeGrid.SetSpacing(probeSize[0]/probeDimension, probeSize[1]/probeDimension, probeSize[2]/probeDimension)

probeFilter = vtk.vtkProbeFilter()
probeFilter.SetInputData(probeGrid)
probeFilter.SetSourceData(meshNode.GetUnstructuredGrid())
probeFilter.SetPassPointArrays(True)
probeFilter.Update()

probeImage = probeFilter.GetOutputDataObject(0)
probeArray = vtk.util.numpy_support.vtk_to_numpy(probeImage.GetPointData().GetArray("Displacement"))
probeArray = numpy.reshape(probeArray, (probeDimension,probeDimension,probeDimension,3))

displacementGridNode = slicer.mrmlScene.GetFirstNodeByName("Displacement")
if not displacementGridNode:
    displacementGridNode = addGridTransformFromArray(probeArray, name="Displacement")
displacementGrid = displacementGridNode.GetTransformFromParent().GetDisplacementGrid()
# TODO: next two lines should be in ijkToRAS of grid node
displacementGrid.SetOrigin(probeImage.GetOrigin())
displacementGrid.SetSpacing(probeImage.GetSpacing())


#
# do the sofa part
#

import Sofa
import SofaRuntime

from stlib3.scene import MainHeader, ContactHeader
from stlib3.solver import DefaultSolver
from stlib3.physics.deformable import ElasticMaterialObject
from stlib3.physics.rigid import Floor
from splib3.numerics import Vec3

dt = 0.05
gravityVector = [0, -1000 * 3, 0]

vonMisesMode = {
    "none": 0,
    "corotational": 1,
    "fullGreen": 2
}

# TODO: calculate the per-element young's moduli
"""
pointsArray = numpy.array(grid.GetPoints().GetData())
cellsArray = numpy.array(grid.GetCells().GetData())
tetrahedraArray = cellsArray.reshape(-1,5)[:,1:5]
centroidsArray = numpy.mean(pointsArray[tetrahedraArray], axis=1)
youngModulusBase = 1.5
youngModulusArray = numpy.ones(centroidsArray.shape[0]) * youngModulusBase
elementsOnRight = numpy.where(centroidsArray[:,0] > 0)[0]
youngModulusArray[elementsOnRight] += 3
"""
youngModulusArray = 1.5

# build the Sofa scene

rootSofaNode = Sofa.Core.Node()

MainHeader(rootSofaNode, plugins=[
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
], dt=dt, gravity=gravityVector)

rootSofaNode.addObject('FreeMotionAnimationLoop',
                   parallelODESolving=True,
                   parallelCollisionDetectionAndFreeMotion=True)

meshSofaNode = rootSofaNode.addChild('Mesh')
meshSofaNode.addObject('EulerImplicitSolver', firstOrder=False, rayleighMass=0.1, rayleighStiffness=0.1)
meshSofaNode.addObject('SparseLDLSolver', name="precond",
                       template="CompressedRowSparseMatrixd",
                       parallelInverseProduct=True)
meshSofaNode.addObject('TetrahedronSetTopologyContainer', name="Container",
                       position=extractedPointsArray,
                       tetrahedra=extractedTetrahedraArray)
meshSofaNode.addObject('TetrahedronSetTopologyModifier', name="Modifier")
meshSofaNode.addObject('MechanicalObject', name="mstate", template="Vec3d")
meshSofaNode.addObject('TetrahedronFEMForceField', name="FEM",
                       youngModulus=youngModulusArray,
                       poissonRatio=0.45,
                       method="large",
                       computeVonMisesStress=vonMisesMode['fullGreen'])
meshSofaNode.addObject('MeshMatrixMass', totalMass=1)

fixedSurface = meshSofaNode.addChild('FixedSurface')
fixedSurface.addObject('FixedConstraint', indices=numpy.where(attachedPoints))

Sofa.Simulation.init(rootSofaNode)
Sofa.Simulation.reset(rootSofaNode)
mechanicalState = meshSofaNode.getMechanicalState()
forceField = meshSofaNode.getForceField(0)

iteration = 0
iterations = 30
simulating = True
def updateSimulation():
    global iteration, iterations, simulating

    Sofa.Simulation.animate(rootSofaNode, rootSofaNode.dt.value)

    # update model from mechanical state
    meshPointsArray = mechanicalState.position.array()
    modelPointsArray = slicer.util.arrayFromModelPoints(meshNode)
    modelPointsArray[:] = meshPointsArray
    slicer.util.arrayFromModelPointsModified(meshNode)

    # update stress from forceField
    stressArray = slicer.util.arrayFromModelCellData(meshNode, "VonMisesStress")
    stressArray[:] = forceField.vonMisesPerElement.array()
    slicer.util.arrayFromModelCellDataModified(meshNode, "VonMisesStress")

    # update grid transform from displacements
    displacementArray = slicer.util.arrayFromModelPointData(meshNode, "Displacement")
    displacementArray[:] = (mechanicalState.position - mechanicalState.rest_position) 
    slicer.util.arrayFromModelPointsModified(meshNode)
    probeFilter.Update()
    probeImage = probeFilter.GetOutputDataObject(0)
    probeVTKArray = probeImage.GetPointData().GetArray("Displacement")
    probeArray = vtk.util.numpy_support.vtk_to_numpy(probeVTKArray)
    probeArrayShape = (probeDimension,probeDimension,probeDimension,3)
    probeArray = probeArray.reshape(probeArrayShape)
    gridArray = slicer.util.arrayFromGridTransform(displacementGridNode)
    gridArray[:] = -1. * probeArray
    slicer.util.arrayFromGridTransformModified(displacementGridNode)

    iteration += 1
    simulating = iteration < iterations
    if iteration % 10 == 0:
        print(f"Iteration {iteration}")
    if simulating:
        qt.QTimer.singleShot(10, updateSimulation)
    else:
        print("Simlation stopped")

updateSimulation()

