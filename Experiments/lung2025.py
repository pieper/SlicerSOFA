import math
import numpy
import vtk.util.numpy_support

targetNodeSize = 20
meshPadding = 2 * targetNodeSize
targetMaxEdge = 1.1 * targetNodeSize

probeDimension = 20

ctTransparentValue = -2048 # well below CT air
ctThresholdValue = -2047
ctAirValue = -1000
ctWindow = 1400
ctLevel = -500

attachmentRAS = numpy.array([38.49312772285862, 159.75102715306605, -225.53033447265625])
attachmentRadius = 42

penetrationPenalty = 15

poissonRatio=0.48
youngModulus = 1.5

dt = 0.025
gravityVector = [-50000, 0, 0]

# set up input data
print("loading...")

ctFilePath = "/opt/data/SlicerLung/EservalRocha/Model case/6 VOL MEDIASTINO.nrrd"
segFilePath = "/opt/data/SlicerLung/EservalRocha/Model case/RightLung.seg.nrrd"

ctNode = slicer.mrmlScene.GetFirstNodeByName("ct")
if not ctNode:
    ctNode = slicer.util.loadVolume(ctFilePath, properties = {"name": "ct"})
    ctLungNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
    ctLungNode.SetName("ctLung")
    ctCavityNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
    ctCavityNode.SetName("ctCavity")
ctIJKToRAS = vtk.vtkMatrix4x4()
ctNode.GetIJKToRASMatrix(ctIJKToRAS)
ctLungNode.SetIJKToRASMatrix(ctIJKToRAS)
ctCavityNode.SetIJKToRASMatrix(ctIJKToRAS)

segNode = slicer.mrmlScene.GetFirstNodeByName("seg")
if not segNode:
    segNode = slicer.util.loadSegmentation(segFilePath, properties = {"name": "seg"})

meshNode = slicer.mrmlScene.GetFirstNodeByName("mesh")
if not meshNode:
    meshNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode")
    meshNode.SetName("mesh")
    meshNode.CreateDefaultDisplayNodes()

originalMeshNode = slicer.mrmlScene.GetFirstNodeByName("originalMesh")
if not originalMeshNode:
    originalMeshNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode")
    originalMeshNode.SetName("originalMesh")
    originalMeshNode.CreateDefaultDisplayNodes()

labelNode = slicer.mrmlScene.GetFirstNodeByName("label")
if not labelNode:
    labelNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
    labelNode.SetName("label")

cavityLabelNode = slicer.mrmlScene.GetFirstNodeByName("cavityLabel")
if not cavityLabelNode:
    cavityLabelNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
    cavityLabelNode.SetName("cavityLabel")

cavityDistanceNode = slicer.mrmlScene.GetFirstNodeByName("cavityDistance")
if not cavityDistanceNode:
    cavityDistanceNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
    cavityDistanceNode.SetName("cavityDistance")

cavityGradientNode = slicer.mrmlScene.GetFirstNodeByName("cavityGradient")
if not cavityGradientNode:
    cavityGradientNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLVectorVolumeNode")
    cavityGradientNode.SetName("cavityGradient")

shrinkTransform = slicer.mrmlScene.GetFirstNodeByName("shrink")
if not shrinkTransform:
    shrinkTransform = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode")
    shrinkTransform.SetName("shrink")

# Create the lung cavity
print("creating cavity...")

segNode.SetReferenceImageGeometryParameterFromVolumeNode(ctNode)
forceToSingleLayer = True
segNode.GetSegmentation().CollapseBinaryLabelmaps(forceToSingleLayer)

slicer.modules.segmentations.logic().ExportVisibleSegmentsToLabelmapNode(segNode, labelNode, ctNode)
labelArray = slicer.util.arrayFromVolume(labelNode)

cavityLabelImage = cavityLabelNode.GetImageData()
if not cavityLabelImage:
    cavityLabelImage = vtk.vtkImageData()
    cavityLabelImage.DeepCopy(labelNode.GetImageData())
    cavityLabelNode.SetAndObserveImageData(cavityLabelImage)
    cavityArray = slicer.util.arrayFromVolume(cavityLabelNode)
    cavityArray[labelArray != 0] = 0 # clear all lung material
    cavityArray[labelArray == 0] = 1 # set all non-lung material
    slicer.util.arrayFromVolumeModified(cavityLabelNode)

if not ctLungNode.GetImageData() or not ctCavityNode.GetImageData():
    ctLungImage = vtk.vtkImageData()
    ctLungImage.DeepCopy(ctNode.GetImageData())
    ctLungNode.SetAndObserveImageData(ctLungImage)
    ctLungArray = slicer.util.arrayFromVolume(ctLungNode)
    ctLungArray[cavityArray == 1] = ctTransparentValue
    slicer.util.arrayFromVolumeModified(ctLungNode)
    ctLungNode.CreateDefaultDisplayNodes()
    ctLungNode.GetDisplayNode().SetApplyThreshold(True)
    ctLungNode.GetDisplayNode().SetAutoWindowLevel(False)
    ctLungNode.GetDisplayNode().SetWindow(ctWindow)
    ctLungNode.GetDisplayNode().SetLevel(ctLevel)
    ctLungNode.GetDisplayNode().SetLowerThreshold(ctThresholdValue)
    ctCavityImage = vtk.vtkImageData()
    ctCavityImage.DeepCopy(ctNode.GetImageData())
    ctCavityNode.SetAndObserveImageData(ctCavityImage)
    ctCavityArray = slicer.util.arrayFromVolume(ctCavityNode)
    ctCavityArray[cavityArray == 0] = ctAirValue
    slicer.util.arrayFromVolumeModified(ctCavityNode)
    ctCavityNode.CreateDefaultDisplayNodes()
    ctCavityNode.GetDisplayNode().SetAutoWindowLevel(False)
    ctCavityNode.GetDisplayNode().SetWindow(ctWindow)
    ctCavityNode.GetDisplayNode().SetLevel(ctLevel)


# make the cavity vector field (gradient)

if not cavityDistanceNode.GetImageData() or not cavityGradientNode.GetImageData():
    cavityCast = vtk.vtkImageCast() # TODO: make 1mm isotropic with identity ijkToRAS
    cavityCast.SetOutputScalarTypeToFloat()
    cavityCast.SetInputData(cavityLabelImage)
    cavityDistance = vtk.vtkImageEuclideanDistance() # TODO sqrt?
    cavityDistance.SetInputConnection(cavityCast.GetOutputPort())
    cavityMath = vtk.vtkImageMathematics()
    cavityMath.SetOperationToSquareRoot()
    cavityMath.SetInputConnection(cavityDistance.GetOutputPort())
    cavityGradient = vtk.vtkImageGradient()
    cavityGradient.SetDimensionality(3)
    cavityGradient.SetInputConnection(cavityMath.GetOutputPort())
    cavityGradient.Update()
    cavityDistanceNode.SetAndObserveImageData(cavityMath.GetOutputDataObject(0))
    cavityGradientNode.SetAndObserveImageData(cavityGradient.GetOutputDataObject(0))
    cavityLabelNode.SetIJKToRASMatrix(ctIJKToRAS)
    cavityDistanceNode.SetIJKToRASMatrix(ctIJKToRAS)
    cavityGradientNode.SetIJKToRASMatrix(ctIJKToRAS)
cavityDistanceArray = slicer.util.arrayFromVolume(cavityDistanceNode)
cavityGradientArray = slicer.util.arrayFromVolume(cavityGradientNode)

# create mesh points
print("meshing...")

segBounds = numpy.ndarray(6)
segNode.GetRASBounds(segBounds)
axes = []
for start,end in segBounds.reshape(-1,2):
    start -= 0*meshPadding
    end += 0*meshPadding
    nodeCount = math.ceil((end-start) / targetNodeSize)
    axes.append(numpy.linspace(start, end, nodeCount))
gridPoints = numpy.transpose(numpy.meshgrid(*axes))
inputPoints = gridPoints.ravel().reshape(-1,3)
homogeneousCoordinates = numpy.ones((inputPoints.shape[0], 1))
inputPoints = numpy.hstack((inputPoints, homogeneousCoordinates))

# run Delaunay on points that are in segmentation

pointSet = vtk.vtkPointSet()
points = vtk.vtkPoints()

rasToIJK = vtk.vtkMatrix4x4()
labelNode.GetRASToIJKMatrix(rasToIJK)

def rasPointToIndex(rasToIJK, rasPoint):
    pointCoordinate = numpy.floor(rasToIJK.MultiplyPoint(rasPoint))
    return tuple(reversed(numpy.array(pointCoordinate[:3], dtype="uint32")))

for p in inputPoints:
    pointIndex = rasPointToIndex(rasToIJK, p)
    # if labelArray[pointIndex]: TESTING
    try:
        if cavityDistanceArray[pointIndex] < meshPadding:
            points.InsertNextPoint(*p[:3])
    except IndexError:
        pass
pointSet.SetPoints(points)

delaunay = vtk.vtkDelaunay3D()
delaunay.SetInputData(pointSet)
delaunay.Update()
meshGrid = delaunay.GetOutputDataObject(0)

meshNode.SetAndObserveMesh(meshGrid)

# Remove elements with centroids outside segmentation or are larger than we want

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
    if maxSize < targetMaxEdge:
        tetrahedraToKeep.InsertNextId(tetraID)

extractCells = vtk.vtkExtractCells()
extractCells.SetInputData(meshGrid)
extractCells.SetCellList(tetrahedraToKeep)
extractCells.Update()

extractedMeshGrid = extractCells.GetOutputDataObject(0)
meshNode.SetAndObserveMesh(extractedMeshGrid)
meshNode.GetDisplayNode().SetEdgeVisibility(True)

originalMeshGrid = vtk.vtkUnstructuredGrid()
originalMeshGrid.DeepCopy(extractedMeshGrid)
originalMeshNode.SetAndObserveMesh(originalMeshGrid)
originalMeshNode.GetDisplayNode().SetRepresentation(slicer.vtkMRMLDisplayNode.WireframeRepresentation)

# create displacement array
print("Configuring variables...")

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
attachedPoints = numpy.linalg.norm(pointsArray - attachmentRAS, axis=1) < attachmentRadius
cellsArray = numpy.array(extractedMeshGrid.GetCells().GetData())
extractedTetrahedraArray = cellsArray.reshape(-1,5)[:,1:5]

# get surface points and triangles

surfaceFilter = vtk.vtkDataSetSurfaceFilter()
surfaceFilter.SetInputData(extractedMeshGrid)
surfaceFilter.SetPassThroughPointIds(True)
surfaceFilter.Update()


surfacePolyData = surfaceFilter.GetOutputDataObject(0)
surfacePointIDs = vtk.util.numpy_support.vtk_to_numpy(surfacePolyData.GetPointData().GetArray("vtkOriginalPointIds"))
surfaceCellsArray = numpy.array(surfacePolyData.GetPolys().GetData())
surfaceTrianglesArray = surfaceCellsArray.reshape(-1,4)[:,1:4]
surfacePointsArray = vtk.util.numpy_support.vtk_to_numpy(surfacePolyData.GetPoints().GetData())

# tetmesh indices of surface triangles
surfaceTrianglesInMesh = surfacePointIDs[surfaceTrianglesArray.flatten()].reshape(-1,3)

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
probeGrid.SetDimensions(probeDimension, probeDimension, probeDimension)
probeGrid.AllocateScalars(vtk.VTK_DOUBLE, 1)
meshBounds = numpy.ndarray(6)
meshNode.GetRASBounds(meshBounds)
meshMins = meshBounds[::2] + meshPadding
meshMaxes = meshBounds[1::2] - meshPadding
probeSize = meshMaxes - meshMins
probeGrid.SetOrigin(*meshMins)
probeGrid.SetSpacing(*(probeSize/probeDimension))

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

# TODO: see if there's a way to regularize the transform to avoid needing to turn off convergence warnings
displacementGridNode.SetGlobalWarningDisplay(0)

ctLungNode.SetAndObserveTransformNodeID(displacementGridNode.GetID())
segNode.SetAndObserveTransformNodeID(displacementGridNode.GetID())

slicer.util.setSliceViewerLayers(background=ctCavityNode, foreground=ctLungNode, foregroundOpacity=1)

#
# do the sofa part
#
print("Initializing SOFA...")

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

todo_this_seems_slower_than_the_default = """
rootSofaNode.addObject('FreeMotionAnimationLoop',
                   parallelODESolving=True,
                   parallelCollisionDetectionAndFreeMotion=True)
"""

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
                       youngModulus=youngModulus,
                       poissonRatio=poissonRatio,
                       method="large",
                       computeVonMisesStress=vonMisesMode['fullGreen'])
meshSofaNode.addObject('MeshMatrixMass', totalMass=1)

forceVectorArray = numpy.zeros_like(surfacePointsArray)
surfaceForces = meshSofaNode.addObject('ConstantForceField', indices=surfacePointIDs, forces=forceVectorArray)

vesselAttachments = meshSofaNode.addChild('VesselAttachments')
vesselAttachments.addObject('FixedConstraint', indices=numpy.where(attachedPoints))

Sofa.Simulation.init(rootSofaNode)
Sofa.Simulation.reset(rootSofaNode)
mechanicalState = meshSofaNode.getMechanicalState()
forceField = meshSofaNode.getForceField(0)

shrinkMatrix = vtk.vtkMatrix4x4()
shrinkMatrix.SetElement(0,0, 0.75)
shrinkMatrix.SetElement(1,1, 0.95)
shrinkMatrix.SetElement(2,2, 0.85)
shrinkTransform.SetMatrixTransformToParent(shrinkMatrix)
displacementGridNode.SetAndObserveTransformNodeID(shrinkTransform.GetID())

print("Starting simulation...")
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

    # update model points and grid transform from displacements
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

    # create reaction forces
    lpsToRAS = numpy.array([-1,-1,1]) # TODO apply rasToRAS to gradients and include spacing
    with surfaceForces.forces.writeableArray() as forces:
        reactionForces = numpy.zeros_like(forces)
        for index in range(len(surfacePointsArray)):
            if attachedPoints[surfacePointIDs[index]]:
                continue
            displacedRAS = modelPointsArray[surfacePointIDs[index]]
            pointIndex = rasPointToIndex(rasToIJK, numpy.array([*displacedRAS,1]))
            try:
                penetration = cavityDistanceArray[pointIndex]
            except IndexError:
                continue
            if penetration > 0:
                gradient = lpsToRAS * cavityGradientArray[pointIndex]
                forces[index] = -1 * penetration * penetrationPenalty * gradient

    # iteration management
    iteration += 1
    simulating = iteration < iterations
    if iteration % 10 == 0:
        print(f"Iteration {iteration}")
    if simulating:
        qt.QTimer.singleShot(10, updateSimulation)
    else:
        print("Simlation stopped")

updateSimulation()
