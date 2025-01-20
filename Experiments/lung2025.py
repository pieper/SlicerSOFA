"""
export SLICER=/opt/sr
export DIR=/Users/pieper/slicer/latest/SOFA
export SOFA_ROOT=${DIR}/SlicerSOFA-build/SOFA-build
${SLICER}/Slicer-build/Slicer \
        --launcher-additional-settings \
            ${DIR}/SlicerSOFA-build/inner-build/AdditionalLauncherSettings.ini
"""

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

#poissonRatio=0.48
poissonRatio=0.38
#youngModulus = 1.5
youngModulus = 1.0

duration = 1
steps = 5000
dt = 0.0125
#dt = duration / steps
#gravityVector = [-50000, 0, 0]
#gravityVector = [50000, 0, 0]
gravityVector = [-60000, 0, 0]

#cavityCollisionMethod = "NewProximityIntersection"
cavityCollisionMethod = "LocalMinDistance"
#cavityCollisionMethod = "MinProximityIntersection"
cavityAlarmDistance = 10.0
cavityContactDistance = 5
cavityFriction=0.001

# set up input data
print("loading...")

ctFilePath = "/opt/data/SlicerLung/EservalRocha/Model case/6 VOL MEDIASTINO.nrrd"
segFilePath = "/opt/data/SlicerLung/EservalRocha/Model case/RightLung.seg.nrrd"
cavityDistanceFilePath = "/opt/data/SlicerLung/EservalRocha/Model case/cavityDistance.nrrd"
#cavityGradientFilePath = "/opt/data/SlicerLung/EservalRocha/Model case/cavityGradient.nrrd"

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

cavitySegmentationNode = slicer.mrmlScene.GetFirstNodeByName("cavitySegmentation")
if not cavitySegmentationNode:
    cavitySegmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
    cavitySegmentationNode.SetName("cavitySegmentation")
    cavitySegmentationNode.CreateDefaultDisplayNodes()

cavityDistanceNode = slicer.mrmlScene.GetFirstNodeByName("cavityDistance")
if not cavityDistanceNode:
    #cavityDistanceNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
    #cavityDistanceNode.SetName("cavityDistance")
    cavityDistanceNode = slicer.util.loadVolume(cavityDistanceFilePath, properties = {"name": "cavityDistance"})

#cavityGradientNode = slicer.mrmlScene.GetFirstNodeByName("cavityGradient")
#if not cavityGradientNode:
    #cavityGradientNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLVectorVolumeNode")
    #cavityGradientNode.SetName("cavityGradient")
    #cavityGradientNode = slicer.util.loadVolume(cavityGradientFilePath, properties = {"name": "cavityGradient"})

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

cavitySegmentCount = cavitySegmentationNode.GetSegmentation().GetNumberOfSegments()
if cavitySegmentCount == 0:
    slicer.vtkSlicerSegmentationsModuleLogic.ImportLabelmapToSegmentationNode(
            cavityLabelNode,
            cavitySegmentationNode)
    segmentID = cavitySegmentationNode.GetSegmentation().GetSegmentIDs()[0]
    segmentationArray = slicer.util.arrayFromSegmentBinaryLabelmap(
                            cavitySegmentationNode, segmentID, ctNode)
    segmentationArray[:] = 1 * numpy.logical_not(cavityArray)
    slicer.util.updateSegmentBinaryLabelmapFromArray(
                            segmentationArray, cavitySegmentationNode, segmentID, ctNode)
    closedSurfaceName = slicer.vtkSegmentationConverter.GetSegmentationClosedSurfaceRepresentationName()
    cavitySegmentationNode.GetSegmentation().CreateRepresentation(closedSurfaceName)
    cavityPolyData = vtk.vtkPolyData()
    cavitySegmentationNode.GetClosedSurfaceRepresentation(segmentID, cavityPolyData)
    cavitySegmentationNode.GetDisplayNode().SetOpacity(0.3)


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

# if not cavityDistanceNode.GetImageData() or not cavityGradientNode.GetImageData():
if not cavityDistanceNode.GetImageData():
    cavityCast = vtk.vtkImageCast() # TODO: make 1mm isotropic with identity ijkToRAS
    cavityCast.SetOutputScalarTypeToFloat()
    cavityCast.SetInputData(cavityLabelImage)
    cavityDistance = vtk.vtkImageEuclideanDistance() # TODO sqrt?
    cavityDistance.SetInputConnection(cavityCast.GetOutputPort())
    cavityMath = vtk.vtkImageMathematics()
    cavityMath.SetOperationToSquareRoot()
    cavityMath.SetInputConnection(cavityDistance.GetOutputPort())
    #cavityGradient = vtk.vtkImageGradient()
    #cavityGradient.SetDimensionality(3)
    #cavityGradient.SetInputConnection(cavityMath.GetOutputPort())
    #cavityGradient.Update()
    cavityDistanceNode.SetAndObserveImageData(cavityMath.GetOutputDataObject(0))
    #cavityGradientNode.SetAndObserveImageData(cavityGradient.GetOutputDataObject(0))
    cavityLabelNode.SetIJKToRASMatrix(ctIJKToRAS)
    cavityDistanceNode.SetIJKToRASMatrix(ctIJKToRAS)
    #cavityGradientNode.SetIJKToRASMatrix(ctIJKToRAS)
cavityDistanceArray = slicer.util.arrayFromVolume(cavityDistanceNode)
#cavityGradientArray = slicer.util.arrayFromVolume(cavityGradientNode)

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
    if maxSize < 2 * targetMaxEdge:
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
attachedPointMap = numpy.linalg.norm(pointsArray - attachmentRAS, axis=1) < attachmentRadius
cellsArray = numpy.array(extractedMeshGrid.GetCells().GetData())
extractedTetrahedraArray = cellsArray.reshape(-1,5)[:,1:5]

# get surface points and triangles

surfaceFilter = vtk.vtkDataSetSurfaceFilter()
surfaceFilter.SetInputData(extractedMeshGrid)
surfaceFilter.SetPassThroughPointIds(True)
surfaceNormals = vtk.vtkPolyDataNormals()
surfaceNormals.SetInputConnection(surfaceFilter.GetOutputPort())
surfaceNormals.Update()

surfacePolyData = surfaceNormals.GetOutputDataObject(0)
surfacePointNormals = vtk.util.numpy_support.vtk_to_numpy(surfacePolyData.GetPointData().GetArray("Normals"))
surfacePointIDs = vtk.util.numpy_support.vtk_to_numpy(surfacePolyData.GetPointData().GetArray("vtkOriginalPointIds"))
surfaceCellsArray = numpy.array(surfacePolyData.GetPolys().GetData())
surfaceTrianglesArray = surfaceCellsArray.reshape(-1,4)[:,1:4]
surfacePointsArray = vtk.util.numpy_support.vtk_to_numpy(surfacePolyData.GetPoints().GetData())

# tetmesh indices of surface triangles
surfaceTrianglesInMesh = surfacePointIDs[surfaceTrianglesArray.flatten()].reshape(-1,3)

# mesh indices of medial 1/2 of the surface 
lrSurfaceNormals = surfacePointNormals.transpose()[0]
medialSurface = lrSurfaceNormals < 0
medialSurfaceMeshIndices = surfacePointIDs[numpy.where(medialSurface)]

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
# - could use a mesh quality filter to look for noninvertible nodes and auto-adjust the timestep
displacementGridNode.SetGlobalWarningDisplay(0)

# Extra visual deflation (not part of simulation)

shrinkMatrix = vtk.vtkMatrix4x4()
shrinkMatrix.SetElement(0,0, 0.75)
shrinkMatrix.SetElement(1,1, 0.95)
shrinkMatrix.SetElement(2,2, 0.85)
shrinkTransform.SetMatrixTransformToParent(shrinkMatrix)
#displacementGridNode.SetAndObserveTransformNodeID(shrinkTransform.GetID())


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
    "MultiThreading",
    "Sofa.Component.AnimationLoop",
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
rootSofaNode.addObject('FreeMotionAnimationLoop')

# Collision of lung with cavity

rootSofaNode.addObject("CollisionPipeline")
rootSofaNode.addObject("ParallelBruteForceBroadPhase")
rootSofaNode.addObject("ParallelBVHNarrowPhase")
rootSofaNode.addObject(cavityCollisionMethod,
                       alarmDistance=cavityAlarmDistance,
                       contactDistance=cavityContactDistance)
rootSofaNode.addObject("CollisionResponse",
                       response="FrictionContactConstraint",
                       responseParams=cavityFriction)
rootSofaNode.addObject("GenericConstraintSolver")

# lung mesh mechanics

lungMeshSofaNode = rootSofaNode.addChild('LungMesh')
lungMeshSofaNode.addObject('EulerImplicitSolver',
                           firstOrder=False,
                           rayleighMass=0.1,
                           rayleighStiffness=0.1)
lungMeshSofaNode.addObject('SparseLDLSolver', name="precond",
                       template="CompressedRowSparseMatrixd",
                       parallelInverseProduct=True)
lungMeshSofaNode.addObject('TetrahedronSetTopologyContainer', name="Container",
                       position=extractedPointsArray,
                       tetrahedra=extractedTetrahedraArray)
lungMeshSofaNode.addObject('MechanicalObject', name="mstate", template="Vec3d")
lungMeshSofaNode.addObject('TetrahedronFEMForceField', name="FEM",
                       youngModulus=youngModulus,
                       poissonRatio=poissonRatio,
                       method="large",
                       computeVonMisesStress=vonMisesMode['fullGreen'])
lungMeshSofaNode.addObject('MeshMatrixMass', totalMass=1)
lungMeshSofaNode.addObject("LinearSolverConstraintCorrection")

lungCollisionSofaNode = lungMeshSofaNode.addChild("LungCollision")
lungCollisionSofaNode.addObject("TriangleSetTopologyContainer")
lungCollisionSofaNode.addObject("TriangleSetTopologyModifier")
lungCollisionSofaNode.addObject("Tetra2TriangleTopologicalMapping")
lungCollisionSofaNode.addObject("PointCollisionModel")
lungCollisionSofaNode.addObject("LineCollisionModel")
lungCollisionSofaNode.addObject("TriangleCollisionModel")

################
#### Sphere ####
################
"""
sphere_node = rootSofaNode.addChild("sphere")
#sphere_node.addObject("MeshOBJLoader", filename=str("./cavityBall.obj"), scale=1.0)
sphere_node.addObject("MeshOBJLoader", filename=str("./bigCavity.obj"), scale=1.0)
sphere_node.addObject("TriangleSetTopologyContainer", src=sphere_node.MeshOBJLoader.getLinkPath())
sphere_node.addObject("TriangleSetTopologyModifier")
sphere_node.addObject("MechanicalObject")
# Paul says: NOTE: The important thing is to set bothSide=True for the collision models, so that both sides of the triangle are considered for collision.
sphere_node.addObject("TriangleCollisionModel", bothSide=True)
sphere_node.addObject("PointCollisionModel")
sphere_node.addObject("LineCollisionModel")
sphere_node.addObject("FixedProjectiveConstraint")
"""

# great vessels that hold lung to heart and medial wall

vesselAttachments = lungMeshSofaNode.addChild('VesselAttachments')
attachedPointIndices = numpy.where(attachedPointMap)[0]
fixedIndices = numpy.unique(numpy.concatenate([attachedPointIndices, medialSurfaceMeshIndices]))
vesselAttachments.addObject('FixedConstraint', indices=fixedIndices)

# Runnning the simulation

Sofa.Simulation.init(rootSofaNode)
Sofa.Simulation.reset(rootSofaNode)
mechanicalState = lungMeshSofaNode.getMechanicalState()
forceField = lungMeshSofaNode.getForceField(0)

# Convert LPS to RAS
"""
with sphere_node.MechanicalObject.position.writeable() as sphereArray:
    sphereArray *= [-1,-1,1]
"""

print("Ready to start...")
iteration = 0
iterations = 30
simulating = True
def updateSimulation():
    global iteration, iterations, simulating

    Sofa.Simulation.animate(rootSofaNode, rootSofaNode.dt.value)


    # Bring back surface calculation and make nodes fixed when they exit cavity

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

    # iteration management
    iteration += 1
    simulating = iteration < iterations
    if iteration % 10 == 0:
        print(f"Iteration {iteration}")
    if simulating:
        qt.QTimer.singleShot(10, updateSimulation)
    else:
        print("Simlation stopped")

print("Starting simulation...")
#updateSimulation()
