"""
export SLICER=/opt/sr
export DIR=/Users/pieper/slicer/latest/SOFA
export SOFA_ROOT=${DIR}/SlicerSOFA-build/SOFA-build
${SLICER}/Slicer-build/Slicer \
        --launcher-additional-settings \
            ${DIR}/SlicerSOFA-build/inner-build/AdditionalLauncherSettings.ini
"""

import json
import math
import numpy
import vtk.util.numpy_support

targetNodeSize = 10
meshPadding = 2 * targetNodeSize
targetMaxEdge = 1.1 * targetNodeSize

probeDimension = 20

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
#gravityVector = [-9000, 0, 0]
#gravityVector = [0, 0, 90000]
gravityVector = [0, 0, 0]

#cavityCollisionMethod = "NewProximityIntersection"
cavityCollisionMethod = "LocalMinDistance"
#cavityCollisionMethod = "MinProximityIntersection"
cavityAlarmDistance = 10.0
cavityContactDistance = 5
cavityFriction=0.001

needleInfluenceRadius = 20
needleForce = 25

# set up input data
print("loading...")

#mrFilePath = "/opt/data/idc/prostate/t2cor.nrrd"
#segFilePath = "/opt/data/idc/prostate/t2cor.seg.nrrd"
#mrFilePath = "/opt/data/idc/prostate/ct-seg/ct cropped.nrrd"
#segFilePath = "/opt/data/idc/prostate/ct-seg/tissues-label croppe_F17V.seg.nrrd"
mrFilePath = "/opt/data/idc/prostate/ct-seg/ct.nrrd"
segFilePath = "/opt/data/idc/prostate/ct-seg/ct.seg.nrrd"
materialsFilePath = "/Users/pieper/slicer/latest/SOFA/SlicerSOFA-pieper/Experiments/ts-materials.json"

materials = json.loads(open(materialsFilePath).read())

mrNode = slicer.mrmlScene.GetFirstNodeByName("mr")
if not mrNode:
    mrNode = slicer.util.loadVolume(mrFilePath, properties = {"name": "mr"})
mrIJKToRAS = vtk.vtkMatrix4x4()
mrNode.GetIJKToRASMatrix(mrIJKToRAS)

segNode = slicer.mrmlScene.GetFirstNodeByName("seg")
if not segNode:
    segNode = slicer.util.loadSegmentation(segFilePath, properties = {"name": "seg"})

roiNode = slicer.mrmlScene.GetFirstNodeByName("pelvis roi")
if not roiNode:
    roiNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsROINode")
    roiNode.SetName("pelvis roi")
    if False:
        roiNode.SetCenter(12, -28, 1183)
        roiNode.SetSize(318, 268, 215)
    else:
        roiNode.SetCenter(19, -27, 1158)
        roiNode.SetSize(153, 160, 166)
    roiNode.CreateDefaultDisplayNodes()
    roiNode.GetDisplayNode().SetVisibility(False)

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

needleNode = slicer.mrmlScene.GetFirstNodeByName("needle")
if not needleNode:
    needleNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsLineNode")
    needleNode.SetName("needle")
    needleNode.AddControlPoint(19, -60, 1063)
    needleNode.AddControlPoint(14, -60, 1120)
    slicer.modules.markups.logic().JumpSlicesToNthPointInMarkup(needleNode.GetID(), 1)

segNode.SetReferenceImageGeometryParameterFromVolumeNode(mrNode)
forceToSingleLayer = True
segNode.GetSegmentation().CollapseBinaryLabelmaps(forceToSingleLayer)

slicer.modules.segmentations.logic().ExportSegmentsToLabelmapNode(segNode, segNode.GetSegmentation().GetSegmentIDs(), labelNode, mrNode, slicer.vtkSegmentation.EXTENT_REFERENCE_GEOMETRY)
labelArray = slicer.util.arrayFromVolume(labelNode)

colorNode = labelNode.GetDisplayNode().GetColorNode()
boneLabelIndices = []
for boneSegmentID in materials['bones']:
    boneName = segNode.GetSegmentation().GetSegment(boneSegmentID).GetName()
    boneIndex = colorNode.GetColorIndexByName(boneName)
    boneLabelIndices.append(boneIndex)


# create mesh points
print("meshing...")

roiBounds = numpy.ndarray(6)
roiNode.GetRASBounds(roiBounds)
axes = []
for start,end in roiBounds.reshape(-1,2):
    start -= 0*meshPadding
    end += 0*meshPadding
    nodeCount = math.ceil((end-start) / targetNodeSize)
    axes.append(numpy.linspace(start, end, nodeCount))
gridPoints = numpy.transpose(numpy.meshgrid(*axes))
inputPoints = gridPoints.ravel().reshape(-1,3)
homogeneousCoordinates = numpy.ones((inputPoints.shape[0], 1))
inputPoints = numpy.hstack((inputPoints, homogeneousCoordinates))

# run Delaunay on points that are in segmentation

includeAllPoints = True

pointSet = vtk.vtkPointSet()
points = vtk.vtkPoints()

rasToIJK = vtk.vtkMatrix4x4()
labelNode.GetRASToIJKMatrix(rasToIJK)

def rasPointToIndex(rasToIJK, rasPoint):
    pointCoordinate = numpy.floor(rasToIJK.MultiplyPoint(rasPoint))
    return tuple(reversed(numpy.array(pointCoordinate[:3], dtype="uint32")))


outsideNodeIndices = []
boneNodeIndices = []
for p in inputPoints:
    pointIndex = rasPointToIndex(rasToIJK, p)
    try:
        if labelArray[pointIndex] or includeAllPoints:
            newPoint = points.InsertNextPoint(*p[:3])
            if labelArray[pointIndex] in boneLabelIndices:
                boneNodeIndices.append(newPoint)
    except IndexError:
        pass
pointSet.SetPoints(points)

delaunay = vtk.vtkDelaunay3D()
delaunay.SetInputData(pointSet)
delaunay.Update()
meshGrid = delaunay.GetOutputDataObject(0)

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

# get points and tetrahedra to pass to sofa

extractedPointsArray = numpy.array(extractedMeshGrid.GetPoints().GetData())
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

# mesh indices pointing down
isSurfaceNormals = surfacePointNormals.transpose()[2]
bottomSurface = isSurfaceNormals < 0
bottomNodeIndices = surfacePointIDs[numpy.where(bottomSurface)]

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


mrNode.SetAndObserveTransformNodeID(displacementGridNode.GetID())
segNode.SetAndObserveTransformNodeID(displacementGridNode.GetID())

slicer.util.setSliceViewerLayers(background=mrNode, foreground=None, foregroundOpacity=1)

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
"""
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
"""

# pelvic mesh mechanics

pelvisMeshSofaNode = rootSofaNode.addChild('PelvisMesh')
pelvisMeshSofaNode.addObject('EulerImplicitSolver',
                           firstOrder=False,
                           rayleighMass=0.1,
                           rayleighStiffness=0.1)
pelvisMeshSofaNode.addObject('SparseLDLSolver', name="precond",
                       template="CompressedRowSparseMatrixd",
                       parallelInverseProduct=True)
pelvisMeshSofaNode.addObject('TetrahedronSetTopologyContainer', name="Container",
                       position=extractedPointsArray,
                       tetrahedra=extractedTetrahedraArray)
pelvisMeshSofaNode.addObject('MechanicalObject', name="mstate", template="Vec3d")
pelvisMeshSofaNode.addObject('TetrahedronFEMForceField', name="FEM",
                       youngModulus=youngModulus,
                       poissonRatio=poissonRatio,
                       method="large",
                       computeVonMisesStress=vonMisesMode['fullGreen'])
pelvisMeshSofaNode.addObject('MeshMatrixMass', totalMass=1)
pelvisMeshSofaNode.addObject("LinearSolverConstraintCorrection")

pelvisCollisionSofaNode = pelvisMeshSofaNode.addChild("LungCollision")
pelvisCollisionSofaNode.addObject("TriangleSetTopologyContainer")
pelvisCollisionSofaNode.addObject("TriangleSetTopologyModifier")
pelvisCollisionSofaNode.addObject("Tetra2TriangleTopologicalMapping")
pelvisCollisionSofaNode.addObject("PointCollisionModel")
pelvisCollisionSofaNode.addObject("LineCollisionModel")
pelvisCollisionSofaNode.addObject("TriangleCollisionModel")

# apply the boundary conditions
vesselAttachments = pelvisMeshSofaNode.addChild('VesselAttachments')
fixedIndices = numpy.unique(numpy.concatenate([boneNodeIndices, bottomNodeIndices]))
vesselAttachments.addObject('FixedConstraint', indices=fixedIndices)

needleForcesArray = numpy.zeros_like(extractedPointsArray)
nodeIndices = numpy.linspace(0, needleForcesArray.shape[0]-1, needleForcesArray.shape[0], dtype='int32')
needleForces = pelvisMeshSofaNode.addObject('ConstantForceField', indices=nodeIndices, forces=needleForcesArray)

def distanceToLineSegment(point, start, end):
    tangent = (end - start) / numpy.linalg.norm(end - start)
    # signed parallel distance components
    s = numpy.dot(start - point, tangent)
    t = numpy.dot(point - end, tangent)
    clampedParallelDistance = numpy.maximum.reduce([s, t, 0])
    perpendicularDistanceComponent = numpy.linalg.norm(numpy.cross(point - start, tangent))
    return numpy.hypot(clampedParallelDistance, perpendicularDistanceComponent)

# Runnning the simulation

Sofa.Simulation.init(rootSofaNode)
Sofa.Simulation.reset(rootSofaNode)
mechanicalState = pelvisMeshSofaNode.getMechanicalState()
forceField = pelvisMeshSofaNode.getForceField(0)

print("Ready to start...")
iteration = 0
iterations = 30
simulating = True
def updateSimulation():
    global iteration, iterations, simulating

    Sofa.Simulation.animate(rootSofaNode, rootSofaNode.dt.value)

    # update mseh from mechanical state
    meshPointsArray = mechanicalState.position.array()
    modelPointsArray = slicer.util.arrayFromModelPoints(meshNode)
    modelPointsArray[:] = meshPointsArray
    slicer.util.arrayFromModelPointsModified(meshNode)

    # update mesh stress stress from forceField
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

    # create needle forces
    with needleForces.forces.writeableArray() as forces:
        base = numpy.array(needleNode.GetNthControlPointPosition(0))
        tip = numpy.array(needleNode.GetNthControlPointPosition(1))
        newForces = numpy.zeros_like(forces)
        for index in nodeIndices:
            originalRAS = extractedPointsArray[index]
            displacedRAS = modelPointsArray[index]
            needleLength = numpy.linalg.norm(tip-base)
            needleTangent = (tip-base) / needleLength
            nodeDistance = distanceToLineSegment(originalRAS, base, tip)
            if nodeDistance < needleInfluenceRadius:
                forces[index] = needleForce

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
updateSimulation()



def onValueChanged(int):
    global displacementGridNode, finalDisplacements
    global slider
    interpolatedDisplacements = slicer.util.arrayFromGridTransform(displacementGridNode)
    animationFactor = slider.value / slider.maximum
    interpolatedDisplacements[:] = animationFactor * finalDisplacements
    slicer.util.arrayFromGridTransformModified(displacementGridNode)
    displacementGridNode.Modified()

slider = qt.QSlider()
slider.orientation = 1
slider.size = qt.QSize(500, 200)
slider.connect("valueChanged(int)", onValueChanged)
slider.show()

go = True
frame = 0
direction = 1
def animate():
    global frame, slider, direction
    slider.value = frame
    frame += direction
    if frame >= 99:
        direction = -1
    if frame <= 0:
        direction = 1
    if go:
        qt.QTimer.singleShot(10, animate)

"""
finalDisplacements = numpy.array(slicer.util.arrayFromGridTransform(displacementGridNode))
animate()
"""
