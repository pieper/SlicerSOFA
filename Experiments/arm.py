import math
import numpy
import vtk.util.numpy_support


scenePath = "/Users/pieper/slicer/latest/SlicerSOFA/Experiments/SimulatedUltrasound_Scene_Updated.mrb"
slicer.mrmlScene.Clear()
try:
    slicer.util.loadScene(scenePath)
except:
    pass

displacementEffect = 100
displacementDimension = 30
displacementSpacing = 1
displacementBound = displacementSpacing * displacementDimension
displacementShape = (displacementDimension, displacementDimension, displacementDimension, 3)
displacementArray = numpy.zeros(displacementShape)
for k in range(1, displacementDimension-1):
    for j in range(1, displacementDimension-1):
        for i in range(1, displacementDimension-1):
            x = i * displacementSpacing - 0.5 * displacementBound
            y = j * displacementSpacing - 0.5 * displacementBound
            z = displacementBound - (k * displacementSpacing)
            toPoint = numpy.array([x,y,z])
            toPointNorm = numpy.linalg.norm(toPoint)
            toPointNorm = max(0.001, toPointNorm)
            pointEffect = displacementEffect / (10 + toPointNorm)
            toPointUnit = toPoint / toPointNorm
            displacementArray[k,j,i] = toPointUnit * pointEffect
print(displacementArray.max())

gridNodeName = "Mechanical Displacement"
try:
    gridNode = slicer.util.getNode(gridNodeName)
except slicer.util.MRMLNodeNotFoundException:
    gridNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLGridTransformNode", gridNodeName)
    gridNode.CreateDefaultDisplayNodes()
displacementGrid = gridNode.GetTransformFromParent().GetDisplacementGrid()
arrayShape = displacementArray.shape
displacementGrid.SetDimensions(arrayShape[2], arrayShape[1], arrayShape[0])
scalarType = vtk.util.numpy_support.get_vtk_array_type(displacementArray.dtype)
displacementGrid.AllocateScalars(scalarType, 3)
gridArray = slicer.util.arrayFromGridTransform(gridNode)
gridArray[:] = displacementArray
slicer.util.arrayFromGridTransformModified(gridNode)
gridNode.SetGlobalWarningDisplay(0)

referenceToTracker = slicer.util.getNode("ReferenceToTracker")
#squishyPartNames = ["radial_artery", "ulnar_artery", "veins"]
squishyPartNames = ["bras", "radial_artery", "ulnar_artery", "veins"]
for partName in squishyPartNames:
    partNode = slicer.util.getNode(partName)
    partNode.HardenTransform()
    partNode.SetAndObserveTransformNodeID(gridNode.GetID())

def updateGridLocation(probeToReference, event):
    probeToWorld = vtk.vtkMatrix4x4()
    probeToReference.GetMatrixTransformFromWorld(probeToWorld)
    probeToWorld.Invert()
    gridNode.GetTransformFromParent().SetGridDirectionMatrix(probeToWorld)
    probeOrigin = [probeToWorld.GetElement(0,3), probeToWorld.GetElement(1,3), probeToWorld.GetElement(2,3)]
    gridNode.GetTransformFromParent().GetDisplacementGrid().SetOrigin(probeOrigin)


probeToReference = slicer.util.getNode("ProbeToReference")
probeToReference.CreateDefaultDisplayNodes()
probeToReference.GetDisplayNode().SetEditorVisibility3D(True)
probeToReference.GetDisplayNode().SetEditorVisibility(True)
probeToReference.AddObserver(slicer.vtkMRMLTransformNode.TransformModifiedEvent, updateGridLocation)

mechanicalTransformVolume = slicer.util.addVolumeFromArray(displacementArray, nodeClassName="vtkMRMLVectorVolumeNode")
mechanicalTransformVolume.SetName("mechanicalTransformVolume")
mechanicalTransformVolume.SetAndObserveTransformNodeID(mechanicalTransformVolume.GetID())
