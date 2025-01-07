import math
import numpy
import random

size = 200
pointCount = 3000
mode = "grid"

markupNode = slicer.mrmlScene.GetFirstNodeByName("p")
if not markupNode:
    markupNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode")
    markupNode.SetName("p")
    markupNode.CreateDefaultDisplayNodes()
markupNode.GetDisplayNode().SetPointLabelsVisibility(False)

if mode == "random":
    inputPoints = size * (-0.5 + numpy.random.rand(pointCount,3))
elif mode == "grid":
    pointsPerAxis = math.ceil(pow(pointCount,1/3))
    raxis = numpy.linspace(-size/2, size/2, num=pointsPerAxis)
    aaxis = numpy.linspace(-size/2, size/2, num=pointsPerAxis)
    saxis = numpy.linspace(-size/2, size/2, num=pointsPerAxis)
    gridPoints = numpy.transpose(numpy.meshgrid(raxis, aaxis, saxis))
    inputPoints = gridPoints.ravel().reshape(-1,3)
else:
    print("Invalid mode")
slicer.util.updateMarkupsControlPointsFromArray(markupNode, inputPoints)

meshNode = slicer.mrmlScene.GetFirstNodeByName("mesh")
if not meshNode:
    meshNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode")
    meshNode.SetName("mesh")
    meshNode.CreateDefaultDisplayNodes()


pointSet = vtk.vtkPointSet()
points = vtk.vtkPoints()
for p in inputPoints:
    points.InsertNextPoint(*p)
pointSet.SetPoints(points)

delaunay = vtk.vtkDelaunay3D()
delaunay.SetInputData(pointSet)
delaunay.Update()
meshGrid = delaunay.GetOutputDataObject(0)

meshNode.SetAndObserveMesh(meshGrid)




