"""

SOFA_ROOT=/media/volume/sdb/Slicer-SOFA-build/SOFA-build \
        /media/volume/sdb/Slicer-superbuild/Slicer-build/Slicer \
        --launcher-additional-settings /media/volume/sdb/Slicer-SOFA-build/inner-build/AdditionalLauncherSettings.ini
        --additional-module-path /media/volume/sdb/SlicerOpenIGTLink-build/inner-build/lib/Slicer-5.7/qt-loadable-modules \
        --additional-module-path /media/volume/sdb/SlicerOpenIGTLink-build/inner-build/lib/Slicer-5.7/qt-scripted-modules

SOFA_ROOT=/media/volume/sdb/Slicer-SOFA-build/SOFA-build \
        /media/volume/sdb/Slicer-superbuild/Slicer-build/Slicer \
        --launcher-additional-settings /media/volume/sdb/Slicer-SOFA-build/inner-build/AdditionalLauncherSettings.ini \
        --launcher-additional-settings /media/volume/sdb/SlicerOpenIGTLink-build/inner-build/AdditionalLauncherSettings.ini

SOFA_ROOT=/media/volume/sdb/Slicer-SOFA-build/SOFA-build \
        /media/volume/sdb/Slicer-superbuild/Slicer-build/Slicer \
        --launcher-additional-settings \
            /media/volume/sdb/Slicer-SOFA-build/inner-build/AdditionalLauncherSettings.ini \
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

class ParameterNode(object):
    dt=0.01
    modelNodeFileName="/home/exouser/right-lung-mesh-low.vtk"
    # grav and roi are in LPS
    def getGravityVector(self):
        #return [0,0,-10000]
        return [0,10000,0]
    def getBoundaryROI(self):
        #[ 0, -220, 0, 30, -170, -300],
        #return [0, 0, 0, 0, 0, 0]
        #return [0, -170, 0, 48, -80, -100]
        return [-300, 300, -380, 300, -300, -200]
        #return [-300, 300, 130, 380, -300, 300]

parameterNode = ParameterNode()

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

rootNode.addObject('VisualStyle', displayFlags='showVisualModels showForceFields')
rootNode.addObject('BackgroundSetting', color=[0.8, 0.8, 0.8, 1])
rootNode.addObject('DefaultAnimationLoop', name="FreeMotionAnimationLoop", parallelODESolving=True)

meshNode = rootNode.addChild('Mesh')
meshNode.addObject('EulerImplicitSolver', firstOrder=False, rayleighMass=0.1, rayleighStiffness=0.1)
meshNode.addObject('SparseLDLSolver', name="precond", template="CompressedRowSparseMatrixd", parallelInverseProduct=True)
meshNode.addObject('MeshVTKLoader', name="loader", filename=parameterNode.modelNodeFileName)
meshNode.addObject('TetrahedronSetTopologyContainer', name="Container", src="@loader")
meshNode.addObject('TetrahedronSetTopologyModifier', name="Modifier")
meshNode.addObject('MechanicalObject', name="mstate", template="Vec3f")
meshNode.addObject('TetrahedronFEMForceField', name="FEM", youngModulus=1.5, poissonRatio=0.45, method="large")
meshNode.addObject('MeshMatrixMass', totalMass=1)

fixedROI = meshNode.addChild('FixedROI')


fixedROI.addObject('BoxROI', template="Vec3", box=parameterNode.getBoundaryROI(),
                   drawBoxes=True, position="@../mstate.rest_position", name="FixedROI",
                   computeTriangles=False,
                   computeTetrahedra=False, computeEdges=False)

fixedROI.addObject('FixedConstraint', indices="@FixedROI.indices")

collisionNode = meshNode.addChild('Collision')
collisionNode.addObject('TriangleSetTopologyContainer', name="Container")
collisionNode.addObject('TriangleSetTopologyModifier', name="Modifier")
collisionNode.addObject('Tetra2TriangleTopologicalMapping', input="@../Container", output="@Container")


modelNode = slicer.util.loadNodeFromFile(parameterNode.modelNodeFileName)

Sofa.Simulation.init(rootNode)
Sofa.Simulation.reset(rootNode)
mechanicalState = meshNode.getMechanicalState()

simulating = True
def updateSimulation():
    Sofa.Simulation.animate(rootNode, rootNode.dt.value)
    meshPointsArray = mechanicalState.position.array()
    modelPointsArray = slicer.util.arrayFromModelPoints(modelNode)
    modelPointsArrayNew = meshPointsArray * numpy.array([-1, -1, 1])
    modelPointsArray[:] = modelPointsArrayNew
    slicer.util.arrayFromModelPointsModified(modelNode)
    if simulating:
        qt.QTimer.singleShot(10, updateSimulation)

updateSimulation()
