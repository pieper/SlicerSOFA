import numpy

import Sofa
import Sofa.Core
import Sofa.Simulation

class Listener(Sofa.Core.Controller):
    def __init__(self, contact_listener) -> None:
        super().__init__()
        self.listener = contact_listener

    def onAnimateBeginEvent(self, _) -> None:

        return
        contacts = self.listener.getContactElements()
        if len(contacts) > 0:
            for contact in contacts:
                # (object, index, object, index)
                index_on_sphere = contact[1] if contact[0] == 0 else contact[3]
                index_on_edge = contact[3] if contact[0] == 0 else contact[1]

                print(f"Contact between sphere {index_on_sphere} and edge {index_on_edge}")

            print(f"Full contact information: {self.listener.getContactData()}")


# Simulation Hyperparameters
liver_mesh_file = "originalMesh.vtk"
sphere_surface_file = "biggerCavity.obj"

slicer.mrmlScene.RemoveNode(originalMeshNode)
slicer.mrmlScene.RemoveNode(sphereNode)

originalMeshNode = slicer.mrmlScene.GetFirstNodeByName("originalMesh")
if not originalMeshNode:
    originalMeshNode = slicer.util.loadModel(liver_mesh_file)
    sphereNode = slicer.util.loadModel(sphere_surface_file)
    sphereNode.GetDisplayNode().SetRepresentation(slicer.vtkMRMLDisplayNode.WireframeRepresentation)

extractedPointsArray = numpy.array(originalMeshNode.GetUnstructuredGrid().GetPoints().GetData())
cellsArray = numpy.array(originalMeshNode.GetUnstructuredGrid().GetCells().GetData())
extractedTetrahedraArray = cellsArray.reshape(-1,5)[:,1:5]

#liver_mesh_file = "collision/meshes/liver2.msh"
#sphere_surface_file = "collision/meshes/sphere.obj"

# Simulation Hyperparameters
dt = 0.01
collision_detection_method = "LocalMinDistance"
alarm_distance = 10.0
contact_distance = 0.8

liver_mass = 30.0
#liver_youngs_modulus = 1.0 * 1000.0
liver_youngs_modulus = 1.0 * 1000.0 * 0.001
liver_poisson_ratio = 0.45

# Create a root node
root = Sofa.Core.Node("root")

plugin_list = [
    "MultiThreading",  # Needed to use components [ParallelBVHNarrowPhase,ParallelBruteForceBroadPhase]
    "Sofa.Component.AnimationLoop",  # Needed to use components [FreeMotionAnimationLoop]
    "Sofa.Component.Collision.Detection.Algorithm",  # Needed to use components [CollisionPipeline]
    "Sofa.Component.Collision.Detection.Intersection",  # Needed to use components [MinProximityIntersection]
    "Sofa.Component.Collision.Geometry",  # Needed to use components [TriangleCollisionModel]
    "Sofa.Component.Collision.Response.Contact",  # Needed to use components [CollisionResponse]
    "Sofa.Component.Constraint.Lagrangian.Correction",  # Needed to use components [LinearSolverConstraintCorrection]
    "Sofa.Component.Constraint.Lagrangian.Solver",  # Needed to use components [GenericConstraintSolver]
    "Sofa.Component.IO.Mesh",  # Needed to use components [MeshGmshLoader]
    "Sofa.Component.LinearSolver.Direct",  # Needed to use components [SparseLDLSolver]
    "Sofa.Component.Mass",  # Needed to use components [UniformMass]
    "Sofa.Component.MechanicalLoad",  # Needed to use components [PlaneForceField]
    "Sofa.Component.ODESolver.Backward",  # Needed to use components [EulerImplicitSolver]
    "Sofa.Component.SolidMechanics.FEM.Elastic",  # Needed to use components [TetrahedralCorotationalFEMForceField]
    "Sofa.Component.StateContainer",  # Needed to use components [MechanicalObject]
    "Sofa.Component.Topology.Container.Dynamic",  # Needed to use components [HexahedronSetTopologyContainer,HexahedronSetTopologyModifier,TetrahedronSetTopology
    "Sofa.Component.Topology.Mapping",  # Needed to use components [Tetra2TriangleTopologicalMapping]
    "Sofa.Component.Visual",  # Needed to use components [VisualStyle]
    "Sofa.Component.Constraint.Projective",  # Needed to use components [FixedProjectiveConstraint]
]
for plugin in plugin_list:
    root.addObject("RequiredPlugin", name=plugin)

# The simulation scene
root.gravity = [-9.81 * 10.0, 0.0, 0.0]
root.dt = dt

root.addObject("FreeMotionAnimationLoop")
root.addObject("VisualStyle", displayFlags=["showForceFields", "showBehaviorModels", "showCollisionModels", "showWireframe"])

root.addObject("CollisionPipeline")
root.addObject("ParallelBruteForceBroadPhase")
root.addObject("ParallelBVHNarrowPhase")
root.addObject(collision_detection_method, alarmDistance=alarm_distance, contactDistance=contact_distance)

# NOTE: There are a few other collision response methods available in SOFA, but in my experience, they tend to be a bit... buggy.
# The responseParams parameter is the friction coefficient. 0.0 is no friction, 1.0 is full friction.
root.addObject("CollisionResponse", response="FrictionContactConstraint", responseParams=0.001)
root.addObject("GenericConstraintSolver")

scene_node = root.addChild("scene")

################
#### Liver ####
################
liver_node = scene_node.addChild("liver")
#liver_node.addObject("MeshGmshLoader", filename=liver_mesh_file, scale=35.0)
#liver_node.addObject("TetrahedronSetTopologyContainer", src=liver_node.MeshGmshLoader.getLinkPath())
#liver_node.addObject("MeshVTKLoader", name="loader", filename=liver_mesh_file, scale=1.0)
#liver_node.addObject("TetrahedronSetTopologyContainer", src="@loader")


liver_node.addObject('TetrahedronSetTopologyContainer', name="Container",
                       position=extractedPointsArray,
                       tetrahedra=extractedTetrahedraArray)

liver_node.addObject("TetrahedronSetTopologyModifier")
liver_node.addObject("EulerImplicitSolver")
liver_node.addObject("SparseLDLSolver", template="CompressedRowSparseMatrixMat3x3d")
liver_node.addObject("MechanicalObject")
liver_node.addObject("TetrahedralCorotationalFEMForceField", youngModulus=liver_youngs_modulus, poissonRatio=liver_poisson_ratio)
liver_node.addObject("UniformMass", totalMass=liver_mass)
liver_node.addObject("LinearSolverConstraintCorrection")

liver_collision_node = liver_node.addChild("collision")
liver_collision_node.addObject("TriangleSetTopologyContainer")
liver_collision_node.addObject("TriangleSetTopologyModifier")
liver_collision_node.addObject("Tetra2TriangleTopologicalMapping")
liver_collision_node.addObject("PointCollisionModel")
liver_collision_node.addObject("LineCollisionModel")
liver_collision_node.addObject("TriangleCollisionModel")

################
#### Sphere ####
################
sphere_node = scene_node.addChild("sphere")
sphere_node.addObject("MeshOBJLoader", filename=sphere_surface_file, scale=1.0)
sphere_node.addObject("TriangleSetTopologyContainer", src=sphere_node.MeshOBJLoader.getLinkPath())
sphere_node.addObject("TriangleSetTopologyModifier")
sphere_node.addObject("MechanicalObject")
# NOTE: The important thing is to set bothSide=True for the collision models, so that both sides of the triangle are considered for collision.
sphere_node.addObject("TriangleCollisionModel", bothSide=True)
sphere_node.addObject("PointCollisionModel")
sphere_node.addObject("LineCollisionModel")
sphere_node.addObject("FixedProjectiveConstraint")

##################
# Contact Listener
##################
contact_listener = scene_node.addObject(
    "ContactListener",
    collisionModel1=sphere_node.TriangleCollisionModel.getLinkPath(),
    collisionModel2=liver_collision_node.PointCollisionModel.getLinkPath(),
)
scene_node.addObject(Listener(contact_listener))




# Initialize the simulation
Sofa.Simulation.init(root)

mechanicalState = liver_node.getMechanicalState()

with sphere_node.MechanicalObject.position.writeable() as sphereArray:
    sphereArray *= [-1,-1,1]

print("Ready to start...")
iteration = 0
iterations = 30
simulating = True
def updateSimulation():
    global iteration, iterations, simulating

    for step in range(10):
        Sofa.Simulation.animate(root, root.dt.value)

    # update model from mechanical state
    meshPointsArray = mechanicalState.position.array()
    modelPointsArray = slicer.util.arrayFromModelPoints(originalMeshNode)
    modelPointsArray[:] = meshPointsArray
    slicer.util.arrayFromModelPointsModified(originalMeshNode)

    """
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
    """

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

