from pathlib import Path

import Sofa
import Sofa.Core

class Listener(Sofa.Core.Controller):
    def __init__(self, contact_listener) -> None:
        super().__init__()
        self.listener = contact_listener

    def onAnimateBeginEvent(self, _) -> None:

        print(f"Found {self.listener.getNumberOfContacts()} contacts")

        contacts = self.listener.getContactElements()
        for contact in contacts:
            # (object, index, object, index)
            index_on_sphere = contact[1] if contact[0] == 0 else contact[3]
            index_on_edge = contact[3] if contact[0] == 0 else contact[1]

            print(f"Contact between sphere {index_on_sphere} and edge {index_on_edge}")

        print(f"Full contact information: {self.listener.getContactData()}")


def createScene(root):
    # Simulation Hyperparameters
    mesh_dir = Path("meshes")
    liver_mesh_file = mesh_dir / "liver2.msh"
    sphere_surface_file = mesh_dir / "sphere.obj"

    # Simulation Hyperparameters
    dt = 0.01
    collision_detection_method = "LocalMinDistance"
    alarm_distance = 10.0
    contact_distance = 0.8

    liver_mass = 3.0
    liver_youngs_modulus = 1.0 * 1000.0
    liver_poisson_ratio = 0.45

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
    root.gravity = [0.0, -9.81 * 10.0, 0.0]
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
    liver_node.addObject("MeshGmshLoader", filename=str(liver_mesh_file), scale=35.0)
    liver_node.addObject("TetrahedronSetTopologyContainer", src=liver_node.MeshGmshLoader.getLinkPath())
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
    sphere_node.addObject("MeshOBJLoader", filename=str(sphere_surface_file), scale=1.0)
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


def main():
    import Sofa
    import Sofa.Core
    import Sofa.Simulation

    # Create a root node
    root = Sofa.Core.Node("root")

    # Create the scene
    createScene(root)

    # Initialize the simulation
    Sofa.Simulation.init(root)

    try:
        while True:
            Sofa.Simulation.animate(root, root.dt.value)

    except KeyboardInterrupt:
        Sofa.Simulation.unload(root)


if __name__ == "__main__":
    main()
