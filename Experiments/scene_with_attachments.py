import numpy as np
from typing import Tuple, Optional


def in_box(point, box):
    return all([box[axis][0] <= point[i] <= box[axis][1] for i, axis in enumerate(["X", "Y", "Z"])])


def hollow_cylinder_hexahedral_topology_data(
    radius_inner: float,
    radius_outer: float,
    height: float,
    num_radius: int,
    num_phi: int,
    num_z: int,
    translation: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Creates a hexahedral topology for a hollow cylinder with inner and outer radii.

    Args:
        radius_inner (float): The inner radius of the hollow cylinder.
        radius_outer (float): The outer radius of the hollow cylinder.
        height (float): The height of the hollow cylinder.
        num_radius (int): Number of points along the radius -> n-1 hexahedra along the radius.
        num_phi (int): Number of points along angle -> n hexahedra around the angle.
        num_z (int): Number of points along the height -> n-1 hexahedra.
        translation (Optional[np.ndarray]): Translation of the hollow cylinder.

    Returns:
        points (List): A list of [x, y, z] coordinates of points.
        hexahedra (List): The list of hexahedra described with 8 indices each corresponding to the points.
    """

    radii = np.linspace(radius_inner, radius_outer, num_radius)
    phis = np.linspace(0, 2 * np.pi, num_phi + 1)[:-1]
    zs = np.linspace(0, height, num_z)

    index_array = np.empty((num_radius, num_phi, num_z), dtype=np.uint64)

    points = []
    i = 0
    for index_z, z in enumerate(zs):
        for index_radius, radius in enumerate(radii):
            for index_phi, phi in enumerate(phis):
                points.append(np.asarray([radius * np.cos(phi), radius * np.sin(phi), z]))
                index_array[index_radius, index_phi, index_z] = i
                i += 1

    points = np.asarray(points)

    hexahedra = []
    for z in range(num_z - 1):
        for r in range(num_radius - 1):
            for phi in range(num_phi):
                phi_upper = (phi + 1) % num_phi
                hexahedron = (
                    index_array[r, phi, z],
                    index_array[r, phi_upper, z],
                    index_array[r, phi_upper, z + 1],
                    index_array[r, phi, z + 1],
                    index_array[r + 1, phi, z],
                    index_array[r + 1, phi_upper, z],
                    index_array[r + 1, phi_upper, z + 1],
                    index_array[r + 1, phi, z + 1],
                )
                hexahedra.append(hexahedron)

    hexahedra = np.asarray(hexahedra)

    if translation is not None:
        points += translation

    return points, hexahedra

import Sofa
import Sofa.Core
import Sofa.Simulation

PYVISTA = True
if PYVISTA:
    import pyvista as pv

    plotter = pv.Plotter()

# Create a root node
root = Sofa.Core.Node("root")


# Simulation Hyperparameters
dt = 0.01
collision_detection_method = "LocalMinDistance"  # Which algorithm to use for collision detection
alarm_distance = 10.0  # This will tell the collision detection algorithm to start checking for actual collisions
contact_distance = 0.8  # This is the distance at which the collision detection algorithm will consider two objects to be in contact

torus_mass = 100.0  # g
torus_youngs_modulus = 2.0 * 1000.0  # 15 kPa -> 15 * 1000 in mm/s/g
torus_poisson_ratio = 0.43

# Plugins required for the components. Usually I just add the components, run Sofa once to get the errors, and use that to add the plugins.
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
    "Sofa.Component.Topology.Mapping",  # Needed to use components [Hexa2TetraTopologicalMapping,Tetra2TriangleTopologicalMapping]
    "Sofa.Component.Visual",  # Needed to use components [VisualStyle]
    "Sofa.Component.SolidMechanics.Spring",  # Needed to use components [RestShapeSpringsForceField]
    "Sofa.Component.Constraint.Projective",  # Needed to use components [FixedProjectiveConstraint]
]
for plugin in plugin_list:
    root.addObject("RequiredPlugin", name=plugin)

# The simulation scene
# root.gravity = [0.0, 0.0, -9.81 * 100.0]
root.gravity = [0.0, 0.0, -9.81 * 10.0]
root.dt = dt

root.addObject("FreeMotionAnimationLoop")  # One of several simulation loops.
root.addObject("VisualStyle", displayFlags=["showForceFields", "showBehaviorModels", "showCollisionModels"])

root.addObject("CollisionPipeline")  # This object will be used to manage the collision detection
root.addObject("ParallelBruteForceBroadPhase")  # The broad phase checks for overlaps in bounding boxes
root.addObject("ParallelBVHNarrowPhase")  # And the narrow phase checks for collisions
root.addObject(collision_detection_method, alarmDistance=alarm_distance, contactDistance=contact_distance)  # Using this algorithm for collision detection

root.addObject("CollisionResponse", response="FrictionContactConstraint")  # This object will be used to manage the collision response
root.addObject("GenericConstraintSolver")  # And this object will be used to solve the constraints resulting from collisions etc.

scene_node = root.addChild("scene")  # The scene node will contain all the objects in the scene

#############
### Torus ###
#############
# We can also manually define the topology
points, hexahedra = hollow_cylinder_hexahedral_topology_data(
    radius_inner=30.0,
    radius_outer=40.0,
    height=20.0,
    num_radius=2,
    num_phi=30,
    num_z=4,
    translation=[0.0, 0.0, 200.0],
)

hexahedra_node = scene_node.addChild("hexa")
hexahedra_node.addObject("HexahedronSetTopologyContainer", position=points, hexahedra=hexahedra)
hexahedra_node.addObject("HexahedronSetTopologyModifier")
hexahedra_node.addObject("EulerImplicitSolver")
hexahedra_node.addObject("SparseLDLSolver", template="CompressedRowSparseMatrixMat3x3d")
hexahedra_node.addObject("MechanicalObject")
hexahedra_node.addObject("UniformMass", totalMass=torus_mass)
hexahedra_node.addObject("LinearSolverConstraintCorrection")

# Subdivide them into Tetrahedra for FEM
tetrahedra_node = hexahedra_node.addChild("tetra")
tetrahedra_node.addObject("TetrahedronSetTopologyContainer")
tetrahedra_node.addObject("TetrahedronSetTopologyModifier")
tetrahedra_node.addObject("Hexa2TetraTopologicalMapping")
tetrahedra_node.addObject("TetrahedralCorotationalFEMForceField", youngModulus=torus_youngs_modulus, poissonRatio=torus_poisson_ratio)

# Determine the surface triangles for collision
triangle_node = tetrahedra_node.addChild("triangle")
triangle_node.addObject("TriangleSetTopologyContainer")
triangle_node.addObject("TriangleSetTopologyModifier")
triangle_node.addObject("Tetra2TriangleTopologicalMapping")

box = {
    "X": [0.0, 100.0],
    "Y": [0.0, 100.0],
    "Z": [210.0, 220.0],
}

points_subset, points_indices = zip(*[(point, index) for index, point in enumerate(points) if in_box(point, box)])
reference_node = scene_node.addChild("reference")
mechanicalObject = reference_node.addObject("MechanicalObject", position=points_subset, showObject=True, showObjectScale=5)

forceField = hexahedra_node.addObject(
    "RestShapeSpringsForceField",
    external_rest_shape=reference_node.MechanicalObject.getLinkPath(),
    stiffness=1e9,
    angularStiffness=1e9,
    points=points_indices,
)

fixed_box = {
    "X": [-100.0, 0.0],
    "Y": [-100.0, 0.0],
    "Z": [200.0, 210.0],
}
fixed_indices = [index for index, point in enumerate(points) if in_box(point, fixed_box)]
hexahedra_node.addObject(
    "FixedProjectiveConstraint",
    indices=fixed_indices,
    showObject=True,
)

#############
### Floor ###
#############
plane_kwargs = {
    "normal": [0, 0, 1],
    "d": 0.0,
    "stiffness": 3000.0,
    "damping": 1.0,
}
hexahedra_node.addObject("PlaneForceField", **plane_kwargs)



# Create the scene
points_indices, fixed_indices = list(points_indices), fixed_indices

# Initialize the simulation
Sofa.Simulation.init(root)

torus_positions = root.scene.hexa.MechanicalObject.position.array()
torus_triangles = root.scene.hexa.tetra.triangle.TriangleSetTopologyContainer.triangles.array()
reference_positions = root.scene.reference.MechanicalObject.position.array()

if PYVISTA:
    faces = np.zeros((torus_triangles.shape[0], 4), dtype=np.uint64)
    faces[:, 1:] = torus_triangles
    faces[:, 0] = 3
    mesh = pv.PolyData(torus_positions, faces)
    plotter.add_mesh(mesh, color="blue", opacity=0.5, show_edges=True, lighting=True)

    fixed_points = torus_positions[fixed_indices]
    plotter.add_points(fixed_points, color="red", point_size=10)

    moving_points = torus_positions[points_indices]
    plotter.add_points(moving_points, color="green", point_size=10)

try:
    count = 0
    while count < 350:
        count += 1

        if count % 5 == 0:
            print ("changing")
            points_indices = [(index+1) % len(points) for index in points_indices]
            forceField.points = points_indices

        Sofa.Simulation.animate(root, root.dt.value)
        slicer.app.processEvents()


        if PYVISTA:
            plotter.meshes[0].points = torus_positions
            plotter.meshes[1].points = torus_positions[fixed_indices]
            plotter.meshes[2].points = torus_positions[points_indices]
            plotter.show(interactive_update=True)
            plotter.update()

        if count == 50:
            hexahedra_node.removeObject(mechanicalObject)
            hexahedra_node.removeObject(mechanicalObject)
            scene_node.removeChild(reference_node)
            print("removing")
            forceField.stiffness = [0]
            forceField.angularStiffness = [0]
        else:
            if count < 50:
                with root.scene.reference.MechanicalObject.position.writeable() as reference_positions:
                    reference_positions[:] += 0.2

except KeyboardInterrupt:
    Sofa.Simulation.unload(root)

print("done")

