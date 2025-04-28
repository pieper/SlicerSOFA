import numpy as np
from pathlib import Path
from typing import Tuple, Optional

try:
  import slicersofa
except ModuleNotFoundError:
  import sys
  sys.path.append(".")
  import slicersofa

import importlib
importlib.reload(slicersofa)
importlib.reload(slicersofa.util)
importlib.reload(slicersofa.meshes)

simulating = True
probeTarget = [150,50,0]


import Sofa
import Sofa.Core
import Sofa.Simulation

PYVISTA = True
if PYVISTA:
    try:
        import pyvista as pv
        plotter = pv.Plotter()
    except ModuleNotFoundError:
        PYVISTA = False

# Create a root node
root = Sofa.Core.Node("root")

# Create the scene

slicer.mrmlScene.Clear()
slicer.app.processEvents()
slicer.app.layoutManager().setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutOneUp3DView)

mesh_dir = Path("meshes")
liver_mesh_file = mesh_dir / "liver2.msh"
#instrument_surface_mesh_file = mesh_dir / "dental_instrument.obj"
instrument_surface_mesh_file="mesh/cylinder_35.56x3.75.obj" #If SOFA_ROOT is correctly set, the loder will try to look for files under ${SHARE_DIR} that is defined in ${SOFA_ROOT}/etc/sofa.ini

# Simulation Hyperparameters
dt = 0.01

alarm_distance = 5  # This will tell the collision detection algorithm to start checking for actual collisions
contact_distance = 3  # This is the distance at which the collision detection algorithm will consider two objects to be in contact

liver_mass = 300.0  # g
#liver_youngs_modulus = 15.0 * 1000.0  # 15 kPa -> 15 * 1000 in mm/s/g
liver_youngs_modulus = 1.50 * 1000.0  # 15 kPa -> 15 * 1000 in mm/s/g
liver_poisson_ratio = 0.45

#torus_mass = 1000.0  # g
torus_mass = 2000.0  # g
torus_youngs_modulus = 30.0 * 1000.0  # 15 kPa -> 15 * 1000 in mm/s/g
torus_poisson_ratio = 0.48

instrument_mass = 100.0  # g
#instrument_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]  # x, y, z, qx, qy, qz, qw
instrument_pose = probeTarget + [0.0, 0.0, 0.0, 1.0]  # x, y, z, qx, qy, qz, qw

slicersofa.util.initPlugins(root)

# The simulation scene
# root.gravity = [0.0, 0.0, -9.81 * 100.0]
root.gravity = [0.0, 0.0, -9.81 * 10.0]
root.dt = dt

root.addObject("FreeMotionAnimationLoop")  # One of several simulation loops.
root.addObject("VisualStyle", displayFlags=["showForceFields", "showBehaviorModels", "showCollisionModels"])

root.addObject("CollisionPipeline")  # This object will be used to manage the collision detection
root.addObject("ParallelBruteForceBroadPhase")  # The broad phase checks for overlaps in bounding boxes
root.addObject("ParallelBVHNarrowPhase")  # And the narrow phase checks for collisions
#root.addObject(collision_detection_method, alarmDistance=alarm_distance, contactDistance=contact_distance)  # Using this algorithm for collision detection
root.addObject("NewProximityIntersection",useLineLine=True,alarmDistance=alarm_distance,contactDistance=contact_distance)  # Using this algorithm for collision detection

root.addObject("CollisionResponse", response="FrictionContactConstraint")  # This object will be used to manage the collision response
root.addObject("GenericConstraintSolver",tolerance=1e-5)  # And this object will be used to solve the constraints resulting from collisions etc.

scene_node = root.addChild("scene")  # The scene node will contain all the objects in the scene

################
#### Liver ####
################
liver_node = scene_node.addChild("liver")
# Load the liver mesh, scale it from m to mm, and move it up by 100 mm
liver_node.addObject("MeshGmshLoader", filename=str(liver_mesh_file), scale=100.0, translation=[0.0, 0.0, 100.0])
liver_node.addObject("TetrahedronSetTopologyContainer", src=liver_node.MeshGmshLoader.getLinkPath())  # This is the container for the tetrahedra
liver_node.addObject("TetrahedronSetTopologyModifier")  # And this loads some algorithms for modifying the topology. Not always necessary, but good to have.
liver_node.addObject("EulerImplicitSolver")  # This is the ODE solver (for the time)
liver_node.addObject("SparseLDLSolver", template="CompressedRowSparseMatrixMat3x3d")  # And the linear solver (for the space).
# CompressedRowSparseMatrixMat3x3d tells it that the vertices are only checked for position interaction, not orientation
liver_node.addObject("MechanicalObject")  # This components holds the positions, velocities, and forces of the vertices
liver_node.addObject("TetrahedralCorotationalFEMForceField", youngModulus=liver_youngs_modulus, poissonRatio=liver_poisson_ratio)  # This is the FEM algorithm that will calculate the forces
liver_node.addObject("UniformMass", totalMass=liver_mass)  # This will give all the vertices the same mass, summing up to the total mass
liver_node.addObject("LinearSolverConstraintCorrection")  # This will compute the corrections to the forces to satisfy the constraints

liver_collision_node = liver_node.addChild("collision")
liver_collision_node.addObject("TriangleSetTopologyContainer")  # Another topology container, this time for the triangles for collision
liver_collision_node.addObject("TriangleSetTopologyModifier")  # And the modifier for the triangles
liver_collision_node.addObject("Tetra2TriangleTopologicalMapping")  # This will map the tetrahedra from the parent node to the triangles in this node
liver_collision_node.addObject("MechanicalObject")  # This components holds the positions, velocities, and forces of the vertices
liver_collision_node.addObject("LineCollisionModel")  # This will create the collision model based on the points stored in the TriangleSetTopologyContainer
liver_collision_node.addObject("IdentityMapping")  # This will create the mapping linking the applied forces to the object


#############
### Torus ###
#############
# We can also manually define the topology
shape = "torus"
if shape == "torus":
  points, hexahedra = slicersofa.meshes.hollow_cylinder_hexahedral_topology_data(
      radius_inner=30.0,
      radius_outer=40.0,
      height=20.0,
      num_radius=2,
      num_phi=30,
      num_z=4,
      translation=[0.0, 0.0, 200.0],
  )
elif shape == "box":
  bbox = (200, 200, 200, 250, 250, 250)
  points, hexahedra = slicersofa.meshes.cube_hexahedral_mesh(bbox, 5, 5, 5)



hexahedra_node = scene_node.addChild("hexa")
hexahedra_node.addObject("HexahedronSetTopologyContainer", name="container", position=points, hexahedra=hexahedra)
hexahedra_node.addObject("HexahedronSetTopologyModifier")
hexahedra_node.addObject("MechanicalObject")


# Subdivide them into Tetrahedra for FEM
tetrahedra_node = scene_node.addChild("tetra")
tetrahedra_node.addObject("EulerImplicitSolver")
tetrahedra_node.addObject("SparseLDLSolver", template="CompressedRowSparseMatrixMat3x3d")

tetrahedra_node.addObject("TetrahedronSetTopologyContainer",name="container")
tetrahedra_node.addObject("TetrahedronSetTopologyModifier")
tetrahedra_node.addObject("Hexa2TetraTopologicalMapping",input="@../hexa/container", output="@container")

tetrahedra_node.addObject("TetrahedralCorotationalFEMForceField", youngModulus=torus_youngs_modulus, poissonRatio=torus_poisson_ratio)
tetrahedra_node.addObject("MechanicalObject")
tetrahedra_node.addObject("UniformMass", totalMass=torus_mass)
tetrahedra_node.addObject("LinearSolverConstraintCorrection")

# Determine the surface triangles for collision
triangle_node = tetrahedra_node.addChild("triangle")
triangle_node.addObject("TriangleSetTopologyContainer")
triangle_node.addObject("TriangleSetTopologyModifier")
triangle_node.addObject("Tetra2TriangleTopologicalMapping")
triangle_node.addObject("MechanicalObject")
triangle_node.addObject("LineCollisionModel")
triangle_node.addObject("IdentityMapping")  # This will create the mapping linking the applied forces to the object

##################
### Instrument ###
##################
# Rigid objects are handled by one vertex that is store in a Rigid3d object, so 7 values (x, y, z, qx, qy, qz, qw) for position and orientation
# If we want to control the position of the object, we can either set velocities and forces in the MechanicalObject, or we use a motion target
# and add some springs, to "fake" a force control. If we set the position directly, collision checking would no longer work, because the object
# "teleports" through the scene.
instrument_motion_target_node = scene_node.addChild("motion_target")
instrument_motion_target_node.addObject("MechanicalObject", template="Rigid3d", position=instrument_pose)



instrument_node = scene_node.addChild("instrument")
instrument_node.addObject("EulerImplicitSolver")
instrument_node.addObject("EigenSparseLU", template="CompressedRowSparseMatrixMat3x3d")


instrument_node.addObject("MechanicalObject", template="Rigid3d", position=instrument_pose)
instrument_node.addObject("UniformMass", totalMass=instrument_mass)
instrument_node.addObject("RestShapeSpringsForceField", external_rest_shape=instrument_motion_target_node.getLinkPath(), stiffness=1e10, angularStiffness=1e10)
instrument_node.addObject("UncoupledConstraintCorrection")  # <- different to the deformable objects, where the points are not uncoupled

instrument_collision_node = instrument_node.addChild("collision")
#instrument_collision_node.addObject("MeshOBJLoader", filename=str(instrument_surface_mesh_file), scale=30.0, translation=[0.0, 0.0, 200.0])
instrument_collision_node.addObject("MeshOBJLoader", filename=str(instrument_surface_mesh_file),scale3d=[0.5, 3, 0.5],rotation=[90,0,0])
instrument_collision_node.addObject("QuadSetTopologyContainer", src=instrument_collision_node.MeshOBJLoader.getLinkPath())
# now we actually have to create a new MechanicalObject, because we load more points, and not just reference them through topological mappings like in the deformable objects
instrument_collision_node.addObject("MechanicalObject", template="Vec3d")
# there are quite a few points in this model. Simulation might slow down. You can just add a different obj file.
instrument_collision_node.addObject("LineCollisionModel")
instrument_collision_node.addObject("RigidMapping")

#############
### Floor ###
#############

boxStiffness = 10000.0

# Just to stop the objects from falling through the ground and off the sides
plane_kwargs = {
    "normal": [0, 0, 1],
    "d": -10.0,
    "stiffness": boxStiffness,
    "damping": 3.0,
}
liver_node.addObject("PlaneForceField", **plane_kwargs)
hexahedra_node.addObject("PlaneForceField", **plane_kwargs)

walls = [ [[0, 1, 0], -200],[[0, -1, 0], -200], [[1, 0, 0], -200],[[-1, 0, 0], -200] ]
for normal,d in walls:
  plane_kwargs = {
      "normal": normal,
      "d": d,
      "stiffness": boxStiffness,
      "damping": 3.0,
  }
  liver_node.addObject("PlaneForceField", **plane_kwargs)
  hexahedra_node.addObject("PlaneForceField", **plane_kwargs)


# Initialize the simulation
Sofa.Simulation.init(root)

# You can get a reference to the memory of the vertices with a call to the array() method of a mechanical object's position attribute
liver_positions = root.scene.liver.MechanicalObject.position.array()
torus_positions = root.scene.hexa.MechanicalObject.position.array()
instrument_positions = root.scene.instrument.collision.MechanicalObject.position.array()

# Similarly you can access the topology of the object
liver_triangles = root.scene.liver.collision.TriangleSetTopologyContainer.triangles.array()
# however, the values of TriangleSetTopologyContainer.position will not change, because the simulated positions are stored in the MechanicalObject
torus_triangles = root.scene.hexa.tetra.triangle.TriangleSetTopologyContainer.triangles.array()
instrument_triangles = root.scene.instrument.collision.TriangleSetTopologyContainer.triangles.array()

if PYVISTA:
    for position_array, triangle_array, color, name in zip([liver_positions, torus_positions, instrument_positions], [liver_triangles, torus_triangles, instrument_triangles], ["red", "blue", "green"], ['liver', 'torus', 'instrument']):
        faces = np.zeros((triangle_array.shape[0], 4), dtype=np.uint64)
        faces[:, 1:] = triangle_array
        faces[:, 0] = 3
        mesh = pv.PolyData(position_array, faces)
        plotter.add_mesh(mesh, color=color, opacity=0.5, show_edges=True, lighting=True)

        modelNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode")
        modelNode.SetName(name)
        modelNode.CreateDefaultDisplayNodes()
        modelNode.SetAndObservePolyData(mesh)
    instrumentPoints = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode")
    instrumentPoints.AddControlPoint(probeTarget)

def timeStep():
    Sofa.Simulation.animate(root, root.dt.value)

    # The array is updated in place, so we can access the new values
    #print(liver_positions.mean(axis=0))

    # We can change the position by getting a writeable reference to the array
    # notice, that we change the position of the motion target, not the actual object
    with root.scene.motion_target.MechanicalObject.position.writeable() as target_positions:
        # first index is the vertex, second index is the x, y, z coordinate
        #target_positions[0, 0] += 0.5
        target_positions[0, 0:3] = slicer.util.arrayFromMarkupsControlPoints(instrumentPoints)[0]

    if PYVISTA:
        # the valuess in the numpy array are update, but pyvista does copy, not reference the data
        for i, position_array in enumerate([liver_positions, torus_positions, instrument_positions]):
            plotter.meshes[i].points = position_array
        #plotter.show(interactive_update=True)
        #plotter.update()

    if simulating:
        qt.QTimer.singleShot(50, timeStep)

timeStep()
