
print("loading")

def hoot():
    print("ba")

def initPlugins(root):

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
    ]
    for plugin in plugin_list:
        root.addObject("RequiredPlugin", name=plugin)
