import numpy as np
from typing import Tuple, Optional

def hollow_cylinder_hexahedral_topology_data(
    radius_inner: float = 10,
    radius_outer: float = 20,
    height: float = 5,
    num_radius: int = 10,
    num_phi: int = 20,
    num_z: int = 3,
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


def cube_hexahedral_mesh(
    bbox: Tuple[float, float, float, float, float, float],
    rows: int,
    cols: int,
    layers: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates a 3D mesh of hexahedral elements stacked in a cube pattern.
    
    Parameters:
    bbox (Tuple[float, float, float, float, float, float]): Bounding box of the mesh (min_x, min_y, min_z, max_x, max_y, max_z)
    rows (int): Number of rows (elements along x-axis)
    cols (int): Number of columns (elements along y-axis)
    layers (int): Number of layers (elements along z-axis)
    
    Returns:
    Tuple[np.ndarray, np.ndarray]: (points, hexahedra)
        points (np.ndarray): Array of [x, y, z] coordinates of mesh points
        hexahedra (np.ndarray): Array of 8-index tuples representing the hexahedral elements
    """
    min_x, min_y, min_z, max_x, max_y, max_z = bbox

    # Generate points
    x = np.linspace(min_x, max_x, rows + 1)
    y = np.linspace(min_y, max_y, cols + 1)
    z = np.linspace(min_z, max_z, layers + 1)

    points = np.array([[xi, yi, zi] for zi in z for yi in y for xi in x])

    # Generate hexahedral elements
    hexahedra = []
    for k in range(layers):
        for j in range(cols):
            for i in range(rows):
                hex_indices = (
                    i + j * (rows + 1) + k * (rows + 1) * (cols + 1),
                    i + 1 + j * (rows + 1) + k * (rows + 1) * (cols + 1),
                    i + 1 + (j + 1) * (rows + 1) + k * (rows + 1) * (cols + 1),
                    i + (j + 1) * (rows + 1) + k * (rows + 1) * (cols + 1),
                    i + j * (rows + 1) + (k + 1) * (rows + 1) * (cols + 1),
                    i + 1 + j * (rows + 1) + (k + 1) * (rows + 1) * (cols + 1),
                    i + 1 + (j + 1) * (rows + 1) + (k + 1) * (rows + 1) * (cols + 1),
                    i + (j + 1) * (rows + 1) + (k + 1) * (rows + 1) * (cols + 1)
                )
                hexahedra.append(hex_indices)
    return points, np.array(hexahedra)
