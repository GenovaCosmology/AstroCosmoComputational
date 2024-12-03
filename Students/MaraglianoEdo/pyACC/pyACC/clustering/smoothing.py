import numpy as np

def cic_interpolator(pos, BoxSize, grid_shape, mass=None):
    """
    CIC (Cloud-In-Cell) interpolator for computing density on a grid.

    This function takes in the positions and masses of particles, as well as the shape of the grid,
    and computes the density on the grid using the CIC interpolation method.

    Parameters:
        pos (array-like): Array of particle positions of shape (N, 3).
        BoxSize (float): Size of the simulation box.
        grid_shape (int): Number of grid cells along one dimension.
        mass (array-like): Array of particle masses of shape (N,). Default is None.
    Returns:
        array-like: The density on the grid.

    """

    # Initialize the density grid
    density = np.zeros((grid_shape, grid_shape, grid_shape))  

    # Find the number of particles and dimensions
    particles = pos.shape[0]  # number of particles
    coord = pos.shape[1]  # number of dimensions (3D)
    
    # Calcola il reciproco della dimensione della cella
    inv_cell_size = grid_shape / BoxSize
    
    # Se non sono forniti i pesi, assume che siano tutti uguali a 1
    if mass is None:
        mass = np.ones(particles)
    
    # Loop over all particles
    for i in range(particles):
        # Initialise arrays to store the weights and indices
        u = np.ones(3)
        d = np.ones(3)
        index_u = np.zeros(3, dtype=int)
        index_d = np.zeros(3, dtype=int)

        # For each dimension, calculate the fractional and complementary distance
        for axis in range(coord):
            dist = pos[i, axis] * inv_cell_size                     # Normalized distance
            u[axis] = dist - int(dist)                              # Fractional part
            d[axis] = 1.0 - u[axis]                                 # Complementary part
            index_d[axis] = int(dist) % grid_shape                  # index of the lower cell (cyclic)
            index_u[axis] = (index_d[axis] + 1) % grid_shape        # index of the upper cell (cyclic)

        # Update the density grid
        weight = mass[i]
        density[index_d[0], index_d[1], index_d[2]] += d[0] * d[1] * d[2] * weight
        density[index_d[0], index_d[1], index_u[2]] += d[0] * d[1] * u[2] * weight
        density[index_d[0], index_u[1], index_d[2]] += d[0] * u[1] * d[2] * weight
        density[index_d[0], index_u[1], index_u[2]] += d[0] * u[1] * u[2] * weight
        density[index_u[0], index_d[1], index_d[2]] += u[0] * d[1] * d[2] * weight
        density[index_u[0], index_d[1], index_u[2]] += u[0] * d[1] * u[2] * weight
        density[index_u[0], index_u[1], index_d[2]] += u[0] * u[1] * d[2] * weight
        density[index_u[0], index_u[1], index_u[2]] += u[0] * u[1] * u[2] * weight

    return density


def nearest_cell_interpolator(particles, masses, grid_shape):
    """
    Inputs:
    particles: 2D array of positions (len(masses), 3)
    masses: 1D array of masses
    grid_shape: tuple of 3 integers defining the shape of the grid (nx, ny, nz)

    Returns:
    rho: 3D array of density values with shape grid_shape
    """
    nx, ny, nz = grid_shape
    rho = np.zeros(grid_shape, dtype=np.float64)

    for p, m in zip(particles, masses):
        i, j, k = np.rint(p).astype(int)

        # Ensure indices are within bounds with periodic boundary conditions
        i %= nx
        j %= ny
        k %= nz

        # Add contribution to the nearest grid point
        rho[i, j, k] += m

    return rho