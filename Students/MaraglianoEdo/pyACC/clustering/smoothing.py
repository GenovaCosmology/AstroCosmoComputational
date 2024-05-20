import numpy as np

def cic_interpolator(particles, masses, grid_shape):
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
        
        # discrete grid
        i, j, k = np.floor(p).astype(int)

        # displacement from mass position to grid point
        dx, dy, dz = p - [i, j, k]

        # weights
        tx, ty, tz = 1 - dx, 1 - dy, 1 - dz

        # Ensure indices are within bounds with periodic boundary conditions
        i0, j0, k0 = i % nx, j % ny, k % nz
        i1, j1, k1 = (i + 1) % nx, (j + 1) % ny, (k + 1) % nz

        # Add contributions to the neighboring grid points
        rho[i0, j0, k0] += m * tx * ty * tz
        rho[i1, j0, k0] += m * dx * ty * tz
        rho[i0, j1, k0] += m * tx * dy * tz
        rho[i1, j1, k0] += m * dx * dy * tz
        rho[i0, j0, k1] += m * tx * ty * dz
        rho[i1, j0, k1] += m * dx * ty * dz
        rho[i0, j1, k1] += m * tx * dy * dz
        rho[i1, j1, k1] += m * dx * dy * dz

    return rho

