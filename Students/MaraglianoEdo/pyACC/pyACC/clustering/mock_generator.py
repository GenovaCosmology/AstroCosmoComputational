import numpy as np

def generate_gaussian_map(pk_func, side, spacing):

    """
    Generates a Gaussian random field from an input power spectrum in a given volume.

    Parameters:
    -----------
    pk_func : function
        A function that takes an array of wavenumbers (k) and returns the power spectrum P(k) at those wavenumbers.
    side : float
        The physical size of the side of the cubic volume.
    spacing : float
        The grid spacing for the Fourier transform. Determines the resolution of the grid.

    Returns:
    --------
    delta_k : ndarray
        The generated Gaussian random field in Fourier space.
    delta_x : ndarray
        The generated Gaussian random field in real space.

    Example:
    --------
    >>> def pk_func(k):
    >>>     return k**-3  # Example power spectrum function
    >>> side = 1.0
    >>> spacing = 0.1
    >>> delta_k, delta_x = generate_gaussian_map(pk_func, side, spacing)
    """

    # set the volume
    volume = side**3

    n_cell = int(side//spacing)

    # set the Fourier grid

    kx = np.fft.fftfreq(n_cell, spacing)*np.pi*2
    ky = np.fft.fftfreq(n_cell, spacing)*np.pi*2
    kz = np.fft.rfftfreq(n_cell, spacing)*np.pi*2

    kx = np.fft.fftshift(kx)
    ky = np.fft.fftshift(ky)

    KX, KY, KZ = np.meshgrid(kx,ky,kz, indexing='xy')

    knorm = np.sqrt(KX**2+KY**2+KZ**2)
    pks = pk_func(knorm)

    #generate a random phase
    phase = np.random.uniform(size=pks.shape)*2*np.pi
    #generate a norm
    delta_k_norm = np.random.normal(size=pks.shape)*np.sqrt(pks*volume)

    delta_k_norm[0,0,0]=0

    delta_k = delta_k_norm*(np.cos(phase)+1j*np.sin(phase))

    delta_x = np.fft.irfftn(delta_k, norm='backward')/spacing**3

    return delta_k, delta_x


def generate_lognormal_map(pk_func, side, spacing):
    """
    Generates a lognormal random field from an input power spectrum in a given volume.
    
    Parameters:
    -----------
    pk_func : function
        A function that takes an array of wavenumbers (k) and returns the power spectrum P(k) at those wavenumbers.
        side : float
        The physical size of the side of the cubic volume.
        spacing : float
        The grid spacing for the Fourier transform. Determines the resolution of the grid.
        
        Returns:
        --------
        delta_x : ndarray
        The generated lognormal density field in real space.
        
        Example:
        --------
        >>> def pk_func(k):
        >>>     return k**-3  # Example power spectrum function
        >>> side = 1.0
        >>> spacing = 0.1
        >>> delta_x = generate_lognormal_map(pk_func, side, spacing)
        """
    
    # set the volume
    volume = side**3
    
    n_cell = side//spacing

    # set the Fourier grid
    
    kx = np.fft.fftfreq(n_cell, spacing)*np.pi*2
    ky = np.fft.fftfreq(n_cell, spacing)*np.pi*2
    kz = np.fft.rfftfreq(n_cell, spacing)*np.pi*2
    
    #kx = np.fft.fftshift(kx)
    #ky = np.fft.fftshift(ky)
    #kz = np.fft.fftshift(kz)
    
    KX, KY, KZ = np.meshgrid(kx,ky,kz, indexing='ij')

    knorm = np.sqrt(KX**2+KY**2+KZ**2)
    pks = pk_func(knorm)

    # get 2PCF on the grid
    xi = np.fft.irfftn(pks) / spacing**3

    if np.any(xi <= -1):
        raise ValueError("Invalid input: xi contains elements <= -1")
    # generate gaussian 2PCF
    xi_g = np.log(1+xi)

    # get PS from xi_g
    pk_g = np.fft.rfftn(xi_g) * spacing**3

    # generate G(k)

    phase = np.random.uniform(size=pks.shape)*2*np.pi
    G_k_norm = np.random.normal(size=pks.shape)*np.sqrt(pk_g*volume)
    G_k_norm[0,0,0] = 0

    G_k = G_k_norm *(np.cos(phase)+1j*np.sin(phase))

    # Compute G(x)
    G_x = np.fft.irfftn(G_k, norm='backward')/spacing**3

    # get delta from G_x using lognormal transform

    var_G = np.var(G_x)
    delta_x = np.exp(G_x-var_G)-1

    return delta_x





## it's called poisson sample map but returns a catalog. maybe change it?
def poisson_sample_map(delta_x, side, spacing, N_objects, seed=666):
    """
    Generates a Poisson realization of a density field from an input density contrast field.

    Parameters:
    -----------
    delta_x : ndarray
        A 3D array representing the density contrast field.
    side : float
        The physical size of the simulation box (e.g., in Mpc/h).
    spacing : float
        The spatial resolution of the grid (e.g., in Mpc/h).
    N_objects : int
        The total number of objects to sample.
    seed : int, optional
        The random seed to use for reproducibility. Default is 666.

    Returns:
    --------
    points : ndarray
        An array of shape `(N_objects, 3)` containing the positions of the sampled objects.

    Example:
    --------
    >>> delta_x = np.random.randn(64, 64, 64)
    >>> side = 100.0
    >>> spacing = 1.5625
    >>> N_objects = 1000
    >>> points = poisson_sample_map(delta_x, side, spacing, N_objects)
    """

    # Set the random state
    np.random.seed(seed)
    # compute grid quantities
    Volume = side**3

    n_bar = N_objects/Volume
    cell_volume = spacing**3

    n_cell = int(side//spacing)

    # Ensure delta_x has the correct shape
    if delta_x.shape != (n_cell, n_cell, n_cell):
        raise ValueError(f"delta_x should have shape ({n_cell}, {n_cell}, {n_cell})")

    # from delta to density
    n_x = n_bar * (1 + delta_x)

    # from number density to mean number (not necessarily integer) of objects per cell

    Nobj_x = n_x * cell_volume

    # Poisson realization: integer number of obj to generate in each cell

    Nobj_x_sample = np.random.poisson(Nobj_x)

    # Generate points
    points = []
    for i in range(n_cell):
        x_min, x_max = i*spacing, (i+1)*spacing
        for j in range(n_cell):
            y_min, y_max = j*spacing, (j+1)*spacing
            for k in range(n_cell):
                z_min, z_max = k*spacing, (k+1)*spacing

                if Nobj_x_sample[i,j,k] > 0 :
                    points.extend(np.random.uniform(size=(Nobj_x_sample[i,j,k], 3), low=[x_min,y_min,z_min], high=[x_max,y_max, z_max]))

    return np.array(points)
