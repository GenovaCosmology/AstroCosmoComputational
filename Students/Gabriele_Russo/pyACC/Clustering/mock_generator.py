import numpy as np

def generate_gaussian_map(pk_func, side, spacing):
    '''
    Generates a Guaussian map from an input P(k) in a given volume

    Parameters:
    pk_func : function
        The input power spectrum function
    side : int
        The size of the volume
    '''

    # Set the Volume
    volume = side**3

    n_cell = int(side//spacing)

    # Set the Fourier Grid
    kx = np.fft.fftfreq(n_cell, d=spacing)*np.pi*2
    ky = np.fft.fftfreq(n_cell, d=spacing)*np.pi*2
    kz = np.fft.rfftfreq(n_cell, d=spacing)*np.pi*2

    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij') # indexing changes the order of the axes of the matrix

    knorm = np.sqrt(KX**2 + KY**2 + KZ**2)

    # Interpolate the Power Spectrum on the Fourier Grid
    pks = pk_func(knorm)

    # Generate a random phase
    phase = np.random.uniform(size=pks.shape) * 2 * np.pi

    # Generate a random norm
    delta_k_norm = np.random.normal(size=pks.shape) * np.sqrt(pks * volume)
    delta_k_norm[0,0,0] = 0 # per garantire trasformata sia reale

    delta_k = delta_k_norm * (np.cos(phase) + 1j*np.sin(phase))

    # Compute the field in configuration space
    delta_x = np.fft.irfftn(delta_k, norm='backward')/spacing**3

    return delta_k, delta_x

def generate_lognormal_map(pk_func, side, spacing):

    # Set the Volume
    volume = side**3

    n_cell = int(side//spacing)

    # Set the Fourier Grid
    kx = np.fft.fftfreq(n_cell, d=spacing)*np.pi*2
    ky = np.fft.fftfreq(n_cell, d=spacing)*np.pi*2
    kz = np.fft.rfftfreq(n_cell, d=spacing)*np.pi*2

    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij') # indexing changes the order of the axes of the matrix

    knorm = np.sqrt(KX**2 + KY**2 + KZ**2)

    # Interpolate the Power Spectrum on the Fourier Grid
    pks = pk_func(knorm)

    # xi 
    xi = np.fft.irfftn(pks, norm='backward')/spacing**3
    
    # log xi
    log_xi = np.log(1 + xi)

    # power spectrum of log xi
    pks_log = np.fft.rfftn(log_xi, norm='backward')*spacing**3
    
    # Generate a random phase
    phase = np.random.uniform(size=pks_log.shape) * 2 * np.pi

    # Generate a random norm
    delta_k_norm = np.random.normal(size=pks_log.shape) * np.sqrt(pks_log * volume)
    delta_k_norm[0,0,0] = 0

    delta_k = delta_k_norm * (np.cos(phase) + 1j*np.sin(phase))

    # Compute the field in configuration space
    delta_Gx = np.fft.irfftn(delta_k, norm='backward')/spacing**3

    delta_x = np.exp(delta_Gx-np.log(np.mean(np.exp(delta_Gx)))) - 1
    return delta_x
# Funzione che prenda una mappa di input e generi dei punti con densità desiderata

def poisson_sample_map(delta_x, side, spacing, N_objects, seed=457):

    # Set the random state
    np.random.seed(seed)

    # Compute grid quantities
    Volume = side**3
    n_bar = N_objects/Volume # densità media del numero di oggetti
    cell_volume = spacing**3
    n_cells = int(side//spacing)

    # From delta_x to density
    n_x = n_bar * (1 + delta_x) # densità sulla nostra cella (da densità media, ho sovra/sottodensità)

    # From number density to number of objects
    Nobj_x = n_x * cell_volume # numero di oggetti sulla cella

    # Poisson Sampling
    Nobj_x_sample = np.random.poisson(Nobj_x) # numero di oggetti campionati sulla cella (realizzazione poissoniana)
    
    # Generate points
    # Al di sotto nostra scala no realizzazione punti, al di sopra si (sotto ho pdf piatta, sopra processo gaussiano)
    points = []
    for i in range(n_cells):
        x_min, x_max = i*spacing, (i+1)*spacing # confini cella 
        for j in range(n_cells):
            y_min, y_max = j*spacing, (j+1)*spacing
            for k in range(n_cells):
                z_min, z_max = k*spacing, (k+1)*spacing
                
                if Nobj_x_sample[i,j,k] > 0: # se ho oggetti genero punti  e coordinate limitate dalla mia griglia
                    points.extend(np.random.uniform(size=(Nobj_x_sample[i,j,k], 3), low=[x_min, y_min, z_min], high=[x_max, y_max, z_max]))


    return np.array(points)