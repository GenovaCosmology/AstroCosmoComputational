import numpy as np

def generate_gaussian_map(pk_func, side, spacing):
    # set volume
    Volume = side**3

    n_cell = int(side//spacing)
    
    # set fourier grid

    kx = np.fft.fftfreq(n_cell, spacing)*2*np.pi
    ky = np.fft.fftfreq(n_cell, spacing)*2*np.pi
    kz = np.fft.rfftfreq(n_cell, spacing)*2*np.pi

    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='xy') # x messa per colonne e y per righe

    knorm = np.sqrt(KX**2 + KY**2 + KZ**2)

    pks = pk_func(knorm)

    phase = np.random.uniform(size=pks.shape)*2*np.pi

    delta_k_norm = np.random.normal(size=pks.shape)*np.sqrt(pks*Volume)
    delta_k_norm[0,0,0] = 0

    delta_k = delta_k_norm*(np.cos(phase) + 1j*np.sin(phase))

    delta_x = np.fft.irfftn(delta_k, norm='backward')/spacing**3

    return delta_k, delta_x

def poisson_sample_map(delta_x, side, spacing,N_objects, seed=666):
    # prende una mappa e crea una distribuzione di punti
    np.random.seed(seed)
    n_cells = int(side//spacing)
    Volume = side**3
    n_bar = N_objects/Volume
    cell_volume = spacing**3

    n_x = n_bar *(1+delta_x)

    Nobj_x = n_x * cell_volume

    Nobj_x_sample = np.random.poisson(Nobj_x) # in ogni cella contiene quanti oggetti ci sono, informazioni sulla densità e sul numero totale di punti

    points = []
    for i in range(n_cells):
        x_min, x_max = i*spacing, (i+1)*spacing
        for j in range(n_cells):
            y_min, y_max = j*spacing, (j+1)*spacing
            for k in range(n_cells):
                z_min, z_max = k*spacing, (k+1)*spacing
                

                if Nobj_x_sample[i,j,k]>0:
                    points.extend(np.random.uniform(size=(Nobj_x_sample[i,j,k],3), low=[x_min, y_min, z_min], high=[x_max, y_max, z_max]))

    return np.array(points)