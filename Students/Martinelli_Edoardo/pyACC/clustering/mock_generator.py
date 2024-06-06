import numpy as np


def generate_gaussian_map(pk_funck,side,spacing):

    '''
    Generate a Gaussian map form an input P(k)  in a given volume.

    Parameters:
    -----------
    pk_funck: function
        The input P(k) function.
    
    '''

    # set the volume
    Volume = side**3
    n_cell = int(side/spacing)

    # set the fourier grid
    kx = np.fft.fftfreq(n_cell,spacing)*np.pi*2
    ky = np.fft.fftfreq(n_cell,spacing)*np.pi*2
    kz = np.fft.fftfreq(n_cell,spacing)*np.pi*2
    KX,KY,KZ = np.meshgrid(kx,ky,kz,indexing='xy')
    knorm = np.sqrt(KX**2 + KY**2 + KZ**2)

    # Innterpolate the power spectrum
    pks = pk_funck(knorm)

    # generate a random phase and a random norm
    phase = np.random.uniform(size=pks.shape)*2*np.pi
    delta_k_norm = np.random.normal(size=pks.shape) * np.sqrt(pks*Volume)
    delta_k_norm[0,0,0] = 0 # to have a real field
    delta_k = delta_k_norm * (np.cos(phase) + 1j*np.sin(phase))

    # compute the field in configuration space
    delta_x = np.fft.irfftn(delta_k,norm='backward')/spacing**3

    return delta_k,delta_x


def poisson_sample_map(delta_x,side,spacing,N_objects,seed=666):

    '''
    Poisson sample a given density field. From the density field we generate 
    a number of objects in each cell following a Poisson distribution. Then we
    generate the objects coordinates in each cell following a uniform distribution.
    '''
    
    np.random.seed(seed)

    # compute grid quantities
    Volume = side**3
    n_bar = N_objects/Volume # mean density
    cell_volume = spacing**3
    n_cells = int(side/spacing)

    # from delta_x to density
    n_x = n_bar*(1+delta_x)

    # from number density to number of objects
    Nobj_x = n_x*cell_volume

    # Poisson sampling
    Nobj_x_sample = np.random.poisson(Nobj_x)

    # generate points
    points=[]
    for i in range(n_cells):
        x_min,x_max = i*spacing,(i+1)*spacing # X boundaries of i-th cell
        for j in range(n_cells):
            y_min,y_max = j*spacing,(j+1)*spacing # Y boundaries of j-th cell
            for k in range(n_cells):
                z_min,z_max = k*spacing,(k+1)*spacing # Z boundaries of k-th cell
                if Nobj_x_sample[i,j,k]>0:
                    points.extend(np.random.uniform(size=(Nobj_x_sample[i,j,k],3),low=[x_min,y_min,z_min],high=[x_max,y_max,z_max]))
    return np.array(points)