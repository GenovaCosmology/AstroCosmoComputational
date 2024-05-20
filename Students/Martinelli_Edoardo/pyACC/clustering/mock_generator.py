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


