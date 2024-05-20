import numpy as np

def generate_gaussian_map(pk_func,side,spacing):
    '''
    Generate a mock map with a Gaussian distribution of points
    '''

    Volume=side**3
    n_cell=int(side//spacing)

    #Fourier grid
    kx=np.fft.fftfreq(n_cell,d=spacing)*2*np.pi
    ky=np.fft.fftfreq(n_cell,d=spacing)*2*np.pi
    kz=np.fft.rfftfreq(n_cell,d=spacing)*2*np.pi

    KX,KY,KZ=np.meshgrid(kx,ky,kz,indexing='ij')
    knorm=np.sqrt(KX**2+KY**2+KZ**2)

    #Interpolate the power spectrum on the grid
    pks=pk_func(knorm)

    #Generate a random field
    phase=np.random.uniform(size=pks.shape)*2*np.pi

    #Generate a random norm
    delta_k_norm=np.random.normal(size=pks.shape)*np.sqrt(pks*Volume)
    delta_k_norm[0,0,0]=0

    delta_k=delta_k_norm*(np.cos(phase)+1j*np.sin(phase))

    #Compute the field in configuration space
    delta_x=np.fft.irfftn(delta_k,norm='backward')/spacing**3

    return delta_k,delta_x