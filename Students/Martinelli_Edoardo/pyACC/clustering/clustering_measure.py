import numpy as np

def get_power_spectrum(delta_x, side, spacing,n_kF=1):
    '''
    Compute the power spectrum from a density field defined on a grid.

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

    # get the density  in fourier space
    delta_k = np.fft.rfftn(delta_x,norm='backward')*spacing**3

    # compute the power spectrum
    delta_k_sq = np.abs(delta_k)**2
    pks=delta_k_sq.flatten()/Volume

    # binning
    kF=2*np.pi/side
    kN=2*np.pi/spacing
    edges = np.arange(kF,kN,n_kF*kF)
    k_bin = (edges[1:]+edges[0:-1])/2

    pk_meas=np.zeros(k_bin.shape)
    for i in range(len(k_bin)):
        k_selection = np.where((knorm.flatten()>edges[i]) & (knorm.flatten()<edges[i+1]))[0]
        pk_meas[i]=np.mean(pks[k_selection])

    return k_bin,pk_meas

