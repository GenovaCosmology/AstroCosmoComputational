import numpy as np

def get_power_spectrum(delta_x, side, spacing, n_kf=1):

    Volume = side**3
    n_cell = int(side//spacing)

    kx = np.fft.fftfreq(n_cell, spacing)*2*np.pi
    ky = np.fft.fftfreq(n_cell, spacing)*2*np.pi
    kz = np.fft.rfftfreq(n_cell, spacing)*2*np.pi

    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='xy') # x messa per colonne e y per righe

    knorm = np.sqrt(KX**2 + KY**2 + KZ**2)


    delta_k = np.fft.rfftn(delta_x, norm='backward')*spacing**3

    delta_k_sq = np.abs(delta_k)**2

    pks = delta_k_sq.flatten()/Volume
    
    kf = 2*np.pi/side
    kN = 2*np.pi/spacing
    edges = np.arange(kf, kN, n_kf*kf)

    k_bin = (edges[1:]+edges[0:-1])/2

    pk_meas = np.zeros(len(edges)-1)

    for i in range(len(k_bin)):
        k_selection = np.where((knorm>edges[i]) & (knorm<edges[i+1]))[0]
        pk_meas[i] = np.mean(pks[k_selection])
    
    return k_bin, pk_meas


