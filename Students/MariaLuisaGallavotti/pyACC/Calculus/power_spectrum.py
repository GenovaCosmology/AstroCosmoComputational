#POWER SPECTRUM ESTIMATOR

import numpy as np
from scipy.fft import fftn, fftfreq

def Power_Spectrum(signal,sampling_frequency):
    #Fourier Transform
    fourier_transform=fftn(signal)
    #Frequencies
    frequencies=[]
    for size in signal.shape:
        frequencies.append(fftfreq(size,1/sampling_frequency))
    #Power Spectrum
    power_spectrum=np.abs(fourier_transform)**2

    return power_spectrum,frequencies