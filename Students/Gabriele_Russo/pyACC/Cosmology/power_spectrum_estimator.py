import numpy as np
import math as m
import matplotlib as plt
def power_spectrum_estimator(Delta_X, Delta_Y, Delta_Z, delta, n):
    '''
    Power spectrum Estimator Function.
    Inputs:
    - Delta_X: numpy 1D array for maximum range covered on x axis
    - Delta_Y: numpy 1D array for maximum range covered on y axis
    - Delta_Z: numpy 1D array for maximum range covered on z axis
    OBS.: ALL STEP ALONG AXIS MUST BE EQUAL
    - delta  : numpy array 3D, field whose power spectrum will be computed
    - n      : int, for rescaling binning pf K_modulus as multiples of the foundamental frequency

    Output:
    -
    '''
    #Computing step along axis
    delta_x = Delta_X[1] - Delta_X[0] 

    #Grid creation
    X, Y, Z = np.meshgrid(Delta_X, Delta_Y, Delta_Z)

    #Fast Fourier Transformation of the grid 
    data_transform = np.fft.rfftn(delta)
    
    # Shape of the 3D grid
    shape = data_transform.shape

    # Sampling frequencies along each axis in the Fourier space
    freq_x = np.fft.fftfreq(shape[0], delta_x)
    freq_y = np.fft.fftfreq(shape[1], delta_x)
    freq_z = np.fft.fftfreq(shape[2], delta_x)

    # Create coordinate grids in Fourier space
    X_fourier, Y_fourier, Z_fourier = np.meshgrid(freq_x, freq_y, freq_z, indexing='ij')
    # Now X_fourier, Y_fourier, and Z_fourier contain the corresponding coordinates in Fourier space
    
    #Modulus of distances in fourier space
    K_modulus = np.sqrt(X_fourier**2 + Y_fourier**2 + Z_fourier**2)

    #Defining foundamental frequency
    k_f = 2*m.pi/len(Delta_X)

    # Compute the histogram of K_modulus
    hist, bin_edges = np.histogram(K_modulus, bins=np.arange(0, K_modulus.max() + k_f, k_f))

    # Get the indices of the bin each element in K_modulus belongs to
    bin_indices = np.digitize(K_modulus.flatten(), bin_edges)
    
    # Bin's number
    num_bins = len(bin_edges) - 1

    plt.hist(K_modulus.flatten(), bins=bin_edges, edgecolor='black', alpha=0.7)
    plt.xlabel('K Modulus')
    plt.ylabel('Frequency')
    plt.title('Histogram of K Modulus with Fixed Size Bins')
    plt.show()
