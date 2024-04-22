import numpy as np

def power_spectrum_estimator(data_array, min_corr, max_corr):
    '''
    Implementing Power Spectrum Estimator starting from 
    2 Point Correletion Function Estimator designed by Landy and Szalay in 1993

    Input:
    - data_array: np array
    '''

    #2PCFE
    DD = 0
    for i in range(0, len(data_array)):
        for j in range(0, len(data_array)):
            r = abs(data_array[i]-data_array[j])
        
        if r>=min_corr and r<=max_corr:
            DD += 1

