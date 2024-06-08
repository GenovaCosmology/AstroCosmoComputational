import numpy as np
from ..calculus import integral,trapezoid
import scipy
import scipy.integrate as integrate

def generate_gaussian_map(pk_func,side,spacing):
    #generates a gaussian map from an imput P(k) in a given volume
    Volume = side**3
    #set the fourier grid
    n_cell = int(side//spacing)
    kx = np.fft.fftfreq(n_cell,spacing)*np.pi*2
    ky = np.fft.fftfreq(n_cell,spacing)*np.pi*2
    kz = np.fft.rfftfreq(n_cell,spacing)*np.pi*2

    kxv, kyv, kzv = np.meshgrid(kx, ky,kz, indexing='xy')

    knorm = np.sqrt(kxv**2+kyv**2+kzv**2)
    #interpolate the pk on the grid
    #pk_int = interp1d(kh,pk[0],kind='cubic',fill_value='extrapolate')
    pks = pk_func(knorm)
    #generate a random phase
    phase = np.random.uniform(size=pks.shape)*2*np.pi #grande come griglia
    #generate a random norm
    delta_k_norm = np.random.normal(size=pks.shape)*np.sqrt(pks*Volume) #genero un numero da distribuz gaussiana con media zero e dev1
    delta_k_norm[0,0,0]=0
    #la normalizzo per un numero perchè voglio che la dev stndard sia P(k) e non 1
    delta_k = delta_k_norm *(np.cos(phase) + 1j*np.sin(phase))
    delta_x = np.fft.irfftn(delta_k,norm="backward")/spacing**3
    return delta_k, delta_x



def poisson_sample_map(delta_x,side,spacing, n_obj,seed=666):
    #set random state, punto da cui parte la generazione random
    np.random.seed(seed)
    #grid
    Volume = side**3
    #set the fourier grid
    n_bar = n_obj/Volume
    cell_vol = spacing**3
    n_cells = int(side//spacing)
    #from delta to density
    n_x = n_bar*(delta_x+1)
    #from number density to number of objects
    nobj_x = n_x * cell_vol
    #poisson sampling: discretizzo
    nobj_x_sample = np.random.poisson(nobj_x)
    #generate points: contiene quanti ogg stanno nela cella e power spectrum
    #NB: sotto la risoluzione non ho controllo ed random (poisson), sopra ho le cose (gaussiane)
    points = []
    for i in range(n_cells):
        x_min,x_max = i*spacing,(i+1)*spacing #volumetto celletta
        for j in range(n_cells):
            y_min,y_max = j*spacing,(j+1)*spacing
            for k in range(n_cells):
                z_min,z_max = k*spacing,(k+1)*spacing
                if nobj_x_sample[i,j,k] > 0:   #se ho punti davvero
                    points.extend(np.random.uniform(size=(nobj_x_sample[i,j,k],3) , low=[x_min,y_min,z_min], high = [x_max,y_max,z_max]))
    return np.array(points)
    



