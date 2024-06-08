import numpy as np
from ..calculus import integral,trapezoid
import scipy
import scipy.integrate as integrate
from scipy.spatial import cKDTree
from tqdm.auto import tqdm

def get_power_spectrum(delta_x,side,spacing,n_kF = 1):
    #compute a power spectrum from a desnity firld defined ona grid
    Volume = side**3
    n_cell = int(side//spacing)
    kx = np.fft.fftfreq(n_cell,spacing)*np.pi*2
    ky = np.fft.fftfreq(n_cell,spacing)*np.pi*2
    kz = np.fft.rfftfreq(n_cell,spacing)*np.pi*2

    kxv, kyv, kzv = np.meshgrid(kx, ky,kz, indexing='xy')

    knorm = np.sqrt(kxv**2+kyv**2+kzv**2)
    #get the density field in the fourier spacing
    #delta_k_norm = np.random.normal(size=pks.shape)*np.sqrt(pks*Volume)
    delta_k = np.fft.rfftn(delta_x,norm="backward")*spacing**3
    #compute the power spectru
    delta_k_sq=np.abs(delta_k)**2
    pks=delta_k_sq.flatten()/Volume
    #binning
    kf=2*np.pi/side #h/Mpc
    kn = 2*np.pi/spacing
    edges = np.arange(kf,kn,kf)
    pk_meas = np.zeros(len(edges)-1)
    k_bin = (edges[1:]+edges[0:-1])/2
    for i in range(len(k_bin)):
        k_selec = np.where((knorm>edges[i]) & (knorm<edges[i+1]))[0]
        pk_meas[i] = np.mean(pks[k_selec])
    return k_bin,pk_meas


def count_pairs(data1,r_edges,data2= None):
    #count auto*croos pairs between sample 1 and 2 with a given separation binning
    #se data 2 non c'e faccio autocoppie
    if data2 is None:
        tree = cKDTree(data1)
        auto = True
    else:
        tree = cKDTree(data2)
        auto = False
    pairs = np.zeros(len(r_edges)-1)

    for i in tqdm(range(len(data1))):
    #for i in range(len(data1)):
        neighbours_idx = np.array(tree.query_ball_point(data1[i],r_edges[-1])) #la funz ritorna gli indici nel cerchio
        #la particella iesima è il centro e r è il raggio
        if auto:
            cut_neighbours = np.where(neighbours_idx>1)[0]
            neighbours_idx = neighbours_idx[cut_neighbours] #conto solo quelle dopo per non contare due volte
    separations = np.linalg.norm(tree.data[neighbours_idx]-data1[i],axis=1)
    pairs += np.histogram(separations,bins=r_edges)[0]
    return pairs
    


