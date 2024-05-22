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

def poisson_sample_map(delta_x,side,spacing,N_objects,seed=666): #N_objects is the number of target objects (I want to sample)

    #random state
    np.random.seed(seed)

    #grid quantities
    Volume=side**3
    n_bar=N_objects/Volume #mean desnity
    cell_volume=spacing**3
    n_cells=int(side//spacing)

    #from delta to density
    n_x=n_bar*(1+delta_x)

    #from number density to number of objects
    Nobj_x=n_x*cell_volume

    #Poisson sampling
    Nobj_x_sample=np.random.poisson(Nobj_x) #poissonian realization of the number of objects in each cell

    #generate points (the catalog)
    points=[]
    for i in range(n_cells):
        x_min,x_max=i*spacing,(i+1)*spacing #limits of each cell in the x axis
        for j in range(n_cells):
            y_min,y_max=j*spacing,(j+1)*spacing #limits of each cell in the y axis
            for k in range(n_cells):
                z_min,z_max=k*spacing,(k+1)*spacing #limits of each cell in the z axis

                if Nobj_x_sample[i,j,k]>0:
                    points.extend(np.random.uniform(size=(Nobj_x_sample[i,j,k],3),low=[x_min,y_min,z_min],high=[x_max,y_max,z_max])) #generates a point in the cell

    return np.array(points)