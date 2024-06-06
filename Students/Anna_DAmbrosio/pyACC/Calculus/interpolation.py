import scipy as sp
import numpy as np


 # array of number of steps

def array_N(n_in,n_fin,n_steps):
    N_int = np.logspace(n_in,n_fin,n_steps)
    return N_int

class Interpolation:

    def __init__(self,f):
        if callable(f)==True:
            self.fun = f
        else:
            raise TypeError("The object is not a function.")


    # INTERPOLATION
            
    def interpol(self,n_steps,n_in,n_fin,x_min,x_max,x_steps):

        '''
        self: fun
              funtion of which to calculate the interpolation
        n_steps: int
                 number of value of number of point for the interpolation
        n_in: int
              minimum value of number of point for the interpolation
        n_fin: int
               maximum value of number of point for the interpolation
        x_min: float
               minimum value of the array in which the function will be evaluated
        x_max: float
               maximum value of the array in which the function will be evaluated
        
        Return: list
                list of interpolated function calculated in np.linspace(x_min, x_max, x_steps)
        '''

        # array of number of steps
        N_int = array_N(n_steps,n_in,n_fin)

        # array for theoretical comparison
        x_th = np.linspace(x_min, x_max, x_steps)

        # array for interpolatioin
        list_x = []
        for i in range (N_int.size):
            x_list = np.linspace(x_min, x_max, int(N_int[i]))
            list_x.append(x_list)

        # calculate the function in the array of points for the interpolation
        list_f = []
        for i in range(N_int.size):
            fun_i = self.fun(list_x[i])
            list_f.append(fun_i)

        # linear interpolatioin of the function
        list_int = []
        for i in range(N_int.size):
            fun_int = sp.interpolate.interp1d(list_x[i], list_f[i])
            list_int.append(fun_int)

       # calculation of function in z_th
        list_fun = []
        for i in range(N_int.size):
            fun_th = list_int[i](x_th)
            list_fun.append(fun_th) 

        return list_fun
    

    # CALCULATION OF THEORETICAL ERROR

    def theoretical_error(self,n_steps,n_in,n_fin,x_min,x_max,x_steps):

        '''
        self: fun
              funtion of which to calculate the interpolation
         n_steps: int
                 number of value of number of point for the interpolation
        n_in: int
              minimum value of number of point for the interpolation
        n_fin: int
               maximum value of number of point for the interpolation
        x_min: float
               minimum value of the array in which the function will be evaluated
        x_max: float
               maximum value of the array in which the function will be evaluated
        x_steps: int
                 number of value of the array in which the function will be evaluated
        
        Return: list
                list of the theoretical error function calculated in np.linspace(x_min, x_max, x_steps)
        '''

       # array of number of steps
        N_int = array_N(n_steps,n_in,n_fin)

        # array for theoretical comparison
        x_th = np.linspace(x_min, x_max, x_steps)

        # calculation of theoretical error        
        f_fun = lambda x: self.fun(x)
        list_error_th = []
        for i in range(N_int.size):
            h = x_max/int(N_int[i])
            der_2 = np.zeros(x_th.size)
            for j,xi in enumerate(x_th):
                der_2[j] = sp.misc.derivative(f_fun,xi.item(),dx=1e-6,n=2)
            error = ((h**2)/8)*np.absolute(der_2)
            list_error_th.append(error)

        return list_error_th        
    

    # COMPARISON BETWEEN THEORETICAL PREDICTION AND INTERPOLATION
            
    def theoretical_comparison(self,n_steps,n_in,n_fin,x_min,x_max,x_steps):

        '''
        self: fun
              funtion of which to calculate the interpolation
         n_steps: int
                 number of value of number of point for the interpolation
        n_in: int
              minimum value of number of point for the interpolation
        n_fin: int
               maximum value of number of point for the interpolation
        x_min: float
               minimum value of the array in which the function will be evaluated
        x_max: float
               maximum value of the array in which the function will be evaluated
        x_steps: int
                 number of value of the array in which the function will be evaluated
                 
        Return: list
                list of the difference between the theoretical function and the interpolated one both calculated in np.linspace(x_min, x_max, x_steps)
        '''

        # array of number of steps
        N_int = array_N(n_steps,n_in,n_fin)

        # array for theoretical comparison
        x_th = np.linspace(x_min, x_max, x_steps)

        # function calculated in x_th for theoretical comparison
        fun_th = self.fun(x_th)

        # calculation of the interpolated function in x_th
        list_fun = []
        for i in range(N_int.size):
            funz = self.interpol(n_steps,n_in,n_fin,x_min,x_max,x_steps)[i]
            list_fun.append(funz)

        # calculation of difference of the two methods
        list_delta = []
        for i in range(N_int.size):
            delta_f = fun_th - list_fun[i]
            list_delta.append(delta_f)

        return list_delta
    

def inter_grid(data, size, N_bin):
    '''
    Parameters
    data: array
          Nx3 distribution of pointa
    size: int
          size of the box filled with the points in dist
    N_bin: int
           number of bins in which each direction of the volume considered is devided
           
    Return
    pdf_norm: array
              normalized probability density function of the points in dist interpolated on the grid of volume size**3 and spacing size/N_bin
    '''

    spacing = size/N_bin
    N = data.shape[0]

    pdf = np.zeros((N_bin+1, N_bin+1, N_bin+1))

    for i in range(N):
        datax = data[i,0]
        datay = data[i,1]
        dataz = data[i,2]

        idx = int(datax/spacing)
        idy = int(datay/spacing)
        idz = int(dataz/spacing)

        if (datax/spacing - int(datax/spacing)) < 0.5:
            idx = int(datax/spacing) + 1

        if (datay/spacing - int(datay/spacing)) < 0.5:
            idy = int(datay/spacing) + 1

        if (dataz/spacing - int(dataz/spacing)) < 0.5:
            idz = int(dataz/spacing) + 1

        pdf[idx, idy, idz] = pdf[idx, idy, idz] + 1

    pdf_norm = pdf/N

    return pdf_norm
