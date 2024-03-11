from scipy import interpolate
import numpy as np

def interp(x,y, extrapolate='True'):
    if(extrapolate):
        return interpolate.interp1d(x,y, fill_value='extrapolate')
    else:
        return interpolate.interp1d(x,y)
    
def interp_3d(x1,x2,x3,y):
    X,Y,Z = np.meshgrid(x1,x2,x3)
    interpolator = interpolate.RegularGridInterpolator((X,Y,Z),y)
    return interpolator
