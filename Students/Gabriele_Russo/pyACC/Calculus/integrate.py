#Definition of a function to integrate with trapezoids:
'''
   parameter of the function:
   integrand: function to be integrated
   low      : point from wich integrate
   up       : point in wich end the integration
   delta_x  : interval of the single iterations (height of the trapezoids) 
'''
import numpy as np

def trapezoid(integrand, low, up, delta_x):

    x = np.arange(low, up + delta_x, delta_x) #np.arange allows you to make an array with n-points or nodes calculated between low and up with space delta_x 
   #WARNING: it's needed up + delta_x in order to ALSO take the last point in the array, otherwise the last trapezoid is neglected!
    
    fx = integrand(x) #evaluation of the function in every point just calculated
    
    #REMAINDER: fx[1:] from 2nd to last element of the array, fx[0:-1] from 1st to second-last element of the array!
    return (np.sum( (fx[1:]+fx[0:-1])/2 * (x[1:]-x[0:-1]) ))   
