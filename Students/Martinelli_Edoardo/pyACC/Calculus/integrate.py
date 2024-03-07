import scipy
import numpy as np
import scipy.integrate as integrate
from datetime import datetime
from typing import Any
import time

def integrate_f(func,x_in=0,x_fin=0): # need to add ,deltax=0 if we want to use the trapezoid method
    
    print("---")  
    print("--> ",datetime.fromtimestamp(time.time()))
    print("---")
    

    '''
    This function integrates a function or a sample:

    func: function or sample to integrate
    x_in: initial point of the integration
    x_fin: final point of the integration
    
    '''
    bool =True
    bool =callable(func)
    if bool == True:
        print("I'm integrating over a function, I used the quad method from scipy.")
        return integrate.quad(func, x_in, x_fin)
    else:
        print("I'm integrating over a sample, I used the trapezoid method from scipy.")
        x=np.linspace(x_in,x_fin,func.size)
        integral= np.sum( ((func[1:]+func[0:-1])/2) * (x[1:]-x[0:-1]) )
        return integral 
        #return integrate.trapezoid(func,dx=(1/len(func)))
        #with this method doesn't work the test, I don't know why. 