import numpy as np
class Integrate:
    def trapezoid(func, low, up,delta_x=1): #function, lower and upper value at which compute integral, dx default=1
        x = np.arange(low, up +delta_x, delta_x) #sample grid, up+delta_x because np.arange does not include last step
        fx = func(x) #compute the function at each value of x
        return delta_x*((fx[0]+fx[-1])/2. + np.sum(fx[1:-1]))
    