from scipy import integrate
import matplotlib.pyplot as plt

class My1DFunction:
    def __init__(self, function):
        self.f = function

    def eval(self, x0):
        return self.f(x0)

    def integrate(self, a, b):
        return integrate.quad(self.f, a, b)
    
    def diff(self, x0, h=0.0001, n=1, acc=1):
        if(n==1):
            if(acc==1):
                central_der = (self.f(x0+h)-self.f(x0-h))/(2*h)
            #if(acc==2):
            #    central_der = 1/12*self.f(x0-2*h)-2/3*self.f(x0-h)+2/3*self.f(x0+h)-1/12*self.f(x0+2*h)
            return central_der
        if(n==2):
            if(acc==1):
                second_der = (self.f(x0+h)+self.f(x0-h)-2*self.f(x0))/h**2
            #if(acc==2):
            #    second_der = -1/12*self.f(x0-2*h)+4/3*self.f(x0-h)-5/2*self.f(x0)+4/3*self.f(x0+h)-1/12*self.f(x0+2*h)
   
            return second_der

    def plot(self, x_array):
        plot=plt.plot(x_array, self.f(x_array))
