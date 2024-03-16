import sys
sys.path.append("../")
from Calculus import trapezoid

class Polyfunctions:

    # Inizialization method (constructure)
    def __init__(self, function):
        self.function = function #self is telling the class to instantiate an object wich is the function I pass as parameter

    #Methods of the class
        
    #Evaluating function in one point
    def evaluate(self, argument):
        return self.function(argument)
    
    #Integrating function 
    def integrate_function(self, x_min, x_max, delta_x):
        return trapezoid(self.function, x_min, x_max, delta_x)
    
    #Derive function at first order

    #Foward derivative method
    def forward_difference_derivative(self, x, h):
        return (self.function(x + h) - self.function(x)) / h
    
    #Central derivative method
    def central_difference_derivative(self, x, h):
        return (self.function(x + h) - self.function(x - h)) / (2 * h)
    
    #Derive function at second order

##TEST
def prova(x):
    return x*x
    
fun = Polyfunctions(prova)

result  = fun.evaluate(5)
result2 = fun.integrate_function(0,1,0.1) 
result3 = fun.forward_difference_derivative(1,0.1)
result4 = fun.central_difference_derivative(1,0.1)
print(result)
print(result2)
print(result3)
print(result4)


        