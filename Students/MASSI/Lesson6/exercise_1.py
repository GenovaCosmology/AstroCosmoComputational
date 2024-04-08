import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import dblquad
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from scipy.optimize import minimize
from scipy.stats import norm



L1 = 1
L2 = 5
def pdf(x, y, lambda1, lambda2, N):
    
    # Compute the unnormalized PDF
    pdf_unnormalized = np.exp(-(x-0.5)**4 / lambda1 - y**4 / lambda2)

    return pdf_unnormalized/N


# define prior box: definisco la funzione nel box di estremi a,b; c,d
a = 4
b = -4
c = 4
d = -4


# sample your reference probability on a 2D regular grid in the prior box
# Generate grid points
x = np.linspace(b, a, 100)
y = np.linspace(d, c, 100)
X, Y = np.meshgrid(x, y)

Z = pdf(X, Y, L1, L2, 1)
print(Z)


# normalize the pdf
x_lower = -np.inf  # Lower limit of integration for x
x_upper = np.inf
y_lower = -np.inf
y_upper = np.inf


# perform the double integration
N, _ = dblquad(pdf, x_lower, x_upper, lambda x: y_lower, lambda x: y_upper, args=(L1,L2,1))   # args è utile per passare i parametri della funzione pdf
print('integrale pdf non norm = ',N)


integral2, __ = dblquad(pdf, x_lower, x_upper, lambda x: y_lower, lambda x: y_upper, args=(L1,L2,N))
print('integrale pdf_norm = ' , integral2)


# contour plot the distribution (on generic iso-contours, the default ones work)


Z1 = pdf(X, Y, L1, L2, N)

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z1, cmap='viridis')

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('PDF Value')
ax.set_title('3D PDF (Non-Random, Non-Gaussian)')

# Show plot
plt.show()


# Plot Contour
plt.figure(figsize=(5, 4))
plt.contour(x, y, Z1, cmap='viridis')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Contour Plot of 2D PDF')
plt.colorbar(label='Probability Density')
plt.grid(True)
plt.show()


