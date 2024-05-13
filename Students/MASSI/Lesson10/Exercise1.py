import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Define the step size
dx = 1  # Change this value to your desired step size

# densità uniforme
delta = 1

a = 0
b = 10

# Calculate the number of steps
num_steps = int((b - a) / dx) + 1


x = np.linspace(a, b, num_steps)

X, Y, Z = np.meshgrid(x, x, x)
D = np.full(X.shape, delta)


# Create a 3D figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the densities
ax.scatter(X, Y, Z, c=D, cmap='viridis')

# Set labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Uniform Densities in 3D Space')

# Add colorbar
cbar = plt.colorbar(ax.scatter(X, Y, Z, c=D, cmap='viridis'))
cbar.set_label('Density')

plt.show()

# Compute the Fourier transform
fourier_transform = np.fft.rfftn(D)
#shape=fourier_transform.shape()

# Get the magnitude of Fourier coefficients
magnitude = np.abs(fourier_transform)

# Plot magnitude without log scale
ax.scatter(X.flatten(), Y.flatten(), Z.flatten(), c=magnitude.flatten(), cmap='viridis')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Fourier Transform of Grid')

plt.show()