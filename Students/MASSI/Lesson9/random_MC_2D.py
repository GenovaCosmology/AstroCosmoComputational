import numpy as np
from scipy.integrate import quad
from scipy.integrate import dblquad
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



# Define the target distribution (e.g., a normal distribution)
def target_distribution1(x,y):
    return np.exp(-x ** 2 / 2) / np.sqrt(2 * np.pi) * np.exp(-y ** 2 / 2) / np.sqrt(2 * np.pi)

def integra(x,y):
    return np.exp(-x ** 4 / 2) * np.exp(-y ** 4 / 2)

N, error = dblquad(integra, -np.inf, np.inf, -np.inf, np.inf)
def target_distribution2(x,y):
    return np.exp(-(x) ** 4 / 2) * np.exp(-y ** 4 / 2) * (1/N)



# Define the rejection sampling algorithm
def rejection_sampling2D(target_pdf, xmin, xmax, ymin, ymax, num_samples):
    samplesx = []
    samplesy = []
    # Trova il massimo della distribuzione target:
    # Define the range
    x_values = np.linspace(xmin, xmax, 1000)
    y_values = np.linspace(ymin, ymax, 1000)

    # Evaluate the PDF function within the range
    pdf_values = target_distribution2(x_values, y_values)

    # Find the maximum value
    max_pdf = np.max(pdf_values)

    # Allargo il box rispetto al max che ho trovato
    zmin = 0
    zmax = max_pdf + 0.1*(max_pdf)

    print("Maximum value of the PDF = ", max_pdf, ", Maximum value of Y_box = ", zmax)

    while len(samplesx) < num_samples:
        x = np.random.uniform(xmin, xmax)
        y = np.random.uniform(xmin, xmax)
        z = np.random.uniform(zmax, zmin)
        if z < target_pdf(x,y):
            samplesx.append(x)
            samplesy.append(y)
    return np.array(samplesx), np.array(samplesy)


# Parameters
num_samples = 50000

xmin = -5
xmax = 5

ymin = -5
ymax = 5


x_values = np.linspace(xmin, xmax, 1000)
y_values = np.linspace(ymin, ymax, 1000)
X, Y = np.meshgrid(x_values, y_values)
Z = target_distribution2(X, Y)

# Generate samples using rejection sampling
x, y = rejection_sampling2D(target_distribution2, xmin, xmax, ymin, ymax, num_samples)

# Plot the results
plt.figure(figsize=(8, 6))

# Overlay PDF
plt.contour(x_values, y_values, Z, levels=10, colors='r', linestyles='solid')
plt.hist2d(x, y, bins=30, cmap='viridis')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('2D Montecarlo Rejection')
plt.grid(True)
plt.show()



# Create 2D histogram
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

hist, xedges, yedges = np.histogram2d(x, y, bins=30)


# Construct arrays for the anchor positions of the bars.
xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
xpos = xpos.ravel()
ypos = ypos.ravel()
zpos = 0

# Construct arrays with the dimensions for the bars.
dx = dy = 0.5 * np.ones_like(zpos)
dz = hist.ravel()
ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')

# Plot analytical distribution
ax.plot_surface(X, Y, *Z, cmap='viridis', alpha=1)


# Add labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Frequency')
ax.set_title('3D Histogram')

plt.show()