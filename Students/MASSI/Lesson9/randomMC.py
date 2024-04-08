import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

# Define the target distribution (e.g., a normal distribution)
def target_distribution1(x):
    return np.exp(-(x) ** 2 / 2) / np.sqrt(2 * np.pi)

def integra(x):
    return np.exp(-(x) ** 4 / 2)

N, error = quad(integra, -np.inf, np.inf)
def target_distribution2(x):
    return np.exp(-(x) ** 4 / 2) * (1/N)



# Define the rejection sampling algorithm
def rejection_sampling1D(target_pdf, xmin, xmax, num_samples):
    i = 0
    j = 0
    samples = []
    # Trova il massimo della distribuzione target:
    # Define the range
    x_values = np.linspace(xmin, xmax, 1000)

    # Evaluate the PDF function within the range
    pdf_values = target_pdf(x_values)

    # Find the maximum value
    max_pdf = np.max(pdf_values)

    # Allargo il box rispetto al max che ho trovato
    ymin = 0
    ymax = max_pdf + 0.1*(max_pdf)

    print("Maximum value of the PDF = ", max_pdf, ", Maximum value of Y_box = ", ymax)

    while len(samples) < num_samples:
        x = np.random.uniform(xmin, xmax)
        y = np.random.uniform(ymax, ymin)
        j = j + 1
        if y < target_pdf(x):
            i = i + 1
            samples.append(x)
    return np.array(samples), i/j

# Parameters
num_samples = 10000
xmin = -5
xmax = 5
x_values = np.linspace(xmin, xmax, 1000)


# Generate samples using rejection sampling
samples, efficiency = rejection_sampling1D(target_distribution2, xmin, xmax, num_samples)

print("efficienza: ",efficiency)

# Plot the results
plt.figure(figsize=(8, 6))
plt.plot(x_values, target_distribution2(x_values), label='Target Distribution')
plt.hist(samples, bins=100, density=True, alpha=0.4, label='Samples')           # 'Density = True' permette di trovare la densità e non i conteggi
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.title('Monte Carlo Rejection Sampling')
plt.legend()
plt.grid(True)
plt.show()
