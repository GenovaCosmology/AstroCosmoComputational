import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt


# Define the rejection sampling algorithm 1D
def rejection_sampling1D(target_pdf, xmin, xmax, num_samples):
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
        if y < target_pdf(x):
            samples.append(x)
    return np.array(samples)



# Define the rejection sampling algorithm 2D
def rejection_sampling2D(target_pdf, xmin, xmax, ymin, ymax, num_samples):
    samplesx = []
    samplesy = []
    # Trova il massimo della distribuzione target:
    # Define the range
    x_values = np.linspace(xmin, xmax, 1000)
    y_values = np.linspace(ymin, ymax, 1000)

    # Evaluate the PDF function within the range
    pdf_values = target_pdf(x_values, y_values)

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