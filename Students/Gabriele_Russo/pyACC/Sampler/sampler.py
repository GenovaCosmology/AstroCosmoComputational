import numpy as np

#**********************#
#Proposal Distributions#
#**********************#

#Uniform distribution
def uniform(x, min_x, max_x):
    if x >= min_x and x <= max_x:
        return 1 / (max_x - min_x) #to have normalization
    else:
        return 0
    
# Probability density function (PDF) of the standard normal distribution (mean 0, sigma 1)
def Gaussian(x):
    return np.exp(-0.5*x**2) / np.sqrt(2*np.pi)
    
#*******************#
#Sampling alghoritms#
#*******************#

# Function to generate random numbers using rejection sampling
def rejection_sampling(num_samples, x_range, y_range):
    '''
    This function wants:
    - num_samples: int (number of samples to generate)
    - x_range: tuple (needed to define x_min, x_max)
    - y_range: tuple (needed to define y_min, y_max)

    '''
    x_min, x_max = x_range
    y_min, y_max = y_range
    samples = []

    while len(samples) < num_samples:
        x = np.random.uniform(x_min, x_max)  # Generate a random number uniformly distributed in an appropriate range
        y = np.random.uniform(y_min, y_max)  # Generate another random variable uniformly distributed above the maximum of the PDF
        if y < Gaussian(x):
            samples.append(x)
    return samples
 