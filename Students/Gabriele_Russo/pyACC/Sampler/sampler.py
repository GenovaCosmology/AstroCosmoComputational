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

# Function to generate random numbers using rejection sampling in 1D
def rejection_sampling(num_samples, x_range, y_range):
    '''
    This function wants:
    - num_samples: int (number of samples to generate)
    - x_range: tuple (needed to define x_min, x_max)
    - y_range: tuple (needed to define y_min, y_max)

    Returns: 
    - samples: array (sampled points)
    - efficiency: int (how many points were generated in total)

    '''
    x_min, x_max = x_range
    y_min, y_max = y_range
    samples = []
    efficiency_counter = 0
    #Number of samples taken and efficiency
    while len(samples) < num_samples:
        x = np.random.uniform(x_min, x_max)  # Generate a random number uniformly distributed in an appropriate range
        y = np.random.uniform(y_min, y_max)  # Generate another random variable uniformly distributed above the maximum of the PDF
        efficiency_counter += 1
        if y <= Gaussian(x):
            samples.append(x)
    
    return samples, efficiency_counter


# Function to generate random numbers using rejection sampling in nD
def rejection_sampling_nd(num_samples, n, ranges, pdf_func):
    '''
    This function wants:
    - num_samples: int (number of samples to generate)
    - n: int (number of dimensions)
    - ranges: 2D numpy array (range for each dimension)
    - pdf_func: function (probability density function of the target distribution)

    Returns:
    - np.array(samples): array (sample points converted in an array)
    - efficiency: int (how many points were generated in total)
    '''
    samples = []
    efficiency_counter = 0
    while len(samples) < num_samples:
        point = np.random.uniform(low=ranges[:, 0], high=ranges[:, 1], size=n)
        '''
        This line generates a random point in n dimensions within the specified ranges. 
        It uses np.random.uniform to generate random numbers uniformly distributed between the low and high values 
        specified in the ranges array for each dimension.
        '''
        efficiency_counter += 1
        if np.random.uniform(0, 1) < pdf_func(*point):
            samples.append(point)
    return np.array(samples), efficiency_counter
 