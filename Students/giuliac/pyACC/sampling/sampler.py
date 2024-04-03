import numpy as np

def rejection(f, q, c, x_min, x_max, N):
    """
    Function to sample a function using the rejection method.
    
    Parameters
    ----------
    f: function
        The target distribution you want to sample from.
    q: function
        The proposal distribution.
    c: float
        A constant such that c*q(x) >= f(x) for all x.
    x_min: float
        Minimum x value.
    x_max: float
        Maximum x value.
    N: int
        Number of samples to generate.
    """
    samples = []
    while len(samples) < N:
        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(0, c*q(x))
        if y < f(x):
            samples.append(x)
    return np.array(samples)

