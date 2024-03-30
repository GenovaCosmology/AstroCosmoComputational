import numpy as np
import scipy as sp

def rejection_sampling(target, domain, n_samples, proposal=None):
    """
    Perform rejection sampling to sample from a probability density function (PDF).

    Parameters:
    - target (callable): The PDF function from which samples are to be generated.
    - domain (array_like): The domain (range of values) over which to sample the PDF.
    - n_samples (int): The number of samples to generate.
    - proposal (scipy.stats frozen random variable, optional): The proposed distribution for sampling.
      Default is uniform distribution over the specified domain.

    Returns:
    - sample (numpy.ndarray): An array containing accepted samples from the target PDF.

    Notes:
    - The function performs rejection sampling by generating samples from the proposal distribution
      and accepting them based on the ratio of the target PDF to the proposal PDF.
    - The acceptance rate is printed to the console for monitoring purposes.
    """

    # Set default proposal distribution if not provided
    if proposal is None:
        proposal = sp.stats.uniform(loc=domain[0], scale=domain[-1] - domain[0])

    # Calculate the scaling factor for the proposal
    
    k = np.max(target(domain) / proposal.pdf(domain))
    print('scaling factor= ', k)
    # Generate samples from the proposal distribution
    xs = proposal.rvs(size=n_samples)
    
    # Generate random comparison values
    cs = np.random.uniform(0, 1, size=n_samples)

    # Acceptance condition
    mask = (k * proposal.pdf(xs)) * cs < target(xs)

    # Print statistics for monitoring
    print('Attempted samples:', n_samples)
    print('Accepted samples:', np.sum(mask))
    print('Acceptance rate:', np.sum(mask) / n_samples)

    norm_proposal = lambda x: k*proposal.pdf(x)

    return xs[mask], norm_proposal

def inverse_transform_sampling():
    pass


