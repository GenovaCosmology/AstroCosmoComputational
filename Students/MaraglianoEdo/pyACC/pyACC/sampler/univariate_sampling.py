import numpy as np
import scipy as sp

def rejection_sampling(target, domain, n_samples, proposal=None):
    """
    Generate samples using rejection sampling.

    Parameters:
        target (callable): The target probability density function.
        domain (array_like): The domain of the target PDF.
        n_samples (int): The number of samples to generate.
        proposal (scipy.stats.rv_continuous): The proposal distribution. If None, a uniform distribution is used.

    Returns:
        numpy.ndarray: Generated samples.
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

def inverse_transform_sampling(pdf, x, n_samples=1000):
    """
    Generate samples using inverse transform sampling.
    
    Parameters:
        pdf (callable): Probability density function.
        x_values (array_like): Values of the variable corresponding to the PDF.
        n_samples (int): Number of samples to generate.
        
    Returns:
        numpy.ndarray: Generated samples.
    """

    # Generate random uniform samples
   
    u = np.random.uniform(0, 1, n_samples)
    
    pdf_val = pdf(x)
    # Compute the CDF

    cdf = np.array([np.trapz(pdf_val[:i], x[:i]) for i in range(1, len(x) + 1)])

    # Compute inverse CDF
    inverse_cdf = sp.interpolate.interp1d(cdf,x)
    
    # Use inverse CDF to get samples
    samples = inverse_cdf(u)
    
    return samples, cdf, inverse_cdf


