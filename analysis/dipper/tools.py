"""Tools and code to find dippers in ZTF data."""

from math import gamma
from scipy.special import erf

def ggd(x, mu, alpha, beta, ofs, amp):
    """
    Generalized Gaussian Distribution modified with an offset term and amplitude to fit the light curves.
    
    Parameters:
    -----------
    - x: array-like
        Input values.
    - mu: float
        Mean of the distribution.
    - alpha: float
        Shape parameter controlling the width of the distribution.
    - beta: float
        Shape parameter controlling the shape of the distribution.
    - ofs: float
        Offset term.
    - amp: float
        Amplitude term.
    
    Returns:
    - y: array-like
        Output values.
    """
    term_1 = beta/(2*alpha*gamma(1/beta))
    abs_term = ((abs(x-mu))/alpha)**beta
    return amp * term_1 * np.exp(-abs_term) + ofs


def depth(m1): 
    """
    Calculate the depth of a light curve.
    
    Parameters:
    - m1: array-like
        Magnitude values of the light curve.
    
    Returns:
    - depth: array-like
        Depth values of the light curve.
    """
    return 10**((m1.min() - m1) / 2.5)


def skew_norm(x, mu, sigma, alpha, ofs, amp):
    """
    Skew-normal distribution modified with an offset term and amplitude to fit the light curves.
    
    Parameters:
    - x: array-like
        Input values.
    - mu: float
        Mean of the distribution.
    - sigma: float
        Standard deviation of the distribution.
    - alpha: float
        Shape parameter controlling the skewness of the distribution.
    - ofs: float
        Offset term.
    - amp: float
        Amplitude term.
    
    Returns:
    - y: array-like
        Output values.
    """
    gaus = 1/np.sqrt(2*np.pi) * np.exp(-(x-mu)**2/sigma**2)
    FI = 0.5 * (1 + erf(alpha*(x-mu)/np.sqrt(2)))
    
    return ofs + amp*2*gaus*FI