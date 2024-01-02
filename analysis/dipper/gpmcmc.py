"""Gaussian process fitting a mean function using a Gaussian Process"""

import george
from george import kernels
from george.modeling import Model as Model_George
import emcee
import numpy as np


class Model(Model_George):
    """Gaussian model.
    
    Parameters
    ----------
    amp : float
        Amplitude of the Gaussian.
    location : float
        Location of the Gaussian.
    log_sigma2 : float    
    """
    parameter_names = ("amp", "location", "sigma","log_sigma2")
    
    def get_value(self, t): 
        return self.amp * np.exp(-0.5*(t.flatten()-self.location)**2/(self.sigma**2) * np.exp(-self.log_sigma2))


class PolynomialModel(Model_George): 
    """Polynomial model: linear function plus Gassian.
    
        Parameters
        ----------
        m : float
            Slope of the linear function.
        b : float
            Intercept of the linear function.
        amp : float
            Amplitude of the Gaussian.
        location : float
            Location of the Gaussian.
        log_sigma2 : float    
    
    """
    parameter_names = ("m", "b", "amp", "location", "sigma", "log_sigma2")

    def get_value(self, t):
        t = t.flatten()
        return (t * self.m + self.b + 
               self.amp * np.exp(-0.5*(t-self.location)**2/(self.sigma**2) * np.exp(-self.log_sigma2)))


def model_gp(X, Y, YERR, window_start, window_end, i0, ell=5):
    """Reuturn the Gaussian process data....
    
    Parameters
    ----------
    X : array_like
        Time array.
    Y : array_like
        Magnitude array.
    YERR : array_like
        Magnitude error array.
    i0 : dict (containing initial parameters)
    """
    
    kwargs = dict(**i0)
    kwargs["bounds"] = dict(m=(None, None), 
                                b=(None, None), 
                                amp=(None, None), 
                                location=(window_start, window_end), # adding boundaries to the location...
                                sigma=(None, None)
                                log_sigma2=(None, None)) 
    mean_model = Model(**kwargs)
                           
    #TODO: what kernel and what scale length to choose?                        
    gp = george.GP(np.var(Y) * kernels.Matern32Kernel(ell), mean=mean_model)
    
    gp.compute(X, YERR) 
        
    def lnprob2_new(p):
        gp.set_parameter_vector(p)
        return gp.log_likelihood(Y, quiet=True) + gp.log_prior()
    
    init = gp.get_parameter_vector()
    ndim, nwalkers = len(init), 32
    
    p0 = init + 1e-8 * np.random.rand(nwalkers, ndim)
    
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob2_new)
    
    ## MCMC might need adjusting on nsamplers
    p0, lp , _ = sampler.run_mcmc(p0, 200)
    sampler.reset()

    p0 = p0[np.argmax(lp)] + 1e-8 * np.random.randn(nwalkers, ndim)
    sampler.reset()
    
    #p0, lp , _ = sampler.run_mcmc(p0, 1_000)
    #sampler.reset()

    sampler.run_mcmc(p0, 1_000)
    
    samples = sampler.flatchain # fetch the flatchain samples
    
    ## What to return? ##
    arg_mu = []
    for i in range(6):
         arg_mu.append(np.median(samples[:,i]))
            
    x = np.linspace(min(X), max(X), 1_000)
    
    gp.set_parameter_vector(arg_mu)
    model_best = gp.sample_conditional(Y, x)
    
    return x, model_best
