"""Gaussian process fitting a mean function using a Gaussian Process"""

import george
from george import kernels
from george.modeling import Model as Model_George
import emcee
import numpy as np
from multiprocessing import Pool
from scipy.optimize import minimize

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


def simple_GP(X, Y, YERR,  ell=10, Niter=1):

    #TODO: what kernel and what scale length to choose?                        
    gp = george.GP(np.var(Y) * kernels.Matern32Kernel(ell))
    
    gp.compute(X, YERR) 
    init = gp.get_parameter_vector()

    def neg_ln_like(p):
        gp.set_parameter_vector(p)
        return -gp.log_likelihood(Y)

    def grad_neg_ln_like(p):
        gp.set_parameter_vector(p)
        return -gp.grad_log_likelihood(Y)
        
    for _ in range(Niter):
        result = minimize(neg_ln_like, gp.get_parameter_vector(), jac=grad_neg_ln_like)
        gp.set_parameter_vector(result.x)

   
    # model predictions
    _x = np.linspace(min(X), max(X), 5_000)
    y_pred, y_var = gp.predict(Y, _x, return_var=True)

    return _x, y_pred, y_var, result


def model_gp_minimize(X, Y, YERR, window_start, window_end, i0, ell=10):
    """Calculate the Gaussian process on the observed data using scipy minimization and fitting a mean function."""
    
    kwargs = dict(**i0)
    kwargs["bounds"] = dict(m=(-np.inf, +np.inf), 
                                b=(-np.inf, +np.inf), 
                                amp=(-np.inf, +np.inf), 
                                location=(-np.inf, +np.inf), # adding boundaries to the location...
                                sigma=(-np.inf, +np.inf),
                                log_sigma2=(-np.inf, +np.inf)) 

    mean_model = Model(**kwargs)
                        
    #TODO: what kernel and what scale length to choose?                        
    gp = george.GP(np.var(Y) * kernels.Matern32Kernel(ell), mean=mean_model)
    
    gp.compute(X, YERR) 
    init = gp.get_parameter_vector()

    def neg_ln_like(p):
        """Negative log-likelihood."""
        gp.set_parameter_vector(p)
        return -gp.log_likelihood(Y, quiet=True)

    bnds = ((0, None), (0, None), (0, 5), (window_start, window_end), (1, 1000), (0, 10))
    result = minimize(neg_ln_like, gp.get_parameter_vector(), bounds=bnds) # containts the fit information
    gp.set_parameter_vector(result.x)

    # model predictions
    _x = np.linspace(min(X), max(X), 5_000)
    y_pred, y_var = gp.predict(Y, _x, return_var=True)

    return _x, y_pred, y_var, result


def model_gp(X, Y, YERR, window_start, window_end, i0, ell=10):
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

    try:
        kwargs = dict(**i0)
        kwargs["bounds"] = dict(m=(None, None), 
                                    b=(None, None), 
                                    amp=(0.01, 10000), 
                                    location=(window_start-10, window_end+10), # adding boundaries to the location...
                                    sigma=(0, 1000),
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

        with Pool() as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob2_new, pool=pool)
            
            ## MCMC might need adjusting on nsamplers
            p0, lp , _ = sampler.run_mcmc(p0, 500)
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

        x = np.linspace(min(X), max(X), 5_000)

        gp.set_parameter_vector(arg_mu)
        model_best = gp.predict(Y, x, return_var=True) #gp.sample_conditional(Y, x)

        return x, model_best[0], model_best[1], {
            "log-like":  gp.log_likelihood(Y, quiet=True),
            "amp_median": np.median(samples[:, 0]),
            "amp_std": np.std(samples[:, 0]),
            "location_median": np.median(samples[:, 1]),
            "location_std": np.std(samples[:, 1]),
            "sigma_median": np.median(samples[:, 2]),
            "sigma_std": np.std(samples[:, 2]),
            "log_sigma2_median": np.median(samples[:, 3]),
            "log_sigma2_std": np.std(samples[:, 3]),
            "m_median": np.median(samples[:, 4]),
            "m_std": np.std(samples[:, 4]),
            "b_median": np.median(samples[:, 5]),
            "b_std": np.std(samples[:, 5])
        },
    except: 
        return None, None, None, None
