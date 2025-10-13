import numpy as np
import ultranest

import tinygp
from tinygp import GaussianProcess
from tinygp import kernels

import jax

jax.config.update('jax_default_device', jax.devices('cpu')[0])

import json
import pickle

from .utils import beta_prior, uniform_prior

class pitchfork_sampler():
    def __init__(self, pitchfork, pitchfork_cov, priors=None, logl_scale=1):
        self.pitchfork = pitchfork
        self.pitchfork_cov = pitchfork_cov
        self.logl_scale = logl_scale

        if not priors:
            mass_prior = beta_prior(0.8, 1.2, a=5, b=2)
            Zinit_prior = beta_prior(0.004, 0.038, a=2, b=5)
            Yinit_prior = beta_prior(0.24, 0.32, a=2, b=5)
            MLT_prior = beta_prior(1.7, 2.5, a=1.2, b=1.2)
            age_prior = beta_prior(0.03, 14, a=1.2, b=1.2)
            a_prior = uniform_prior(-10, 2)
            b_prior = uniform_prior(4.4, 5.25)
            
            self.priors = [mass_prior, Zinit_prior, Yinit_prior, MLT_prior, age_prior, a_prior, b_prior]

        else:
            self.priors = priors
            
        
    def load_star_data(self, star_name, gp_var = 4, gp_ls_factor = 7):
        star_data_filepath = f'stars/{star_name}/{star_name}.json'
        
        # -----------------------
        # compute covariance matrix
        # -----------------------
        with open(star_data_filepath, 'r') as fp:
            star_dict = json.load(fp)
        
        ## unpacking dict
        def unpack_star_dict(star_dict, keys):
            values = [star_dict[key][0] for key in keys]
            uncs = [star_dict[key][1] for key in keys]
            return values, uncs
        
        self.nu_max = star_dict['nu_max'][0]
        self.dnu = star_dict['dnu'][0]
        
        classical_keys = ['calc_effective_T', 'luminosity', 'star_feh']
        classical_vals, classical_uncs = unpack_star_dict(star_dict, classical_keys)
        
        nu_keys = [key for key in list(star_dict.keys()) if 'nu_0_' in key]
        frequency_vals, frequency_uncs = unpack_star_dict(star_dict, nu_keys)
        
        n_ints = [int(nu_key.replace('nu_0_','')) for nu_key in nu_keys]
        self.n_min = n_ints[0]
        self.n_max = n_ints[-1]
        
        ###------------- sigma calc ----------------------
        ## sigma_obs
        self.obs_vals = np.array(classical_vals + frequency_vals)
        obs_uncs = np.array(classical_uncs + frequency_uncs)
        obs_var = obs_uncs * obs_uncs
        sigma_obs = (obs_var)*(np.identity(len(obs_var)))
        
        ## sigma_psi
        pitchfork_pred_idxs = [0,1,2]+[n_idx for n_idx in range(self.n_min-3, self.n_max-2)]
        sigma_psi = self.pitchfork_cov[np.ix_(pitchfork_pred_idxs, pitchfork_pred_idxs)]
        
        ## sigma_gp
        gp_kernel = gp_var*kernels.ExpSquared(scale=gp_ls_factor*self.dnu)
        gp_noise = tinygp.noise.Dense(value=np.zeros((len(frequency_vals),len(frequency_vals))))
        gp_cov = tinygp.solvers.DirectSolver.init(gp_kernel, np.array(frequency_vals), noise=gp_noise).covariance()
        sigma_gp = np.pad(gp_cov, (3,0))
        
        ## sigma
        sigma = sigma_obs + sigma_psi + sigma_gp
        ###------------------------------------------------

        _, log_sigma_det = np.linalg.slogdet(sigma)
        
        self.logl_constant = -(len(self.obs_vals)*0.5*np.log(2*np.pi))-(0.5*log_sigma_det)   
        self.sigma_inv = np.linalg.inv(sigma)
        self.matmul_path = np.einsum_path('ij, jk, ik->i', np.array(self.obs_vals).reshape(-1,1), self.sigma_inv, np.array(self.obs_vals).reshape(-1,1), optimize='optimal')[0]

    def ptform(self, u):
        theta = np.array([self.priors[i].ppf(u[:,i]) for i in range(len(self.priors))]).T
        return theta

    def surface_correction(self, freqs, a, b):
        return freqs + a*((freqs/self.nu_max)**b)

    def logl(self, theta):
        
        preds = self.pitchfork.predict(theta[:,:-2], n_min =self.n_min, n_max = self.n_max)

        a_arr = np.expand_dims(theta[:,-2],1)

        b_arr = np.expand_dims(theta[:,-1],1)

        preds[:,3:] = self.surface_correction(preds[:,3:], a_arr, b_arr)

        residual_matrix = np.array(preds - self.obs_vals)

        ll = self.logl_constant-0.5*np.einsum('ij, jk, ik->i', residual_matrix, self.sigma_inv, residual_matrix, optimize=self.matmul_path)

        return self.logl_scale * ll

    def __call__(self, star_name, ndraw_min=2**15, ndraw_max=2**15, draw_multiple=True, save=False, **run_kwargs):
        
        self.load_star_data(star_name)
        
        self.sampler = ultranest.ReactiveNestedSampler(['initial_mass', 'initial_Zinit', 'initial_Yinit', 'initial_MLT', 'star_age','a','b'], 
                                                       self.logl, 
                                                       transform = self.ptform, 
                                                       vectorized=True, 
                                                       ndraw_min=ndraw_min, 
                                                       ndraw_max=ndraw_max,
                                                       draw_multiple=draw_multiple,
                                                      )
        
        results = self.sampler.run(**run_kwargs)

        results['priors'] = self.priors
        
        if save:
            with open(f'stars/{star_name}/{star_name}_results.pkl', 'wb') as fp:
                pickle.dump(results, fp, protocol=pickle.HIGHEST_PROTOCOL)
        
        return results