import numpy as np
import scipy
import corner
import pickle
import matplotlib.pyplot as plt

plt.style.use("Solarize_Light2")
plt.rcParams.update({"axes.edgecolor": "black"})
plt.rcParams.update({"text.color": "black"})
plt.rcParams.update({"axes.labelcolor": "black"})
plt.rcParams.update({"xtick.color": "black"})
plt.rcParams.update({"ytick.color": "black"})
plt.rcParams.update({"font.family": "monospace"})

## prior funcs
def uniform_prior(prior_min, prior_max):
    return scipy.stats.uniform(loc=prior_min, scale=prior_max-prior_min)

def beta_prior(prior_min, prior_max, a=1, b=1):
    return scipy.stats.beta(loc=prior_min, scale=prior_max-prior_min, a=a, b=b)

## plotting funcs

def posterior_plot(results, star_name=None, color = '#D33682', include_prior = False, n_prior_samples = 10000):
    
    labels = ['initial_mass', 'initial_Zinit', 'initial_Yinit', 'initial_MLT', 'star_age', 'a', 'b']
    
    if include_prior:
        priors = results['priors']
        prior_samples = np.array([prior.rvs(size=n_prior_samples) for prior in priors])
        
        figure = corner.corner(prior_samples.T, labels = labels, color='black', hist_kwargs={'density':True}, smooth=True);
    
        corner.corner(results['samples'], fig=figure, color=color, hist_kwargs={'density':True}, smooth=True,show_titles=True);
        
        if star_name:
            plt.suptitle(f'posterior samples for {star_name}')
    
    else:
        figure = corner.corner(results['samples'], color=color, labels = labels, hist_kwargs={'density':True}, smooth=True,show_titles=True);
        
        if star_name:
            plt.suptitle(f'posterior samples for {star_name}')
            
    return figure