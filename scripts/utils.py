import scipy
import corner
import pickle
import matplotlib.pyplot as plt

## prior funcs
def uniform_prior(prior_min, prior_max):
    return scipy.stats.uniform(loc=prior_min, scale=prior_max-prior_min)

def beta_prior(prior_min, prior_max, a=1, b=1):
    return scipy.stats.beta(loc=prior_min, scale=prior_max-prior_min, a=a, b=b)

## plotting funcs

def inspect_star(star_name, color = '#DC322F', include_prior = True, n_prior_samples = 10000):
    
    labels = ['initial_mass', 'initial_Zinit', 'initial_Yinit', 'initial_MLT', 'star_age', 'a', 'b']
    
    with open(f'stars/{star_name}/{star_name}_results.pkl', 'rb') as fp:
        results = pickle.load(fp)
        
    if include_prior:
        priors = results['priors']
        prior_samples = np.array([prior.rvs(size=n_prior_samples) for prior in priors])
        
        figure = corner.corner(prior_samples.T, labels = labels, color='black', hist_kwargs={'density':True}, smooth=True);
    
        corner.corner(results['samples'], fig=figure, color=color, hist_kwargs={'density':True}, smooth=True,show_titles=True);

        plt.show()

    else:
        figure = corner.corner(results['samples'], color=color, hist_kwargs={'density':True}, smooth=True,show_titles=True);
        
        plt.show()