# pitchfork
*rapid emulation of individual radial modes of solar like oscillators using a branching neural network*
---
Accurately measuring the ages and internal structures of stars is tough!

We can use individual asteroseismic modes of oscillation to improve precision, but typically this comes at a hefty computational cost when interpolating grids to high dimenstions.

In this repo we present *pitchfork* - a branching multilayer perceptron capable of rapidly emulating the indivudual radial modes of a grid of solar-like oscillators.

For the remainder of this README, I'll present the relevant info regarding the training data for pitchfork and our prediction accuracy on a test set.

Then, we can take a look at the `inference-example.ipynb` notebook to go over our process of stellar parameter inference.

Hope you enjoy!

See the supporting publication here:
- arXiv: *url here*
- MNRAS: *url here*

## training data
Let's take a look at the data used to train *pitchfork*.

We use the grid of solar-like oscillators described in Lyttle et al. 2021[^Lyttle_2021], which I recommend checking out if you want more details. In fact, I recommend checking it out in general - it's a great paper!

[^Lyttle_2021]: Lyttle, Alexander J., et al. “Hierarchically Modelling Kepler Dwarfs and Subgiants to Improve Inference of Stellar Properties with Asteroseismology.” \mnras, vol. 505, no. 2, Aug. 2021, pp. 2427–46, https://doi.org/10.1093/mnras/stab1368.

Without going into too much detail, the stellar models were generated using MESA **REF**, and the asteroseismic terms were generated with GYRE **REF**.

This means we supply inputs (stellar fundamental properties) of **M, T, Zini, Yini, and MLT** and end up with outputs (observables) **L, Fe/H, Teff** (the *classical* obserables), and a host of individual radial modes of radial orders **$6<n<40$** (the *classical* observables) for each of these models.

Because we are trying to emulate the behaviour of the stellar evolution codes, we use these inputs and outputs for *pitchfork* too.

The grid contains **X** tracks for a total of **X** stellar models, for which we set aside 5% for both the testing and validation sets.

A summary plot of the input distributions is shown below:

