import camb
from camb import model, initialpower

# Set up a set of cosmological parameters
pars = camb.CAMBparams()
pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122)
pars.InitPower.set_params(As=2e-9, ns=0.965)

# Set up the redshifts at which to calculate outputs
z = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]

# Calculate results for these parameters
results = camb.get_results(pars)
print(results.get_unlensed_scalar_array_cls(1000))

# Get matter power spectra at the desired redshifts
# kha, z, pk = results.get_matter_power_spectrum(minkh=1e-4, maxkh=1, npoints=200)

# You can now use 'kha', 'z', and 'pk' to plot the power spectrum, or do other analyses
