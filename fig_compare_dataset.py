from matplotlib import pyplot as plt
import numpy as np
import metal_corrfunc as mcf
from enigma.reion_forest.compute_model_grid import read_model_grid

# Compare the mean correlation function for different datasets with different spectral resolution

path = '/Users/suksientie/Research/CIV_forest/nyx_sim_data/'

hires_modelfile = path + 'igm_cluster/corr_func_models_fwhm_10.000_samp_3.000_SNR_50.000_nqsos_20.fits' # R=30k
xshooter_modelfile = path + 'igm_cluster/corr_func_models_fwhm_30.000_samp_3.000_SNR_50.000_nqsos_20.fits' # R=10k
deimos_modelfile = path + 'igm_cluster/corr_func_models_fwhm_60.000_samp_3.000_SNR_50.000_nqsos_20.fits' # R=5k
iZ = 12

params_deimos, xi_mock_array_deimos, xi_model_array_deimos, _, _, _ = read_model_grid(deimos_modelfile)
params_xshooter, xi_mock_array_xshooter, xi_model_array_xshooter, _, _, _ = read_model_grid(xshooter_modelfile)
params_hires, xi_mock_array_hires, xi_model_array_hires, _, _, _ = read_model_grid(hires_modelfile)

logZ_coarse = np.array(params_deimos['logZ'].flatten())

mcf.plot_corrfunc(params_hires, xi_model_array_hires[0][iZ], label='hires (fwhm=%0.1f)' % params_hires['fwhm'])
mcf.plot_corrfunc(params_xshooter, xi_model_array_xshooter[0][iZ], label='xshooter (fwhm=%0.1f)' % params_xshooter['fwhm'])
mcf.plot_corrfunc(params_deimos, xi_model_array_deimos[0][iZ], label='deimos (fwhm=%0.1f)' % params_deimos['fwhm'])
plt.title(r'logZ = $%0.2f$' % logZ_coarse[iZ], fontsize=17)
plt.xlim([0, 1200])
plt.show()


