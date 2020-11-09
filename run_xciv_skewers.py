"""
Generates x_metal skewers for corresponding Nyx skewers and save the results
"""


from cloudy_runs.cloudy_utils import read_cloudy_koki
from metal_frac_skewers import make_xmetal_skewer
from astropy.table import Table
import time

data_path = '/Users/suksientie/research/CIV_forest/nyx_sim_data/'
cloudy_path = '/Users/suksientie/research/CIV_forest/cloudy_runs/'
outfile = data_path + 'rand_skewers_z45_ovt_tau_xciv.fits'

tau_params = Table.read(data_path + 'rand_skewers_z45_ovt_tau.fits', hdu=1)
tau_skewers = Table.read(data_path + 'rand_skewers_z45_ovt_tau.fits', hdu=2)
cloudy_lookup = read_cloudy_koki(cloudy_path + 'output/cloudy_grid_more')

start = time.time()
#make_xciv_skewer(tau_params, tau_skewers, cloudy_lookup, outfile)
make_xmetal_skewer(tau_params, tau_skewers, cloudy_lookup, 'IONI CARB 4 1', 'X_CIV', outfile)
end = time.time()

print('Done...', (end-start)/60.)