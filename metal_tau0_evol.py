import numpy as np
from matplotlib import pyplot as plt
import cloudy_utils
import metal_frac_skewers

nh_log10 = -4.5 # mean density
temp_log10 = np.log10(6000) # temp at mean density

cldy_outpath = 'cloudy_runs/output/'
cldy_outfiles = [cldy_outpath + 'cloudy_grid_more', cldy_outpath + 'cloudy_grid_z4p8', cldy_outpath + 'cloudy_grid_z5p1', \
                 cldy_outpath + 'cloudy_grid_z5p4', cldy_outpath + 'cloudy_grid_z5p7', cldy_outpath + 'cloudy_grid_z6']
z_range = [4.5, 4.8, 5.1, 5.4, 5.7, 6.0] # redshifts of the cloudy models

metal_ion_ls = ['C IV', 'Si IV', 'N V', 'O VI']
c4_tau0f = []
si4_tau0f = []
n5_tau0f = []
o6_tau0f = []

all_tau0f = []
for i, file in enumerate(cldy_outfiles):
    lookup = cloudy_utils.read_cloudy_koki(file, verbose=False)
    z = z_range[i]
    t0f = metal_frac_skewers.get_tau0_frac(lookup, nh_log10, temp_log10, z, logZ=-3.5)
    all_tau0f.append(t0f)

for i in range(len(z_range)):
    plt.plot(all_tau0f[i], 'o')

plt.show()
