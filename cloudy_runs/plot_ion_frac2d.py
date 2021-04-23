import numpy as np
from matplotlib import pyplot as plt
import cloudy_utils as cu

Z = -3.5
nh_bar_log10 = -4.5 # mean density at z=4.5
temp_log10 = 4.0 # temp at mean density

#Z = None
#nh_bar_log10 = np.log10(6.52846623340864e-05) # z=5.1
#temp_log10 = np.log10(6000)

#out_filename = 'carbon_ion_frac2d_Z_-3.5.pdf'
out_filename = 'plots/oxy_frac2d.pdf'

met_lookup = cu.read_cloudy_koki('output/cloudy_grid_more')

metal_ion_ls = ['IONI HYDR 1 1', 'IONI HYDR 2 1']
metal_ion_ls = ['IONI CARB 2 1', 'IONI CARB 3 1', 'IONI CARB 4 1', 'IONI CARB 5 1', 'IONI CARB 6 1']#, 'IONI CARB 7 1']
#metal_ion_ls = ['IONI SILI 2 1', 'IONI SILI 3 1', 'IONI SILI 4 1', 'IONI SILI 5 1', 'IONI SILI 6 1', 'IONI SILI 7 1']
#metal_ion_ls = ['IONI NITR 2 1', 'IONI NITR 3 1', 'IONI NITR 4 1', 'IONI NITR 5 1', 'IONI NITR 6 1', 'IONI NITR 7 1']
#metal_ion_ls = ['IONI OXYG 2 1', 'IONI OXYG 3 1', 'IONI OXYG 4 1', 'IONI OXYG 5 1', 'IONI OXYG 6 1', 'IONI OXYG 7 1']

title = ['$C\ I$', '$C\ II$', '$C\ III$', '$C\ IV$', '$C\ V$', '$C\ VI$']
plt.figure(figsize=(12,8))
for i in range(len(metal_ion_ls)):
    selected_ion_frac, nh_grid, temp_grid = cu.get_ion_frac1d(met_lookup, metal_ion_ls[i], Z)

    # getting the total number, lower bound, and higher bound of the gridpoints
    N_nh_grid, nh_grid_lo, nh_grid_hi = len(np.unique(nh_grid)), np.unique(nh_grid)[0], np.unique(nh_grid)[-1]
    print(N_nh_grid, nh_grid_lo, nh_grid_hi)
    N_temp_grid, temp_grid_lo, temp_grid_hi = len(np.unique(temp_grid)), np.unique(temp_grid)[0], np.unique(temp_grid)[-1]
    print(N_temp_grid, temp_grid_lo, temp_grid_hi)

    # double-checked reshaping is correct
    selected_ion_frac = np.reshape(selected_ion_frac, (N_nh_grid, N_temp_grid))

    plt.subplot(2,3,i+1)
    plt.title(title[i], fontsize=15)

    # origin='lower' means (0,0) is at lower left
    # extent defines the limit of the axes labels
    plt.imshow(selected_ion_frac.transpose(), vmin=0, vmax=1, origin='lower', \
               extent=[nh_grid_lo, nh_grid_hi, temp_grid_lo, temp_grid_hi], \
               interpolation='none')
    plt.colorbar()
    plt.plot(nh_bar_log10, temp_log10, 'r*', ms=10)
    plt.ylabel(r'log$_{10}T (K)$', fontsize=13)
    plt.xlabel(r'log$_{10}n_H$', fontsize=13)

plt.tight_layout()
plt.show()
#plt.savefig(out_filename)
#plt.clf()
