import numpy as np
from matplotlib import pyplot as plt
import cloudy_utils as cu

Z = -3.5
nh_bar_log10 = -4.5 # mean density at z=4.5
temp_log10 = 4.0 # temp at mean density

met_lookup = cu.read_cloudy_koki('output/cloudy_grid_more')
metal_ion_ls = ['IONI CARB 2 1', 'IONI CARB 3 1', 'IONI CARB 4 1', 'IONI CARB 5 1', 'IONI CARB 6 1', 'IONI CARB 7 1']


title = ['$C\ I$', '$C\ II$', '$C\ III$', '$C\ IV$', '$C\ V$', '$C\ VI$']

vmin = -3.0
vmax = 0

plt.figure(figsize=(12,8))
for i in range(len(metal_ion_ls)):
    plt.subplot(2, 3, i + 1)
    plt.title(title[i], fontsize=15)

    metal_ind = np.where(met_lookup['METALS= %'] == Z)[0]
    nh_grid = np.array(met_lookup['HDEN=%f L'][metal_ind])  # log10 unit
    temp_grid = np.array(met_lookup['CONSTANT'][metal_ind])  # log10 unit
    ion_frac = np.array(met_lookup[metal_ion_ls[i]][metal_ind])

    #plt.scatter(np.log10(met_lookup['hden']), np.log10(met_lookup['Te']), c=met_lookup[metal_ion_ls[i]], s=20, vmin=vmin, vmax=vmax)
    plt.scatter(nh_grid, temp_grid, c=np.log10(ion_frac), s=8, vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.plot(nh_bar_log10, temp_log10, 'r*', ms=10)
    plt.ylabel(r'log$_{10}T (K)$', fontsize=13)
    plt.xlabel(r'log$_{10}n_H$', fontsize=13)

plt.tight_layout()
plt.show()
#plt.savefig(out_filename)