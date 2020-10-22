import numpy as np
from matplotlib import pyplot as plt
import cloudy_utils as cu

Z = -3.5
nh_bar_log10 = -4.5 # mean density
temp_log10 = 4.0 # temp at mean density
metal_ion_ls = ['IONI CARB 1 1', 'IONI CARB 2 1', 'IONI CARB 3 1', 'IONI CARB 4 1', 'IONI CARB 5 1', 'IONI CARB 6 1']
#metal_ion_ls = ['IONI SILI 1 1', 'IONI SILI 2 1', 'IONI SILI 3 1', 'IONI SILI 4 1', 'IONI SILI 5 1', 'IONI SILI 6 1']
#out_filename = 'carbon_ion_frac2d_Z_-3.5.pdf'
out_filename = '../plots/debug_carbon_ion_frac1d.pdf'

met_lookup = cu.read_cloudy_koki('output/cloudy_grid')

plt.figure(figsize=(10,8))
for i in range(len(metal_ion_ls)):
    selected_ion_frac, nh_grid, temp_grid = cu.get_ion_frac1d(met_lookup, metal_ion_ls[i], Z)
    selected_ion_frac = np.reshape(selected_ion_frac, (61, 31))

    plt.subplot(2,3,i+1)
    plt.title(metal_ion_ls[i], fontsize=15)
    plt.imshow(selected_ion_frac, vmin=0, vmax=1, origin='lower', extent=[3, 6, -7, -1]) # 3 to 6 are the grid limits for log10(T)
                                                                         # -7 to -1 are the grid limits for log10(hden)
    plt.plot(temp_log10, nh_bar_log10, 'r*', ms=10)
    plt.colorbar()
    #plt.xticks(np.arange(3, 6.5, 0.5))
    #plt.yticks(np.arange(-7, -0.5, 0.5))
    plt.xlabel('log10(T)', fontsize=13)
    plt.ylabel('log10(hden)', fontsize=13)

plt.tight_layout()
#plt.show()
plt.savefig(out_filename)
plt.clf()