import numpy as np
from matplotlib import pyplot as plt
import cloudy_utils as cu

# plotting the relative difference in ion fraction between logZ=-3.5 and logZ=-1.5 in the Cloudy outputs

metal_ion_ls = ['IONI CARB 1 1', 'IONI CARB 2 1', 'IONI CARB 3 1', 'IONI CARB 4 1', 'IONI CARB 5 1', 'IONI CARB 6 1']
#metal_ion_ls = ['IONI SILI 1 1', 'IONI SILI 2 1', 'IONI SILI 3 1', 'IONI SILI 4 1', 'IONI SILI 5 1', 'IONI SILI 6 1']
out_filename = 'plots/carb_reldiff.pdf'

met_lookup = cu.read_cloudy_koki('output/cloudy_grid_more')

plt.figure(figsize=(10,8))
for i in range(len(metal_ion_ls)):
    ion1_frac, nh_grid, temp_grid = cu.get_ion_frac1d(met_lookup, metal_ion_ls[i], -3.5) # logZ = -3.5
    ion2_frac, nh_grid, temp_grid = cu.get_ion_frac1d(met_lookup, metal_ion_ls[i], -1.5) # logZ = -1.5

    # getting the total number, lower bound, and higher bound of the gridpoints
    N_nh_grid, nh_grid_lo, nh_grid_hi = len(np.unique(nh_grid)), np.unique(nh_grid)[0], np.unique(nh_grid)[-1]
    N_temp_grid, temp_grid_lo, temp_grid_hi = len(np.unique(temp_grid)), np.unique(temp_grid)[0], np.unique(temp_grid)[
        -1]

    ion1_frac = np.reshape(ion1_frac, (N_nh_grid, N_temp_grid))
    ion2_frac = np.reshape(ion2_frac, (N_nh_grid, N_temp_grid))

    #ion1, nh_grid, temp_grid = rc.get_ion_frac(met_lookup, metal_ion_ls[i], -3.5)
    #ion1 = np.reshape(ion1, (61, 31))
    #ion2, nh_grid, temp_grid = rc.get_ion_frac(met_lookup, metal_ion_ls[i], -1.5)
    #ion2 = np.reshape(ion2, (61, 31))

    diff = (ion1_frac - ion2_frac)/ion1_frac

    plt.subplot(2,3,i+1)
    plt.title(metal_ion_ls[i], fontsize=12)
    plt.imshow(diff, extent=[temp_grid_lo, temp_grid_hi, nh_grid_lo, nh_grid_hi])
    plt.colorbar()
    #plt.xticks(np.arange(3, 6.5, 0.5))
    #plt.yticks(np.arange(-6, -0.5, 0.5))
    plt.xlabel('log10(T)', fontsize=12)
    plt.ylabel('log10(hden)', fontsize=12)

plt.suptitle('Frac relative difference', fontsize=13)
plt.tight_layout()
plt.savefig(out_filename)
plt.clf()
#plt.show()