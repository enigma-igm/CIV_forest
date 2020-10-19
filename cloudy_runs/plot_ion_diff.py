import numpy as np
from matplotlib import pyplot as plt
import read_cloudy as rc

metal_ion_ls = ['IONI CARB 1 1', 'IONI CARB 2 1', 'IONI CARB 3 1', 'IONI CARB 4 1', 'IONI CARB 5 1', 'IONI CARB 6 1']
#metal_ion_ls = ['IONI SILI 1 1', 'IONI SILI 2 1', 'IONI SILI 3 1', 'IONI SILI 4 1', 'IONI SILI 5 1', 'IONI SILI 6 1']
out_filename = 'carbon_ion_reldiff.pdf'

met_lookup = rc.read_cloudy_koki('output/cloudy_grid')

plt.figure(figsize=(10,8))
for i in range(len(metal_ion_ls)):
    ion1, nh_grid, temp_grid = rc.get_ion_frac(met_lookup, metal_ion_ls[i], -3.5)
    ion1 = np.reshape(ion1, (61, 31))
    ion2, nh_grid, temp_grid = rc.get_ion_frac(met_lookup, metal_ion_ls[i], -1.5)
    ion2 = np.reshape(ion2, (61, 31))

    diff = (ion1 - ion2)/ion1

    plt.subplot(2,3,i+1)
    plt.title(metal_ion_ls[i], fontsize=12)
    plt.imshow(diff, extent=[3, 6, -6, -1]) # 3 to 6 are the grid limits for log10(T)
                                                           # -6 to -1 are the grid limits for log10(hden)
    plt.colorbar()
    plt.xticks(np.arange(3, 6.5, 0.5))
    plt.yticks(np.arange(-6, -0.5, 0.5))
    plt.xlabel('log10(T)', fontsize=12)
    plt.ylabel('log10(hden)', fontsize=12)

plt.suptitle('Relative difference', fontsize=13)
plt.tight_layout()
plt.savefig(out_filename)
plt.clf()
#plt.show()