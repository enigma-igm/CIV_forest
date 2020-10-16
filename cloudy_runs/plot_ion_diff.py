import numpy as np
from matplotlib import pyplot as plt
import read_cloudy as rc

metal_ion_ls = ['CARB1', 'CARB2', 'CARB3', 'CARB4', 'CARB5', 'CARB5']
out_filename = 'carbon_ion_diff.pdf'

met_lookup = rc.create_metal_lookup('output/cloudy_grid')

plt.figure(figsize=(12,10))
for i in range(len(metal_ion_ls)):
    ion1 = rc.arr_lookup(met_lookup, -3.5, metal_ion=metal_ion_ls[i])
    ion2 = rc.arr_lookup(met_lookup, -1.5, metal_ion=metal_ion_ls[i])
    diff = ion1 - ion2

    plt.subplot(3,3,i+1)
    plt.title(metal_ion_ls[i], fontsize=15)
    plt.imshow(diff, extent=[3, 6, -6, -1]) # 3 to 6 are the grid limits for log10(T)
                                                           # -6 to -1 are the grid limits for log10(hden)
    plt.colorbar()
    plt.xticks(np.arange(3, 6.5, 0.5))
    plt.yticks(np.arange(-6, -0.5, 0.5))
    plt.xlabel('log10(T)', fontsize=13)
    plt.ylabel('log10(hden)', fontsize=13)

plt.tight_layout()
plt.savefig(out_filename)
plt.clf()