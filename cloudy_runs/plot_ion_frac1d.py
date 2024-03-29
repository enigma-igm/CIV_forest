import numpy as np
from matplotlib import pyplot as plt
import cloudy_utils as cu

# from skewers param file:
# EOS-logT0 = 3.9574533249435015
# nH_bar = 3.1315e-05 cm^-3

Z = -3.5
nh_bar_log10 = -4.5 # mean density
temp_log10 = 4.0    # temperature at mean density

metal_ion_ls = ['IONI CARB 1 1', 'IONI CARB 2 1', 'IONI CARB 3 1', 'IONI CARB 4 1', 'IONI CARB 5 1', 'IONI CARB 6 1', 'IONI CARB 7 1']
metal_ion_ls = ['IONI SILI 2 1', 'IONI SILI 3 1', 'IONI SILI 4 1', 'IONI SILI 5 1', 'IONI SILI 6 1', 'IONI SILI 7 1']
metal_ion_ls = ['IONI NITR 2 1', 'IONI NITR 3 1', 'IONI NITR 4 1', 'IONI NITR 5 1', 'IONI NITR 6 1', 'IONI NITR 7 1']
#metal_ion_ls = ['IONI OXYG 2 1', 'IONI OXYG 3 1', 'IONI OXYG 4 1', 'IONI OXYG 5 1', 'IONI OXYG 6 1', 'IONI OXYG 7 1']

out_filename = 'plots/nitro_frac1d.pdf'

met_lookup = cu.read_cloudy_koki('output/cloudy_grid_more')
#total_frac_fixed_hden = np.zeros(51)
#total_frac_fixed_temp = np.zeros(71)

#plt.figure(figsize=(10,9))
plt.figure(figsize=(10,5))
plt.subplot(121)
for metal_ion in metal_ion_ls:
    selected_ion_frac, nh_grid, temp_grid = cu.get_ion_frac1d(met_lookup, metal_ion, Z, want_hden_value=nh_bar_log10)
    #total_frac_fixed_hden += selected_ion_frac
    plt.plot(temp_grid, selected_ion_frac, label=metal_ion)
plt.title('fixed log10(hden)=$%0.2f$' % nh_bar_log10, fontsize=13)
plt.xlabel('log10(T)', fontsize=13)
plt.ylabel('ion fraction', fontsize=13)
plt.ylim([0,1.0])
plt.legend()

#plt.subplot(222)
#plt.plot(temp_grid, total_frac_fixed_hden)
#plt.xlabel('log10(T)', fontsize=13)
#plt.ylabel('Sum(ion fraction)', fontsize=13)

plt.subplot(122)
for metal_ion in metal_ion_ls:
    selected_ion_frac, nh_grid, temp_grid = cu.get_ion_frac1d(met_lookup, metal_ion, Z, want_temp_value=temp_log10)
    #total_frac_fixed_temp += selected_ion_frac
    plt.plot(nh_grid, selected_ion_frac, label=metal_ion)

plt.title('fixed log10(T)=%0.2f' % temp_log10, fontsize=13)
plt.xlabel('log10(hden)', fontsize=13)
plt.ylabel('ion fraction', fontsize=13)
plt.ylim([0,1.0])
#plt.yscale('log')
plt.legend()

#plt.subplot(224)
#plt.plot(nh_grid, total_frac_fixed_temp)
#plt.xlabel('log10(hden)', fontsize=13)
#plt.ylabel('Sum(ion fraction)', fontsize=13)

plt.tight_layout()
#plt.show()
plt.savefig(out_filename)
plt.clf()