import numpy as np
from matplotlib import pyplot as plt
import cloudy_runs.cloudy_utils as cloudy_utils
import metal_frac_skewers

#nh_log10 = -4.5 # mean density
nh_log10_ls = np.log10([3.166686791788229e-05, 3.71365044820066e-05, 4.320226805030104e-05, 4.9894992778153774e-05, 5.724551282095286e-05, 6.52846623340864e-05])
temp_log10 = np.log10(10000)

cldy_outpath = 'cloudy_runs/output/'
cldy_outfiles = [cldy_outpath + 'cloudy_grid_more', cldy_outpath + 'cloudy_grid_z4p8', cldy_outpath + 'cloudy_grid_z5p1', \
                 cldy_outpath + 'cloudy_grid_z5p4', cldy_outpath + 'cloudy_grid_z5p7', cldy_outpath + 'cloudy_grid_z6']

z_range = [4.5, 4.8, 5.1, 5.4, 5.7, 6.0] # redshifts of the cloudy models

# list of metal ions
cldy_metal_ion_ls = ['IONI CARB 4 1', 'IONI SILI 4 1', 'IONI NITR 5 1', 'IONI OXYG 6 1']
metal_ion_ls = ['C IV', 'Si IV', 'N V', 'O VI']

lookup_all = []
for i in range(len(z_range)):
    lookup = cloudy_utils.read_cloudy_koki(cldy_outfiles[i], verbose=False)
    lookup_all.append(lookup)

all_tau0 = []
all_ion_frac = []
all_tau0f = []

for i in range(len(cldy_metal_ion_ls)):
    temp1 = []
    temp2 = []
    temp3 = []
    for j in range(len(z_range)):
        lookup = lookup_all[j]
        #tau0f = metal_frac_skewers.get_tau0_frac(lookup, cldy_metal_ion_ls[i], metal_ion_ls[i], nh_log10, temp_log10, z_range[j], logZ=-3.5)
        nh_log10 = nh_log10_ls[j]
        tau0, ion_frac = metal_frac_skewers.get_tau0_frac(lookup, cldy_metal_ion_ls[i], metal_ion_ls[i], nh_log10, temp_log10,
                                                 z_range[j], logZ=-3.5)
        temp1.append(tau0)
        temp2.append(ion_frac)
        temp3.append(np.array(tau0)*np.array(ion_frac))

    all_tau0.append(temp1)
    all_ion_frac.append(temp2)
    all_tau0f.append(temp3)

for i in range(len(cldy_metal_ion_ls)):

    plt.subplot(131)
    plt.plot(z_range, all_tau0[i], label='%s' % metal_ion_ls[i], marker='o', linestyle='--')
    plt.legend()
    plt.yscale('log')
    plt.xlabel('z', fontsize=15)
    plt.ylabel(r'$\tau_0$', fontsize=15)

    plt.subplot(132)
    plt.plot(z_range, all_ion_frac[i], label='%s' % metal_ion_ls[i], marker='o', linestyle='--')
    plt.legend()
    plt.yscale('log')
    plt.xlabel('z', fontsize=15)
    plt.ylabel(r'$f_{ion}$', fontsize=15)

    plt.subplot(133)

    plt.plot(z_range, all_tau0f[i], label='%s' % metal_ion_ls[i], marker='o', linestyle='--')
    plt.legend()
    plt.yscale('log')
    plt.xlabel('z', fontsize=15)
    plt.ylabel(r'$\tau_0 \times f_{ion}$', fontsize=15)

plt.suptitle(r'log$_{10}(\bar{n}_H(z))$, log$_{10}$(T) = %0.1f' % (temp_log10), fontsize=15)
plt.show()