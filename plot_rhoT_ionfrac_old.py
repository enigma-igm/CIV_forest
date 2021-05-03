import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from astropy.table import Table
from matplotlib.colors import LogNorm
import cloudy_runs.cloudy_utils as cu
from matplotlib.ticker import AutoMinorLocator
from scipy.spatial import ConvexHull

#TODO: plot shaded locus for each ion, show contours of CIV like Illustris paper
#shading the region spanned by all grid points a single color and using transparency

#setting the figure
font = {'family' : 'serif', 'weight' : 'normal'}#, 'size': 11}
plt.rc('font', **font)
mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['xtick.major.width'] = 1.5
mpl.rcParams['ytick.major.width'] = 1.5
mpl.rcParams['xtick.minor.width'] = 1.5
mpl.rcParams['ytick.minor.width'] = 1.5
mpl.rcParams['xtick.major.size'] = 7
mpl.rcParams['xtick.minor.size'] = 4
mpl.rcParams['ytick.major.size'] = 7
mpl.rcParams['ytick.minor.size'] = 4

logZ_cloudy = -3.5
cloudy_file = 'cloudy_runs/output/cloudy_grid_more'
metal_ion_ls = ['IONI CARB 1 1', 'IONI CARB 2 1', 'IONI CARB 3 1', 'IONI CARB 4 1', 'IONI CARB 5 1', 'IONI CARB 6 1', 'IONI CARB 7 1']

met_lookup = cu.read_cloudy_koki(cloudy_file)
metal_ind = np.where(met_lookup['METALS= %'] == logZ_cloudy)[0]
nh_grid = np.array(met_lookup['HDEN=%f L'][metal_ind])  # log10 unit
temp_grid = np.array(met_lookup['CONSTANT'][metal_ind])  # log10 unit
domion, nh_grid, temp_grid = cu.dominant_ion(lookup=cloudy_file, ion_ls=metal_ion_ls, logZ=logZ_cloudy)

### plotting ion fraction from Cloudy
plt.figure(figsize=(10,8))
title = ['$C\ I$', '$C\ II$', '$C\ III$', '$C\ IV$', '$C\ V$', '$C\ VI$', '$C\ VII$']
color = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:olive', 'tab:cyan']
#alpha = np.linspace(0, 1, 7)
# use plt.colorbar but set the colorbar to be the ion!

n = np.zeros(len(metal_ion_ls))
for i in range(len(nh_grid)):
    n[domion[i]] += 1
    if n[domion[i]] == 1:
        #plt.plot(nh_grid[i], temp_grid[i], 's', ms=10, color=color[domion[i]], label=title[domion[i]])
        plt.plot(nh_grid[i], temp_grid[i], 's', ms=5, color='tab:red', alpha=alpha[domion[i]], label=title[domion[i]], mew=0)
    else:
        plt.plot(nh_grid[i], temp_grid[i], 's', ms=5, color='tab:red', alpha=alpha[domion[i]], mew=0)
print(n, np.sum(n))
"""
for i in range(len(metal_ion_ls)):
    ion_frac = np.array(met_lookup[metal_ion_ls[i]][metal_ind])
    ind = np.where(ion_frac > 0.3)[0]
    plt.plot(nh_grid[ind], temp_grid[ind], 's', ms=3, label=title[i])
    #plt.scatter(nh_grid[ind], temp_grid[ind], marker=marker[i], s=6, label=title[i])
"""
plt.legend(loc=2,fontsize=14)

"""
### plotting TDR using Nyx skewers
# borrowing things from Vikram: https://github.com/qsopairs/enigma/blob/master/enigma/whim/my_plotting_routines/paper_plots/weighted_bN_igm.py
par = Table.read('nyx_sim_data/rand_skewers_z45_ovt_tau.fits', hdu=1)
ske = Table.read('nyx_sim_data/rand_skewers_z45_ovt_tau.fits', hdu=2)
nbins = 300 # number of bins for log(rho) and log(T)

nH_bar = par['nH_bar']
oden = ske['ODEN'].flatten()
temp = ske['T'].flatten()

hist2d, xe, ye = np.histogram2d(np.log10(oden), np.log10(temp), bins=nbins) # 2d histogram
norm_hist2d = hist2d/np.sum(hist2d) # normalizing such that the histrogram is density
y_coord = 0.5 *(ye[:-1]+ ye[1:]) # bin centers
x_coord = 0.5 *(xe[:-1]+ xe[1:])

# converting x-axis from log(overdensity) to log(nH)
plt.pcolormesh(np.log10(nH_bar*(10**x_coord)), y_coord, norm_hist2d.T, norm=LogNorm(), cmap=plt.cm.gist_gray) # LogNorm uses a log-scale colorbar
cbar = plt.colorbar()
cbar.set_label('Density', rotation=270, fontsize=14, labelpad=15)
plt.gca().tick_params(right=True, which='both')
plt.gca().minorticks_on()

# demarcating different IGM phases
max_log_oden = 2.0 # maximum oden for diffuse IGM
plt.axhline(5.0, ls='--', lw=2, color='k')
plt.vlines(np.log10(nH_bar*(10**max_log_oden)), ymin=2.0, ymax=5.0, ls='--', lw=2, color='k')
#plt.xlabel(r'log$\mathrm{(n_H)}$ $\mathrm{[cm^{-3}]}$', fontsize=18)
#plt.ylabel(r'log$\mathrm{(T)}$ $\mathrm{[K]}$', fontsize=18)
plt.xlabel(r'log(n$_\mathrm{H}$) [cm$^{-3}$]', fontsize=18)
plt.ylabel('log(T) [K]', fontsize=18)
plt.gca().tick_params(axis="x", labelsize=16)
plt.gca().tick_params(axis="y", labelsize=16)

bbox = dict(boxstyle="round", fc="0.9") # fc is shading of the box, sth like alpha
plt.gca().annotate('Condensed', xy=(-1.4, 2.2), xytext=(-1.4, 2.2), textcoords='data', xycoords='data', annotation_clip=False, fontsize=14, bbox=bbox)
plt.gca().annotate('Diffuse', xy=(-6.8, 2.2), xytext=(-6.8, 2.2), textcoords='data', xycoords='data', annotation_clip=False, fontsize=14, bbox=bbox)
plt.gca().annotate('WHIM', xy=(-1., 6.7), xytext=(-1., 6.7), textcoords='data', xycoords='data', annotation_clip=False, fontsize=14, bbox=bbox)

# including ODEN on the top axis
min_oden = np.log10((10**nh_grid.min())/nH_bar)
max_oden = np.log10((10**nh_grid.max())/nH_bar)
atwin = plt.gca().twiny()
atwin.set_xlabel(r'log$(\Delta)$ $[\rho/\bar{\rho}]$', fontsize=18, labelpad=8)
atwin.xaxis.tick_top()

atwin.axis([min_oden, max_oden, temp_grid.min(), temp_grid.max()])
atwin.tick_params(top=True)
atwin.xaxis.set_minor_locator(AutoMinorLocator())
atwin.yaxis.set_minor_locator(AutoMinorLocator())
atwin.tick_params(axis="x", labelsize=16)
"""
plt.tight_layout()
plt.show()