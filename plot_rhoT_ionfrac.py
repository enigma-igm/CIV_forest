import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from astropy.table import Table
from matplotlib.colors import LogNorm
import cloudy_runs.cloudy_utils as cu
from matplotlib.ticker import AutoMinorLocator
import matplotlib.style as style
style.use('seaborn-colorblind')

#setting the figure
font = {'family' : 'serif', 'weight' : 'normal'}#, 'size': 11}
plt.rc('font', **font)
mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['xtick.major.width'] = 1.5
mpl.rcParams['ytick.major.width'] = 1.5
mpl.rcParams['xtick.minor.width'] = 1.5
mpl.rcParams['ytick.minor.width'] = 1.5
mpl.rcParams['xtick.major.size'] = 6
mpl.rcParams['xtick.minor.size'] = 3
mpl.rcParams['ytick.major.size'] = 6
mpl.rcParams['ytick.minor.size'] = 3

# TODO: annotate diffuse, whim, and hot gas on TDR
# maybe plot the dominant ion at each grid point?

logZ_cloudy = -3.5
cloudy_file = 'cloudy_runs/output/cloudy_grid_more'
metal_ion_ls = ['IONI CARB 1 1', 'IONI CARB 2 1', 'IONI CARB 3 1', 'IONI CARB 4 1', 'IONI CARB 5 1', 'IONI CARB 6 1', 'IONI CARB 7 1']

met_lookup = cu.read_cloudy_koki(cloudy_file)
metal_ind = np.where(met_lookup['METALS= %'] == logZ_cloudy)[0]
nh_grid = np.array(met_lookup['HDEN=%f L'][metal_ind])  # log10 unit
temp_grid = np.array(met_lookup['CONSTANT'][metal_ind])  # log10 unit
domion, nh_grid, temp_grid = cu.dominant_ion(lookup=cloudy_file, ion_ls=metal_ion_ls, logZ=logZ_cloudy)

plt.figure(figsize=(10,8))
title = ['$C\ I$', '$C\ II$', '$C\ III$', '$C\ IV$', '$C\ V$', '$C\ VI$', '$C\ VII$']
color = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:olive', 'tab:cyan']

n = np.zeros(len(metal_ion_ls))
for i in range(len(nh_grid)):
    n[domion[i]] += 1
    if n[domion[i]] == 1:
        plt.plot(nh_grid[i], temp_grid[i], 's', ms=3, color=color[domion[i]], label=title[domion[i]])
    else:
        plt.plot(nh_grid[i], temp_grid[i], 's', ms=3, color=color[domion[i]])
print(n, np.sum(n))
"""
for i in range(len(metal_ion_ls)):
    ion_frac = np.array(met_lookup[metal_ion_ls[i]][metal_ind])
    ind = np.where(ion_frac > 0.3)[0]
    plt.plot(nh_grid[ind], temp_grid[ind], 's', ms=3, label=title[i])
    #plt.scatter(nh_grid[ind], temp_grid[ind], marker=marker[i], s=6, label=title[i])
"""
plt.legend(loc=2,fontsize=14)

par = Table.read('nyx_sim_data/rand_skewers_z45_ovt_tau.fits', hdu=1)
ske = Table.read('nyx_sim_data/rand_skewers_z45_ovt_tau.fits', hdu=2)
nbins = 300

nH_bar = par['nH_bar']
oden = ske['ODEN'].flatten()
temp = ske['T'].flatten()

hist2d, xe, ye = np.histogram2d(np.log10(oden), np.log10(temp), bins=nbins)
norm_hist2d = hist2d/np.sum(hist2d)
y_coord = 0.5 *(ye[:-1]+ ye[1:])
x_coord = 0.5 *(xe[:-1]+ xe[1:])

# converting x-axis from overdensity to nH
plt.pcolormesh(np.log10(nH_bar*(10**x_coord)), y_coord, norm_hist2d.T, norm=LogNorm(), cmap=plt.cm.gist_gray)
cbar = plt.colorbar()
cbar.set_label('Density', rotation=270, fontsize=14, labelpad=15)
plt.gca().tick_params(right=True, which='both')
plt.gca().minorticks_on()

max_log_oden = 2.0
plt.axhline(5.0, ls='--', lw=2, color='k')
plt.vlines(np.log10(nH_bar*10**max_log_oden), ymin=2.0, ymax=5.0, ls='--', lw=2, color='k')
plt.xlabel(r'log$(n_H)$ $[cm^{-3}]$', fontsize=18)
plt.ylabel(r'log$(T)$ $[K]$', fontsize=18)
plt.gca().tick_params(axis="x", labelsize=16)
plt.gca().tick_params(axis="y", labelsize=16)

bbox = dict(boxstyle="round", fc="0.9") # fc is shading of the box, sth like alpha
plt.gca().annotate('Condensed', xy=(-1.4, 2.2), xytext=(-1.4, 2.2), textcoords='data', xycoords='data', annotation_clip=False, fontsize=14, bbox=bbox)
plt.gca().annotate('Diffuse', xy=(-6.8, 2.2), xytext=(-6.8, 2.2), textcoords='data', xycoords='data', annotation_clip=False, fontsize=14, bbox=bbox)
plt.gca().annotate('WHIM', xy=(-1., 6.7), xytext=(-1., 6.7), textcoords='data', xycoords='data', annotation_clip=False, fontsize=14, bbox=bbox)

min_oden = np.log10(10**nh_grid.min()/nH_bar)
max_oden = np.log10(10**nh_grid.max()/nH_bar)
atwin = plt.gca().twiny()
atwin.set_xlabel(r'log$(\Delta)$ $[\rho/\bar{\rho}]$', fontsize=18, labelpad=8)
atwin.xaxis.tick_top()

atwin.axis([min_oden, max_oden, temp_grid.min(), temp_grid.max()])
atwin.tick_params(top=True)
atwin.xaxis.set_minor_locator(AutoMinorLocator())
atwin.yaxis.set_minor_locator(AutoMinorLocator())
atwin.tick_params(axis="x", labelsize=16)

plt.tight_layout()
plt.show()