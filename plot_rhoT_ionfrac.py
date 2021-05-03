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

# setting the figure
font = {'family' : 'serif', 'weight' : 'normal'}
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

plt.figure(figsize=(12., 8))
plt.subplots_adjust(left=0.07, bottom=0.1, right=0.95, top=0.9, wspace=0, hspace=0)

cbar_fontsize = 12
xytick_size = 16
annotate_text_size = 16
xylabel_fontsize = 20
ionfrac_alpha = 1.0
tdr_alpha = 0.7

logZ_cloudy = -3.5
cloudy_file = 'cloudy_runs/output/cloudy_grid_more'
metal_ion_ls = ['IONI CARB 1 1', 'IONI CARB 2 1', 'IONI CARB 3 1', 'IONI CARB 4 1', 'IONI CARB 5 1', 'IONI CARB 6 1', 'IONI CARB 7 1']

met_lookup = cu.read_cloudy_koki(cloudy_file)
metal_ind = np.where(met_lookup['METALS= %'] == logZ_cloudy)[0]
nh_grid = np.array(met_lookup['HDEN=%f L'][metal_ind])  # log10 unit
temp_grid = np.array(met_lookup['CONSTANT'][metal_ind])  # log10 unit
domion, nh_grid, temp_grid = cu.dominant_ion(lookup=cloudy_file, ion_ls=metal_ion_ls, logZ=logZ_cloudy)

N_nh_grid, nh_grid_lo, nh_grid_hi = len(np.unique(nh_grid)), np.unique(nh_grid)[0], np.unique(nh_grid)[-1]
print(N_nh_grid, nh_grid_lo, nh_grid_hi)
N_temp_grid, temp_grid_lo, temp_grid_hi = len(np.unique(temp_grid)), np.unique(temp_grid)[0], np.unique(temp_grid)[-1]
print(N_temp_grid, temp_grid_lo, temp_grid_hi)

# reshaping from 1d to 2d, for plt.imshow
cmap = plt.cm.get_cmap('BuPu', len(set(domion))) # discrete colormap
domion = np.reshape(domion, (N_nh_grid, N_temp_grid))

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
x_coord = np.log10(nH_bar*(10**x_coord))
mat2 = plt.pcolormesh(x_coord, y_coord, norm_hist2d.T, norm=LogNorm(), cmap=plt.cm.gist_gray, alpha=tdr_alpha) # LogNorm uses a log-scale colorbar
#plt.imshow(norm_hist2d.T, norm=LogNorm(), cmap=plt.cm.gist_gray, origin='lower', \
#           extent=[x_coord.min(), x_coord.max(), y_coord.min(), y_coord.max()], alpha=tdr_alpha)

cbar2 = plt.colorbar(mat2, fraction=0.047, pad=0.047)
cbar2.ax.tick_params(labelsize=cbar_fontsize)
#cbar2.set_label('Density', rotation=270, fontsize=cbar_fontsize, labelpad=15)

plt.gca().tick_params(right=True, which='both')
plt.gca().minorticks_on()
plt.gca().set_xlim([nh_grid.min(), nh_grid.max()])
plt.gca().set_ylim([temp_grid.min(), temp_grid.max()])

# demarcating different IGM phases
max_log_oden = 2.0 # maximum oden for diffuse IGM
plt.axhline(5.0, ls='--', lw=2, color='k')
plt.vlines(np.log10(nH_bar*(10**max_log_oden)), ymin=2.0, ymax=5.0, ls='--', lw=2, color='k')
plt.xlabel(r'log(n$_\mathrm{H}$) [cm$^{-3}$]', fontsize=xylabel_fontsize)
plt.ylabel('log(T) [K]', fontsize=xylabel_fontsize)
plt.gca().tick_params(axis="both", labelsize=xytick_size)

bbox = dict(boxstyle="round", fc="0.9") # fc is shading of the box, sth like alpha
plt.gca().annotate('Condensed', xy=(-1.2, 2.2), xytext=(-1.2, 2.2), textcoords='data', xycoords='data', annotation_clip=False, fontsize=annotate_text_size, bbox=bbox)
plt.gca().annotate('Diffuse', xy=(-6.8, 2.2), xytext=(-6.8, 2.2), textcoords='data', xycoords='data', annotation_clip=False, fontsize=annotate_text_size, bbox=bbox)
plt.gca().annotate('WHIM', xy=(-1., 6.7), xytext=(-1., 6.7), textcoords='data', xycoords='data', annotation_clip=False, fontsize=annotate_text_size, bbox=bbox)

title = ['C II', 'C III', 'C IV', 'C V', 'C VI', 'C VII']
mat = plt.imshow(domion.transpose(), cmap=cmap, vmin=np.min(domion) - 0.5, vmax=np.max(domion) + 0.5, \
      origin='lower', extent=[nh_grid_lo, nh_grid_hi, temp_grid_lo, temp_grid_hi], interpolation='None', alpha=ionfrac_alpha)
cbar1 = plt.colorbar(mat, ticks=np.arange(np.min(domion), np.max(domion)+1), fraction=0.047, pad=0.02)#, orientation='horizontal')
cbar1.ax.set_yticklabels(title, fontsize=cbar_fontsize)

# including ODEN on the top axis
min_oden = np.log10((10**nh_grid.min())/nH_bar)
max_oden = np.log10((10**nh_grid.max())/nH_bar)
atwin = plt.gca().twiny()
atwin.set_xlabel(r'log($\Delta$) [$\rho/\bar{\rho}$]', fontsize=xylabel_fontsize, labelpad=8)
atwin.xaxis.tick_top()
atwin.axis([min_oden, max_oden, temp_grid_lo, temp_grid_hi])
atwin.tick_params(top=True)

atwin.xaxis.set_minor_locator(AutoMinorLocator())
atwin.yaxis.set_minor_locator(AutoMinorLocator())
atwin.tick_params(axis="x", labelsize=16)

plt.show()