import numpy as np
from matplotlib import pyplot as plt
import cloudy_runs.cloudy_utils as cloudy_utils
from astropy.io import fits
from astropy.table import Table
import halos_skewers

par, ske, halos = halos_skewers.init_all() # load halo catalog and skewerfile

logM_min = 8.0
slice_thickness = 1.0
Zc = 50

maskfile = 'nyx_sim_data/rand_skewers_z45_halomask_test2.fits'
mask = Table.read(maskfile)
mask00 = mask['mask00'] # R, logM_min = (1.375, 10)
mask01 = mask['mask01'] # (0.34375, 9)
mask02 = mask['mask02']

R00, logM00 = 1.375, 10
R01, logM01 = 0.34375, 9
R02, logM02 = 2.75, 9

fm00, fv00 = halos_skewers.calc_fm_fv(mask00, ske)
fm01, fv01 = halos_skewers.calc_fm_fv(mask01, ske)
fm02, fv02 = halos_skewers.calc_fm_fv(mask02, ske)

imass = np.where(np.log10(halos['MASS']) >= logM_min)[0]
halos = halos[imass]
print("after mass cut, N(halos): ", len(halos))

Zmin = Zc - slice_thickness/2.
Zmax = Zc + slice_thickness/2.
slice = (halos['ZHALO'] >= Zmin) * (halos['ZHALO'] < Zmax)
halo_slice = halos[slice]

plt.figure(figsize=(12,4))
plt.subplot(131)
plt.plot(halo_slice['XHALO'], halo_slice['YHALO'], 'k.', ms=3, alpha=0.7)
for ihalo in halo_slice:
    if np.log10(ihalo['MASS']) > logM00:
        circ = plt.Circle((ihalo['XHALO'], ihalo['YHALO']), R00, color='r', fill=True)
        plt.gca().add_patch(circ)

plt.title('R = %0.3f Mpc, logM = %d \n fv = %0.3f, fm = %0.3f' % (R00, logM00, fv00, fm00), fontsize=15)
plt.axis('equal')
plt.xlabel('XHALO', fontsize=12)
plt.ylabel('YHALO', fontsize=12)
plt.xlim([10,50])
plt.ylim([10,50])

plt.subplot(132)
plt.plot(halo_slice['XHALO'], halo_slice['YHALO'], 'k.', ms=3, alpha=0.7)
for ihalo in halo_slice:
    if np.log10(ihalo['MASS']) > logM01:
        circ = plt.Circle((ihalo['XHALO'], ihalo['YHALO']), R01, color='r', fill=True)
        plt.gca().add_patch(circ)
plt.title('R = %0.3f Mpc, logM = %d \n fv = %0.3f, fm = %0.3f' % (R01, logM01, fv01, fm01), fontsize=15)
plt.axis('equal')
plt.xlabel('XHALO', fontsize=12)
plt.ylabel('YHALO', fontsize=12)
plt.xlim([10,50])
plt.ylim([10,50])

plt.subplot(133)
plt.plot(halo_slice['XHALO'], halo_slice['YHALO'], 'k.', ms=3, alpha=0.7)
for ihalo in halo_slice:
    if np.log10(ihalo['MASS']) > logM02:
        circ = plt.Circle((ihalo['XHALO'], ihalo['YHALO']), R02, color='r', fill=True)
        plt.gca().add_patch(circ)
plt.title('R = %0.3f Mpc, logM = %d \n fv = %0.3f, fm = %0.3f' % (R02, logM02, fv02, fm02), fontsize=15)
plt.axis('equal')
plt.xlabel('XHALO', fontsize=12)
plt.ylabel('YHALO', fontsize=12)
plt.xlim([10,50])
plt.ylim([10,50])

plt.tight_layout()
plt.show()