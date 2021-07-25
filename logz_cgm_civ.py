import numpy as np
from matplotlib import pyplot as plt
from astropy.io import fits
from astropy.table import Table
import glob
import os
from scipy import integrate
from scipy.interpolate import interp1d
from scipy.ndimage.filters import gaussian_filter1d
from scipy.ndimage import zoom

import mpmath
from scipy import special
from scipy import integrate
from astropy.cosmology import FlatLambdaCDM
from astropy.cosmology import Planck15
from astropy import constants as const
from astropy import units as u
from tqdm.auto import tqdm
#from progressbar import ProgressBar, Counter, Timer
from sklearn.neighbors import KDTree
from linetools.lists.linelist import LineList
from linetools.abund import solar as labsol
from IPython import embed
from enigma.reion_forest import utils

####### repurposed from enigma.reion_forest.logz_cgm.py

def civ_W_b_Ngrid(nN, cgm_dict):
    """
    Function that creates a grid of logN_MgII and returns b(N) and W(N) evaluated on this grid
    Args:
        nN:
        cgm_dict:

    Returns:

    """

    b_weak = cgm_dict['b_weak']
    b_strong = cgm_dict['b_strong']
    logN_MgII_min = cgm_dict['logN_metal_min']
    logN_MgII_max = cgm_dict['logN_metal_max']
    logN_strong = cgm_dict['logN_strong']
    logN_trans = cgm_dict['logN_trans']
    W_min = cgm_dict['W_min']
    W_max = cgm_dict['W_max']

    dvpix = 2.0
    vgrid_min = 0.0
    v_mgII = utils.vel_metal_doublet('C IV').value
    vgrid_max = 5.0*np.max(np.array([b_strong, v_mgII]))
    vmid = vgrid_min + (vgrid_max - vgrid_min)/2.0
    vel_grid = np.arange(vgrid_min, vgrid_max, dvpix)
    logN_MgII = logN_MgII_min + (logN_MgII_max - logN_MgII_min)*np.arange(nN)/(nN - 1) # grid of logN_mgii values
    v_abs = np.full(nN, vmid) # Just center these for determining the EW logN_MgII relation
    # sigmoid logistic function allows for smooth transition between b_weak and b_strong at the activiation locatino
    # logN_strong, over an interval of ~ 4*logN_trans about logN_strong
    sigmoid_arg = (logN_MgII - logN_strong)/logN_trans
    b_val = b_weak + (b_strong - b_weak)*special.expit(sigmoid_arg)
    tau_weak, W_2796 = utils.metal_voigt(vel_grid, v_abs, b_val, logN_MgII)
    if (W_2796.value.max() < W_max) or (W_2796.value.min() > W_min):
        raise ValueError('The N and b you are using cannot describe the full range of W requested. Revisit cgm_dict params')

    return logN_MgII, b_val, W_2796


def omega_civ(cgm_dict, z, cosmo=Planck15, X=0.76, logN_ref=12.0):
    """
    Args:
        cgm_dict (dict):
            Dictionary containing cgm model params for EW distribution
        cosmo (astropy.cosmology):
            astropy cosmology object, default=Planck15
        X (float):
            Mass fraction of baryons that are hydrogen, default=0.76
    Returns:
        omega_MgII, nMgII_out, logZ
        omega_MgII (float):
            Cosmological Omega of Mg atoms as traced by the MgII ions
        nMgII_out (Quantity):
            Proper Number density of Mg atoms per unit volume as traced by MgII ions
        logZ (float):
            Metallicity in solar units implied by nMgII_out. This is
            obtained by dividing by the cosmic hydrogen density and comparing to the solar abundance
    """

    # Create an interpolator for logN_MgII(W)
    nN = 1001
    logN_MgII, b_val, W_2796 = civ_W_b_Ngrid(nN, cgm_dict)
    logN_interp = interp1d(W_2796.value, logN_MgII, kind='cubic', bounds_error=True)  # W-logN relation
    #logN_ref = 12.0 # reference column density
    print("logN_ref", logN_ref)

    # Number density of hydrogen
    #a = 1.0 / (1.0 + cgm_dict['z'])
    a = 1.0 / (1.0 + z)
    rho_crit0 = cosmo.critical_density0
    rho_b_bar = cosmo.Ob0 * rho_crit0 / a ** 3
    nH_bar = (X * rho_b_bar / const.m_p).to('1/cm3')

    def n_integrand(W):
        integ = utils.mgii_dNdzdW(cgm_dict['n_star'], cgm_dict['alpha'], cgm_dict['W_star'], W)*np.power(10.0, logN_interp(W) - logN_ref)
        return integ

    nMg_int, nMg_err = integrate.quad(n_integrand, cgm_dict['W_min'], cgm_dict['W_max'], epsabs=1e-10, epsrel=1e-10)

    # This absorption distance converts dN/dz/dW to dN/dX/dW
    dXdz = cosmo.abs_distance_integrand(z)
    D_H = (const.c / cosmo.H0).to('cm')  # Hubble distance
    # number of Mg atoms as traced by MgII per cm^3
    nMgII_out = ((np.power(10.0,logN_ref)/u.cm/u.cm)*nMg_int/dXdz/D_H).to('1/cm3')
    # (SS) logN_ref and nMg_int cancels each other out, such that nNgII_out is invariant to value of logN_ref

    # mass fraction of Mg atoms as traced by MgII
    omega_MgII = ((12.0 * const.m_p) * nMgII_out/rho_crit0).decompose()

    # Copute this as a metallicity (relative to solar)
    sol = labsol.SolarAbund(verbose=False)
    n_MgbyH_sol = np.power(10.0,sol['C'] - 12.0) # number density of metal relative to number density of H in the Sun

    n_MgIIbyH = (nMgII_out/nH_bar).decompose()
    logZ = np.log10(n_MgIIbyH/n_MgbyH_sol)
    return omega_MgII.value, nMgII_out, logZ.value


cosmo= FlatLambdaCDM(H0=67.038, Om0=0.3192, Ob0=0.04964, Tcmb0=2.725)

"""
# Read in the model grid
zstr = 'z75'
outpath = '/Users/joe/reion_forest/Nyx_output/' + zstr + '/'
xhi_path = os.path.join(outpath, 'xHI')

# Fiducial model
xhi = 0.74
logZ = -3.7
rantaufile = os.path.join(xhi_path, 'ran_skewers_' + zstr + '_OVT_' + 'xHI_{:4.2f}'.format(xhi) + '_tau.fits')
"""

rantaufile = '/Users/suksientie/Research/CIV_forest/nyx_sim_data/igm_cluster/enrichment_models/tau/rand_skewers_z45_ovt_xciv_tau_R_0.30_logM_9.50.fits'
params = Table.read(rantaufile, hdu=1)
skewers = Table.read(rantaufile, hdu=2)
z = params['z'][0]

# Number density of hydrogen
a = 1.0/(1.0 + z)
rho_crit0 = cosmo.critical_density0
rho_b_bar = cosmo.Ob0 * rho_crit0 / a ** 3
X = 0.76
nH_bar = (X * rho_b_bar / const.m_p).to('1/cm3')

import civ_cgm
cgm_dict = civ_cgm.init_metal_cgm_dict(alpha=-1.1, n_star=5, W_star = 0.45)
#cgm_dict['logN_MgII_min'] = 8.0

logN_ref_ls = np.array([11.0, 12.0, 13.0])
for logN_ref in logN_ref_ls:
    omega_MgII, nMgII, logZ = omega_civ(cgm_dict, z, cosmo=cosmo, logN_ref=logN_ref)
    print(logN_ref, omega_MgII, logZ)

"""
# What about Carbon
z_CIV =  4.5
rho_crit0 = cosmo.critical_density0
omega_CIV =1e-7
# Copute this as a metallicity (relative to solar)
sol = labsol.SolarAbund(verbose=False)
n_CbyH_sol = np.power(10.0, sol['C'] - 12.0)  # number density of metal relative to number density of H in the Sun
n_CIV = (omega_CIV*rho_crit0/(12.0*const.m_p)).decompose()
# Number density of hydrogen
a = 1.0 / (1.0 + z_CIV)
rho_crit0 = cosmo.critical_density0
rho_b_bar = cosmo.Ob0 * rho_crit0 / a ** 3
nH_bar = (X * rho_b_bar / const.m_p).to('1/cm3')
n_CIVbyH = (n_CIV/nH_bar).decompose()
logZ_CIV = np.log10(n_CIVbyH/n_CbyH_sol)
"""

# If you want to investigate sensitivity to the cutoffs
#nW=20
#W_min = np.logspace(-5,np.log10(cgm_dict['W_min']), nW)
#W_max = np.logspace(-2.1,np.log10(cgm_dict['W_max']), nW)
#omega_vec = np.zeros(nW)
#logZ_vec = np.zeros(nW)

#for iw in range(nW):
#    cgm_dict['W_max'] = W_max[iw]
#    omega_vec[iw], _, logZ_vec[iw] = omega_mgii(cgm_dict)

#plt.plot(W_max, logZ_vec)
#plt.xscale('log')
#plt.show()

#N_star = cgm_dict['N_star']
#alpha = cgm_dict['alpha']
#W_star = cgm_dict['W_star']
#W_min = cgm_dict['W_min']
#W_max = cgm_dict['W_max']

#nN = 101
#logN_MgII, b_val, W_2796 = utils.mgii_W_b_Ngrid(nN, cgm_dict)
#logN_interp = interp1d(W_2796.value, logN_MgII, kind='cubic', bounds_error=True) # W-logN relation
#b_interp = interp1d(W_2796.value, b_val, kind='cubic', bounds_error=True) # W-b value relation

#nW = 500
#W = np.logspace(np.log10(W_min), np.log10(W_max), nW)
#logN_out = logN_interp(W)
#b_out = b_interp(W)
#logN_ref = 12.0

#dXdz = cosmo.abs_distance_integrand(z)
#integrand = utils.mgii_dNdzdW(N_star, alpha, W_star, W)*np.power(10.0, logN_out - logN_ref)

#plt.plot(W, integrand)
#plt.xscale('log')
#plt.yscale('log')
#plt.show()