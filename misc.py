import numpy as np
from linetools.lists.linelist import LineList
from astropy import constants as const
from astropy import units as u
from astropy.cosmology import Planck15
from scipy import integrate

def metalsline_info():
    strong = LineList('Strong', verbose=False)
    metals = ['CIV 1548', 'SiIV 1393', 'NV 1238', 'OVI 1031']
    ion_lo = [47.9, 33.49, 77.47, 113.9]
    ion_hi = [64.49, 45.14, 97.89, 113.9]
    ion_mean = (np.array(ion_lo) + np.array(ion_hi))/2
    ion_mean_ryd = ion_mean/13.6

    ion_temp = ion_mean/const.k_B.to('eV/K').value
    for i in range(len(metals)):
        print(metals[i], ion_mean_ryd[i], ion_temp[i]/1e4)

def get_nH_bar(z, cosmo=Planck15, X=0.76):
    # X is mass fraction of hydrogen

    a = 1.0 / (1.0 + z)
    rho_crit0 = cosmo.critical_density0
    rho_b_bar = cosmo.Ob0 * rho_crit0 / a ** 3
    nH_bar = (X * rho_b_bar / const.m_p).decompose().to('1/cm3')

    return nH_bar

def convert_pathlength_cMpc(z, vmin, vmax, cosmo=Planck15):
    Hz = cosmo.H(z) # Hubble parameter at z
    a = 1.0 / (1.0 + z)
    rmin = (vmin*u.km/u.s/a/Hz).to('Mpc').value
    rmax = (vmax*u.km/u.s/a/Hz).to('Mpc').value

    return rmin, rmax

def comov_dist(z, cosmo=Planck15):
    h = cosmo.H0.value / 100
    return h*cosmo.comoving_distance(z).value # comoving Mpc/h

def convert_resolution(R_val):
    r_kms = const.c.to('km/s')/R_val
    return r_kms

def convert_to_energy(value, type):
    if type == 'wave':
        value *= u.Angstrom
        energy_mks = const.h*const.c/value
    elif type == 'freq':
        value *= u.Hz
        energy_mks = const.h*value
    energy_ryd = energy_mks.to('Ry')

    return energy_ryd

def get_npath(z, delta_z, nqsos):
    vmin = 1.7433470381205325 # vel_lores.min()
    vmax = 12931.743347038268 # vel_lores.max()
    vside = vmax - vmin
    zmin = z - delta_z
    zeff = (z + zmin)/2.
    c_light = (const.c.to('km/s')).value

    dv_path = delta_z / (1.0 + zeff) * c_light
    dz_side = (vside / c_light) * (1.0 + zeff)
    npath_float = nqsos * dv_path / vside
    npath = int(np.round(npath_float))
    dz_tot = dz_side * npath

    print(dv_path)
    print(dz_side)
    print(dz_tot)
    print(npath)

def get_nqsos(z, delta_z, npath):
    vmin = 1.7433470381205325 # vel_lores.min()
    vmax = 12931.743347038268 # vel_lores.max()
    vside = vmax - vmin
    zmin = z - delta_z
    zeff = (z + zmin)/2.
    c_light = (const.c.to('km/s')).value

    dv_path = delta_z / (1.0 + zeff) * c_light
    dz_side = (vside / c_light) * (1.0 + zeff)
    nqsos = (npath/dv_path)*vside

    #npath_float = nqsos * dv_path / vside
    #npath = int(np.round(npath_float))
    #dz_tot = dz_side * npath
    print(dv_path)
    print(dz_side)
    print(nqsos)

### testing functions for CIV distribution functions from literatures.
### double-check before using
def civ_dNdzdW(W, z, type, k=None, alpha=None):
    # dN, where N = number, not column density

    if k == None:
        if type == 'Cooksey': # Table 4 and <z> = 3.25860
            k = 1.82
        elif type == 'DOdorico': # Cooksey's fit to D'Odorico et al. (2010); Sec 4.3
            k = 3.72
        elif type == 'Songaila': # Cooksey's fit to Songaila (2001); Sec 4.3
            k = 2.29

    if alpha == None:
        if type == 'Cooksey':
            alpha = -2.61
        elif type == 'DOdorico':
            alpha = -2.65
        elif type == 'Songaila':
            alpha = -2.58

    # exponential form from Cooksey et al. (2013)
    dN_dXdW = k*np.exp(alpha*W)

    # converting to dNdzdW
    omega_m = Planck15.Om0
    omega_lambda = 1 - omega_m
    dX_dz = ((1 + z)**2)*(omega_m*(1 + z)**3 + omega_lambda)**(-0.5) # correct for discrete dz??
    dN_dzdW = dN_dXdW * dX_dz

    return dN_dzdW

def civ_dndNdX(B, alpha, N_CIV):
    # alpha = 1.71 or 1.8 from D'Odorico et al. (2010)
    # alpha - 1.44 for Ellison et al.(2000)
    # what are obs values for B?

    dn_dNdX = B*N_CIV**(-alpha) # power law form for column density distribution function (CDDF)
    return dn_dNdX

# want to match to Simcoe et al. (2011) or Songaila (2001) dN/dz (which one?) by adjusting alpha and k
def civ_dNdz(k, alpha, z, W_min, W_max, nW):

    W = np.linspace(W_min, W_max, nW)
    dN_dXdW = k * np.exp(alpha * W)
    dN_dX = integrate.simps(dN_dXdW, W) # integrate dN_dXdW over dW to get dN_dX

    # convert dN_dX to dN_dz
    omega_m = Planck15.Om0
    omega_lambda = 1 - omega_m
    dX_dz = ((1 + z) ** 2) * (omega_m * (1 + z) ** 3 + omega_lambda) ** (-0.5)
    dN_dz = dN_dX * dX_dz

    return dN_dX, dN_dz