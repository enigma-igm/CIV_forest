import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate, special
from scipy.interpolate import interp1d
from astropy.cosmology import Planck15
import mpmath
from astropy import constants as const
from astropy import units as u
import enigma.reion_forest.utils as reion_utils

def civ_dndNdz_sch(n_star, alpha, N_star, N, z=None):

    # Schechter function for CIV column density (N) distribution.
    # note: small n refers to number

    dn_dNdz = (n_star / N_star) * np.power(N / N_star, alpha) * np.exp(-N/ N_star)

    if z != None:
        omega_m = Planck15.Om0
        omega_lambda = 1 - omega_m
        dz_dX = 1/(((1 + z) ** 2) * (omega_m * (1 + z) ** 3 + omega_lambda) ** (-0.5))
        dn_dNdX = dn_dNdz * dz_dX
        return dn_dNdX
    else:
        return dn_dNdz

def civ_dndNdz_sch2(normalization, alpha, N_star, N, z=None):

    # Schechter function for CIV column density (N) distribution.
    # same as above, but replacing absorbing 'n_star' into 'normalization'

    dn_dNdz = normalization * np.power(N / N_star, alpha) * np.exp(-N/ N_star)
    if z != None:
        omega_m = Planck15.Om0
        omega_lambda = 1 - omega_m
        dz_dX = 1/(((1 + z) ** 2) * (omega_m * (1 + z) ** 3 + omega_lambda) ** (-0.5))
        print("dz_dX", dz_dX)
        dn_dNdX = dn_dNdz * dz_dX
        return dn_dNdX
    else:
        return dn_dNdz

def civ_dndz_schechter(n_star, alpha, N_star, N_min, N_max):
    """
    Compute Schechter integral over [N_min, N_max] interval using the incomplete gamma functions.
    """

    z = alpha + 1 # slope
    upper = N_max / N_star
    lower = N_min / N_star
    # \Gamma(z, l, u) = \int_lower^upper x^(z-1) exp(-x) dx, where x = W/W_star
    if isinstance(N_max,float):
        I = float(mpmath.gammainc(z, lower, upper))
    elif isinstance(N_max,np.ndarray):
        I = np.zeros_like(N_max)
        for indx in range(N_max.size):
            I[indx] = float(mpmath.gammainc(z, lower, upper[indx]))

    dn_dz = n_star * I
    return dn_dz

def civ_dndz_schechter2(normalization, alpha, N_star, N_min, N_max):
    """
    Compute Schechter integral over [N_min, N_max] interval using the incomplete gamma functions.
    """

    z = alpha + 1 # slope
    upper = N_max / N_star
    lower = N_min / N_star
    # \Gamma(z, l, u) = \int_lower^upper x^(z-1) exp(-x) dx, where x = W/W_star
    if isinstance(N_max,float):
        I = float(mpmath.gammainc(z, lower, upper))
    elif isinstance(N_max,np.ndarray):
        I = np.zeros_like(N_max)
        for indx in range(N_max.size):
            I[indx] = float(mpmath.gammainc(z, lower, upper[indx]))

    dn_dz = (normalization * N_star) * I
    return dn_dz

def convert_dXdz(omega_m, omega_lambda, z):
    #omega_m = Planck15.Om0
    #omega_lambda = 1 - omega_m
    dX_dz = ((1 + z) ** 2) * (omega_m * (1 + z) ** 3 + omega_lambda) ** (-0.5)
    return dX_dz

def convert_W2N_civ(W):
    # N in 1/cm2, W in Angstrom
    N = 1e14 * (W/0.6)
    return N

def convert_W2N_civ_old(W):

    # all in SI units
    ec = const.e.value
    me = const.m_e.value
    c = const.c.value
    f = 0.1899  # oscillator strength for CIV 1548
    wrest = 1.548204e-07 * u.m # rest wavelength of CIV 1548 in SI

    N = ((me*c**2)/(np.pi*ec**2))* (W * u.m)/(wrest*f) # 1/m2
    N /= (u.m * u.m)

    # W ~ 0.6A should be logN ~ 14 (Cooksey+2010), but not getting this...
    return N

##### EW distribution function from Cooksey et al. (2010) #####
def civ_dndzdW(W, z, type, k=None, alpha=None):
    # dN, where N = number, not column density
    # W = np.arange(0.03, 2.5, 0.01)

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
    dn_dXdW = k*np.exp(alpha*W)

    # converting to dNdzdW
    omega_m = Planck15.Om0
    omega_lambda = 1 - omega_m
    dX_dz = ((1 + z)**2)*(omega_m*(1 + z)**3 + omega_lambda)**(-0.5) # correct for discrete dz??
    dn_dzdW = dn_dXdW * dX_dz

    return dn_dzdW, dn_dXdW

# want to match to Simcoe et al. (2011) or Songaila (2001) dN/dz (which one?) by adjusting alpha and k
def civ_dndz_exp(k, alpha, z, W_min, W_max, nW):

    W = np.linspace(W_min, W_max, nW)
    dN_dXdW = k * np.exp(alpha * W)
    dN_dX = integrate.simps(dN_dXdW, W) # integrate dN_dXdW over dW to get dN_dX

    # convert dN_dX to dN_dz
    omega_m = Planck15.Om0
    omega_lambda = 1 - omega_m
    dX_dz = ((1 + z) ** 2) * (omega_m * (1 + z) ** 3 + omega_lambda) ** (-0.5)
    dN_dz = dN_dX * dX_dz

    return dN_dX, dN_dz
###############################################################

def civ_dndNdX_pl(B, alpha, N_CIV):
    # alpha = 1.71 or 1.8 from D'Odorico et al. (2010)
    # alpha = 1.75 for D'Odorico et al. (2013), Figure 18, 4.35 < z < 5.3
    # what are obs values for B?
    #    - log(B) = 10.3, by eye based on Figure 18 (i.e. for logN = 12.4 and logf = -12.2) of D'Odorico et al (2013)

    dn_dNdX = B*N_CIV**(-alpha) # power law form for column density distribution function (CDDF)
    return dn_dNdX

def civ_dndNdX_pl_sch(N_star):
    # attaching an exponential cutoff to D'Odorico et al. (2013) power law fit
    B = 10 ** 10.3
    logN_CIV = np.arange(12.4, 15.2, 0.1)
    alpha = 1.75

    f = civ_dndNdX_pl(B, alpha, 10 ** logN_CIV)
    N_Nstar = (10**logN_CIV)/N_star
    f_sch = f * np.exp(-N_Nstar)

    return f_sch

######### data observations #########
def cooksey2013_dndz():
    # Table 4 (CIV results summary)
    z_median = [1.96, 1.56, 1.66, 1.74, 1.82, 1.91, 2.02, 2.15, 2.36, 2.72, 3.26]
    z_min = [1.47, 1.47, 1.61, 1.70, 1.78, 1.87, 1.96, 2.08, 2.24, 2.51, 2.97]
    z_max = [4.54, 1.61, 1.70, 1.78, 1.87, 1.96, 2.08, 2.24, 2.51, 2.97, 4.54]
    z_err_lo = np.array(z_median) - np.array(z_min)
    z_err_hi = np.array(z_max) - np.array(z_median)

    dndz = [0.92, 0.95, 1.00, 1.08, 1.09, 1.12, 1.04, 1.04, 0.96, 0.83, 0.59]
    dndz_err_hi = [0.02, 0.03, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.03, 0.02]
    dndz_err_lo = [0.01, 0.03, 0.03, 0.04, 0.04, 0.04, 0.04, 0.04, 0.03, 0.03, 0.02]

    # first point in the array is the largest bin that includes all subsequent smaller bins
    plt.errorbar(z_median[1:], dndz[1:], xerr=(z_err_lo[1:], z_err_hi[1:]), yerr=(dndz_err_lo[1:], dndz_err_hi[1:]), fmt='o')

    #plt.plot(z_median[0], dndz[0], 'r*')
    #plt.hlines(dndz[0], dndz_err_lo[0] - dndz[0], dndz_err_hi[0] + dndz[0], color='r')
    #plt.vlines(z_median[0], z_median[0] - z_err_lo[0], z_median[0] + z_err_hi[0], color='r')

    plt.yscale('log')
    plt.xlabel('z', fontsize=13)
    plt.ylabel('dn/dz', fontsize=13)
    plt.tight_layout()
    plt.show()

# option 1: by-eye estimate the data points for f(N), then convert data points to dX --> dz
# option 2: get dn/dN and provided you have dX (or dz), then just divide to get dn/dN/dZ
# option 3: make use of omega_civ
def dodorico2013_cddf():
    # D'Odorico et al. (2013)
    # data points for CDDF not provided, so estimating the points by eye from Figure 18
    # for 4.35 < z < 5.3

    logN_CIV = np.array([12.82, 13.45, 13.64, 14.04])
    logf = np.array([-12.98, -12.97, -13.59, -14.39]) # f = dn/dN/dX

    # note: returning the log10 values
    return logN_CIV, logf

def reproduce_dodorico2013_fig18():
    # to check if my by-eye estimate of the data points looks ok
    B = 10**10.3
    logN_CIV = np.arange(12.4, 15.2, 0.1)
    alpha = 1.75

    f = civ_dndNdX_pl(B, alpha, 10**logN_CIV)
    data_logN_CIV, data_logf = dodorico2013_cddf()

    plt.plot(data_logN_CIV, data_logf, 'kx')
    plt.plot(logN_CIV, np.log10(f), ':', label=r'f(N) = B N$^{-\alpha}$')

    plt.legend(fontsize=13)
    plt.xlabel('log N(CIV)', fontsize=13)
    plt.ylabel('log f', fontsize=13)
    plt.xlim([12.4, 15.2])
    plt.ylim([-17.4, -11.2])
    plt.show()

def fit_dodorico2013_schechter():
    B = 10 ** 10.3
    logN_CIV = np.arange(12.4, 15.2, 0.1)
    alpha = 1.75

    # convert dX to dz using cosmology from D'Odorico et al. (2013)
    omega_m = 0.26
    omega_lambda = 1 - omega_m
    z = 4.8
    dX_dz = convert_dXdz(omega_m, omega_lambda, z)

    # D'Odorico data and fit
    f = civ_dndNdX_pl(B, alpha, 10 ** logN_CIV)
    data_logN_CIV, data_logf = dodorico2013_cddf()

    f_dz = f*dX_dz
    data_logf_dz = np.log10(dX_dz*10**(data_logf)) # dn/dN/dz

    # Schechter's fit
    norm, alpha, N_star = 1e-13, -0.80, 10 ** 14.0
    dn_dNdz_sch = civ_dndNdz_sch2(norm, alpha, N_star, 10 ** logN_CIV)

    # PL + exp fit
    dn_dNdX_sch2 = civ_dndNdX_pl_sch(N_star)
    dn_dNdz_sch2 = dX_dz * dn_dNdX_sch2

    # plotting
    plt.figure(figsize=(8,6))
    plt.plot(data_logN_CIV, data_logf_dz, 'kx', label='4.35 < z < 5.3', ms=8, mew=2)
    plt.plot(logN_CIV, np.log10(f_dz), ':', label=r"D'Odorico fit: f(N) = B N$^{-\alpha'}$")
    plt.plot(logN_CIV, np.log10(dn_dNdz_sch), '--', label=r"Schechter fit: $A_{norm}$ $(N/N*)^{\alpha}$ $e^{-N/N*}$")
    plt.plot(logN_CIV, np.log10(dn_dNdz_sch2), '--', label=r"PL + Exp fit: B N$^{-\alpha'}$ $e^{-N/N*}$")

    plt.legend(fontsize=11)
    plt.xlabel('log N(CIV)', fontsize=13)
    plt.ylabel('log (dn/dN/dz)', fontsize=13)
    plt.xlim([12.4, 15.2])
    #plt.ylim([-17.4, -11.2])
    plt.show()

def fit_cooksey(try_norm):
    # in progress

    W = np.arange(0.03, 2.5, 0.01)
    z = 3.25
    dn_dzdW, dn_dXdW = civ_dndzdW(W, z, type='Cooksey')

    converted_N = convert_W2N_civ(W)
    plt.plot(np.log10(converted_N), np.log10(try_norm*dn_dXdW))

    ####
    data_logN_CIV, data_logf = dodorico2013_cddf()
    plt.plot(data_logN_CIV, data_logf, 'kx', label='4.35 < z < 5.3', ms=8, mew=2)
    ####
    logN_CIV = np.arange(12.4, 15.2, 0.1)
    norm, alpha, N_star, z = 1e-13, -0.80, 10 ** 14.0, 4.8
    dn_dNdX_sch = civ_dndNdz_sch2(norm, alpha, N_star, 10 ** logN_CIV, z=z)
    plt.plot(logN_CIV, np.log10(dn_dNdX_sch), '--', label=r"Schechter fit: $A_{norm}$ $(N/N*)^{\alpha}$ $e^{-N/N*}$")

    plt.show()

#####################
def metal_W2bN(W, metal_ion='C IV'):
    # see enigma.reion_forest.utils.mgii_W2bN

    # hack for now
    cgm_dict = {'b_weak': 20.0, 'b_strong': 200.0, 'logN_metal_min': 10.0, 'logN_metal_max': 17.0, 'logN_strong': 16.0, 'logN_trans': 0.25, \
                'W_min': W.min(), 'W_max': W.max()}

    b_weak = cgm_dict['b_weak']
    b_strong = cgm_dict['b_strong']
    logN_metal_min = cgm_dict['logN_metal_min']
    logN_metal_max = cgm_dict['logN_metal_max']
    logN_strong = cgm_dict['logN_strong']
    logN_trans = cgm_dict['logN_trans']
    W_min = cgm_dict['W_min']
    W_max = cgm_dict['W_max']

    dvpix = 2.0
    vgrid_min = 0.0
    v_metal = reion_utils.vel_metal_doublet(metal_ion).value
    vgrid_max = 5.0*np.max(np.array([b_strong, v_metal]))
    vmid = vgrid_min + (vgrid_max - vgrid_min)/2.0 # midpoint velocity values on grid
    vel_grid = np.arange(vgrid_min, vgrid_max, dvpix)

    # start here
    nabs = W.shape[0]
    nN = 101
    logN_metal = logN_metal_min + (logN_metal_max - logN_metal_min)*np.arange(nN)/(nN - 1) # grid of logN_metal values
    v_abs = np.full(nN, vmid) # Just center these for determining the EW logN_MgII relation

    # sigmoid logistic function allows for smooth transition between b_weak and b_strong at the activiation locatino
    # logN_strong, over an interval of ~ 4*logN_trans about logN_strong
    sigmoid_arg = (logN_metal - logN_strong)/logN_trans
    b_val = b_weak + (b_strong - b_weak)*special.expit(sigmoid_arg)

    # calculate W numerically from a grid of N and b values, which allows you to construct a W-logN and W-b relations
    # interpolate these relations to obtain N and b for the desired W
    tau_weak, W_blue = reion_utils.metal_voigt(vel_grid, v_abs, b_val, logN_metal, metal_ion=metal_ion)
    if (W_blue.value.max() < W_max) or (W_blue.value.min() > W_min):
        raise ValueError('The N and b you are using cannot describe the full range of W requested. Revisit cgm_dict params')
    N_interp = interp1d(W_blue.value, logN_metal, kind='cubic', bounds_error=True) # W-logN relation
    b_interp = interp1d(W_blue.value, b_val, kind='cubic', bounds_error=True) # W-b value relation
    logN_out = N_interp(W)
    b_out = b_interp(W)
    return logN_out, b_out