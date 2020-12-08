import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from astropy.cosmology import Planck15

# is this right?
def civ_dndNdz(n_star, alpha, N_star, N):

    # Schechter function for CIV column density (N) distribution.
    # note: small n refers to number

    dn_dNdz = (n_star / N_star) * np.power(N / N_star, alpha) * np.exp(-N/ N_star)
    return dn_dNdz

# dn_dNdX_sch = cgm.civ_dndNdz_test(1e-13, -0.8, 10**14, 10**logN_CIV, z=4.5) seems ok
# when compared with dodorico2013_cddf()
def civ_dndNdz_test(normalization, alpha, N_star, N, z=None):

    # Schechter function for CIV column density (N) distribution.
    # note: small n refers to number

    dn_dNdz = normalization * np.power(N / N_star, alpha) * np.exp(-N/ N_star)
    if z != None:
        omega_m = Planck15.Om0
        omega_lambda = 1 - omega_m
        dz_dX = 1/(((1 + z) ** 2) * (omega_m * (1 + z) ** 3 + omega_lambda) ** (-0.5))
        dn_dNdX = dn_dNdz * dz_dX
        return dn_dNdX
    else:
        return dn_dNdz

def civ_dndz(n_star, alpha, N_star, N_min, N_max):
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

##### EW distribution function #####
def civ_dndzdW(W, z, type, k=None, alpha=None):
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
    dn_dXdW = k*np.exp(alpha*W)

    # converting to dNdzdW
    omega_m = Planck15.Om0
    omega_lambda = 1 - omega_m
    dX_dz = ((1 + z)**2)*(omega_m*(1 + z)**3 + omega_lambda)**(-0.5) # correct for discrete dz??
    dn_dzdW = dn_dXdW * dX_dz

    return dn_dzdW, dn_dXdW

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
##############################

def civ_dndNdX(B, alpha, N_CIV):
    # alpha = 1.71 or 1.8 from D'Odorico et al. (2010)
    # alpha = 1.75 for D'Odorico et al. (2013), Figure 18, 4.35 < z < 5.3
    # what are obs values for B?
    #    - log(B) = 10.3, by eye based on Figure 18 (i.e. for logN = 12.4 and logf = -12.2) of D'Odorico et al (2013)

    dn_dNdX = B*N_CIV**(-alpha) # power law form for column density distribution function (CDDF)
    return dn_dNdX

# to be verified if correct
def civ_dndNdz(B, alpha, N_CIV, z):
    dn_dNdX = B*N_CIV**(-alpha)

    # converting to dNdzdW
    omega_m = Planck15.Om0
    omega_lambda = 1 - omega_m
    dX_dz = ((1 + z) ** 2) * (omega_m * (1 + z) ** 3 + omega_lambda) ** (-0.5)  # correct for discrete dz??

    dn_dNdz = dn_dNdX * dX_dz
    return dn_dNdz

##### data observations #####
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
def dodorico2013_cddf(z=None):
    # D'Odorico et al. (2013)
    # data points for CDDF not provided, so estimating the points by eye from Figure 18
    # for 4.35 < z < 5.3

    logN_CIV = [12.82, 13.45, 13.64, 14.04]
    logf = [-12.98, -12.97, -13.59, -14.39] # f = dn/dN/dX

    """
    if z != None:
        omega_m = Planck15.Om0
        omega_lambda = 1 - omega_m
        dX_dz = ((1 + z) ** 2) * (omega_m * (1 + z) ** 3 + omega_lambda) ** (-0.5)
        logf = np.array(logf) * dX_dz
    """
    return logN_CIV, logf

def reproduce_dodorico2013_fig18():
    B = 10**10.3
    logN_CIV = np.arange(12.4, 15.2, 0.1)
    alpha = 1.75

    f = civ_dndNdX(B, alpha, 10**logN_CIV)
    data_logN_CIV, data_logf = dodorico2013_cddf()

    plt.plot(logN_CIV, np.log10(f), '--', label=r'f(N) = B N$^{-\alpha}$')
    plt.plot(data_logN_CIV, data_logf, 'x')
    plt.legend(fontsize=13)
    plt.xlabel('log N(CIV)', fontsize=13)
    plt.ylabel('log f', fontsize=13)
    plt.xlim([12.4, 15.2])
    plt.ylim([-17.4, -11.2])
    plt.show()