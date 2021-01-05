'''
Functions here:
    - civ_dndzdW_sch
    - convert_dXdz
    - civ_dndzdW
    - civ_dndz_cooksey
    - civ_dndNdX_pl
    - cooksey2013_dndz
    - dodorico2013_cddf
    - reproduce_dodorico2013_fig18
    - metal_W2bN
    - plot_multiple_cog
    - reproduce_cooksey_w
    - dwdn_numerical
    - dwdn_theory
    - fit_alldata_dW
    - fit_alldata_dN
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate, special
from scipy.interpolate import interp1d
from astropy.cosmology import Planck15
import mpmath
from astropy import constants as const
from astropy import units as u
import enigma.reion_forest.utils as reion_utils

########## Schechter function
def civ_dndzdW_sch(W, W_star, n_star, alpha, z=None):

    # Eqn (15) from Joe's paper, where for Mg II forest, (alpha, W_star, N_star) = -0.8, 1.0 A, 2.34
    dn_dzdW = (n_star / W_star) * np.power(W / W_star, alpha) * np.exp(-W/ W_star)

    # if z is provided, return dn/dX/dW instead
    if z != None:
        omega_m = Planck15.Om0
        omega_lambda = 1 - omega_m
        dz_dX = 1/(((1 + z) ** 2) * (omega_m * (1 + z) ** 3 + omega_lambda) ** (-0.5))
        dn_dWdX = dn_dzdW * dz_dX
        return dn_dWdX
    else:
        return dn_dzdW

########## convenience function
def convert_dXdz(omega_m, omega_lambda, z):

    #omega_m = Planck15.Om0
    #omega_lambda = 1 - omega_m
    dX_dz = ((1 + z) ** 2) * (omega_m * (1 + z) ** 3 + omega_lambda) ** (-0.5)
    return dX_dz

########## fits from literature
def civ_dndzdW(W, z, type, k=None, alpha=None):
    # dn/dz/dW from Cooksey et al. (2013)
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

def civ_dndz_cooksey(k, alpha, z, W_min, W_max, nW):

    # in progress: integral of dn/dz/dW above
    # want to match to Simcoe et al. (2011) or Songaila (2001) dN/dz (which one?) by adjusting alpha and k
    W = np.linspace(W_min, W_max, nW)
    dN_dXdW = k * np.exp(alpha * W)
    dN_dX = integrate.simps(dN_dXdW, W) # integrate dN_dXdW over dW to get dN_dX

    # convert dN_dX to dN_dz
    omega_m = Planck15.Om0
    omega_lambda = 1 - omega_m
    dX_dz = ((1 + z) ** 2) * (omega_m * (1 + z) ** 3 + omega_lambda) ** (-0.5)
    dN_dz = dN_dX * dX_dz

    return dN_dX, dN_dz

def civ_dndNdX_pl(B, alpha, N_CIV):
    # power law fit for dn/dN/dX from D'Odorico et al. (2013)
    # alpha = 1.71 or 1.8 from D'Odorico et al. (2010)
    # alpha = 1.75 for D'Odorico et al. (2013), Figure 18, 4.35 < z < 5.3
    # what are obs values for B?
    #    - log(B) = 10.3, by eye based on Figure 18 (i.e. for logN = 12.4 and logf = -12.2) of D'Odorico et al (2013)

    dn_dNdX = B*N_CIV**(-alpha) # power law form for column density distribution function (CDDF)
    return dn_dNdX

########## converting W to N
def metal_W2bN(W, cgm_dict=None, b_in=None, metal_ion='C IV', return_wtau=False, plot=False):
    """
    Modified from enigma.reion_forest.utils.mgii_W2bN

    Params:
        W: list of equivalent widths in Angstrom
        cgm_dict: Python dictionary. If not provided, use default model below.
        b_in: b-value in km/s. If not provided, then model it using sigmoid function.
                 If provided, then assume constant b-value.
    """

    if cgm_dict == None:
        cgm_dict = {'b_weak': 20.0, 'b_strong': 150.0, 'logN_metal_min': 10.0, 'logN_metal_max': 22.0, 'logN_strong': 14.5, 'logN_trans': 0.25, 'W_min': W.min(), 'W_max': W.max()}

    b_weak = cgm_dict['b_weak']
    b_strong = cgm_dict['b_strong']
    logN_metal_min = cgm_dict['logN_metal_min'] # minimum N value to construct interpolation grid
    logN_metal_max = cgm_dict['logN_metal_max'] # maximum N value to construct interpolation grid
    logN_strong = cgm_dict['logN_strong']
    logN_trans = cgm_dict['logN_trans']
    W_min = cgm_dict['W_min']
    W_max = cgm_dict['W_max']

    dvpix = 2.0
    vgrid_min = 0.0
    v_metal = reion_utils.vel_metal_doublet(metal_ion).value
    vgrid_max = 5.0*np.max(np.array([b_strong, v_metal]))
    vmid = vgrid_min + (vgrid_max - vgrid_min)/2.0 # location of absorbers on vel_grid
    vel_grid = np.arange(vgrid_min, vgrid_max, dvpix) # vel grid to place absorbers
    print("vgrid_min", vgrid_min, "vgrid_max", vgrid_max)

    nabs = W.shape[0]
    nN = 101
    logN_metal = logN_metal_min + (logN_metal_max - logN_metal_min)*np.arange(nN)/(nN - 1) # grid of logN_metal values
    v_abs = np.full(nN, vmid) # Just center these for determining the EW logN_MgII relation

    # sigmoid logistic function allows for smooth transition between b_weak and b_strong at the activiation locatino
    # logN_strong, over an interval of ~ 4*logN_trans about logN_strong
    if b_in == None:
        sigmoid_arg = (logN_metal - logN_strong)/logN_trans
        b_val = b_weak + (b_strong - b_weak)*special.expit(sigmoid_arg)
    else:
        b_val = np.ones(nN) * b_in

    # calculate W numerically from a grid of N and b values, which allows you to construct a W-logN and W-b relations
    # interpolate these relations to obtain N and b for the desired W
    tau_tot, W_blue = reion_utils.metal_voigt(vel_grid, v_abs, b_val, logN_metal, metal_ion=metal_ion)

    if (W_blue.value.max() < W_max) or (W_blue.value.min() > W_min):
        print(W_blue.min(), W_blue.max())
        raise ValueError('The N and b you are using cannot describe the full range of W requested. Revisit cgm_dict params')

    if return_wtau:
        return W_blue, logN_metal
    else:
        N_interp = interp1d(W_blue.value, logN_metal, kind='cubic', bounds_error=True) # W-logN relation
        b_interp = interp1d(W_blue.value, b_val, kind='cubic', bounds_error=True) # W-b value relation
        logN_out = N_interp(W)
        b_out = b_interp(W)

        if plot:
            plt.axhline(np.log10(0.6), ls='--')
            plt.plot(logN_out, np.log10(W), 'k')
            plt.xlabel('log(N)', fontsize=13)
            plt.ylabel(r'log($W_{blue}$)', fontsize=13)
            plt.title('b_weak=%0.1f, b_strong=%0.1f, logN_strong=%0.1f' % (b_weak, b_strong, logN_strong), fontsize=13)
            plt.grid()
            plt.show()

        return logN_out, b_out

def plot_multiple_cog(W, b_list):
    # plotting multiple COG at various b-values

    for b in b_list:
        logN_out, _ = metal_W2bN(W, b_in=b)
        plt.plot(logN_out, np.log10(W), label='b = %d km/s' % b)

    plt.axhline(np.log10(0.6), c='r', ls='--')
    plt.xlabel('log(N)', fontsize=13)
    plt.ylabel(r'log($W_{blue}$)', fontsize=13)
    plt.grid()
    plt.legend()
    plt.show()

def dwdn_theory():
    # calculate theoretical value of dW/dN on the linear part of the COG

    #w_lambda = N * f * np.pi * (ec ** 2) / (me * c ** 2) * (wrest_civ ** 2)
    wrest_civ = 1548 * u.Angstrom
    ec = (const.e.esu.value) * (u.cm) ** (3 / 2) * (u.g) ** (1 / 2) / u.s
    me = const.m_e.to('g')
    c = const.c.to('cm/s')
    f = 0.1899  # oscillator strength for CIV 1548

    dW_dN = f * np.pi * (ec ** 2) / (me * c ** 2) * wrest_civ.to('cm') # W is dimensionless (W = Wlambda / lambda)
    dWlambda_dN = dW_dN * wrest_civ # units are Angstrom * cm2

    return dWlambda_dN

def dwdn_numerical(cgm_dict, b_in, plot=False):
    # compute dW/dN numerically

    W = np.arange(0.01, 5.0, 0.01)
    logN_out, b_out = metal_W2bN(W, cgm_dict=cgm_dict, b_in=b_in)

    if b_in == None:
        dw_db = np.gradient(W, b_out, edge_order=2)
        db_dn = np.gradient(b_out, 10**logN_out)
        dw_dn = dw_db * db_dn # in units of Angstrom cm^2
        dw_dn2 = np.gradient(W, 10**logN_out, edge_order=2) # same as dw_dn ...
    else:
        dw_dn = np.gradient(W, 10**logN_out, edge_order=2)

    # equivalent to results obtained below
    #W_border = np.array((W - 0.01 / 2).tolist() + [W[-1] + 0.01/2])
    #logN_out_border, b_out_border = metal_W2bN(W_border, b_in=b_in)
    #dw_dn2 = np.diff(W_border) / np.diff(10 ** logN_out_border)
    #dw_dn = dw_dn2

    if plot:
        plt.figure(figsize=(12,5))
        plt.subplot(131)
        plt.plot(logN_out, np.log10(W))
        plt.xlabel('log N(CIV)', fontsize=13)
        plt.ylabel('log (W)', fontsize=13)
        plt.axhline(np.log10(0.6), c='r', ls='--', label='0.6 A')
        plt.legend()
        plt.grid()

        plt.subplot(132)
        plt.plot(logN_out, b_out)
        plt.xlabel('log N(CIV)', fontsize=13)
        plt.ylabel('b (km/s)', fontsize=13)
        plt.grid()

        plt.subplot(133)
        dwdn_linear = dwdn_theory().value
        plt.plot(logN_out, dw_dn)
        plt.axhline(dwdn_linear, color='r', ls='--', label='theory dW/dN')
        plt.legend()
        plt.xlabel('log N(CIV)', fontsize=13)
        plt.ylabel('dW/dN', fontsize=13)
        plt.grid()

        plt.tight_layout()
        plt.show()

    return W, dw_dn, logN_out

########## data ##########
def cooksey2013_dndz():
    # Table 4 (CIV results summary) of Cooksey et al. (2013)
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

def dodorico2013_cddf():
    # D'Odorico et al. (2013), Figure 18
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

def reproduce_cooksey_w():
    # Cooksey+ (2013) claims W_1548 = 0.6A is saturated, which is logN ~ 14
    # Here trying to estimate what b value is required to get W=0.6A, assuming linear COG.
    # this gives b ~ 66 km/s and logN ~ 14.17

    wrest_civ  = 1548 * u.Angstrom
    W_lambda_saturate = 0.6 * u.Angstrom
    tau_saturate = 1.0

    c = const.c.to('km/s')
    b_linear_out = (W_lambda_saturate/tau_saturate) * c/np.sqrt(np.pi) * 1/wrest_civ # km/s

    # getting N now
    ec = const.e.esu # cgs
    ec = (const.e.esu.value) * (u.cm) ** (3 / 2) * (u.g) ** (1 / 2) / u.s
    me = const.m_e.to('g')
    c = const.c.to('cm/s')
    f = 0.1899  # oscillator strength for CIV 1548
    N = (tau_saturate * b_linear_out.to('cm/s')) / (f * wrest_civ.to('cm')) * me * c / (np.sqrt(np.pi) * ec**2)

    return N, b_linear_out

########## fitting data with Schechter function ##########
def fit_alldata_dW():
    # do fitting in terms of W

    W = np.arange(0.01, 5.0, 0.01)  # range for Cooksey and Schechter

    # D'Odorico data, converted to dW
    omega_m = 0.26
    omega_lambda = 1 - omega_m
    z = 4.8
    dX_dz = convert_dXdz(omega_m, omega_lambda, z)

    data_logN_CIV, data_logf = dodorico2013_cddf()
    data_logf_dz = np.log10(dX_dz * 10 ** (data_logf))  # f = dn/dN/dz
    data_f_dz = 10 ** data_logf_dz

    W_blue, logN_metal = metal_W2bN(W, return_wtau=True)
    W_interp = interp1d(logN_metal, W_blue.value, kind='cubic', bounds_error=True)
    W_out_data = W_interp(data_logN_CIV)
    dw_dn2 = np.gradient(W_out_data, 10 ** data_logN_CIV, edge_order=2)
    dn_dzdW_do = data_f_dz / dw_dn2
    dn_dXdW_do = (10** data_logf) / dw_dn2

    # D'Odorico fit parameters
    B = 10 ** 10.3
    logN_CIV = np.arange(12.4, 15.2, 0.1)  # range from D'Odorico
    alpha = 1.75
    f = civ_dndNdX_pl(B, alpha, 10 ** logN_CIV)
    f_dz = f * dX_dz  # converting data points from dX to dz

    W_out_fit = W_interp(logN_CIV)
    dw_dn3 = np.gradient(W_out_fit, 10 ** logN_CIV, edge_order=2)
    f_dW = f/dw_dn3
    f_dzdW = f_dz/dw_dn3

    # Cooksey's fit at z=3.25
    dn_dzdW_cook, dn_dXdW_cook = civ_dndzdW(W, 3.25, 'Cooksey')

    # Schechter's fit
    W_star, n_star, alpha = 0.4, 15.0, -0.5 # these fit DO data well
    W_star, n_star, alpha = 0.4, 2.5, -0.02 # these fit Cooksey fit well
    W_star, n_star, alpha = 0.45, 28.0, -0.2 # trying to fit both DO (small W) and Cooksey (large W)
    dn_dzdW_sch = civ_dndzdW_sch(W, W_star, n_star, alpha)
    dn_dXdW_sch = civ_dndzdW_sch(W, W_star, n_star, alpha, z=4.5)

    ##### plotting #####
    plt.figure(figsize=(8,6))
    #plt.plot(np.log10(W_out_fit), np.log10(f_dzdW), ':', label="D'Odorico fit")
    plt.plot(np.log10(W), np.log10(13.0 * dn_dzdW_cook), '-', label='Cooksey fit (x arbitrary norm)')
    plt.plot(np.log10(W), np.log10(dn_dzdW_sch), '--', label=r"Schechter fit ($W*=%0.2f, N*=%0.2f, \alpha=%0.2f$" % (W_star, n_star, alpha))
    plt.plot(np.log10(W_out_data), np.log10(dn_dzdW_do), 'kx', label="D'Odorico data (4.35 < z < 5.3)", ms=8, mew=2)

    plt.legend(fontsize=11, loc=3)
    plt.xlabel('log W', fontsize=13)
    plt.ylabel('log (dn/dW/dz)', fontsize=13)
    plt.show()

def fit_alldata_dN():
    # do the fitting in terms of N

    W = np.arange(0.01, 5.0, 0.01) # range for Cooksey and Schechter

    # D'Odorico data
    omega_m = 0.26
    omega_lambda = 1 - omega_m
    z = 4.8
    dX_dz = convert_dXdz(omega_m, omega_lambda, z)

    data_logN_CIV, data_logf = dodorico2013_cddf()
    data_logf_dz = np.log10(dX_dz * 10 ** (data_logf))  # f = dn/dN/dz

    # dw/dN for converting Schechter and Cooksey
    _, dw_dn, logN_out = dwdn_numerical(None, None) # using default cgm_dict and sigmoid function for b-value

    # Cooksey's fit
    dn_dzdW_cook, dn_dXdW_cook = civ_dndzdW(W, 3.25, 'Cooksey')
    dn_dzdN_cook = dn_dzdW_cook * dw_dn

    # Schechter's fit
    W_star, n_star, alpha = 0.4, 15.0, -0.5  # these fit DO data well
    W_star, n_star, alpha = 0.4, 2.5, -0.02  # these fit Cooksey fit well
    W_star, n_star, alpha = 0.45, 28.0, -0.2 # trying to fit both DO (small W) and Cooksey (large W)
    dn_dzdW_sch = civ_dndzdW_sch(W, W_star, n_star, alpha)
    dn_dXdW_sch = civ_dndzdW_sch(W, W_star, n_star, alpha, z=4.5)
    dn_dzdN_sch = dn_dzdW_sch * dw_dn

    ##### plotting #####
    plt.figure(figsize=(8,6))
    plt.plot(logN_out, np.log10(13.0 * dn_dzdN_cook), '-', label='Cooksey fit (x arbitrary norm)')
    plt.plot(logN_out, np.log10(dn_dzdN_sch), '--', lw=2.5, label=r"Schechter fit ($W*=%0.2f, N*=%0.2f, \alpha=%0.2f$" % (W_star, n_star, alpha))
    plt.plot(data_logN_CIV, data_logf_dz, 'kx', label="D'Odorico data (4.35 < z < 5.3)", ms=8, mew=2)

    plt.legend(fontsize=11, loc=3)
    plt.xlabel('log N(CIV)', fontsize=13)
    plt.ylabel('log (dn/dN/dz)', fontsize=13)
    plt.show()
