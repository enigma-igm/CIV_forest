import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import integrate, special
from scipy.interpolate import interp1d
from astropy.cosmology import Planck15
import mpmath
from astropy import constants as const
from astropy import units as u
import enigma.reion_forest.utils as reion_utils
import matplotlib.style as style
style.use('tableau-colorblind10')

########## literature data ##########
def dodorico2013_cddf_bosman():
    dl_n = np.array([12.85, 13.25, 13.65, 14.05]) # log10(N)
    dl = np.array([-12.8942474, -13.0310060, -13.58779213, -14.5044219]) # log10(d^2 n/(dN dX))
    dl_bot = np.array([-13.19, -13.11406, -13.68934, -14.7106])
    dl_top = np.array([-12.71, -12.96130, -13.50554, -14.3651])
    dl_err = np.array([dl-dl_bot, dl_top-dl])

    # returning log10(N), log10(d^2 n/dN/dX), error on log10(d^2 n/dN/dX)
    return dl_n, dl, dl_err

def simcoe2011_cddf_bosman():

    from astropy.cosmology import FlatLambdaCDM
    cosmo_this = FlatLambdaCDM(H0=67.74, Om0=0.3089, Tcmb0=2.725) # what Sarah had in her code
    #cosmo_this = FlatLambdaCDM(H0=65, Om0=1.0)

    def abspath(z1, z2):
        return cosmo_this.absorption_distance(z1) - cosmo_this.absorption_distance(z2)

    # S11 length
    xp = abspath(4.475, 4.024) + abspath(4.353, 4.000) + abspath(4.397, 4.122)

    # S11 bins
    binn = np.arange(11.8, 14.6, 0.3)
    binlen = 0.3

    # S11 list of systems (paper)
    systems = [3.17e12, 1.45e12, 4.48e12, 6.62e12, 6.53e12, 1.406e13, 1.499e13, 9.96e12, 9.18e12, 9.76e12, 1.9e13,
               6.301e13, 3.027e12, 2.055e13, 1.854e12, 1.298e13, 2.136e12, 6.06e11, 6.524e12, 2.4015e12, 2.833e12,
               1.3346e13, 4.20e12, 1.187e12, 1.896e13, 2.149e13, 4.352e13, 1.933e12, 1.329e13, 2.229e12, 2.185e13,
               3.049e12, 9.325e12, 6.065e12, 1.95e13, 3.804e12, 2.932e12, 1.792e14, 7.748e12, 5.041e12, 2.864e12]

    # calculating density
    logsys = np.log10(systems)
    a = np.histogram(logsys, binn - (binlen / 2.))
    dn = 10 ** (binn + (binlen / 2)) - 10 ** (binn - (binlen / 2)) # logN
    df = a[0] / (xp * dn[:-1]) # d^2 n/dN/dX, in linear scale

    # calculating uncertainty
    toperr = np.zeros_like(a[0]).astype('float')
    boterr = np.zeros_like(a[0]).astype('float')
    # approximate-Poisson error bars (not great but still better than original S11!)
    for i in range(len(a[0])):
        if a[0][i] == 1:
            toperr[i] = 3.3
            boterr[i] = 0.33
        else:
            toperr[i] = np.sqrt(a[0][i].astype('float')) + a[0][i]
            boterr[i] = a[0][i] - np.sqrt(a[0][i].astype('float'))

    dftop = (toperr) / (xp * dn[:-1])
    dfbot = (boterr) / (xp * dn[:-1])
    y_error = [np.log10(df) - np.log10(dfbot), np.log10(dftop) - np.log10(df)]

    out_logN = binn[:-1]
    out_logf = np.log10(df)
    out_logf_err = np.array(y_error)
    #out_logf_bot = np.log10(dfbot)
    #out_logf_top = np.log10(dftop)

    return out_logN, out_logf, out_logf_err #out_logf_bot, out_logf_top

def cooksey_dndXdW(W):
    # W = np.arange(0.03, 2.5, 0.01), set manually

    # exponential form fit to Cooksey et al. (2013) data at z=3.25
    # Table 4: The frequency distribution was fit with an exponential for absorbers with Wr >= 0.6 A
    k = 1.82
    alpha = -2.61
    dn_dXdW = k * np.exp(alpha * W)

    return dn_dXdW

def plot_alldata_raw():
    # plot data in their raw units as presented in the paper
    do_logN_CIV, do_logf, do_logf_err = dodorico2013_cddf_bosman()
    simcoe_logN_CIV, simcoe_logf, simcoe_logf_err = simcoe2011_cddf_bosman()

    Win = np.arange(0.6, 3.0, 0.01)
    cooksey_datafit = cooksey_dndXdW(Win)

    plt.figure(figsize=(10,5))
    plt.subplot(121)
    plt.errorbar(do_logN_CIV, do_logf, yerr=do_logf_err, marker='x', color='k', mew=2, ms=8, label=r"D'Odorico xshooter data ($z_{med}$ = 4.8)")
    plt.errorbar(simcoe_logN_CIV, simcoe_logf, yerr=simcoe_logf_err, marker='+', color='b', mew=2, ms=10, label="Simcoe MIKE data (z=4.25)")

    #plt.axvline(12.0, color='b', ls='--', alpha=0.7, label='Simcoe rough detection limit')
    #plt.axvline(13.3, color='k', ls='--', alpha=0.7, label="D'Odorico 85% complete")
    plt.xlim([11, 15])
    plt.ylim([-17, -10])
    plt.xlabel('log N(CIV)', fontsize=13)
    plt.ylabel(r'log($d^2n/dN/dX$)', fontsize=13)
    plt.legend(loc=3)

    plt.subplot(122)
    plt.plot(np.log10(Win), np.log10(cooksey_datafit), 'k-', label='Cooksey data fit ($<z>$=3.25)')
    plt.xlabel('log (W)', fontsize=13)
    plt.ylabel(r'log($d^2n/dW/dX$)', fontsize=13)
    plt.legend()

    plt.tight_layout()
    plt.show()

########## converting functions
def convert_dXdz(z, omega_m=0.3, omega_lambda=0.7):
    # default values are Nyx cosmology
    dX_dz = ((1 + z) ** 2) * (omega_m * (1 + z) ** 3 + omega_lambda) ** (-0.5) # Eqn(1) from DO+(2013) paper
    return dX_dz

def convert_data_dXtodz(logf, logf_err, z):

    dX_dz = convert_dXdz(z)
    logf_bot = logf - logf_err[0]
    logf_top = logf + logf_err[1]

    logf_dz = np.log10(dX_dz * 10 ** (logf))
    logf_dz_bot = np.log10(dX_dz * 10 ** (logf_bot))
    logf_dz_top = np.log10(dX_dz * 10 ** (logf_top))
    logf_dz_err = np.array([logf_dz - logf_dz_bot, logf_dz_top - logf_dz])

    return logf_dz, logf_dz_err

def convert_data_dNtodW(logN, logf, logf_err, W_range, cgm_dict):

    logf_bot  = logf - logf_err[0]
    logf_top = logf + logf_err[1]

    W_blue, logN_metal = reion_utils.metal_W2bN(W_range, cgm_dict=cgm_dict, return_wtau=True)
    W_interp = interp1d(logN_metal, W_blue.value, kind='cubic', bounds_error=True)
    W_out = W_interp(logN)  # interpolated EWs

    dw_dn = np.gradient(W_out, 10 ** logN, edge_order=2)  # dW/dN
    #f_new = 10**logf / dw_dn
    #f_bot_new = 10**logf_bot / dw_dn
    #f_top_new = 10**logf_top / dw_dn

    logf_new = np.log10(10 ** logf / dw_dn)
    logf_bot_new = np.log10(10 ** logf_bot / dw_dn)
    logf_top_new = np.log10(10 ** logf_top / dw_dn)

    logf_err_new = np.array([logf_new - logf_bot_new, logf_top_new - logf_new])

    return W_out, logf_new, logf_err_new

########## Schechter function ##########
def civ_dndzdW_sch(W, W_star, n_star, alpha, z=None):

    # Eqn (15) from Joe's paper, where for Mg II forest, (alpha, W_star, N_star) = -0.8, 1.0 A, 2.34
    dn_dzdW = (n_star / W_star) * np.power(W / W_star, alpha) * np.exp(-W/ W_star)

    # if z is provided, convert dz--> dX and return dn/dX/dW instead
    if z != None:
        print("converting dz to dX at z = %0.2f" % z)
        #omega_m = Planck15.Om0
        #omega_lambda = 1 - omega_m
        dX_dz = convert_dXdz(z)
        dn_dWdX = dn_dzdW * (1/dX_dz)
        return dn_dWdX
    else:
        return dn_dzdW

def civ_dndz_sch(n_star, alpha, W_star, W_min, W_max):
    """
    Compute Schechter integral (civ_dndzdW_sch above) over [W_min, W_max] interval using the incomplete gamma functions.
    """

    z = alpha + 1 # changing variable and integrating as a function of z
    upper = W_max / W_star
    lower = W_min / W_star

    # \Gamma(z, l, u) = \int_lower^upper x^(z-1) exp(-x) dx, where x = W/W_star
    if isinstance(W_max,float):
        I = float(mpmath.gammainc(z, lower, upper))
    elif isinstance(W_max,np.ndarray):
        I = np.zeros_like(W_max)
        for indx in range(W_max.size):
            I[indx] = float(mpmath.gammainc(z, lower, upper[indx]))
    dn_dz = n_star * I
    return dn_dz

########## fitting data with Schechter function ##########
def fit_alldata_dW(cgm_dict, cooksey_norm):
    # do fitting in terms of dW (and dz)

    # (1) Schechter function fit
    W_star, n_star, alpha = cgm_dict['W_star'], cgm_dict['n_star'], cgm_dict['alpha']
    W_min, W_max = cgm_dict['W_min'], cgm_dict['W_max']
    W_range = np.arange(W_min, W_max+0.01, 0.01)
    d2n_dzdW_sch = civ_dndzdW_sch(W_range, W_star, n_star, alpha)

    # (2) D'Odorico data (from Sarah Bosman)
    DO_logN_CIV, DO_logf, DO_logf_err = dodorico2013_cddf_bosman()
    DO_logf, DO_logf_err = convert_data_dXtodz(DO_logf, DO_logf_err, 4.8) # converting data dX to dz
    DO_W_out, DO_logf_new, DO_logf_err_new = convert_data_dNtodW(DO_logN_CIV, DO_logf, DO_logf_err, W_range, cgm_dict) # converting data from dN to dW

    # (3) Simcoe data
    simcoe_logN_CIV, simcoe_logf, simcoe_logf_err = simcoe2011_cddf_bosman()
    simcoe_logf, simcoe_logf_err = convert_data_dXtodz(simcoe_logf, simcoe_logf_err, 4.25) # converting data dX to dz
    simcoe_W_out, simcoe_logf_new, simcoe_logf_err_new = convert_data_dNtodW(simcoe_logN_CIV, simcoe_logf, simcoe_logf_err, W_range, cgm_dict) # converting from dN to dW

    # (4) Cooksey data fit
    W_cooksey = np.arange(0.6, W_max+0.01, 0.01) # starting at 0.6 A where the fit is applicable
    cooksey_datafit = cooksey_dndXdW(W_cooksey)
    dX_dz = convert_dXdz(3.25)
    cooksey_datafit = cooksey_datafit * dX_dz

    ##### plotting (paper plot settings) #####
    font = {'family': 'serif', 'weight': 'normal'}  # , 'size': 11}
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

    plt.figure(figsize=(8, 6))
    plt.subplots_adjust(left=0.12, bottom=0.14, right=0.97, top=0.97)

    xytick_size = 16
    xylabel_fontsize = 20
    legend_fontsize = 14
    linewidth = 2.5

    plt.errorbar(DO_W_out, DO_logf_new, yerr=DO_logf_err_new, fmt='o', ms=6, label=r"D'Odorico et al. (2013), $z$ = 4.8")
    plt.errorbar(simcoe_W_out, simcoe_logf_new, yerr=simcoe_logf_err_new, fmt='o', ms=6, label=r"Simcoe(2011), $z$ = 4.25")
    plt.errorbar(W_cooksey, np.log10(cooksey_norm * cooksey_datafit), fmt='r--', lw=3.0, zorder=1, label=r'Cooksey et al. (2013), $z$ = 3.25')
    plt.errorbar(W_range, np.log10(d2n_dzdW_sch), fmt='k-', alpha=0.7, lw=linewidth, zorder=-1, label=r"Schechter fit ($\alpha=%0.2f, W*=%0.2f, n*=%0.2f$)" % (alpha, W_star, n_star))

    plt.xscale('log')
    plt.gca().minorticks_on()
    plt.gca().tick_params(which='both', labelsize=xytick_size)
    plt.legend(fontsize=legend_fontsize, loc=3)
    plt.xlabel(r'W [$\AA$]', fontsize=xylabel_fontsize)
    plt.ylabel(r'log(d$^2$n/dW/dz)', fontsize=xylabel_fontsize)
    plt.show()

########## cgm model dictionary ##########
def init_metal_cgm_dict(alpha=-0.50, W_star = 0.45, n_star = 28.0, \
                        W_min=0.001, W_max=5.0, b_weak=10.0, b_strong=150.0, \
                        logN_metal_min=10.0, logN_metal_max=22.0, logN_strong=14.5, logN_trans=0.35):

    cgm_dict = dict(n_star=n_star, alpha=alpha, W_star=W_star, W_min=W_min, W_max=W_max, b_weak=b_weak, b_strong=b_strong, \
                    logN_metal_min=logN_metal_min, logN_metal_max=logN_metal_max, logN_strong=logN_strong, logN_trans=logN_trans)

    return cgm_dict

########## various presets ##########
def cgm_model1():
     out_dict = init_metal_cgm_dict() # default
     fit_alldata_dW(out_dict, 1.0)

def cgm_model2():
    out_dict = init_metal_cgm_dict(alpha=-1.1, n_star=5)
    fit_alldata_dW(out_dict, 1.0)

def cgm_model3():
    out_dict = init_metal_cgm_dict(alpha=-0.95, n_star=5)
    fit_alldata_dW(out_dict, 1.0)