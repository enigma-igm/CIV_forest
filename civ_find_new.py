import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d, interp2d
from scipy.ndimage.filters import gaussian_filter1d
from scipy.ndimage import correlate1d
from scipy import optimize

from scipy import integrate
from scipy import stats
from astropy import constants as const
from astropy import units as u
from tqdm.auto import tqdm
#from progressbar import ProgressBar, Counter, Timer
from linetools.lists.linelist import LineList
from IPython import embed
from enigma.reion_forest import utils
#from pypeit.core.arc import detect_peaks
from sklearn.neighbors import KDTree
#from numba import jit


# changing values to civ values
A_VOIGT_B_MGII = utils.a_voigt_b('C IV').value  # 0.00578569
strong_lines = LineList('Strong', verbose=False)
mgii_2796 = strong_lines['CIV 1548'] # blue
mgii_2803 = strong_lines['CIV 1550'] # red

# the rest remains unchanged
wave_2796 = mgii_2796['wrest']
wave_2803 = mgii_2803['wrest']
f_2796 = mgii_2796['f']
f_2803 = mgii_2803['f']
wave_eff = (wave_2796 + wave_2803) / 2.0
c_light = const.c.to('km/s')
v_mgII = c_light*(wave_2803 - wave_2796)/wave_eff # doublet separation in km/s
v_mgII = v_mgII.value
nu_mgII = wave_2796.to(u.Hz, equivalencies=u.spectral())
print("v_CIV", v_mgII)

# these are used if you use MgiiFinder.fit and mgii_fit
logN_fid = 13.0
N_fid = np.power(10.0, logN_fid)/u.cm/u.cm
tau_c_2796_fid = (((np.pi * const.e.esu ** 2 * f_2796) / (const.m_e * nu_mgII) * N_fid).to('km/s')).value
tau_c_2803_fid = tau_c_2796_fid * f_2803 / f_2796
#print("tau_c_2796_fid", tau_c_2796_fid)
#print("tau_c_2803_fid", tau_c_2803_fid)

def mgii_kernel(vgrid, fwhm, logN=13.5):
    imid = np.round(vgrid.size/2).astype(int)
    dv_pix = np.median(np.diff(vgrid))
    v_kern0 = vgrid - vgrid[imid]
    v_abs = np.array([0.0])
    bval = np.array([np.sqrt(2.0) * fwhm / 2.35])
    logN_MgII = np.array([logN])
    #tau_tot, W_2796 = utils.mgii_voigt(v_kern0, v_abs, bval, logN_MgII)
    tau_tot, W_2796 = utils.metal_voigt(v_kern0, v_abs, bval, logN_MgII, metal_ion='C IV')  # utils.mgii_voigt(v_kern0, v_abs, bval, logN_MgII)

    kernel0 = 1.0 - np.exp(-tau_tot.flatten())
    # Center the kernel about zero
    nkern = 2 * np.ceil((bval[0] / dv_pix) * 20).astype(int)
    kernel = kernel0[imid - nkern:imid + nkern + 1]
    v_kern = v_kern0[imid - nkern:imid + nkern + 1]
    # kernel = kernel[::-1]/np.sum(kernel)
    kernel = kernel / np.sum(kernel)

    return v_kern, kernel


class MgiiFinder(object):
    def __init__(self, vgrid, flux, ivar, fwhm, signif_thresh, gpm=None,
                 signif_mask_nsigma=None, signif_mask_dv=300.0, one_minF_thresh=np.inf, model_fit_thresh=1e-4,
                 find_alias=False, logN=13.5, Pfit_min=-3.0, LR_max=-10.0, prob_cuts=True, trim_igm=True,
                 bval_igm=150.0, W_2796_igm=1.0, debug=False, diffevol=False, min_peak_dist=None):
        """

        Args:
            vgrid (ndarray):
               Velocity grid (nspec,)
            flux (ndarray):
               Flux array (nskew, nspec)
            ivar (ndarray):
               Inverse variance array (nskew, nspec)
            fwhm (float):
               Spectral FWHM
            signif_thresh (float):
               Significance threshold for matched filter convolution
            gpm (ndarray):
               Good pixel mask, optional
            signif_mask_nsigma (float):
               A threshold to be applied to the MgII kernel convolved significance spectrum. The signif_gpm sets
               all pixels above this value to be bad pixels (False). We also identify all peaks above this
               significance and mask a region of width signif_mask_dv about each peak
            signif_mask_dv (float):
               Velocity window to mask on each side (i.e. +- signif_mask_dv) of a peak in the significance spectrum
               that is higher than the signif_mask_nsigma threshold. Note that the absorption regions for both 2796 and
               2803 are masked here.
            one_min_Fthresh (float):
               Threshold in 1-F to be used to create the flux_gpm mask. All values above this flux threshold will be masked.
            logN (float):
               Column density used to create matched filter
            Pfit_min (float): default = -3
               Goodness of fit probability threshold. The Pfit is the log10 of the value of the chi^2 cumualtive
               P(> chi^2 | dof) distribution for the number of degrees of freedom. The default=-3,  i.e. a cumulative
               probability of 1e-3. Good absorbers will have a large value of Pfit. Choose Pfit_min = -np.inf to not cut anything
            LR_max (float): default=-10
               The likelihood ratio is defined to be LR = exp(-chi^2_null/2)/exp(-chi^2_opt/2), where chi^2_null is
               the null hypothesis of no absorber and chi^2_opt is the chi^2 minimum at the best fit parameter values.
               The LR is bounded between [0.0, 1.0]. We take the log10(LR), and LR_thresh is a threshold in this quantity.
               If an absorber is clearly favored, then LR = 0.0, whereas if the null hypothesis is just as good a fit
               as the best fit model, LR = 1.0. Good absorbers will have a value of LR very close to zero.
               Choose LR_max = np.inf to keep everything
            trim_igm (bool): default=True
               If true trim out IGM absorbers as described below.
            bval_igm (float): default = 150.0
               IGM absorbers tend to be fit with large b-values and low W_2796. We filter out absorbers with:

                  IGM_abs = (bval_fit > bval_igm) & (W_2796_fit1 < W_2796_igm)
            W_2796_igm (float): default = 1.0
               See above.

            debug:

        Returns:

        """

        # Initialize input attributes
        self.vgrid = vgrid
        self.flux = flux
        self.ivar = ivar
        self.fwhm = fwhm
        self.signif_thresh = signif_thresh
        self.input_gpm = gpm if gpm is not None else np.ones_like(flux, dtype=bool)
        self.signif_mask_nsigma = signif_mask_nsigma
        self.signif_mask_dv = signif_mask_dv
        self.one_minF_thresh = one_minF_thresh
        self.model_fit_thresh = model_fit_thresh
        self.find_alias = find_alias
        self.logN = logN
        self.Pfit_min = Pfit_min
        self.LR_max = LR_max
        self.prob_cuts = prob_cuts
        self.trim_igm = trim_igm
        self.bval_igm = bval_igm
        self.W_2796_igm = W_2796_igm
        self.debug = debug
        self.diffevol = diffevol
        self.nskew, self.nspec = flux.shape

        # Good pixel max based on the 1-F values
        self.flux_gpm = (1.0 - self.flux) < self.one_minF_thresh

        # Build the kernel for convolutions
        var_noise = (self.ivar > 0.0) / (np.abs(self.ivar) + (self.ivar == 0.0))
        v_kern, kernel = mgii_kernel(self.vgrid, self.fwhm, logN=self.logN)
        # Convolve the spectrum with the kernel
        conv_out = correlate1d(1.0 - self.flux, kernel, axis=-1, mode='constant', cval=0.0)
        conv_noise = correlate1d(var_noise, kernel ** 2, axis=-1, mode='constant', cval=0.0)
        self.signif = (conv_noise > 0.0)*conv_out/(np.sqrt(conv_noise) + (conv_noise == 0.0))
        # For noise that is heteroscedastic we may want to reconsider taking the median?
        # self.signif = conv_out / np.sqrt(np.median(conv_noise))

        # Now loop over skewers, find peaks in the significance
        #dv_pix = np.median(np.diff(self.vgrid))
        #min_peak_dist = self.fwhm / dv_pix
        if min_peak_dist == None:
            dv_pix = np.median(np.diff(self.vgrid))
            min_peak_dist = self.fwhm / dv_pix
        else:
            print("using input min_peak_dist of", min_peak_dist)

        pbar = tqdm(total=self.nskew, desc="Peak Finding Skewers")
        peak_pix0 = []
        peak_signif0 = []
        peak_vel0 = []
        peak_skew0 = []
        for iskew in range(self.nskew):
            pix = detect_peaks(self.signif[iskew, :], mph=self.signif_thresh, mpd=min_peak_dist, show=self.debug)
            peak_pix0 += pix.tolist()
            peak_vel0 += self.vgrid[pix].tolist()
            peak_signif0 += self.signif[iskew, pix].tolist()
            peak_skew0 += [iskew] * len(pix)
            pbar.update(1)

        self.peak_skew0 = np.array(peak_skew0)
        self.peak_pix0 = np.array(peak_pix0)
        self.peak_vel0 = np.array(peak_vel0)
        self.peak_signif0 = np.array(peak_signif0)

        # good_peak = True are not aliased peaks
        self.alias_peak_gpm0 = np.ones_like(peak_pix0, dtype=bool)
        # Remove the aliasing?
        if self.find_alias:
            n_neigh0 = np.zeros_like(peak_pix0, dtype=int)
            dvtol = self.fwhm / 2.0
            for iskew in range(self.nskew):
                ithis = peak_skew0 == iskew
                if np.any(ithis):
                    self.alias_peak_gpm0[ithis], n_neigh0[ithis] = remove_aliasing(peak_vel0[ithis], peak_signif0[ithis], dvtol)

        if self.signif_mask_nsigma is not None:
            # Mask the significance spectrum where it crosses the nsigma thershold
            self.signif_gpm = self.signif < self.signif_mask_nsigma
            # Identify the peaks that have a significance above this nsigma threshold.  These are "bad peaks" = True
            self.signif_peak_bpm0 = self.peak_signif0 > self.signif_mask_nsigma
            # Loop over all the peaks, for any that are above the signif_mask_nsigma, grow the mask by +- signif_mask_dv
            for ipeak, iskew in enumerate(peak_skew0):
                if self.signif_peak_bpm0[ipeak]:
                    # Grow the mask about both transitions
                    mask_2796 = np.abs(self.vgrid - self.peak_vel0[ipeak]) < self.signif_mask_dv
                    mask_2803 = np.abs(self.vgrid - (self.peak_vel0[ipeak] + v_mgII)) < self.signif_mask_dv
                    self.signif_gpm[iskew, :] = self.signif_gpm[iskew, :] & np.invert(mask_2796) & np.invert(mask_2803) & (
                                self.signif[iskew, :] < self.signif_mask_nsigma)
        else:
            self.signif_gpm = np.ones_like(self.flux, dtype=bool)
            self.signif_peak_bpm0 = np.zeros_like(self.peak_skew0, dtype=bool)

        # Final spectral gpm is the and of the input mask, the flux mask, and the significance mask
        self.fit_gpm = self.input_gpm &  self.flux_gpm & self.signif_gpm

        # Peaks to be fit are the those that are:
        # 1) not aliased (if de-aliasing was requested),
        # 2) are below the signif_mask_nsigma if that was set
        self.good_peak0 = self.alias_peak_gpm0 & np.invert(self.signif_peak_bpm0)
        self.peak_skew = self.peak_skew0[self.good_peak0]
        self.peak_pix = self.peak_pix0[self.good_peak0]
        self.peak_vel = self.peak_vel0[self.good_peak0]
        self.peak_signif = self.peak_signif0[self.good_peak0]
        self.npeak_init = self.peak_skew.size


    def fit(self):


        # Allocate arrays to store things
        success = np.zeros(self.npeak_init, dtype=bool)
        vel_fit = np.zeros(self.npeak_init)
        logN_fit = np.zeros(self.npeak_init)
        bval_fit = np.zeros(self.npeak_init)
        fwhm_fit = np.zeros(self.npeak_init)
        W_2796_fit = np.zeros(self.npeak_init)
        ndof = np.zeros(self.npeak_init, dtype=int)
        chi2 = np.zeros(self.npeak_init)
        chi2_null = np.zeros(self.npeak_init)
        log10P_fit = np.full(self.npeak_init, -np.inf)
        log10LR = np.full(self.npeak_init, np.inf)


        # This could be made more memory efficient
        F_fit = np.zeros((self.npeak_init, self.nspec))
        pbar = tqdm(total=self.npeak_init, desc="Fitting Absorbers    ")
        for ipeak, iskew in enumerate(self.peak_skew):
            if self.peak_vel[ipeak] > (v_mgII - 100):
                success[ipeak], (vel_fit[ipeak], bval_fit[ipeak], logN_fit[ipeak], fwhm_fit[ipeak],
                                 W_2796_fit[ipeak]), F_fit[ipeak, :], ndof[ipeak], chi2[ipeak], \
                chi2_null[ipeak], log10P_fit[ipeak], log10LR[ipeak] = mgii_fit(
                    self.vgrid, self.flux[iskew, :], self.ivar[iskew, :], self.peak_vel[ipeak], self.fwhm,
                    diffevol=self.diffevol,
                    gpm=self.fit_gpm[iskew, :])
            else:
                # Try with this peak being 2796
                success_2796, (
                    vel_fit_2796, bval_fit_2796, logN_fit_2796, fwhm_fit_2796, W_2796_fit_2796), F_fit_2796, \
                ndof_2796, chi2_2796, chi2_null_2796, log10P_fit_2796, log10LR_2796 = mgii_fit(
                    self.vgrid, self.flux[iskew, :], self.ivar[iskew, :], self.peak_vel[ipeak], self.fwhm, diffevol=self.diffevol,
                    gpm=self.fit_gpm[iskew, :])
                # Try with this peak being 2803
                success_2803, (
                    vel_fit_2803, bval_fit_2803, logN_fit_2803, fwhm_fit_2803, W_2796_fit_2803), F_fit_2803, \
                ndof_2803, chi2_2803, chi2_null_2803, log10P_fit_2803, log10LR_2803 = mgii_fit(
                    self.vgrid, self.flux[iskew, :], self.ivar[iskew, :], self.peak_vel[ipeak] - v_mgII, self.fwhm, diffevol=self.diffevol,
                    gpm=self.fit_gpm[iskew, :])
                if (log10P_fit_2796 > log10P_fit_2803) & success_2796:
                    success[ipeak] = True
                    (vel_fit[ipeak], bval_fit[ipeak], logN_fit[ipeak], fwhm_fit[ipeak], W_2796_fit[ipeak]) = \
                        vel_fit_2796, bval_fit_2796, logN_fit_2796, fwhm_fit_2796, W_2796_fit_2796
                    F_fit[ipeak, :] = F_fit_2796
                    ndof[ipeak], chi2[ipeak], chi2_null[ipeak], log10P_fit[ipeak], log10LR[ipeak] = \
                        ndof_2796, chi2_2796, chi2_null_2796, log10P_fit_2796, log10LR_2796
                elif (log10P_fit_2796 <= log10P_fit_2803) & success_2803:
                    success[ipeak] = True
                    (vel_fit[ipeak], bval_fit[ipeak], logN_fit[ipeak], fwhm_fit[ipeak], W_2796_fit[ipeak]) = \
                        vel_fit_2803, bval_fit_2803, logN_fit_2803, fwhm_fit_2803, W_2796_fit_2803
                    F_fit[ipeak, :] = F_fit_2803
                    ndof[ipeak], chi2[ipeak], chi2_null[ipeak], log10P_fit[ipeak], log10LR[ipeak] = \
                        ndof_2803, chi2_2803, chi2_null_2803, log10P_fit_2803, log10LR_2803

            pbar.update(1)


        # Filter out the good valuessel
        # This line exploits the fact that the CGM absorbers have different velocity structure than IGM absorbers.
        igood = (log10P_fit > self.Pfit_min) & (log10LR < self.LR_max) & success if self.prob_cuts is True else success
        IGM_abs = (bval_fit > self.bval_igm) & (W_2796_fit < self.W_2796_igm)
        igood = igood & np.invert(IGM_abs) if self.trim_igm else igood
        self.npeak = np.sum(igood)
        self.peak_pix_fit = self.peak_pix[igood]
        self.peak_vel_fit = self.peak_vel[igood]
        self.peak_signif_fit = self.peak_signif[igood]
        self.peak_skew_fit = self.peak_skew[igood]
        self.vel_fit = vel_fit[igood]
        self.bval_fit = bval_fit[igood]
        self.logN_fit = logN_fit[igood]
        self.fwhm_fit = fwhm_fit[igood]
        self.W_2796_fit = W_2796_fit[igood]
        self.F_fit = F_fit[igood, :]
        self.ndof = ndof[igood]
        self.chi2 = chi2[igood]
        self.chi2_null = chi2_null[igood]
        self.log10P_fit = log10P_fit[igood]
        self.log10LR = log10LR[igood]
        print('A total of npeak={:d} abosrbers survived all cuts'.format(self.npeak))

        # Construct a model of the absorbers that we fit and return a model_gpm
        tau_fit = np.zeros_like(self.flux)
        for ipeak, iskew in enumerate(self.peak_skew_fit):
            tau_fit[iskew, :] += -np.log(self.F_fit[ipeak, :])
        self.F_model = np.exp(-tau_fit)
        self.model_gpm = (1.0 - self.F_model) < self.model_fit_thresh

        # Final GPM is model_gpm & input_gp & signif_gpm & flux_gpm
        self.total_gpm = self.model_gpm & self.fit_gpm
        self.good_path_frac = np.sum(self.total_gpm)/self.flux.size

def detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising',
                 kpsh=False, valley=False, show=False, ax=None):

    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
        if mph is not None:
            mph = -mph
    # find indices of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan - 1, indnan + 1))), invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size - 1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind] - x[ind - 1], x[ind] - x[ind + 1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                       & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])

    if show:
        if indnan.size:
            x[indnan] = np.nan
        if valley:
            x = -x
            if mph is not None:
                mph = -mph
        _plot(x, mph, mpd, threshold, edge, valley, ax, ind)

    return ind


def _plot(x, mph, mpd, threshold, edge, valley, ax, ind):
    """Plot results of the detect_peaks function, see its help."""

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(8, 4))

    ax.plot(x, 'b', lw=1)
    if ind.size:
        label = 'valley' if valley else 'peak'
        label = label + 's' if ind.size > 1 else label
        ax.plot(ind, x[ind], '+', mfc=None, mec='r', mew=2, ms=8,
                label='%d %s' % (ind.size, label))
        ax.legend(loc='best', framealpha=.5, numpoints=1)
    ax.set_xlim(-.02 * x.size, x.size * 1.02 - 1)
    ymin, ymax = x[np.isfinite(x)].min(), x[np.isfinite(x)].max()
    yrange = ymax - ymin if ymax > ymin else 1
    ax.set_ylim(ymin - 0.1 * yrange, ymax + 0.1 * yrange)
    ax.set_xlabel('Data #', fontsize=14)
    ax.set_ylabel('Amplitude', fontsize=14)
    mode = 'Valley detection' if valley else 'Peak detection'
    ax.set_title("%s (mph=%s, mpd=%f, threshold=%s, edge='%s')"
                 % (mode, str(mph), mpd, str(threshold), edge))
    # plt.grid()
    plt.show()

def remove_aliasing(peak_vel, peak_signif, dvtol):

    keep_peak = np.ones_like(peak_vel, dtype=bool)
    n_neigh = np.zeros_like(peak_vel, dtype=int)
    # This computes all pairs of distances
    data = np.array([peak_vel])
    data = data.transpose()
    tree = KDTree(data)
    npeaks = len(peak_vel)
    ind, dist = tree.query_radius(data, v_mgII + dvtol, return_distance=True)
    for ipeak in range(npeaks):
        ibin = (dist[ipeak] >= (v_mgII - dvtol)) & (dist[ipeak] <= (v_mgII + dvtol))
        if np.any(ibin):
            n_neigh[ipeak] = np.sum(ibin)
            ind_neigh = (ind[ipeak])[ibin]
            # Keep the peak if the neighbors all have a lower value of significance, otherwise discard the peak
            keep_peak[ipeak] = np.all(peak_signif[ind_neigh] < peak_signif[ipeak])

    return keep_peak, n_neigh

def mgii_fit(vgrid, F, ivar, v_guess, fwhm, gpm=None, fit_width = 500.0, max_vshift=None,
             min_bval=None, max_bval=300.0, diffevol=False):
    """
    Fit a Vogit profile to region of spectrum
    Args:
        vgrid:
        oneminF:
        sigma:

    Returns:

    """


    max_vshift = fwhm if max_vshift is None else fwhm
    min_bval = 5.0 if min_bval is None else min_bval

    logN_min = 10.0
    logN_max = 17.0


    oneminF = 1.0 - F
    var_noise = (ivar > 0.0)/(np.abs(ivar) + (ivar == 0.0))
    sig_noise = np.sqrt(var_noise)

    gpm_use = np.ones_like(vgrid, dtype=bool) if gpm is None else gpm
    #v_mgII = utils.vel_mgii().value
    velmask = ((vgrid - v_guess) < (fit_width + v_mgII)) & ((v_guess - vgrid) < fit_width)
    fitmask = velmask & gpm_use
    ndof = np.sum(fitmask)
    if ndof == 0:
        #print('No good pixels to fit in mgii_fit')
        return False, np.full(5,0.0), np.zeros_like(vgrid), ndof, 0.0, 0.0, -np.inf, 0.0

    # Fit only the locations that are not masked
    v_fit = vgrid[fitmask]
    oneminF_fit = oneminF[fitmask]
    sig_noise_fit = sig_noise[fitmask]
    ivar_fit = ivar[fitmask]

    # Obtain guesses for the parameters

    # Estimate Gaussian parameters based on the
    singmask = (np.abs(vgrid - v_guess) < max_vshift)
    denom = np.sum(oneminF*gpm_use*singmask)
    sigma = np.sqrt(np.abs(np.sum((vgrid-v_guess)**2*gpm_use*singmask*oneminF)/denom)) if (denom > 0.0) else fwhm/2.35
    b_guess = np.clip(np.sqrt(2.0)*sigma, min_bval, max_bval)

    # Calculate ampl from pixels within +/- sigma/2
    cen_pix = (np.abs(vgrid - v_guess)< sigma/2.0) & gpm_use
    if np.any(cen_pix):
        ampl = np.abs(np.median(oneminF[cen_pix]))
    elif np.any(gpm_use):
        # Just take the median value of the noise
        ampl = np.median(sig_noise[gpm_use])
    else:
        ampl = 0.1

    # On the linear part of the COG the amplitude is related to column and b_valu via the below
    tau_const = ((np.sqrt(np.pi)* const.e.esu **2*f_2796) / (const.m_e * nu_mgII)/(b_guess*u.km/u.s)).decompose().to('cm2')
    logN_guess = np.clip(np.log10(ampl) - np.log10(tau_const.value), logN_min, logN_max)

    p0 = np.array([v_guess, b_guess, logN_guess, fwhm])
    bounds = ([v_guess - max_vshift, min_bval, logN_min, 0.7*fwhm], [v_guess + max_vshift, max_bval, logN_max, 1.3*fwhm])
    if diffevol:
        bounds = [(v_guess - max_vshift, v_guess + max_vshift), (min_bval, max_bval),
                  (logN_min, logN_max), (0.7 * fwhm, 1.3 * fwhm)]
        result = optimize.differential_evolution(mgii_voigt_diff, bounds, args=(v_fit, oneminF_fit, ivar_fit), disp=True)
        popt = result.x
    else:
        try:
            popt, pcov = optimize.curve_fit(mgii_voigt_eval, v_fit, oneminF_fit, bounds=bounds, p0=p0, sigma=sig_noise_fit)
        except RuntimeError:
            bounds = [(v_guess - max_vshift, v_guess + max_vshift), (min_bval, max_bval), (logN_min, logN_max),
                      (0.7 * fwhm, 1.3 * fwhm)]
            result = optimize.differential_evolution(mgii_voigt_diff, bounds, args=(v_fit, oneminF_fit, ivar_fit),
                                                     disp=True)
            popt = result.x

    oneminF_best = mgii_voigt_eval(vgrid, popt[0], popt[1], popt[2], popt[3])

    W_2796 = mgii_voigt_EW(popt[1], popt[2])
    chi2_best = np.sum(np.square(oneminF-oneminF_best)*fitmask*ivar)
    chi2_null = np.sum(np.square(oneminF)*fitmask*ivar)

    fit_tuple = (popt[0], popt[1], popt[2], popt[3], W_2796.value)
    # log10P_fit is the log10 of the cumulative probability of obtaiing > chi2 for the number of dof.
    # Return the log of the survival function, i.e. P(>x)
    rv = stats.chi2(ndof - 4)
    # prob_fit is the log_10 of the survival function of the chi2 distribution
    log10P_fit = rv.logsf(chi2_best)/np.log(10.0)
    # LR is the likelihood ratio, which is bounded between:
    #   0 = null hypothesis is a much worse fit than the optimum (lnLR is a large negative number)
    #   1 = null hypothesis is as good a fit as the optimum (lnLR approaches zero)
    lnLR = -0.5 *(chi2_null - chi2_best)
    log10LR = lnLR/np.log(10.0)

    return True, fit_tuple, (1-oneminF_best), ndof, chi2_best, chi2_null, log10P_fit, log10LR

def mgii_voigt_diff(theta, vgrid, oneminF, ivar):

    v_abs, b_val, logN_MgII, fwhm = theta
    oneminF_model = mgii_voigt_eval(vgrid, v_abs, b_val, logN_MgII, fwhm)
    chi2 = np.sum(np.square(oneminF-oneminF_model)*ivar)
    return chi2

def mgii_voigt_eval(vdata, v_abs, b_val, logN_MgII, fwhm):

    vmin = vdata.min() - 5*fwhm
    vmax = vdata.max() + 5*fwhm
    dvpix = b_val/10.0
    npix = (vmax - vmin)/dvpix
    vgrid = vmin + dvpix*np.arange(npix)

    du_2796 = (vgrid - v_abs)/b_val
    du_2803 = du_2796 - v_mgII/b_val

    phi_nu_2796 = utils.H_voigt(A_VOIGT_B_MGII/b_val, du_2796)/b_val/np.sqrt(np.pi)
    phi_nu_2803 = utils.H_voigt(A_VOIGT_B_MGII/b_val, du_2803)/b_val/np.sqrt(np.pi)

    N_by_fid = np.power(10.0, logN_MgII-logN_fid)
    tau_c_2796 = N_by_fid*tau_c_2796_fid
    tau_c_2803 = N_by_fid*tau_c_2803_fid

    tau_2796 = phi_nu_2796 * tau_c_2796
    tau_2803 = phi_nu_2803 * tau_c_2803
    tau_tot = tau_2796 + tau_2803
    flux_hires = np.exp(-tau_tot)
    # Now convole with resolution
    sigma_resolution = (fwhm / 2.35483) /dvpix  # fwhm = 2.3548 sigma
    flux_sm = gaussian_filter1d(flux_hires, sigma_resolution, mode='constant', cval=1.0)
    flux_lores = interp1d(vgrid, flux_sm, fill_value=0.0, kind='cubic', bounds_error=True)(vdata)
    oneminF = 1.0 - flux_lores

    return oneminF

# Returns model EW of absorber
def mgii_voigt_EW(b_val, logN_MgII):

    vmin = -10.0*b_val
    vmax =  10.0*b_val
    dvpix = b_val/5.0
    npix = (vmax - vmin)/dvpix
    vgrid = vmin + dvpix*np.arange(npix)

    du_2796 = vgrid/b_val
    #du_2803 = du_2796 - v_mgii.value/b_val

    phi_nu_2796 = utils.H_voigt(A_VOIGT_B_MGII/b_val, du_2796)/b_val/np.sqrt(np.pi)
    N_by_fid = np.power(10.0, logN_MgII-logN_fid)
    tau_c_2796 = N_by_fid*tau_c_2796_fid
    tau_2796 = phi_nu_2796 * tau_c_2796
    W_2796 = integrate.simps(1.0 - np.exp(-tau_2796), vgrid)/c_light.value * wave_2796
    return W_2796
