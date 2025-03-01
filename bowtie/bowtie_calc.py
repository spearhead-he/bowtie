#!/usr/bin/env python3
"""
A module for the bow-tie (https://www.utupub.fi/handle/10024/152846 and references therein)
analysis of a response function of a particle instrument.
"""
__author__ = "Philipp Oleynik"
__credits__ = ["Philipp Oleynik", "Christian Palmroos"]

import math
from matplotlib import rcParams
from scipy import interpolate
from scipy import optimize
import matplotlib.cm as cm
import matplotlib.colors as clr
import matplotlib.pyplot as plt
import numpy as np
from statistics import geometric_mean

from . import plotutil as plu

RESPONSE_AXIS_COLOR = "blue"
DEFAULT_TITLE_FONTSIZE = 18
DEFAULT_AXIS_LABEL_SIZE = 14

def plot_multi_geometric(geometric_factors, response_data,
                         emin=1.0, emax=100.0, gmin=1.0E-15, gmax=1.0E10,
                         save=False, saveidx="0", integral=False, save_path='',
                         channel:str=None):
    """
    Plot differential geometric factor and optionally save the plot.
    :param response_data: channel response data
    :type response_data: a dictionary. 'grid' defines the energy grid, 'resp' defines the response function.
    :param integral: if True, geometric factors are assumed to be computed for an integral channel.
    :param geometric_factors: N_gamma by N_energy matrix. N_gamma denotes amount of different power law indices.
                              N_energy denotes the number of energy bins
    :param energy_grid_plot: N_energy array with midpoint energies of each energy bin
    :param emin: Minimum on energy axis
    :param emax: Maximum on energy axis
    :param gmin: Minimum on geometric factor axis
    :param gmax: Maximum on geometric factor axis
    :param saveidx: String suffix for the plot filename
    :param save: if True, save a Gdiff_saveidx.png file, show the plot otherwise
    :param save_path: Base path for saving the plot
    :type save_path: basestring
    :channel : {str} optional. Adds channel to the title of the figure if provided.
    """
    energy_grid_plot = response_data['grid']
    plu.setup_latex(rcParams)
    plu.setup_plotstyle(rcParams)
    rcParams["figure.figsize"] = [6.4, 4.8]
    grid_kws = {"height_ratios": (0.7, 0.3), "hspace": .1}
    fig, (ax, subax) = plt.subplots(2, sharex='col', sharey='none', gridspec_kw=grid_kws)
    plu.set_log_axes_simple(ax, grid=False)

    if integral:
        ax.set_ylabel(r'G(E) [cm$^2$ sr]', fontsize=DEFAULT_AXIS_LABEL_SIZE, color='black')
        subax.set_xlabel(r'Threshold energy, MeV', fontsize=DEFAULT_AXIS_LABEL_SIZE, color='black')
    else:
        ax.set_ylabel(r'G$\delta$E [cm$^2$ sr MeV]', fontsize=DEFAULT_AXIS_LABEL_SIZE, color='black')
        subax.set_xlabel(r'Effective energy [MeV]', fontsize=DEFAULT_AXIS_LABEL_SIZE, color='black')
    gamma_steps_ = geometric_factors.shape[0]
    energy_steps_ = geometric_factors.shape[1]

    # apply color palette to the plot
    gamma_norm_ = clr.Normalize(vmin=0, vmax=gamma_steps_ - 1)

    # cm.cmap() doesn't exist in matplotlib 3.9.0 -> workaround to call plt.get_cmap() instead
    # gamma_colormap_ = cm.ScalarMappable(norm=gamma_norm_, cmap=cm.get_cmap('viridis'))
    gamma_colormap_ = cm.ScalarMappable(norm=gamma_norm_, cmap=plt.get_cmap("viridis"))
    ax.set_prop_cycle('c', [gamma_colormap_.to_rgba(ii) for ii in range(gamma_steps_)])

    # gamma_steps_ - number of different power law indices in the calculated geometric factors
    # x data is a repetition of the same energies that ^^^ number of times.
    # energy_steps_ - number of energy values for which the geometric factors are calculated
    # y data is just geometric factors themselves for different power law indices, transposed due to the initial choice of the array definition
    ax.plot([np.full(gamma_steps_, energy_grid_plot['midpt'][jj]) for jj in range(energy_steps_)],
            geometric_factors.T)

    # Initialize a parallel y-axis for which to plot the response function
    ax1 = ax.twinx()

    # Plotting the response function and setting the axis
    ax1.set_yscale("log")
    ax1.tick_params(which="both", direction="in", color=RESPONSE_AXIS_COLOR, labelcolor=RESPONSE_AXIS_COLOR)
    ax1.set_ylabel(r"Response [cm$^2$ sr]", color=RESPONSE_AXIS_COLOR, fontsize=DEFAULT_AXIS_LABEL_SIZE)
    ax1.plot(response_data['grid']['midpt'], response_data['resp'], c=RESPONSE_AXIS_COLOR)

    # These for the lower panel ->
    non_zero_geof = np.mean(geometric_factors, axis=0) > 0
    means_ = np.mean(np.log(geometric_factors[:, non_zero_geof]), axis=0) + 1E-124
    stddev_ = np.std((geometric_factors[:, non_zero_geof]), axis=0) / np.exp(means_)
    stddev_ /= np.min(stddev_)

    subax.plot(energy_grid_plot['midpt'][non_zero_geof], stddev_, c="r")
    subax.set_ylim(0, 5)
    subax.grid(True, which='both', alpha=0.3, zorder=0)
    subax.set_ylabel(r'$\sigma$', fontsize=DEFAULT_AXIS_LABEL_SIZE, color='black')
    subax.set_xlim(emin, emax)
    ax.set_xlim(emin, emax)
    ax.set_ylim(gmin, gmax)

    if isinstance(channel, str):
        ax.set_title(f"{channel} response function and bowtie", fontsize=DEFAULT_TITLE_FONTSIZE)
    if integral:
        fname_ = save_path + 'Gint_np_{0:s}.png'.format(saveidx)
    else:
        fname_ = save_path + 'Gdiff_np_{0:s}.png'.format(saveidx)

    if save:
        plt.savefig(fname_, format='png', dpi=150)
        print(fname_)
    else:
        plt.show(block=True)
        return fig, (ax, subax)


def generate_pwlaw_spectra(energy_grid_dict,
                           gamma_pow_min=-3.5, gamma_pow_max=-1.5,
                           num_steps=100, use_integral_bowtie=False):
    model_spectra = []  # generate power-law spectra for folding
    if use_integral_bowtie:
        for power_law_gamma in np.linspace(gamma_pow_min, gamma_pow_max, num=num_steps, endpoint=True):
            model_spectra.append({
                'gamma': power_law_gamma,
                'spect': generate_powerlaw_np(energy_grid=energy_grid_dict, power_index=power_law_gamma),
                'intsp': generate_integral_powerlaw_np(energy_grid=energy_grid_dict,
                                                       power_index=power_law_gamma)
            })
    else:
        for power_law_gamma in np.linspace(gamma_pow_min, gamma_pow_max, num=num_steps, endpoint=True):
            model_spectra.append({
                'gamma': power_law_gamma,
                'spect': generate_powerlaw_np(energy_grid=energy_grid_dict, power_index=power_law_gamma)
            })
    return model_spectra


def generate_exppowlaw_spectra(energy_grid_dict,
                               gamma_pow_min=-3.5, gamma_pow_max=-1.5,
                               num_steps=100, use_integral_bowtie=False,
                               cutoff_energy=1.0):
    model_spectra = []  # generate exponential cutoff power-law spectra for folding
    if use_integral_bowtie:
        print("Integral bowtie is not yet implemented!")
        return None
    else:
        for power_law_gamma in np.linspace(gamma_pow_min, gamma_pow_max, num=num_steps, endpoint=True):
            spectrum = 1.0 * np.power(energy_grid_dict['midpt'], power_law_gamma) * \
                       np.exp(-cutoff_energy / (energy_grid_dict['midpt'] - cutoff_energy))

            index_cutoff = np.searchsorted(energy_grid_dict['midpt'], cutoff_energy)
            np.put(spectrum, range(0, index_cutoff + 1), 1.0E-30)
            model_spectra.append({
                'gamma': power_law_gamma,
                'spect': spectrum
            })
    return model_spectra


def generate_integral_powerlaw_np(*, energy_grid=None,
                                  power_index=-3.5, sp_norm=1.0):
    """

    :param energy_grid:
    :param power_index:
    :param sp_norm:
    :return:
    """
    if energy_grid is not None:
        spectrum = - sp_norm * np.power(energy_grid['enlow'], power_index + 1) / (power_index + 1)
        return spectrum
    else:
        return None


def generate_powerlaw_np(*, energy_grid=None, power_index=-2, sp_norm=1.0):
    """

    :param energy_grid:
    :param power_index:
    :param sp_norm:
    :return:
    """
    spectrum = sp_norm * np.power(energy_grid['midpt'], power_index)
    return spectrum


def fold_spectrum_np(*, grid=None, spectrum=None, response=None):
    """
    Folds incident spectrum with an instrument response. Int( spectrum * response * dE)
    :param grid: energy grid, midpoints of each energy bin
    :param spectrum: intensities defined at the midpoint of each energy bin
    :param response: geometric factor curve defined at the midpoint of each energy bin
    :return: countrate in the channel described by the response.
    """
    if grid is None:
        return math.nan
    if spectrum is None or response is None:
        return 0
    if (len(spectrum) == len(response)) and (len(spectrum) == len(grid['midpt'])):
        result = np.trapezoid(np.multiply(spectrum, response), grid['midpt'])
        return result
    else:
        return 0


def calculate_bowtie_gf(response_data,
                        model_spectra,
                        emin=0.01, emax=1000,
                        gamma_index_steps=100,
                        use_integral_bowtie=False,
                        sigma=3,
                        plot=False,
                        gfactor_confidence_level=0.9,
                        return_gf_stddev=False,
                        channel:str=None):
    """
    Calculates the bowtie geometric factor for a single channel
    :param return_gf_box: True if the margin of the channel geometric factor is requested.
    :type return_gf_box: bool
    :param response_data: The response data for the channel.
    :type response_data: A dictionary, must have 'grid', the energy_grid_data (dictionary, see make_energy_grid),
                         and 'resp', the channel response (an array of a length of energy_grid_data['nstep'])
    :param model_spectra: The model spectra for the analysis.
    :type model_spectra: A dictionary (see generate_pwlaw_spectra)
    :param emin: the minimal energy to consider
    :type emin: float
    :param emax: the maximum energy to consider
    :type emax: float
    :param gamma_index_steps:
    :type gamma_index_steps:
    :param use_integral_bowtie:
    :type use_integral_bowtie:
    :param sigma: Cutoff sigma value for the energy margin.
    :type sigma: float
    :return: (The geometric factor, the effective energy, lower margin for the effective energy, upper margin for the effective energy)
    :rtype: list
    :channel : {str} optional. Adds channel id to plot if both enabled.
    """
    energy_grid_local = response_data['grid']['midpt']

    index_emin = np.searchsorted(energy_grid_local, emin)  # search for an index corresponding to start energy
    index_emax = np.searchsorted(energy_grid_local, emax)
    multi_geometric_factors = np.zeros((gamma_index_steps, response_data['grid']['nstep']), dtype=float)
    # for each model spectrum do the folding.

    for model_spectrum_idx, model_spectrum in enumerate(model_spectra):

        spectral_folding_int = fold_spectrum_np(grid=response_data['grid'],
                                                spectrum=model_spectrum['spect'],
                                                response=response_data['resp'])
        # TODO response error margin!
        if use_integral_bowtie:
            spectrum_data = model_spectrum['intsp']
        else:
            spectrum_data = model_spectrum['spect']

        multi_geometric_factors[model_spectrum_idx, index_emin:index_emax] = spectral_folding_int / spectrum_data[index_emin:index_emax]

    # Create a discrete standard deviation vector for each energy in the grid.
    # This standard deviation is normalized to the local mean, so that a measure of spreading of points is obtained.
    # Mathematically, this implies normalization of the random variable to its mean.
    non_zero_gf = np.mean(multi_geometric_factors, axis=0) > 0
    multi_geometric_factors_usable = multi_geometric_factors[:, non_zero_gf]
    # print(len(multi_geometric_factors), len(multi_geometric_factors_usable))
    means = np.exp(np.mean(np.log(multi_geometric_factors_usable), axis=0))  # logarithmic mean of geometric factor curves by energy bin

    # Cross-channel contamination: PE4 causes ValueError: zero-size array to reduction operation minimum which has no identity for 
    # gf_stddev_norm = gf_stddev / np.min(gf_stddev)
    # For now, 2024-08-23 I'll replace the normed stddev with a negative tenth (It should never be a negative number). 
    try : 
        gf_stddev_abs = np.std(multi_geometric_factors_usable, axis=0)
        gf_stddev = gf_stddev_abs / means
        gf_stddev_norm = gf_stddev / np.min(gf_stddev)
    except ValueError:
        gf_stddev_norm = -1e-1

    bowtie_cross_index = np.argmin(gf_stddev_norm)  # The minimal standard deviation point - bowtie crossing point.

    # Interpolate the discrete standard deviation so that it could be used in a equation solver.
    # The discrete standard deviation is normalized to 1 in the minimum, so that 1.0 must be subtracted
    # before sigma level to make a discrete "equation" for the the interpolator because
    # the optimize.bisect looks for zeroes of a function, which is the interpolator.
    stddev_interpolator = interpolate.interp1d(energy_grid_local[non_zero_gf], gf_stddev_norm - 1.0 - sigma)
    try:
        (channel_energy_low) = optimize.bisect(stddev_interpolator,
                                               energy_grid_local[non_zero_gf][0],
                                               energy_grid_local[non_zero_gf][bowtie_cross_index])  # to the left of bowtie_cross_index
    except ValueError:
        channel_energy_low = 0

    try:
        (channel_energy_high) = optimize.bisect(stddev_interpolator,
                                                energy_grid_local[non_zero_gf][bowtie_cross_index],
                                                energy_grid_local[non_zero_gf][-1])  # to the right of bowtie_cross_index
    except ValueError:
        channel_energy_high = 0

    # gf = np.mean(multi_geometric_factors_usable, axis = 0)  # Average geometric factor for all model spectra
    # gf_cross = gf[bowtie_cross_index]  # The mean geometric factor for the bowtie crossing point

    gf_cross = geometric_mean(multi_geometric_factors_usable[:, bowtie_cross_index])
    energy_cross = energy_grid_local[bowtie_cross_index]

    if plot:
        fig, axes = plot_multi_geometric(geometric_factors=multi_geometric_factors, response_data=response_data,
                             emin=emin, emax=emax, gmin=1E-5, gmax=10, channel=channel)

    if return_gf_stddev:
        gf_upper = np.quantile(multi_geometric_factors_usable[:, bowtie_cross_index], gfactor_confidence_level)
        gf_lower = np.quantile(multi_geometric_factors_usable[:, bowtie_cross_index], 1 - gfactor_confidence_level)

        if plot:
            return gf_cross, {'gfup': gf_upper, 'gflo': gf_lower}, energy_cross, channel_energy_low, channel_energy_high, fig, axes
        return gf_cross, {'gfup': gf_upper, 'gflo': gf_lower}, energy_cross, channel_energy_low, channel_energy_high 

    return gf_cross, energy_cross, channel_energy_low, channel_energy_high
