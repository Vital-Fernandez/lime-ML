import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt, cm, rc_context

from tools import sn_formula, params_labels_dict, normal_curve, params_title_dict, params_units_dict, detection_function


PLOT_CONF = {'figure.figsize': (12, 12),
             'axes.titlesize': 16,
             'axes.labelsize': 16,
             'legend.fontsize': 14,
             'xtick.labelsize': 14,
             'ytick.labelsize': 14}

line_type_color = {True: 'palegreen',
                   'True': 'palegreen',
                   False: 'xkcd:salmon',
                   'False': 'xkcd:salmon',
                   None: 'white',
                   'None': 'white',
                   np.nan: 'white',
                   'undecided': 'lightskyblue'}

line_type_color_invert = {'palegreen': True,
                          'xkcd:salmon': False,
                          'white': None,
                          'lightskyblue': 'undecided'}


def sn_evolution_plot(amp_array, noise_array, sigma_g_array, SN_low_limit=3, SN_upper_limit=5,
                      wavelength_ref=None, c_speed=299792.458):

    plt_conf = {'figure.figsize': (10, 8),
                'axes.titlesize': 16,
                'axes.labelsize': 16,
                'legend.fontsize': 14,
                'xtick.labelsize': 14,
                'ytick.labelsize': 14}

    with rc_context(plt_conf):

        cmap = cm.get_cmap()

        fig, ax = plt.subplots()

        for i, Ag_noise in enumerate(amp_array):
            SN_curve = sn_formula(Ag_noise, noise=noise_array[i], sigma=sigma_g_array, mu=None, lambda_step=None)

            label = r'$\frac{A_{g}}{\sigma_{noise}}$'
            color_curve = cmap(i / len(amp_array))

            if wavelength_ref is None:
                x_array = sigma_g_array
                x_label = r'$\sigma_{gas} (\AA)$'
            else:
                x_array = c_speed * sigma_g_array/wavelength_ref
                x_label = r'$\sigma_{gas} (km/s)$' + f' at ({wavelength_ref}$\AA$)'

            ax.plot(x_array, SN_curve, label=f'{label} = {Ag_noise}', color=color_curve)

        ax.axhspan(SN_low_limit, SN_upper_limit, label=r'$S/N=3-5$ ', alpha=0.30, color='lightskyblue')
        # _ax.axhline(lineCheck_sample=SN_limit, color='black', linestyle='--', label=f'S/N = {SN_limit}')

        ax.set_yscale('log')
        ax.update({'xlabel': x_label, 'ylabel': r'$\frac{S}{N}$',
                   'title': 'Signal-to-noise ratio versus gas velocity dispersion'})
        ax.legend(ncol=4, loc=4)
        plt.tight_layout()
        plt.show()

    return


def plot_distribution(x_array, y_array, density_function, dist_label=None, title=None, x_label=None, verbose=True):

    if verbose:
        with rc_context(PLOT_CONF):
            fig, ax = plt.subplots()
            ax.plot(x_array, density_function, label=dist_label)
            ax.hist(y_array, density=True, histtype='stepfilled', alpha=0.2, bins=50)
            ax.legend(loc='lower center')
            ax.update({'xlabel': x_label, 'ylabel': 'Variable value count', 'title': title})
            plt.show()

    return


def plot_ratio_distribution(y_array, param, units, x_label=None, label=None, title=None):

    with rc_context(PLOT_CONF):

        if x_label is None:
            x_label = x_label if x_label is not None else params_labels_dict[param]
            x_label += '' if units == 'none' else f' ({params_units_dict[units]})'

        title = title if title is not None else params_title_dict[param]

        fig, ax = plt.subplots()
        ax.hist(y_array, histtype='stepfilled', alpha=0.2, bins=1000, label=label)
        ax.update({'xlabel': x_label, 'ylabel': 'Variable value count', 'title': title})
        if label is not None:
            ax.legend(loc='upper center')
        plt.show()

    return


def plot_line_window(x_array, y_array, amp_noise_ratio, sigma_lambda_ratio, line_check):

    with rc_context(PLOT_CONF):
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.step(x_array, y_array)
        visual_check = amp_noise_ratio > detection_function(sigma_lambda_ratio)
        ax.set_facecolor(line_type_color[line_check])
        ax.update({'title': f'{params_labels_dict["amp_noise_ratio"]}={amp_noise_ratio:.2f};  '
                            f'{params_labels_dict["sigma_lambda_ratio"]}={sigma_lambda_ratio:.2f}; '
                            f'f(detect) = {visual_check}'})
        plt.show()

    return


class LineVisualizationMapper:

    def __init__(self, db_address, param_j, param_i, seed_value=0, **kwargs):

        self.db_address = Path(db_address)

        if self.db_address.is_file():
            self.selections_df = pd.read_csv(self.db_address, delim_whitespace=True, header=0, index_col=0)

        else:
            self.selections_df = pd.DataFrame(columns=['amp', 'mu', 'sigma', 'noise', 'lambda_step', 'line'])

        self.fig = None
        self.ax = None
        self.idx_ax = None

        self.grid_params = {}

        self.param_j, self.param_i =param_j, param_i

        with rc_context(PLOT_CONF):

            j_array, i_array = kwargs[param_j], kwargs[param_i]

            self.fig, ax_array = plt.subplots(nrows=len(j_array), ncols=len(i_array),
                                         figsize=(len(i_array) * 2, len(j_array) * 2))

            for j, value_j in enumerate(j_array):

                for i, value_i in enumerate(i_array):

                    self.ax = ax_array[j, i]

                    # Gaussian curve
                    gaussian_params = {**kwargs}
                    gaussian_params.update({param_j: value_j, param_i: value_i})

                    # Add new entry to dataframe and store the paremeters for the replotting
                    idx_line = self.add_to_dataframe(**gaussian_params)
                    self.grid_params[idx_line] = gaussian_params

                    # Check if the lines has been measured before
                    line_check = self.selections_df.loc[idx_line, 'line']

                    # Plot the line
                    self.plot_line(self.ax, self.grid_params[idx_line], line_check=line_check, title=idx_line)

            # Grid guides
            bbox_props = dict(boxstyle="rarrow, pad=0.3", fc="white", lw=1)
            arrow_props = dict(facecolor='black', shrink=0.05)

            # Columns arrow
            ax_0 = ax_array[0, 0]
            ax_0.annotate('', xy=(1.1 * len(i_array), 1.5), xycoords='axes fraction',
                          xytext=(0.2, 1.5), textcoords='axes fraction', arrowprops=arrow_props)
            ax_0.text(1.2 * len(i_array) / 2, 1.7, params_labels_dict[param_i], ha="center", va="center", size=15,
                      transform=ax_0.transAxes)

            # Rows arrow
            ax_0 = ax_array[0, 0]
            ax_0.annotate('', xy=(-0.3, -len(j_array)), xycoords='axes fraction',
                          xytext=(-0.3, 0.5), textcoords='axes fraction', arrowprops=arrow_props)
            ax_0.text(-0.5, -len(j_array) / 2, params_labels_dict[param_j], ha="center", va="center", size=15,
                      transform=ax_0.transAxes,
                      rotation=90)

            bpe = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
            aee = self.fig.canvas.mpl_connect('axes_enter_event', self.on_enter_axes)

            # Save the log
            self.save_log()

            plt.show()
            plt.close(self.fig)

        return

    def add_to_dataframe(self, **kwargs):

        idx_line = (self.selections_df.amp.values == kwargs['amp']) & \
                   (self.selections_df.mu.values == kwargs['mu']) & \
                   (self.selections_df.sigma.values == kwargs['sigma']) & \
                   (self.selections_df.noise.values == kwargs['noise']) & \
                    (self.selections_df.lambda_step.values == kwargs['lambda_step'])

        if ~np.any(idx_line):
            idx_line = len(self.selections_df.index)
            self.selections_df.loc[idx_line, 'amp':'line'] = kwargs['amp'], kwargs['mu'], \
                                                             kwargs['sigma'], kwargs['noise'], \
                                                             kwargs['lambda_step'], None
        else:
            idx_line = self.selections_df.loc[idx_line].index.values[0]

        return idx_line

    def plot_line(self, ax, parameters_dict, line_check=None, title=None):

        # Gaussian curve
        # value_i, value_j = parameters_dict[param_i], parameters_dict[param_j]
        x_array, y_array = normal_curve(**parameters_dict)

        # Limits checks
        SN = sn_formula(**parameters_dict)
        n_pix = (6.0 * parameters_dict['sigma']) / parameters_dict['lambda_step']
        total_pix = x_array.size

        # Plot
        label_curve = f'S/N = {SN:.2f}\n' \
                      f'Line width = {n_pix:.2f} pixels\n' \
                      f'Total pix = {total_pix:.2f}'

        # Clear axis everywhere but edges
        ax.axes.yaxis.set_visible(False)
        ax.yaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_ticklabels([])
        ax.xaxis.set_major_locator(plt.NullLocator())

        ax.step(x_array, y_array, where='mid', label=label_curve)
        ax.set_facecolor(line_type_color[line_check])

        # Title used for the reference:
        ax.set_title(f'{title}', fontsize=1)

        return

    def on_click(self, event):

        line_check = None

        if event.button == 1:
            line_check = True

        if event.button == 2:
            line_check = 'undecided'

        if event.button == 3:
            line_check = False

        # Save line label to dataframe
        self.selections_df.loc[self.idx_ax, 'line'] = line_check
        self.save_log()

        # Plot
        self.ax.clear()
        self.plot_line(self.ax, self.grid_params[self.idx_ax], line_check, title=self.idx_ax)
        self.fig.canvas.draw()

        return

    def on_enter_axes(self, event):

        # Assign new axis
        self.in_fig = event.canvas.figure
        self.ax = event.inaxes
        self.idx_ax = int(self.ax.get_title())
        self.line_check = line_type_color_invert[self.ax._facecolor]

        # TODO we need a better way to index than the latex label
        # Recognise line line
        # idx_line = self.log.index == self.in_ax.get_title()
        # self.line = self.log.loc[idx_line].index.values[0]
        # self.mask = self.log.loc[idx_line, 'w1':'w6'].values[0]

    def save_log(self):

        with open(self.db_address, 'wb') as output_file:
            string_DF = self.selections_df.to_string()
            output_file.write(string_DF.encode('UTF-8'))

        return