import numpy as np
from lime.plots import theme
from matplotlib import pyplot as plt, rc_context

def scatter_plot(ax, x_arr, y_arr, labels_arr, feature_list, id_dict, color_dict, alpha=0.5, idx_target=None):

    for feature in feature_list:
        idcs_class = labels_arr == feature
        x_feature = x_arr[idcs_class]
        y_feature = y_arr[idcs_class]
        ax.scatter(x_feature, y_feature, label=id_dict[feature], color=color_dict[id_dict[feature]], alpha=alpha, edgecolor='none')

    if idx_target is not None:
        ax.scatter(x_arr[idx_target], y_arr[idx_target], marker='x', label='selection', color='black')


    return


def ax_wording(ax, ax_cfg=None, legend_cfg=None, yscale=None):


    ax.update(ax_cfg)

    if legend_cfg is not None:
        ax.legend(**legend_cfg)

    if yscale is not None:
        ax.set_yscale(yscale)

    return


def parse_fig_cfg(fig_cfg=None, ax_diag=None, ax_line=None):

    # Input configuration updates default
    fig_cfg = fig_cfg if fig_cfg is not None else {'axes.labelsize': 10, 'axes.titlesize': 10,
                                                'figure.figsize': (12, 6), 'hatch.linewidth': 0.3, 'legend.fontsize': 8}
    fig_cfg = theme.fig_defaults(fig_cfg)

    ax_diag = {} if ax_diag is None else ax_diag
    ax_diag = {'xlabel': r'$\frac{\sigma_{gas}}{\Delta\lambda_{inst}} = \sigma_{pixels}$ (Gaussian sigma in pixels)',
               **ax_diag}

    ax_line = {} if ax_line is None else ax_line
    ax_line = {'xlabel': 'Feature Number', 'ylabel': 'value', **ax_line}

    return {'fig': fig_cfg, 'ax1': ax_diag, 'ax2': ax_line}

class SampleReviewer:

    def __init__(self, sample_df, id_dict, color_dict, fig_cfg=None, ax_diag=None, ax_line=None, base=10000):

        self.y_base = base
        self.x_coords = sample_df['res_ratio'].to_numpy()
        self.y_coords = sample_df['int_ratio'].to_numpy()
        self.y_coords_log = np.log10(self.y_coords)/np.log10(self.y_base)
        self.id_arr = sample_df['spectral_number'].to_numpy()
        self.classes = np.sort(np.unique(self.id_arr))
        self.data_df = sample_df.iloc[:, 3:]
        self.wave_range = np.arange(self.data_df.columns.size)

        self.idx_current = None
        self.id_dict = id_dict
        self.color_dict = color_dict

        self.fig_format = parse_fig_cfg(fig_cfg, ax_diag, ax_line)
        self._fig, self._ax1, self._ax2 = None, None, None

        return

    def interactive_plot(self):

        # Generate the figure
        with rc_context(self.fig_format['fig']):

            # Create the figure
            self._fig, (self._ax1, self._ax2) = plt.subplots(1, 2)

            # Diagnostic plot
            scatter_plot(self._ax1, self.x_coords, self.y_coords, self.id_arr, self.classes, self.id_dict, self.color_dict,
                         idx_target=self.idx_current)
            ax_wording(self._ax1, self.fig_format['ax1'], legend_cfg={'loc': 'lower center', 'ncol':2, 'framealpha':0.95},
                       yscale='log')

            # Line plot
            self.index_target()
            self.line_plot()
            ax_wording(self._ax2, self.fig_format['ax2'])

            # Interactive widget
            self._fig.canvas.mpl_connect('button_press_event', self._on_click)

            # Display the plot
            plt.tight_layout()
            plt.show()

        return

    def _on_click(self, event):

        if event.inaxes == self._ax1 and event.button == 1:

            user_point = (event.xdata, np.log10(event.ydata) / np.log10(self.y_base))

            # Get index point
            self.index_target(user_point)

            # Replot the figures
            self._ax1.clear()
            scatter_plot(self._ax1, self.x_coords, self.y_coords, self.id_arr, self.classes, self.id_dict, self.color_dict,
                         idx_target=self.idx_current)
            ax_wording(self._ax1, self.fig_format['ax1'], legend_cfg={'loc': 'lower center', 'ncol':2, 'framealpha':0.95},
                       yscale='log')

            self._ax2.clear()
            self.line_plot()
            self._fig.canvas.draw()

        return

    def index_target(self, mouse_coords=None):

        # If no selection use first point
        if mouse_coords is None:
            self.idx_current = 0
            print(f'Reseting location')

        else:
            print('Click on:', mouse_coords)
            distances = np.sqrt((self.x_coords - mouse_coords[0]) ** 2 + (self.y_coords_log - mouse_coords[1]) ** 2)
            self.idx_current = np.argmin(distances)

        return

    def line_plot(self):

        feature = self.id_dict[self.id_arr[self.idx_current]]
        self._ax2.step(self.wave_range, self.data_df.iloc[self.idx_current, :].to_numpy(), label=feature,
                      color=self.color_dict[feature], where='mid')

        return