import numpy as np
from lime.plots import theme
from matplotlib import pyplot as plt, rc_context
from lime.recognition import detection_function, cosmic_ray_function
from matplotlib.lines import Line2D


def scatter_plot(ax, x_arr, y_arr, labels_arr, feature_list, color_dict, alpha=0.5, idx_target=None,
                 detection_range=None):

    for feature in feature_list:
        idcs_class = labels_arr == feature
        x_feature = x_arr[idcs_class]
        y_feature = y_arr[idcs_class]
        label = f'{feature} ({y_feature.size})'
        ax.scatter(x_feature, y_feature, label=label, color=color_dict[feature], alpha=alpha, edgecolor='none')

    if idx_target is not None:
        ax.scatter(x_arr[idx_target], y_arr[idx_target], marker='x', label='selection', color='black')

    if detection_range is not None:
        ax.plot(detection_range, detection_function(detection_range))

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
               'ylabel': r'$\frac{A_{gas}}{\sigma_{noise}}$ (Signal-to-noise)',
               **ax_diag}

    ax_line = {} if ax_line is None else ax_line
    ax_line = {'xlabel': 'Feature Number', 'ylabel': 'value', **ax_line}

    return {'fig': fig_cfg, 'ax1': ax_diag, 'ax2': ax_line}


def diagnostics_plot(model_cfg=None, categories=None, missmatch_df=None, fig_cfg=None, ax_cfg=None, color_dict=None,
                     output_address=None):

    fig_cfg = theme.fig_defaults({'axes.labelsize': 10, 'axes.titlesize':10, 'figure.figsize': (4, 4),
                                  'hatch.linewidth': 0.3, "legend.fontsize" : 8})

    ax_cfg = {**ax_cfg, **{'xlabel': r'$\frac{\sigma_{gas}}{\Delta\lambda_{inst}} = \sigma_{pixels}$ (Gaussian sigma in pixels)',
               'ylabel': r'$\frac{A_{gas}}{\sigma_{noise}}$ (Signal-to-noise)'}}

    # Plot limits
    res_ratio_min = model_cfg['res-ratio_min']
    res_ratio_max = model_cfg['box_pixels']/model_cfg['n_sigma']
    int_ratio_min = model_cfg['int-ratio_min']
    int_ratio_max = model_cfg['int-ratio_max']

    # Compute boundary lines values
    x_detection = np.linspace(res_ratio_min, res_ratio_max, 50)
    y_detection = detection_function(x_detection)

    x_pixel_lines = np.linspace(0, 0.6, 100)
    y_pixel_lines = cosmic_ray_function(x_pixel_lines)
    idcs_crop = y_pixel_lines > 5

    # Cosmic ray and pixel-line boundary
    cr_limit = model_cfg['cosmic-ray']['cosmic-ray_boundary']
    detection_limit_pixel = 0.5439164154163089
    x_cr_low = np.linspace(0, detection_limit_pixel, 100)
    y_cr_low = cosmic_ray_function(x_cr_low)
    idcs_high = y_cr_low > cr_limit
    y_cr_low[idcs_high] = cr_limit

    x_cr_high = x_cr_low[idcs_high]
    y_cr_high = cosmic_ray_function(x_cr_high)

    # Doublet
    x_doublet = np.linspace(model_cfg['doublet']['min_res_ratio'], model_cfg['doublet']['max_res_ratio'], 50)
    y_doublet_min = detection_function(x_doublet) * model_cfg['doublet']['min_detection_factor']
    y_doublet_max =  np.full_like(x_doublet, int_ratio_max)

    # Broad
    x_broad = np.linspace(cosmic_ray_function(int_ratio_max, res_ratio_check=False), res_ratio_max, 100)
    y_broad = cosmic_ray_function(x_broad)
    broad_int_min_factor = model_cfg['broad']['min_detection_factor']
    broad_int_max = int_ratio_max * model_cfg['broad']['narrow_broad_max_factor']
    idx_detect = y_broad < broad_int_min_factor * detection_function(x_broad)
    y_broad[idx_detect] = broad_int_min_factor * detection_function(x_broad)[idx_detect]


    with (rc_context(fig_cfg)):

        fig, ax = plt.subplots()

        # Detection boundary
        ax.plot(x_detection, y_detection, color='black', label='Detection boundary')

        # Single pixel line boundary
        ax.plot(x_pixel_lines[idcs_crop], y_pixel_lines[idcs_crop], linestyle='--', color='black', label='Single pixel boundary')

        if 'emission' in categories:
            ax.fill_between(x_detection, y_detection, int_ratio_max, alpha=1, color=color_dict['emission'], label='emission',
                            edgecolor='none')

        if 'white-noise' in categories:
            ax.fill_between(x_detection, 0, y_detection, alpha=1, color=color_dict['white-noise'], label='white-noise',
                            edgecolor='none')

        if 'pixel-line' in categories:
            ax.fill_between(x_cr_low, detection_function(x_cr_low), y_cr_low, color=color_dict['pixel-line'], label='pixel-line', edgecolor='none')

        if 'cosmic-ray' in categories:
            ax.fill_between(x_cr_high, cr_limit, y_cr_high, color=color_dict['cosmic-ray'], label='cosmic-ray', edgecolor='none')

        if 'doublet' in categories:
            ax.fill_between(x_doublet, y_doublet_min, y_doublet_max, color=color_dict['doublet'], alpha=0.5, label='doublet',
                            edgecolor='none', zorder=2)

        if 'broad' in categories:
            ax.fill_between(x_broad, y_broad, broad_int_max, color=color_dict['broad'], alpha=0.5, label='broad',
                            edgecolor='none', zorder=2)

        # Add the bad detections
        if missmatch_df is not None:
            false_categories = missmatch_df['shape_class'].unique()
            for category in false_categories:
                idcs_fake = missmatch_df['shape_class'] == category
                x_coords = missmatch_df.loc[idcs_fake, 'res_ratio'].to_numpy()
                y_coords = missmatch_df.loc[idcs_fake, 'int_ratio'].to_numpy()
                color_arr = missmatch_df.loc[idcs_fake, 'pred_values'].map(color_dict).to_numpy()
                ax.scatter(x_coords, y_coords, color=color_arr, marker='o', alpha=0.5, edgecolor='none')

        # Grid
        ax.grid(axis='x', color='0.95', zorder=1)
        ax.grid(axis='y', color='0.95', zorder=1)

        # Axis format
        ax.set_yscale('log')
        ax.set_xlim(res_ratio_min, res_ratio_max)
        ax.set_ylim(int_ratio_min, int_ratio_max)
        ax.update(ax_cfg)

        # Legend
        ax.legend(loc='lower center', ncol=2, framealpha=0.95)

        # Missmatch entry
        if missmatch_df is not None:
            custom_marker = Line2D([0], [0], marker='o', color='black', markerfacecolor='none', markeredgecolor='black',
                                   label='Miss-identified object', linestyle='none')

            handles, labels = plt.gca().get_legend_handles_labels()
            handles.append(custom_marker)
            labels.append('Miss-identified object')
            ax.legend(handles=handles, labels=labels, loc='lower center', ncol=2, framealpha=0.5)

        plt.tight_layout()
        if output_address is not None:
            plt.savefig(output_address, )
            plt.savefig(f'hyper_parameter_search.png', bbox_inches='tight')

        plt.show()

    return



class SampleReviewer:

    def __init__(self, sample_df, color_dict, fig_cfg=None, ax_diag=None, ax_line=None,
                 base=10000, sample_size = 5000, column_labels='shape_class'):


        crop_df = sample_df.sample(sample_size)

        self.y_base = base
        self.x_coords = crop_df['res_ratio'].to_numpy()
        self.y_coords = crop_df['int_ratio'].to_numpy()
        self.y_coords_log = np.log10(self.y_coords)/np.log10(self.y_base)
        self.id_arr = crop_df[column_labels].to_numpy()
        self.classes = np.sort(np.unique(self.id_arr))
        self.data_df = crop_df.loc[:, 'Pixel0':]
        self.wave_range = np.arange(self.data_df.columns.size)

        self.idx_current = None
        self.color_dict = color_dict

        self.fig_format = parse_fig_cfg(fig_cfg, ax_diag, ax_line)
        self._fig, self._ax1, self._ax2 = None, None, None

        self.detection_range=np.linspace(self.x_coords.min(), self.x_coords.max(), 50)

        return

    def interactive_plot(self):

        # Generate the figure
        with rc_context(self.fig_format['fig']):

            # Create the figure
            self._fig, (self._ax1, self._ax2) = plt.subplots(1, 2)

            # Diagnostic plot
            scatter_plot(self._ax1, self.x_coords, self.y_coords, self.id_arr, self.classes, self.color_dict,
                         idx_target=self.idx_current, detection_range=self.detection_range)
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
            scatter_plot(self._ax1, self.x_coords, self.y_coords, self.id_arr, self.classes, self.color_dict,
                         idx_target=self.idx_current, detection_range=self.detection_range)
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

        feature = self.id_arr[self.idx_current]
        self._ax2.step(self.wave_range, self.data_df.iloc[self.idx_current, :].to_numpy(), label=feature,
                      color=self.color_dict[feature], where='mid')

        return