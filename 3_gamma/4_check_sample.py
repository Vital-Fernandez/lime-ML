import numpy as np
import pandas as pd
import lime
from lime.plots import theme
from lime.recognition import detection_function, cosmic_ray_function, broad_component_function
from pathlib import Path
from matplotlib import pyplot as plt, rc_context


# Read sample configuration
cfg_file = 'training_sample_v3.toml'
sample_params = lime.load_cfg(cfg_file)

version = sample_params['data_labels']['version']
cfg = sample_params[f'training_data_{version}']
inverted_class_dict = {value: key for key, value in cfg['classes'].items()}

# Read the sample database
output_folder = Path(sample_params['data_labels']['output_folder'])/version
sample_database = f'{output_folder}/training_multi_sample_{version}.csv'
database_df = pd.read_csv(sample_database)

# Figure format
fig_cfg = theme.fig_defaults({'axes.labelsize': 10,
                              'axes.titlesize':10,
                              'figure.figsize': (3, 3),
                              'hatch.linewidth': 0.3,
                              "legend.fontsize" : 8})

# Creat the figure
n_points = 500
with rc_context(fig_cfg):

    # Loop throught the sample and generate the figures
    fig, ax = plt.subplots()
    for label_feature, number_feature in cfg['classes'].items():
        if number_feature > 0:
            print(label_feature, number_feature)

            # Filter the DataFrame by the category
            idcs_feature = database_df['spectral_number'] == number_feature

            if idcs_feature.sum() > 0:
                feature_df = database_df.loc[database_df['spectral_number'] == number_feature].sample(n=n_points)

                int_ratio = feature_df.loc[:, 'int_ratio'].to_numpy()
                res_ratio = feature_df.loc[:, 'res_ratio'].to_numpy()
                color = cfg[label_feature]['color']

                ax.scatter(res_ratio, int_ratio, color=color, label=label_feature, alpha= 0.5, edgecolor='none')

                # Narrow component case
                if label_feature == 'broad':
                    feature_df = database_df.loc[database_df['spectral_number'] == number_feature + 0.5].sample(n=n_points)

                    int_ratio = feature_df.loc[:, 'int_ratio'].to_numpy()
                    res_ratio = feature_df.loc[:, 'res_ratio'].to_numpy()
                    color = cfg[label_feature]['color']

                    ax.scatter(res_ratio, int_ratio, marker='x', color='black', label='narrow', alpha= 0.5, edgecolor='none')



    # Wording
    ax.update({'xlabel': r'$\frac{\sigma_{gas}}{\Delta\lambda_{inst}} = \sigma_{pixels}$ (Gaussian sigma in pixels)',
               'ylabel': r'$\frac{A_{gas}}{\sigma_{noise}}$ (Signal-to-noise)'})
    ax.legend(loc='lower center', ncol=2, framealpha=0.95)

    # Axis format
    ax.set_yscale('log')
    # ax.set_xlim(0, 10)
    # ax.set_ylim(0.01, 10000)

    # Upper axis
    ax2 = ax.twiny()
    ticks_values = ax.get_xticks()
    ticks_labels = [f'{tick:.0f}' for tick in ticks_values*6]
    ax2.set_xticks(ticks_values)  # Set the tick positions
    ax2.set_xticklabels(ticks_labels)
    ax2.set_xlabel(r'$b_{pixels}$ (detection box width in pixels)')

    # Grid
    ax.grid(axis='x', color='0.95', zorder=1)
    ax.grid(axis='y', color='0.95', zorder=1)

    plt.tight_layout()
    plt.show()