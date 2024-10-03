import numpy as np
import pandas as pd
from lime.plots import theme
from matplotlib import pyplot as plt, rc_context, cm
from scipy.interpolate import griddata


format_figure = {'figure.figsize': (4, 4), 'figure.dpi': 800, "xtick.labelsize" : 5, "ytick.labelsize" : 5}
# theme.set_style(fig_cfg='figure.figsize')


# Read the CSV file
summary_txt_file = "/home/vital/Astrodata/LiMe_ml/v3/results/training_multi_sample_v3_min-max_4_categories_v1_search_80000points_min_samples_1-5-10-15percent_mult_sqrt_features_summary.txt"
results_df = pd.read_csv(summary_txt_file, sep='\s+', header=0, index_col=0)

estimators_depth_pairs = [[20, 4], [20, 6], [20, 8],
                          [40, 4], [40, 6], [40, 8],
                          [40, 4], [40, 6], [40, 8]]

max_features_list = ['sqrt']

# Compute mean f1 score
idx_max = results_df['mean_test_score'].idxmax()

# Define a color map (you can change 'viridis' to other color maps)
cmap = cm.get_cmap('viridis')

legend_size = 3
text_size = 8

axis_limits_values = np.array([800, 4000, 8000, 12000])
ticks_labels = [f'{tick:.0f}%' for tick in axis_limits_values / 80000 * 100]
intervals_grid = 6

# Make a figure
with rc_context(theme.fig_defaults(format_figure)):

    fig, axes = plt.subplots(3, 3)
    axes_list = axes.ravel()
    norm = plt.Normalize(0.90, 0.95)

    for i, efficiency_pairs in enumerate(estimators_depth_pairs):

        n_estimators, max_depth = efficiency_pairs
        max_features = max_features_list[0]

        idcs = ((results_df.param_max_features == max_features) &
                (results_df.param_max_depth == max_depth) &
                (results_df.param_n_estimators == n_estimators))

        selection_df = results_df.loc[idcs]

        x_arr = selection_df["param_min_samples_leaf"].to_numpy()
        y_arr = selection_df["param_min_samples_split"].to_numpy()
        f1_arr = selection_df["mean_test_score"].to_numpy()

        # Grid format
        grid_x, grid_y = np.meshgrid(np.linspace(x_arr.min(), x_arr.max(), intervals_grid), np.linspace(x_arr.min(), y_arr.max(), intervals_grid))
        grid_f1 = griddata((x_arr, y_arr), f1_arr, (grid_x, grid_y), method='cubic')

        sc = axes_list[i].pcolormesh(grid_x, grid_y, grid_f1, cmap=cmap, norm=norm, shading='auto')
        # axes_list[i].update({'title': f'Number estimators {n_estimators}, Number estimators {max_depth}'})

        axes_list[i].text(0.5, 0.90, f'Number estimators {n_estimators}\n\nMaximum depth {max_depth}',
                            ha='center', va='top', transform=axes_list[i].transAxes,
                            fontsize=legend_size, color='black',
                            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5',linewidth=0.25))

        if axes_list[i] in axes[2, :]:
            axes_list[i].set_xticks(axis_limits_values)  # Set the tick positions
            axes_list[i].set_xticklabels(ticks_labels)
        else:
            axes_list[i].set_xticklabels([])
            axes_list[i].set_xticks([])
            axes_list[i].set_xlabel('')

        if axes_list[i] in axes[:, 0]:
            axes_list[i].set_yticks(axis_limits_values)  # Set the tick positions
            axes_list[i].set_yticklabels(ticks_labels)
        else:
            axes_list[i].set_yticklabels([])
            axes_list[i].set_yticks([])
            axes_list[i].set_ylabel('')


        fig.text(0.5, 0.01, 'Minimum number for a leaf (% category sample size)', ha='center', fontsize=text_size)
        fig.text(0.01, 0.5, 'Minimum number for split (% category sample size)', va='center', rotation='vertical', fontsize=text_size)

    # Add the color bar
    fig.colorbar(sc, ax=axes[:, 2], orientation='vertical', label='F1 score')#, fraction=0.02, pad=0.04)
    plt.savefig(f'hyper_parameter_search.png', bbox_inches='tight')
    plt.show()
