import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.colors import LogNorm, LinearSegmentedColormap, to_rgba_array
from mpl_toolkits.axes_grid1 import make_axes_locatable
from statsmodels.stats.multitest import multipletests
from pathlib import Path
import pickle as pkl
import scipy.stats as stats
import numpy as np

# local imports
import sys
sys.path.append(str(Path(__file__).parents[2])) # adds the parent directory to the path so that the utils module can be imported

colours = ['#9ad8b1', "#9dbde6", '#5DBB63FF']

# set font for all plots
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['image.cmap'] = 'RdBu_r'
plt.rcParams['image.interpolation'] = 'bilinear'
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['legend.title_fontsize'] = 12
plt.rcParams['figure.titlesize'] = 14
plt.rcParams['figure.dpi'] = 300



def flatten_remove_nans(arr):
    # flatten the array
    arr = arr.flatten()

    # remove nans
    arr = arr[~np.isnan(arr)]

    return arr
    

def explained_variance(y_true, y_pred):
    """
    Compute the explained variance score.
    
    Explained variance measures the proportion of the variance in the dependent variable
    that is predictable from the independent variables.
    
    Parameters:
    y_true (array-like): Actual values
    y_pred (array-like): Predicted values
    
    Returns:
    float: Explained variance score
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    variance_y_true = np.var(y_true, ddof=1)
    variance_residuals = np.var(y_true - y_pred, ddof=1)
    
    if variance_y_true == 0:
        return 1.0 if variance_residuals == 0 else 0.0
    
    return 1 - (variance_residuals / variance_y_true)

def explained_variance_relative_to_mean(y_true, y_pred):
    """
    Computes the explained variance relative to just using the mean of y_true.

    Parameters:
    - y_true: array-like, true values.
    - y_pred: array-like, predicted values.

    Returns:
    - Explained variance score (R^2-like measure).
    """
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)  # Total sum of squares
    ss_residual = np.sum((y_true - y_pred) ** 2)  # Residual sum of squares

    if ss_total == 0:  # Handle edge case where variance in y_true is zero
        return 1.0 if ss_residual == 0 else 0.0

    return 1 - (ss_residual / ss_total)


def get_explained_variance_diagonal(predicted, true):
    """
    Calculate the explained variance between the predicted and true values for each timepoint
    """
    n_cv = predicted.shape[1]

    explained_variance_diagonal = np.zeros((250, n_cv))
    for i in range(250):
        for n in range(n_cv):
            # take only the non-nan values and flatten the array
            tmp_predicted = flatten_remove_nans(predicted[i, n, :])
            tmp_true = flatten_remove_nans(true[i, n, :])

            # calculate the correlation
            explained_variance_diagonal[i, n] = explained_variance_relative_to_mean(tmp_true, tmp_predicted)

    return explained_variance_diagonal


def prepare_dicts(results_path):

    results = {}
    print(results_path)
    # loop over the files in the path 
    for i, file in enumerate(results_path.iterdir()):
        with open(file, 'rb') as fr:
            try:
                while True:
                    tmp_data = pkl.load(fr)
                    results[file.name] = tmp_data
          

            except EOFError:
                pass
    
    output = {}     
    # loop over the results and permuted results    
    for key, value in results.items():
        # add to the output dictionary
        output[key] = {
            "diagonal": get_explained_variance_diagonal(value["predicted"], value["true"])
        }
    return output



def custom_colormap():
    """
    Copied from https://github.com/vidaurre/glhmm/blob/main/glhmm/graphics.py
    Generate a custom colormap consisting of segments from red to blue.

    Returns:
    --------
    A custom colormap with defined color segments.
    """
    # Retrieve existing colormaps
    coolwarm_cmap = plt.get_cmap('coolwarm').reversed()
    coolwarm_cmap2 = plt.get_cmap('autumn')
    copper_cmap = plt.get_cmap('copper').reversed()
    # Define the colors for the colormap
    copper_color1 = to_rgba_array(copper_cmap(1))[0][:3]
    # Define the colors for the colormap
    red = (1,0,0)
    red2 = (66/255, 13/255, 9/255)
    orange =(1, 0.5, 0)
    # red_color1 = to_rgba_array(coolwarm_cmap(0))[0][:3]
    warm_color2 = to_rgba_array(coolwarm_cmap2(0.8))[0][:3]
    blue_color1 = to_rgba_array(coolwarm_cmap(0.6))[0][:3]
    blue_color2 = to_rgba_array(coolwarm_cmap(1.0))[0][:3] # Extract the blue color from coolwarm

    # Define the color map with three segments: red to white, white, and white to blue
    cmap_segments = [
        (0.0, red2),
        #(0.002, orange),
        (0.005, red),   # Intermediate color
        (0.02, orange),   # Intermediate color
        #(0.045, warm_color1),
        (0.040, warm_color2),  # Intermediate color
        (0.05, copper_color1),
        (0.09,blue_color1),
        (1, blue_color2)
    ]

    # Create the custom colormap
    custom_cmap = LinearSegmentedColormap.from_list('custom_colormap', cmap_segments)

    return custom_cmap


def interpolate_colormap(cmap_list):     
    """
    Copied from https://github.com/vidaurre/glhmm/blob/main/glhmm/graphics.py
    Create a new colormap with the modified color_array.

    Parameters:
    --------------
    cmap_list (numpy.ndarray): 
        Original color array for the colormap.

    Returns:
    ----------  
    modified_cmap (numpy.ndarray): 
        Modified colormap array.
    """
    # Create a new colormap with the modified color_array
    modified_cmap  = np.ones_like(cmap_list)

    for channel_idx in range(3):
        # Extract the channel values from the colormap
        channel_values = cmap_list[:, channel_idx]

        # Get unique values, their indices, and counts
        unique_values, unique_indices, counts = np.unique(channel_values, return_index=True, return_counts=True)

        # Create a copy unique_indices that is will get reduced for every interation
        remaining_indices = unique_indices.copy()
        remaining_counts = counts.copy()
        # Create a list to store the interpolated values
        new_map_list = []

        for _ in range(len(unique_values)-1):
            # Find the minimum value
            min_value = np.min(remaining_indices)
            # Locate the index
            min_idx =np.where(unique_indices==min_value)
            # Remove the minimum value from the array
            remaining_counts = remaining_counts[remaining_indices != min_value]
            remaining_indices = remaining_indices[remaining_indices != min_value]
            
            # Find the location of the next minimum value from remaining_indices
            next_min_value_idx =np.where(unique_indices==np.min(remaining_indices))
            # Calculate interpolation space difference
            space_diff = (unique_values[next_min_value_idx]-unique_values[min_idx])/int(counts[min_idx])
            # Append interpolated values to the list
            new_map_list.append(np.linspace(unique_values[min_idx], unique_values[next_min_value_idx]-space_diff, int(counts[min_idx])))
        last_val =np.where(unique_indices==np.min(remaining_indices))
        for _ in range(int(remaining_counts)):
            # Append the last value to the new_map_list
            new_map_list.append([unique_values[last_val]])
        con_values= np.squeeze(np.concatenate(new_map_list))
        # Insert values into the new color map
        modified_cmap [:,channel_idx]=con_values
    
    return modified_cmap


def create_cmap(num_colors:int = 259, pval_min = -3):    
    # Convert to log scale
    color_array = np.logspace(pval_min, 0, num_colors).reshape(1, -1)

    # Create custom colormap
    coolwarm_cmap = custom_colormap()
    # Create a new colormap with the modified color_array

    cmap_list = coolwarm_cmap(color_array)[0]
    cmap_list = interpolate_colormap(cmap_list)
   
    cmap = LinearSegmentedColormap.from_list('custom_colormap', cmap_list)    
    
    return cmap

def bootstrap_p_value(bootstrap_variances, alternative = "larger", popmean = 0):
    """
    Computes the one-sample bootstrap p-value
    """
    if alternative == "larger":
        return np.mean(bootstrap_variances <= popmean)
    elif alternative == "smaller":
        return np.mean(bootstrap_variances >= popmean)
    elif alternative == "two-sided":
        return np.mean(np.abs(bootstrap_variances) >= np.abs(popmean)) * 2  # Two-tailed test
    else:
        raise ValueError("alternative must be 'larger', 'smaller', or 'two-sided'")


def plot_results_diagonals(diag_dict, save_path=None, y_min = -0.75, y_max = 0.75, tick_positions=[0.001, 0.01, 0.05, 0.1, 0.3, 1]):    
    fig, ax = plt.subplots(1, 1, figsize=(10, 4), dpi=300)

    all_diagonals = [] 

    for i, (key, value) in enumerate(diag_dict.items()):
        diagonal = value["diagonal"].mean(-1)
        all_diagonals.append(diagonal)
        ax.plot(diagonal, color= "grey", alpha=0.1, linewidth=0.5, label = "bootstraps" if i == 0 else None)

    all_diagonals = np.array(all_diagonals)

    # create colour map for p_values
    p_vals = [bootstrap_p_value(all_diagonals[:, i]) for i in range(all_diagonals.shape[-1])]

    # Apply FDR correction (Benjamini-Hochberg)
    _, p_vals, _, _ = multipletests(p_vals, alpha=0.05, method='fdr_bh')


    pval_min = -3
    p_vals = np.clip(p_vals, 10**pval_min, [1])
    cmap = create_cmap(pval_min=pval_min)
    norm = LogNorm(vmin=10**pval_min , vmax=1)


    avg_diagonal = np.mean(all_diagonals, axis=0)
    max_index = np.argmax(avg_diagonal)
    print(max_index)


    # plot the diagonal by plotting the line between each sample at a time in colors corresponding to the p-value
    for i, pval, avg in zip(range(len(avg_diagonal)), p_vals, avg_diagonal):
        color = cmap(norm(pval))
        try:
            ax.plot([i, i + 1 ],[avg, avg_diagonal[i+1]], color=color, linewidth=2)
        except (IndexError):
            pass
    

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Explained variance")

    ax.set_xticks(np.arange(0, 251, step=50))
    ax.set_xticklabels([0, 0.2, 0.4, 0.6, 0.8, 1])

    ax.set_ylim(y_min, y_max)
    ax.set_xlim(0, 250)

    ax.legend()

    # Add colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3.5%", pad=0.05)
    colorbar = plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cax, ticks=tick_positions, format="%.3g"
    )

            
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    plt.close()



def plot_results_diagonals_v2(diag_dict, save_path=None, y_min = -0.75, y_max = 0.75, tick_positions=[0.001, 0.01, 0.05, 0.1, 0.3, 1]):    
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), dpi=300, gridspec_kw={'height_ratios': [5, 2], "width_ratios":[10, 1], "wspace":0.05})

    all_diagonals = [] 

    for i, (key, value) in enumerate(diag_dict.items()):
        diagonal = value["diagonal"].mean(-1)
        all_diagonals.append(diagonal)
        axes[0,0].plot(diagonal, color= "forestgreen", alpha=0.1, linewidth=0.5, label = "bootstraps" if i == 0 else None)


    all_diagonals = np.array(all_diagonals)
    avg_diagonal = np.mean(all_diagonals, axis=0)

    axes[0, 0].plot(avg_diagonal, color = "k", linewidth=2, label = "Average")

    # create colour map for p_values
    p_vals = [bootstrap_p_value(all_diagonals[:, i]) for i in range(all_diagonals.shape[-1])]

    # Apply FDR correction (Benjamini-Hochberg)
    _, p_vals, _, _ = multipletests(p_vals, alpha=0.05, method='fdr_bh')


    pval_min = -3
    p_vals = np.clip(p_vals, 10**pval_min, [1])
    cmap = create_cmap(pval_min=pval_min)
    norm = LogNorm(vmin=10**pval_min , vmax=1)


    # plot the diagonal by plotting the line between each sample at a time in colors corresponding to the p-value
    #for i, pval, avg in zip(range(len(avg_diagonal)), p_vals, avg_diagonal):
    #    color = cmap(norm(pval))
    #    try:
    #        axes[1].plot([i, i + 1 ],[avg, avg_diagonal[i+1]], color=color, linewidth=2)
    #    except (IndexError):
    #        pass

    for i, pval in zip(range(len(avg_diagonal)), p_vals):
        color = cmap(norm(pval))
        try:
            axes[1,0].plot([i, i + 1 ],[norm(pval), norm(p_vals[i+1])], color=color, linewidth=2)
        except (IndexError):
            pass
    
    axes[1,0].set_xlabel("Time (s)")
    axes[0,0].set_ylabel("Explained variance")
    axes[0,0].set_ylim(y_min, y_max)
    
    axes[0,0].axhline(0, linestyle = "--", alpha = 0.2, color = "k", linewidth = 0.5)

    for ax in [axes[1, 0], axes[1,1], axes[0,1]]:
        ax.axis('off')
        ax.set_ylim(0, 1)

    for ax in axes[:, 0]:
        ax.set_xticks(np.arange(0, 251, step=50))
        ax.set_xticklabels([0, 0.2, 0.4, 0.6, 0.8, 1])
        ax.set_xlim(0, 250)

    # Add colorbar
    divider = make_axes_locatable(axes[1,1])
    cax = divider.append_axes("left", size="80%", pad=0.05)  # Increase "size" for a bigger color bar


    colorbar = plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cax, ticks=tick_positions, format="%.3g"
    )

    colorbar.ax.tick_params(labelsize=8)  # Make tick labels smaller


    """
    # Add a colorbar linked to the figure instead of a single axis
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes, 
        orientation='vertical', fraction=0.02, pad=0.02
    )
    cbar.set_ticks(tick_positions)
    cbar.ax.set_yticklabels([f"{tick:.3g}" for tick in tick_positions])

    """  
            
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    plt.close()



if __name__ == "__main__":
    path = Path(__file__).parent

    save_path = path / 'plots'

    # make dirs for saving if they don't exist
    if not save_path.exists():
        save_path.mkdir()

        
    dict_path = path / "dicts"
    
    if not dict_path.exists():
        dict_path.mkdir()

    # looping over each trial type, session type and prediction type
    for trial_type in ["animate"]:#, "inanimate"]:
        for session_type in ["visual"]:
            for predict in ["session day", "session number"]:

                file_path = dict_path / f"correlation_dict_{trial_type}_{session_type}_{predict.replace(' ', '')}.npy"

                if (file_path).exists():                           
                    print(f"Loading explained variance for {trial_type} {session_type} {predict} from file")
                    correlation_dict = np.load(file_path, allow_pickle=True).item()

                else:
                    print(f"Calculating explained variance for {trial_type} {session_type} {predict}")
                    results_path = path / "results" / f"{trial_type}_{session_type}_{predict.replace(' ', '_')}"
                    correlation_dict = prepare_dicts(results_path)
                    np.save(file_path, correlation_dict) # save the dictionary to file to be loaded in if script is run again
    

                plot_results_diagonals(
                    correlation_dict,
                    save_path = save_path / f'diagonals_{trial_type}_{session_type}_{predict.replace(" ", "")}.png'
                    )
                
                plot_results_diagonals_v2(
                    correlation_dict,
                    save_path = save_path / f'diagonals_v2_{trial_type}_{session_type}_{predict.replace(" ", "")}.png'
                    )
    