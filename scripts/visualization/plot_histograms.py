import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import numpy as np

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import pandas as pd

def plot_histograms(df):
    """
    Plots histograms for each numerical column in the DataFrame.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing numerical columns.

    Returns:
        None

    Plots histograms with density curves and empirical rule indicators for each numerical column in the DataFrame.
    The histograms display the frequency distribution of the data along with probability density functions.
    The empirical rule is represented by dashed lines at ±1, ±2, and ±3 standard deviations from the mean.

    Usage:
        plot_histograms(dataframe)
    """
    # Filter out only numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns

    num_cols = len(numerical_cols)
    num_rows = num_cols // 5 + (num_cols % 5 > 0)
    fig, axs = plt.subplots(nrows=num_rows, ncols=5, figsize=(20, num_rows*4))
    axs = axs.flatten()

    for i, col in enumerate(numerical_cols):
        data = df[col]
        mu, std = np.mean(data), np.std(data)

        axs[i].hist(data, bins=10, density=True, alpha=0.7, color='skyblue')
        xmin, xmax = axs[i].get_xlim()
        x = np.linspace(xmin, xmax, 100)
        p = stats.norm.pdf(x, mu, std)
        axs[i].plot(x, p, 'k', linewidth=2)
        axs[i].axvline(mu-3*std, color='r', linestyle='--', linewidth=2, alpha=0.5)
        axs[i].axvline(mu-2*std, color='y', linestyle='--', linewidth=2, alpha=0.5)
        axs[i].axvline(mu-std, color='g', linestyle='--', linewidth=2, alpha=0.5)
        axs[i].axvline(mu, color='purple', linestyle='-', linewidth=2, alpha=0.5)
        axs[i].axvline(mu+std, color='g', linestyle='--', linewidth=2, alpha=0.5)
        axs[i].axvline(mu+2*std, color='y', linestyle='--', linewidth=2, alpha=0.5)
        axs[i].axvline(mu+3*std, color='r', linestyle='--', linewidth=2, alpha=0.5)
        axs[i].set_xlabel(col)
        axs[i].set_ylabel('Relative Frequency')

    for i in range(num_cols, num_rows*5):
        fig.delaxes(axs[i])

    fig.suptitle('Histograms with density curves and empirical rule indicators', fontsize=15)
    fig.tight_layout()
    plt.show()
