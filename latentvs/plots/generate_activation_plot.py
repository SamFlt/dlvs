import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import sys
import re
if __name__ == '__main__':
    """
    Generate matplotlib plot comparing the performance of different activation functions for servoing
    Takes as argument the path to a result .csv
    """
    path = sys.argv[1]
    df = pd.read_csv(path, sep=';', index_col=0)
    model_name_key = df.columns.values[0]
    model_names = df[:][model_name_key]
    convs = df[:][r'Converged, % of samples']
    print(convs.keys())
    print(convs)
    model_start = 'resnet_32_'
    model_name_to_plot_name = {
        model_start + 'relu_ESM': 'ReLU',
        model_start + 'softplus_ESM': 'Softplus',
        model_start + 'tanh_ESM': 'Tanh',
        model_start + 'leaky_relu_0.01_ESM': 'Leaky \nReLU, 0.01',
        model_start + 'leaky_relu_0.1_ESM': 'Leaky \n ReLU, 0.1',
        model_start + 'leaky_relu_0.5_ESM': 'Leaky \nReLU, 0.5',
    }
    

    fig = plt.figure()
    ax = fig.gca()
    names = [n for n in convs.keys()]
    names = [model_name_to_plot_name[n] for n in names]
    rates = [c for c in convs]
    
    plt.grid()
    plt.bar(names, rates)
    plt.ylim([0, 1])
    
    plt.ylabel('Convergence rate, %', fontsize=14)
    # fig.gca().get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    plt.tight_layout()
    plt.savefig('activations.pdf', format='pdf')
    plt.show()

    # print(df.columns.values)