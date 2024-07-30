import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import sys
import re

def get_conv_rates(path):
    df = pd.read_csv(path, sep=';')
    model_name_key = df.columns.values[0]
    model_names = df[:][model_name_key]
    model_latent_dims = np.array([next(map(int, re.findall('\d+', s))) for s in model_names]) # First int of model name is considered the latent dim
    sort_indices = np.argsort(model_latent_dims)
    model_latent_dims = model_latent_dims[sort_indices]
    conv_rates = np.array(df[:][r"Converged, % of samples"])
    conv_rates = conv_rates[sort_indices] * 100
    return model_latent_dims, conv_rates

if __name__ == '__main__':
    plt.rc('text', usetex=True)
    #plt.rc('font', family='serif')
    path = sys.argv[1]
    path_screw = sys.argv[2]
    # path_pca = sys.argv[2]
    
    latent_dims_aevs, conv_rates_aevs = get_conv_rates(path)
    latent_dims_aevs, conv_rates_screw_aevs = get_conv_rates(path_screw)
    
    # latent_dim_pca, conv_rate_pca = get_conv_rates(path_pca)
    fig = plt.figure()
    ax = fig.gca()
    plt.grid()
    plt.plot(latent_dims_aevs, conv_rates_aevs, color='red', marker='x')
    plt.plot(latent_dims_aevs, conv_rates_screw_aevs, color='blue', marker='+')
    
    plt.ylim([0, 100])
    plt.xlabel(r'Latent vector $\mathbf{z}$ size', fontsize=14)
    plt.ylabel('Convergence rate, \\%', fontsize=14)
    plt.semilogx(2)
    plt.xticks(latent_dims_aevs)
    plt.minorticks_off()
    fig.gca().get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    plt.legend(['Look-at test', 'Screw motion test'], loc='lower right', fontsize=14)
    plt.tight_layout()
    plt.savefig('latent_dim_vs_convergence.pdf', format='pdf')
    plt.show()

    # print(df.columns.values)