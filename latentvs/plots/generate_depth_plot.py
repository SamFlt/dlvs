import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import sys
import re
def read_csv(path):
    df = pd.read_csv(path, sep=';')
    # print(df)
    model_name_key = df.columns.values[0]
    model_names = df[:][model_name_key]
    depth_run = np.array([list(map(int, re.findall('\d+', s)))[-1] for s in model_names]) # Last int of model name is considered the depth run
    sort_indices = np.argsort(depth_run)
    depth_run = depth_run[sort_indices]
    conv_rates = np.array(df[:][r"Converged, % of samples"])
    conv_rates = conv_rates[sort_indices] * 100.0
    indices = [d + 1 for d in depth_run]

    return indices, conv_rates
if __name__ == '__main__':
    plt.rc('text', usetex=True)
    #plt.rc('font', family='serif')
    path_look_at = sys.argv[1]
    path_screw = sys.argv[2]
    
    indices, conv_la = read_csv(path_look_at)
    _, conv_sc = read_csv(path_screw)
    
    fig = plt.figure()
    ax = fig.gca()
    plt.grid()
    
    plt.plot(indices, conv_la, color='red', marker='x')
    plt.plot(indices, conv_sc, color='blue', marker='x')
    
    # plt.xlim([indices[0], indices[-1]])
    ax.set_xlim(indices[0], indices[-1])
    plt.ylim([0, 100])
    plt.xlabel(r'Number of residual blocks', fontsize=14)
    plt.ylabel('Convergence rate, \\%', fontsize=14)
    plt.xticks(indices)
    # fig.gca().get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    plt.legend(['Look-at test', 'Screw motion test'], loc='lower right', fontsize=14)
    plt.tight_layout()
    plt.savefig('depth_vs_convergence.pdf', format='pdf')
    plt.show()

    # print(df.columns.values)