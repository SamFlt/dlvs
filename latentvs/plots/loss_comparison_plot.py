import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import sys
import numpy as np
def read_results(df):
    
    # model_names = df[:][model_name_key]
    # print(model_names)
    convs = df[:][r'Converged, % of samples']
    print(convs)
    is_bce_loss = lambda x: 'bce' in x
    bce_res = []
    dct_res = []
    for i in range(len(convs)):
        if is_bce_loss(df.index[i]):
            bce_res.append(convs[i] * 100)
        else:
            dct_res.append(convs[i] * 100)
    return np.array(bce_res), np.array(dct_res)
if __name__ == '__main__':
    look_at_path = sys.argv[1]
    screw_path = sys.argv[2]
    low = 75

    look_at_df = pd.read_csv(look_at_path, sep=';', index_col=0)
    screw_df = pd.read_csv(screw_path, sep=';', index_col=0)
    bce_look_at, dct_look_at = read_results(look_at_df)
    bce_screw, dct_screw = read_results(screw_df)
    bce_look_at, dct_look_at = bce_look_at - low, dct_look_at - low
    bce_screw, dct_screw = bce_screw - low, dct_screw - low

    bce_look_at_median = np.median(bce_look_at)
    bce_look_at_max = np.max(bce_look_at)
    bce_look_at_min = np.min(bce_look_at)
    print('BCE look_at {} + {} - {}'.format(bce_look_at_median, bce_look_at_max, bce_look_at_min))
    bce_screw_median = np.median(bce_screw)
    bce_screw_max = np.max(bce_screw)
    bce_screw_min = np.min(bce_screw)
    print('BCE screw {} + {} - {}'.format(bce_screw_median, bce_screw_max, bce_screw_min))

    dct_look_at_median = np.median(dct_look_at)
    dct_look_at_max = np.max(dct_look_at)
    dct_look_at_min = np.min(dct_look_at)
    print('DCT look_at {} + {} - {}'.format(dct_look_at_median, dct_look_at_max, dct_look_at_min))
    dct_screw_median = np.median(dct_screw)
    dct_screw_max = np.max(dct_screw)
    dct_screw_min = np.min(dct_screw)
    print('DCT screw {} + {} - {}'.format(dct_screw_median, dct_screw_max, dct_screw_min))

    labels = ['Look at test', 'Screw motion test']
    bce = [bce_look_at_median, bce_screw_median]
    dct = [dct_look_at_median, dct_screw_median]

    x = np.arange(len(labels))
    width = 0.35
    
    fig = plt.figure()
    ax = plt.gca()
    bce_yerr = [[abs(bce_look_at_min - bce_look_at_median), abs(bce_screw_min - bce_screw_median)], [abs(bce_look_at_max - bce_look_at_median), abs(bce_screw_max - bce_screw_median)]]
    dct_yerr = [[abs(dct_look_at_min - dct_look_at_median), abs(dct_screw_min - dct_screw_median)], [abs(dct_look_at_max - dct_look_at_median), abs(dct_screw_max - dct_screw_median)]]
    
    rects1 = ax.bar(x - width/2, bce, width, yerr=bce_yerr, label='BCE loss', bottom=low)
    rects2 = ax.bar(x + width/2, dct, width, yerr=dct_yerr, label='DCT loss', bottom=low)

    ax.set_ylabel('Convergence rate, %')
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc='lower right')
    # ax.set_yticks(np.arange(low, 100, 5))
    ax.set_ylim([75, 100])
    plt.grid()
    fig.tight_layout()

    plt.show()
    