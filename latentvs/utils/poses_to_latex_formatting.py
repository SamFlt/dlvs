import numpy as np
import sys
import os
from pathlib import Path
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from geometry import *

def latex_format(pose):
    s = '('
    for i in range(3):
        s += '{}cm, '.format(np.round(pose[i] * 100.0, 2))
    for i in range(3, 6):
        s += '{}\\textdegree, '.format(np.round(np.degrees(pose[i]), 2))
    s = s[:-2] # remove trailing comma and space
    s += ')'
    return s
if __name__ == '__main__':
    path = sys.argv[1]
    array = np.loadtxt(path, delimiter='\t', skiprows=1, usecols=(1, 4, 7, 10, 13, 16))
    first_pose = array[0]
    end_pose = array[-1]
    print('$\\Delta \\mathbf{r}_0 = $' + latex_format(first_pose))
    print('$\\Delta \\mathbf{r}_{final} = $' +latex_format(end_pose))
    
    
