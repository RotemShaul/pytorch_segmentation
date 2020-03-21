import numpy as np
import os
import sys

# WARNING: this will work on little-endian architectures (eg Intel x86) only!
if '__main__' == __name__:
    file_name = '/Users/rotem/Documents/wis/Research/data/elbit data/Elbit/output/work/inference/run.epoch-0-flow-field/000010.flo'
    f = open(file_name, 'rb')
    magic = np.fromfile(f, np.float32, count=1)
    if 202021.25 != magic:
        print('Magic number incorrect. Invalid .flo file')
    else:
        w = np.fromfile(f, np.int32, count=1)[0]
        h = np.fromfile(f, np.int32, count=1)[0]
        print('Reading %d x %d flo file' % (w, h))
        data = np.fromfile(f, np.float32, count=2*w*h)
        # Reshape data into 3D array (columns, rows, bands)
        data2D = np.resize(data, (h, w, 2))
        print(data2D)