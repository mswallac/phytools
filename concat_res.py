from matplotlib.backends.backend_pdf import PdfPages
# from phylib.io.traces import get_ephys_reader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import os
import importlib
from matplotlib import cm
import sys
from glob import glob
from outlier_bench import obench_plot

# For development pipeline convenience
importlib.reload(obench_plot)

maindir = 'C:\\Users\\black\\Desktop\\*\\'


dirs = glob(maindir)
filt = np.nonzero(['eel6' in x for x in dirs])[0]
datasets = [dirs[x] for x in filt]
hyb_data_dirs = []
for directory in datasets:
    this_dir = directory+'*\\'
    this_glob = glob(this_dir)
    good_dirs = ['_bu']
    filt = np.nonzero([np.any([x in d for x in good_dirs]) for d in this_glob])[0]
    hyb_datasets = [this_glob[x] for x in filt]
    hyb_data_dirs.extend(hyb_datasets)

split_res = []
out_res = []
cont = False
for directory in hyb_data_dirs:  
    if 'bu' in directory:
        files = os.listdir(directory)
        filts = np.nonzero(['_res.npy' in x for x in files])[0]
        if filts:
            res_files = [files[x] for x in filts]
            print('Found %s in dir: %s'%(res_files[0],directory))
            split_res.append((np.load(directory+res_files[0],allow_pickle=True,))[()])
            print(split_res)

split_dict = {}
for result in split_res:
    for key in list(result.keys()):
        try:
            split_dict[key].extend(result[key])
        except:
            split_dict.update({key:[]})
            split_dict[key].extend(result[key])
            
pp = PdfPages('summary.pdf')
obench_plot.generate_outlier_plots(split_dict,pp)
pp.close()
