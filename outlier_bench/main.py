from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from hyb_clu import hyb_clu
import numpy as np
import pandas as pd
import outlier
import time
import os
import importlib

# For development pipeline convenience
importlib.reload(outlier)

# First, data directories, ground truth (GT) clusters, artificially added clusters,
# and clusters associated with the former two types.
# Data directories
raw_data_dir = r'C:\Users\black\Desktop\eel6_2020-03-01'
hyb_data_dir = os.getcwd()
# Artificial Clusters (from output of hybridfactory)
art_units = pd.read_csv(hyb_data_dir+r'\artificial_units-test.csv')
true_units = art_units['true_unit']
gt_clus = np.unique(true_units)
center_channels = np.array(art_units['center_channel'])
# Load spikes associated with GT clusters, incl. time and raw cluster ID
raw_spike_clus = np.load(raw_data_dir+r'\spike_clusters.npy')
raw_spike_times = np.load(raw_data_dir+r'\spike_times.npy')
# Load spikes associated with hybrid clusters
hyb_spike_clus = np.load(hyb_data_dir+r'\spike_clusters.npy')
hyb_spike_times = np.load(hyb_data_dir+r'\spike_times.npy')

diffs = []
idxs = []
exp_dict = {}
run_ct = 0

if 'hyb_clu_list' in dir():
    #if len(hyb_clu_list)==len(gt_clus):
    if True:
        for i,clu in enumerate(gt_clus):
            for x in hyb_clu_list[i].exp_clusts:
                outlier.run_exp_outlier(x,s,m,c)
    else:
        print('Old data does not match current # of clusters!')
else:
    hyb_clu_list = []
    for i,clu in enumerate(gt_clus):
        hfact_idx = np.where(true_units == clu)
        true_hyb_spike_times = np.asarray(art_units['timestep'])[hfact_idx]
        chans = np.unique(center_channels[hfact_idx])
        assert len(chans)==1
        hyb_spike_times = np.load(hyb_data_dir+r'\spike_times.npy')
        spike_idxs = np.where(raw_spike_clus == clu)[0]
        spike_times = raw_spike_times[spike_idxs][:,0]
        print('Artificial cluster (based on %d) is on channel(s): %s'%(clu,np.unique(center_channels[hfact_idx])))
        hyb_clu_list.append(hyb_clu(clu,true_hyb_spike_times,s,m,c,chans))
        hyb_clu_list[i].link_hybrid(hyb_spike_times,hyb_spike_clus)
        for x in hyb_clu_list[i].exp_clusts:
            outlier.run_exp_outlier(x,s,m,c)

#timestr_pdf = time.strftime("%Y%m%d-%H%M%S")+('_n%d.pdf'%nclusts)
#pp = PdfPages(timestr_pdf)

#for i,clu in enumerate(exp_dict['gt_clu']):
#    fig,axes=plt.subplots(nrows=1,ncols=2,figsize=(8,4),sharey=True,sharex=True)
#    axes[0].bar(np.arange(1,nclusts+1),(np.flip(np.sort(exp_dict['clu_precision'][i]))*100))
#    axes[0].set_xlabel('Sub-cluster (idx)')
#    axes[0].set_ylabel('Precision (%)')
#    axes[0].set_title('GT-%d / Artificial-%d (art%% %2.3f)'%(clu,exp_dict['hyb_clu'][i],(exp_dict['art%'][i]*100)))
#    f1_merged_scores = np.array(exp_dict['f1_scores'][i])
#    axes[1].plot(np.arange(1,nclusts+1),(f1_merged_scores*100))
#    axes[1].set_xlabel('Sub-clusters combined (count)')
#    axes[1].set_ylabel('F1 Score (%) of combined clust.')
#    axes[1].set_title('GT-%d / Artificial-%d (art%% %2.3f)'%(clu,exp_dict['hyb_clu'][i],(exp_dict['art%'][i]*100)))
#    pp.savefig(plt.gcf())
#pp.close()