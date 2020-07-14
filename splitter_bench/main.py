import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from hyb_clu import hyb_clu
import split
import time
import os
# First, data directories, ground truth (GT) clusters, artificially added clusters,
# and clusters associated with the former two types.
# Data directories
raw_data_dir = r'C:\Users\black\Desktop\eel6_2020-03-01'
hyb_data_dir = os.getcwd()
print(hyb_data_dir)
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
exp_dict = {'gt_clu':[],'hyb_clu':[],'clust_art_%':[],'split_real_%':[],'split_hyb_%':[]}
run_ct = 0
nclusts = 20
if 'hyb_clu_list' in dir():
    for i,clu in enumerate(gt_clus):
            for x in hyb_clu_list[i].exp_clusts:
                art_pct,real_res,hyb_res,merged = split.run_exp_split(x,s,m,c,nclusts)
                if real_res:
                    exp_dict['gt_clu'].append(clu)
                    exp_dict['hyb_clu'].append(x['id'])
                    exp_dict['split_real_%'].append(real_res*100)
                    exp_dict['split_hyb_%'].append(hyb_res*100)
                    exp_dict['clust_art_%'].append(art_pct*100)
                    print('Successful split!')
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
            art_pct,real_res,hyb_res,merged = split.run_exp_split(x,s,m,c,nclusts)
            if real_res:
                exp_dict['gt_clu'].append(clu)
                exp_dict['hyb_clu'].append(x['id'])
                exp_dict['split_real_%'].append(real_res)
                exp_dict['split_hyb_%'].append(hyb_res)
                exp_dict['clust_art_%'].append(art_pct)
                print('Successful split!')

timestr = time.strftime("%Y%m%d-%H%M%S")+('_n%d.csv'%nclusts)
timestr_pdf = time.strftime("%Y%m%d-%H%M%S")+('_n%d.pdf'%nclusts)
exp_data = pd.DataFrame(data=exp_dict)
axes=exp_data[['clust_art_%','split_hyb_%','split_real_%']].plot.bar(rot=0)
fig = axes.figure
axes.set_xlabel('trial #')
axes.set_ylabel('cluster purity (%)')
axes.set_title('Automated splitter performance, n_clusts = %d'%nclusts)
fig.show()
fig.savefig(timestr_pdf)
exp_data.to_csv(timestr)