import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from hyb_clu import hyb_clu
import split

# First, data directories, ground truth (GT) clusters, artificially added clusters,
# and clusters associated with the former two types.
# Data directories
raw_data_dir = r'C:\Users\black\Desktop\eel6_2020-03-01'
hyb_data_dir = raw_data_dir+r'\test_hyb'

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

# Create data objects which describe one GT cluster.
hyb_clu_list = []

# For each GT cluster we will store: 
    # raw clustering ID, raw spike idx (all), time
    # Add hybrid spike idx (all? -- as many as possible) after using spike times
diffs = []
idxs = []
prev_done = []
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
    	if x['id'] not in prev_done:
	    	prev_done.append(x['id'])
	    	split.run_exp_split(x,s,m,c)