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

exp_dict.update({'prec_onstep':[]})
exp_dict.update({'cum_nouts':[]})
exp_dict.update({'hyb_clu':[]})
exp_dict.update({'art%':[]})

if 'hyb_clu_list' in dir():
    #if len(hyb_clu_list)==len(gt_clus):
    if True:
        for i,clu in enumerate(gt_clus[0:1]):
            for x in hyb_clu_list[i].exp_clusts:
                final_prec,prec_onstep,cum_n_outs_onstep = outlier.run_exp_outlier(x,s,m,c)
                if final_prec:
                    print('Final precision for clu. %d : %.3f'%(x['id'],final_prec))
                    exp_dict['prec_onstep'].append(prec_onstep)
                    exp_dict['cum_nouts'].append(prec_onstep)
                    exp_dict['art%'].append(x['art_pct'])
                    exp_dict['hyb_clu'].append(x['id'])
    else:
        print('Old data does not match current # of clusters!')
else:
    hyb_clu_list = []
    for i,clu in enumerate(gt_clus[0:1]):
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
                final_prec,prec_onstep,cum_n_outs_onstep = outlier.run_exp_outlier(x,s,m,c)
                if final_prec:
                    print('Final precision for clu. %d : %.3f'%(x['id'],final_prec))
                    exp_dict['prec_onstep'].append(prec_onstep)
                    exp_dict['cum_nouts'].append(prec_onstep)
                    exp_dict['art%'].append(x['art_pct'])
                    exp_dict['hyb_clu'].append(x['id'])

timestr_pdf = time.strftime("outlier_%Y%m%d-%H%M%S.pdf")
pp = PdfPages(timestr_pdf)

for i,clu in enumerate(exp_dict['hyb_clu']):
    fig,axes=plt.subplots(nrows=1,ncols=1,figsize=(8,4),sharey=True,sharex=True)
    axes.plot(exp_dict['cum_nouts'][i],exp_dict['prec_onstep'][i])
    print(exp_dict['cum_nouts'][i],exp_dict['prec_onstep'][i])
    axes.set_xlabel('N spikes removed (count)')
    axes.set_ylabel('Precision (%)')
    axes.set_title('Unit %d (art%% %2.3f)'%(exp_dict['hyb_clu'][i],(exp_dict['art%'][i]*100)))
    pp.savefig(plt.gcf())
pp.close()