
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import LineCollection
from phylib.io.traces import get_ephys_reader
import matplotlib.pyplot as plt
from matplotlib import cm
from hyb_clu import hyb_clu
import numpy as np
import pandas as pd
import importlib
import outlier
import time
from glob import glob
import os
import sys
import obench_plot

# Load raw traces for calculation of SNR
trace_args = ['sample_rate', 'n_channels_dat', 'dtype', 'offset']
trace_vals = [[25000],[64],[np.int16],[0]]
kwargs = {trace_args[x]: trace_vals[x][0] for x in range(len(trace_args))}
data_dir = os.getcwd()+r'\ConcatenatedData_Probe1.GT.bin'
traces = get_ephys_reader(data_dir, **kwargs)
pbounds = slice(traces.part_bounds[0],traces.part_bounds[1])
traces = traces._get_part(pbounds,0)

# Noise estimate for spikesorting data
noise_est = lambda data : np.median((np.abs(data))/0.6745)

# Load human performances
hperfs = glob('hperf_*')
hperf_ids = [int(x.split('_')[1].split('.')[0]) for x in hperfs]

# Replace stdout with text file.
timestr = time.strftime("outlier_%Y%m%d-%H%M%S")
orig_stdout = sys.stdout
sys.stdout = open(timestr+'.txt','w')

# For development pipeline convenience
importlib.reload(outlier)
importlib.reload(obench_plot)

# First, data directories, ground truth (GT) clusters, artificially added clusters,
# and clusters associated with the former two types.
# Data directories
raw_data_dir = r'C:\Users\black\Desktop\eel6_2020-03-01_prb1'
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
try:
    del exp_dict
    exp_dict = {}
except:
    exp_dict = {}

run_ct = 0
exp_dict.update({'snr':[],'nspikes':[],'obi':[],'fdr_ba':[],'fdr_onstep':[],'f1_ba':[],'f1_onstep':[],'cum_nouts':[],'hyb_clu':[],'art%':[],'f1_max_idx':[]})

lfc = [275]
useclus = True

if useclus:
    clus = np.nonzero(np.in1d(gt_clus,lfc))[0]
    gt_clus = np.array(gt_clus)[clus]
else:
    gt_clus = np.array(gt_clus)[:]

hybclu_structs = []
if 'hyb_clu_list' in dir():
    if len(hyb_clu_list)>=len(gt_clus):
        for i,clu in enumerate(gt_clus):
            print('GT Cluster %d'%clu)
            for x in hyb_clu_list[i].exp_clusts:
                fdr_onstep,cum_n_outs_onstep,fdr_ba,f1_onstep,f1_max_idx,f1_ba,obi = outlier.run_exp_outlier(x,s,m,c)
                if fdr_onstep:
                    exp_dict['fdr_onstep'].append(fdr_onstep)
                    exp_dict['f1_onstep'].append(f1_onstep)
                    exp_dict['cum_nouts'].append(cum_n_outs_onstep)
                    exp_dict['art%'].append(x['art_pct'])
                    exp_dict['hyb_clu'].append(x['id'])
                    exp_dict['fdr_ba'].append(fdr_ba)
                    exp_dict['f1_ba'].append(f1_ba)
                    exp_dict['f1_max_idx'].append(f1_max_idx)
                    exp_dict['obi'].append(obi)
                    hybclu_structs.append(x)
                    # Primary channel signal for this cluster
                    trace = traces[:,x['best_c']]
                    n_est = (noise_est(trace))
                    
                    # Get spike waveforms
                    all_spikes = []
                    all_spikes.extend(x['hyb'][:])
                    all_spikes.extend(x['real'][:])
                    exp_dict['nspikes'].append(len(all_spikes))
                    waveform = m.get_cluster_spike_waveforms(x['id'])[:,:,0]
                    mean_waveform = np.mean(waveform,axis=0)
                    s_est = np.max(np.abs(mean_waveform))
                
                    # Maybe plot mean waveform labeled with noise estimate line?
                    # Plus all traces with low opacity?

                    SNR = (s_est/n_est)
                    exp_dict['snr'].append(SNR)
                    hybclu_structs.append(x)
    else:
        print('Old data does not match current # of clusters!')
else:
    hyb_clu_list = []
    for i,clu in enumerate(gt_clus):
        print('GT Cluster %d'%clu)
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
            fdr_onstep,cum_n_outs_onstep,fdr_ba,f1_onstep,f1_max_idx,f1_ba,obi = outlier.run_exp_outlier(x,s,m,c)
            if fdr_onstep:
                exp_dict['fdr_onstep'].append(fdr_onstep)
                exp_dict['f1_onstep'].append(f1_onstep)
                exp_dict['cum_nouts'].append(cum_n_outs_onstep)
                exp_dict['art%'].append(x['art_pct'])
                exp_dict['hyb_clu'].append(x['id'])
                exp_dict['fdr_ba'].append(fdr_ba)
                exp_dict['f1_ba'].append(f1_ba)
                exp_dict['f1_max_idx'].append(f1_max_idx)
                exp_dict['obi'].append(obi)
                hybclu_structs.append(x)


                # Primary channel signal for this cluster
                trace = traces[:,x['best_c']]
                n_est = (noise_est(trace))

                # Get spike waveforms
                all_spikes = []
                all_spikes.extend(x['hyb'][:])
                all_spikes.extend(x['real'][:])
                exp_dict['nspikes'].append(len(all_spikes))
                waveform = m.get_cluster_spike_waveforms(x['id'])[:,:,0]
                mean_waveform = np.mean(waveform,axis=0)
                s_est = np.max(np.abs(mean_waveform))
                
                # Maybe plot mean waveform labeled with noise estimate line?
                # Plus all traces with low opacity?

                SNR = (s_est/n_est)
                exp_dict['snr'].append(SNR)

timestr_pdf = timestr+".pdf"
timestr_dict_npy = timestr+"_res.npy"
plt.close('all')
np.save(timestr_dict_npy,exp_dict,allow_pickle=True)

pp = PdfPages(timestr_pdf)

obench_plot.generate_outlier_plots(exp_dict,pp)
obench_plot.human_perf_plot(exp_dict,hybclu_structs,pp,s,m,c,hperf_ids)
obench_plot.isi_fdr_comparison(exp_dict,hybclu_structs,pp,s,m,c)


pp.close()
sys.stdout.close()
sys.stdout = orig_stdout

import winsound
winsound.Beep(450,900)