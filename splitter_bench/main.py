from matplotlib.backends.backend_pdf import PdfPages
from phylib.io.traces import get_ephys_reader
import matplotlib.pyplot as plt
from hyb_clu import hyb_clu
import numpy as np
import pandas as pd
import split
import time
import os
import importlib
import sys

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

# Replace stdout with text file.
timestr = time.strftime("splitter_%Y%m%d-%H%M%S")
orig_stdout = sys.stdout
sys.stdout = open(timestr+'.txt','w')

# For development pipeline convenience
importlib.reload(split)

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

# Initialize variables to record results and run experiments.
diffs = []
idxs = []
exp_dict = {'snr':[],'nspikes':[],'gt_clu':[],'hyb_clu':[],'f1_scores':[],'clu_precision':[],'merged':[],'art%':[],'f1_ba':[],'sts':[]}
run_ct = 0
nclusts = 15
split_dict = {}

lfc = [274]
useclus = False

if useclus:
    clus = np.nonzero(np.in1d(gt_clus,lfc))[0]
    gt_clus = np.array(gt_clus)[clus]
else:
    gt_clus = np.array(gt_clus)[:]

if 'hyb_clu_list' in dir():
    if len(hyb_clu_list)==len(gt_clus):
        for i,clu in enumerate(gt_clus):
            for x in hyb_clu_list[i].exp_clusts:
                art_pct,clust_precs,f1s_merged,merged_clusts,f1_ba,sts = split.run_exp_split(x,s,m,c,nclusts)
                if not (clust_precs is None):
                    exp_dict['gt_clu'].append(clu)
                    exp_dict['hyb_clu'].append(x['id'])
                    exp_dict['clu_precision'].append(clust_precs)
                    exp_dict['f1_scores'].append(f1s_merged)
                    exp_dict['merged'].append(merged_clusts)
                    exp_dict['art%'].append(art_pct)
                    exp_dict['f1_ba'].append(f1_ba)
                    exp_dict['sts'].append(sts)
                    
                    # Primary channel signal for this cluster
                    trace = traces[:,x['best_c']]
                    print(trace.shape)
                    n_est = (noise_est(trace))
                    
                    # Get spike waveforms
                    all_spikes = []
                    all_spikes.extend(x['hyb'][:])
                    all_spikes.extend(x['real'][:])
                    exp_dict['nspikes'].append(len(all_spikes))
                    waveform = m.get_cluster_spike_waveforms(x['id'])[:,:,0]
                    mean_waveform = np.mean(waveform,axis=0)
                    s_est = np.max(np.abs(mean_waveform))
                    
                    SNR = (s_est/n_est)
                    
                    print('SNR: %f'%SNR)
                    exp_dict['snr'].append(SNR)

                    tmp_splt = {x['id']: (np.flip(np.sort(s.shown_cluster_ids))[:2])}
                    split_dict.update(tmp_splt)
                    print('%d - > %s' %(x['id'],np.flip(np.sort(s.shown_cluster_ids))[:2]))
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
            art_pct,clust_precs,f1s_merged,merged_clusts,f1_ba,sts = split.run_exp_split(x,s,m,c,nclusts)
            if not (clust_precs is None):
                exp_dict['gt_clu'].append(clu)
                exp_dict['hyb_clu'].append(x['id'])
                exp_dict['clu_precision'].append(clust_precs)
                exp_dict['f1_scores'].append(f1s_merged)
                exp_dict['merged'].append(merged_clusts)
                exp_dict['art%'].append(art_pct)
                exp_dict['f1_ba'].append(f1_ba)
                exp_dict['sts'].append(sts)
                
                    
                # Primary channel signal for this cluster
                trace = traces[:,x['best_c']]
                print(np.mean(trace),np.max(trace),np.min(trace))
                n_est = (noise_est(trace))
                
                # Get spike waveforms
                all_spikes = []
                all_spikes.extend(x['hyb'][:])
                all_spikes.extend(x['real'][:])
                exp_dict['nspikes'].append(len(all_spikes))
                waveform = m.get_cluster_spike_waveforms(x['id'])[:,:,0]
                mean_waveform = np.mean(waveform,axis=0)
                s_est = np.max(np.abs(mean_waveform))
                
                SNR = (s_est/n_est)
                
                print('SNR: %f'%SNR)
                exp_dict['snr'].append(SNR)

                tmp_splt = {x['id']: (np.flip(np.sort(s.shown_cluster_ids))[:2])}
                split_dict.update(tmp_splt)
                print('%d - > %s' %(x['id'],np.flip(np.sort(s.shown_cluster_ids))[:2]))

# Prepare strings for output files
timestr = time.strftime("splitter_%Y%m%d-%H%M%S")
timestr_pdf = timestr+('_n%d.pdf'%nclusts)
timestr_dict_npy = timestr+('_res.npy')

# Save output files and init. PDF to save graphs to
np.save(timestr_dict_npy,exp_dict,allow_pickle=True)
np.save(timestr+'_splits.npy',split_dict,allow_pickle=True)
pp = PdfPages(timestr_pdf)

# Prepare arrays for plotting
f1_ba_arr = np.array(exp_dict['f1_ba'])
f1_diff_arr = (f1_ba_arr[:,1]-f1_ba_arr[:,0])/(1-f1_ba_arr[:,0])

# Plot presets
xlocs = [1,2]
xlabels = ['Before', 'After']

# PLOTS TO SHOW CHANGES IN FDR / F1

# F1
fig1=plt.figure()
for i,clu in enumerate(exp_dict['hyb_clu']):
    plt.plot(xlocs,exp_dict['f1_ba'][i]-exp_dict['f1_ba'][i][0],'-ok',alpha = .8)
plt.title('F1 Before/After automated curation')
plt.xticks(xlocs,xlabels,fontsize='x-large')
plt.xlim(0,3)
plt.ylabel('F1 Score')
fig1.tight_layout()
plt.draw()
pp.savefig(plt.gcf())

# Plot F1 score before vs after
fig1=plt.figure()
plt.scatter(f1_ba_arr[:,0],f1_ba_arr[:,1],s=.2)
for i in range(f1_ba_arr.shape[0]):
    plt.text(f1_ba_arr[i,0]*1.002,f1_ba_arr[i,1]*1.002,'%d'%exp_dict['hyb_clu'][i],size='xx-small')
plt.plot(np.linspace(0,np.max(f1_ba_arr.flatten()),100),np.linspace(0,np.max(f1_ba_arr.flatten()),100),'k--',lw=.1)
plt.ylabel('F1 After')
plt.xlabel('F1 Before')
fig1.tight_layout()
plt.draw()
pp.savefig(plt.gcf())
plt.close()

# Plot SNR vs dF1
fig1=plt.figure()
plt.scatter(exp_dict['snr'],f1_diff_arr,s=.2)
for i in range(f1_ba_arr.shape[0]):
    plt.text(exp_dict['snr'][i]*1.002,f1_diff_arr[i]*1.002,'%d'%exp_dict['hyb_clu'][i],size='xx-small')
plt.axhline(y=0,c='k',ls='--')
plt.ylabel('dF1 (fraction of possible +dF1)')
plt.xlabel('SNR')
fig1.tight_layout()
plt.draw()
pp.savefig(plt.gcf())

# Plot nspikes vs dF1
fig1=plt.figure()
plt.scatter(exp_dict['nspikes'],f1_diff_arr,s=.2)
for i in range(f1_ba_arr.shape[0]):
    plt.text(exp_dict['nspikes'][i]*1.002,f1_diff_arr[i]*1.002,'%d'%exp_dict['hyb_clu'][i],size='xx-small')
plt.axhline(y=0,c='k',ls='--')
plt.ylabel('dF1 (fraction of possible +dF1)')
plt.xlabel('Number of spikes (count)')
fig1.tight_layout()
plt.draw()
pp.savefig(plt.gcf())

# Plot per-cluster representation of splitting process.
for i,clu in enumerate(exp_dict['hyb_clu']):
    fig,axes=plt.subplots(nrows=1,ncols=2,figsize=(8,4),sharey=False,sharex=True)
    axes[0].bar(np.arange(1,nclusts+1),(np.flip(np.sort(exp_dict['clu_precision'][i]))*100))
    axes[0].set_xlabel('Sub-cluster (idx)')
    axes[0].set_ylabel('Precision (%)')
    axes[0].set_title('Unit %d'%(clu))
    f1_merged_scores = np.array(exp_dict['f1_scores'][i])
    best_idx = np.argmax(f1_merged_scores)
    axes[1].plot(np.arange(1,nclusts+1),(f1_merged_scores*100))
    axes[1].plot(np.arange(1,nclusts+1)[best_idx],(f1_merged_scores*100)[best_idx],'ro')
    axes[1].set_xlabel('Sub-clusters combined (count)')
    axes[1].set_ylabel('F1 Score (%) of combined clust.')
    axes[1].set_title('Unit %d'%(clu))
    fig.tight_layout()
    pp.savefig(plt.gcf())

# Close opened files, reassign stdout, beep to signal done.
pp.close()
sys.stdout.close()
sys.stdout = orig_stdout
import winsound
winsound.Beep(450,900)