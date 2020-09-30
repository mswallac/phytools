
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
n_minima = 6
exp_dict.update({'snr':[],'nspikes':[],'fdr_ba':[],'fdr_onstep':[],'f1_ba':[],'f1_onstep':[],'prec_rem_onstep':[],'cum_nouts':[],'hyb_clu':[],'art%':[],'fdr_idxs':[]})

lfc = [274]
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
                prec_rem_onstep,fdr_onstep,cum_n_outs_onstep,fdr_ba,f1_onstep,fdr_min_idxs,f1_ba = outlier.run_exp_outlier(x,s,m,c,n_minima)
                if fdr_onstep:
                    exp_dict['fdr_onstep'].append(fdr_onstep)
                    exp_dict['f1_onstep'].append(f1_onstep)
                    exp_dict['prec_rem_onstep'].append(prec_rem_onstep)
                    exp_dict['cum_nouts'].append(cum_n_outs_onstep)
                    exp_dict['art%'].append(x['art_pct'])
                    exp_dict['hyb_clu'].append(x['id'])
                    exp_dict['fdr_ba'].append(fdr_ba)
                    exp_dict['f1_ba'].append(f1_ba)
                    exp_dict['fdr_idxs'].append(fdr_min_idxs)
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
            prec_rem_onstep,fdr_onstep,cum_n_outs_onstep,fdr_ba,f1_onstep,fdr_min_idxs,f1_ba = outlier.run_exp_outlier(x,s,m,c,n_minima)
            if fdr_onstep:
                exp_dict['fdr_onstep'].append(fdr_onstep)
                exp_dict['f1_onstep'].append(f1_onstep)
                exp_dict['prec_rem_onstep'].append(prec_rem_onstep)
                exp_dict['cum_nouts'].append(cum_n_outs_onstep)
                exp_dict['art%'].append(x['art_pct'])
                exp_dict['hyb_clu'].append(x['id'])
                exp_dict['fdr_ba'].append(fdr_ba)
                exp_dict['f1_ba'].append(f1_ba)
                exp_dict['fdr_idxs'].append(fdr_min_idxs)
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

f1_ba_arr = np.array(exp_dict['f1_ba'])
fdr_ba_arr = np.array(exp_dict['fdr_ba'])
pdf1_arr = np.zeros(f1_ba_arr.shape)

for i in range(f1_ba_arr.shape[0]):
    temp_f1_arr = f1_ba_arr[i,:]
    pdf1_arr[i,:] = np.array([(x-temp_f1_arr[0])/(1-temp_f1_arr[0]) for x in temp_f1_arr])

# Get cmap
cmap = plt.get_cmap('plasma').colors
cmap = [cmap[x] for x in np.linspace(0,252,n_minima).astype(int)]
cmap.reverse()
labels = ['']
ext_lbls = [str(x+1) for x in range(n_minima)]
labels.extend(ext_lbls)

# Plot before F1 vs after F1
fig1=plt.figure()
plt.plot(np.linspace(0,np.max(f1_ba_arr.flatten()),100),np.linspace(0,np.max(f1_ba_arr.flatten()),100),'k--',lw=.15)
for x in range(n_minima): 
    plt.scatter(f1_ba_arr[:,0],f1_ba_arr[:,x],c=cmap[x],s=7)
for i in range(f1_ba_arr.shape[0]):
    plt.text(f1_ba_arr[i,0]*.998,f1_ba_arr[i,0]*1.002,'%d'%exp_dict['hyb_clu'][i],size='xx-small')
plt.ylabel('F1 After')
plt.xlabel('F1 Before')
plt.legend(labels)
fig1.tight_layout()
plt.draw()
pp.savefig(plt.gcf())

# Plot before FDR vs after FDR
fig1=plt.figure()
plt.plot(np.linspace(0,np.max(fdr_ba_arr.flatten()),100),np.linspace(0,np.max(fdr_ba_arr.flatten()),100),'k--',lw=.15)
for x in range(n_minima): 
    plt.scatter(fdr_ba_arr[:,0],fdr_ba_arr[:,x],c=cmap[x],s=7)
for i in range(fdr_ba_arr.shape[0]):
    plt.text(fdr_ba_arr[i,0]*.998,fdr_ba_arr[i,0]*1.002,'%d'%exp_dict['hyb_clu'][i],size='xx-small')
plt.ylabel('FDR After')
plt.xlabel('FDR Before')
plt.legend(labels)
fig1.tight_layout()
plt.draw()
pp.savefig(plt.gcf())

# Plot SNR vs dF1
fig1=plt.figure()
for x in range(n_minima): 
    plt.scatter(exp_dict['snr'],pdf1_arr[:,x],c=cmap[x],s=7)
for i in range(pdf1_arr.shape[0]):
    plt.text(exp_dict['snr'][i]*.998,pdf1_arr[i,0]*1.002,'%d'%exp_dict['hyb_clu'][i],size='xx-small')

plt.axhline(y=0,c='k',ls='--')
plt.ylabel('dF1 (fraction of possible +dF1)')
plt.xlabel('SNR')
plt.legend(labels)
fig1.tight_layout()
plt.draw()
pp.savefig(plt.gcf())

# Plot nspikes vs dF1
fig1=plt.figure()
for x in range(n_minima): 
    plt.scatter(exp_dict['nspikes'],pdf1_arr[:,x],c=cmap[x],s=7)
for i in range(pdf1_arr.shape[0]):
    plt.text(exp_dict['nspikes'][i]*.998,pdf1_arr[i,0]*1.002,'%d'%exp_dict['hyb_clu'][i],size='xx-small')

plt.axhline(y=0,c='k',ls='--')
plt.ylabel('dF1 (fraction of possible +dF1)')
plt.xlabel('Number of spikes (count)')
plt.legend(labels)
fig1.tight_layout()
plt.draw()
pp.savefig(plt.gcf())

for i,clu in enumerate(exp_dict['hyb_clu']):
    if clu in hperf_ids:
        fig,axes=plt.subplots(nrows=2,ncols=2,figsize=(10,6),sharey='row',sharex=True)
        axes[0][0].plot(exp_dict['cum_nouts'][i],exp_dict['fdr_onstep'][i])
        for x,d in enumerate(exp_dict['fdr_idxs'][i]):
            axes[0][0].plot(exp_dict['cum_nouts'][i][d],exp_dict['fdr_onstep'][i][d],c=cmap[x],marker='o')
        axes[0][0].set_xlabel('N spikes removed (count)')
        axes[0][0].set_ylabel('False discovery rate')
        axes[0][0].set_title('Unit %d (auto.)'%(clu))
        
        axes[1][0].plot(exp_dict['cum_nouts'][i],exp_dict['f1_onstep'][i])
        axes[1][0].set_xlabel('N spikes removed (count)')
        axes[1][0].set_ylabel('F1 Score')
        axes[1][0].set_title('Unit %d (auto.)'%(clu))
        hclust = hybclu_structs[i]
        hperf = np.load('hperf_%d.npy'%clu,allow_pickle=True).item()
        prec_rem_onstep,fdr_onstep,cum_n_outs_onstep,fdr_ba,f1_onstep,fdr_min_idxs,f1_ba=outlier.human_bench(hclust,s,m,c,hperf)
        axes[0][1].plot(cum_n_outs_onstep,fdr_onstep)
        for x,d in enumerate(fdr_min_idxs):
            axes[0][1].plot(cum_n_outs_onstep[d],fdr_onstep[d],c=cmap[x],marker='o')
        axes[0][1].set_xlabel('N spikes removed (count)')
        axes[0][1].set_ylabel('False discovery rate')
        axes[0][1].set_title('Unit %d (human)'%(clu))

        axes[1][1].plot(cum_n_outs_onstep,f1_onstep)
        axes[1][1].set_xlabel('N spikes removed (count)')
        axes[1][1].set_ylabel('F1 Score')
        axes[1][1].set_title('Unit %d (human)'%(clu))
    else:
        fig,axes=plt.subplots(nrows=1,ncols=2,figsize=(10,6),sharey='col',sharex=True)
        axes[0].plot(exp_dict['cum_nouts'][i],exp_dict['fdr_onstep'][i])
        for x,d in enumerate(exp_dict['fdr_idxs'][i]):
            axes[0].plot(exp_dict['cum_nouts'][i][d],exp_dict['fdr_onstep'][i][d],c=cmap[x],marker='o')
        axes[0].set_xlabel('N spikes removed (count)')
        axes[0].set_ylabel('False discovery rate')
        axes[0].set_title('Unit (auto.) %d'%(clu))
        axes[1].plot(exp_dict['cum_nouts'][i],exp_dict['f1_onstep'][i])
        axes[1].set_xlabel('N spikes removed (count)')
        axes[1].set_ylabel('F1 Score')
        axes[1].set_title('Unit %d (auto.)'%(clu))


    fig.tight_layout()
    plt.draw()
    pp.savefig(plt.gcf())

pp.close()
sys.stdout.close()
sys.stdout = orig_stdout
import winsound
winsound.Beep(450,900)