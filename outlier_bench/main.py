from matplotlib.backends.backend_pdf import PdfPages
from phylib.io.traces import get_ephys_reader
import matplotlib.pyplot as plt
from hyb_clu import hyb_clu
import numpy as np
import pandas as pd
import outlier
import time
import os
import importlib
from matplotlib import cm
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
exp_dict = {}
run_ct = 0

exp_dict.update({'snr':[],'nspikes':[],'f1_ba':[],'f1_onstep':[],'prec_rem_onstep':[],'cum_nouts':[],'hyb_clu':[],'art%':[]})
gt_clus = gt_clus

if 'hyb_clu_list' in dir():
    #if len(hyb_clu_list)==len(gt_clus):
    if True:
        for i,clu in enumerate(gt_clus):
            for x in hyb_clu_list[i].exp_clusts:
                prec_rem_onstep,f1_onstep,cum_n_outs_onstep,f1_ba = outlier.run_exp_outlier(x,s,m,c)
                if f1_onstep:
                    exp_dict['f1_onstep'].append(f1_onstep)
                    exp_dict['prec_rem_onstep'].append(prec_rem_onstep)
                    exp_dict['cum_nouts'].append(cum_n_outs_onstep)
                    exp_dict['art%'].append(x['art_pct'])
                    exp_dict['hyb_clu'].append(x['id'])
                    exp_dict['f1_ba'].append(f1_ba)
                    # Primary channel signal for this cluster
                    trace = traces[:,x['best_c']]
                    n_est = (noise_est(trace))
                    
                    # Get spike waveforms
                    all_spikes = []
                    all_spikes.extend(x['hyb'][:])
                    all_spikes.extend(x['real'][:])
                    exp_dict['nspikes'].append(len(all_spikes))

                    waveform = m.get_waveforms(all_spikes,[x['best_c']])
                    mean_waveform = np.mean(waveform,axis=1)
                    s_est = np.max(np.abs(mean_waveform))
                    
                    SNR = (s_est/n_est)
                    
                    print('SNR: %f'%SNR)
                    exp_dict['snr'].append(SNR)
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
                prec_rem_onstep,f1_onstep,cum_n_outs_onstep,f1_ba = outlier.run_exp_outlier(x,s,m,c)
                if f1_onstep:
                    exp_dict['f1_onstep'].append(f1_onstep)
                    exp_dict['prec_rem_onstep'].append(prec_rem_onstep)
                    exp_dict['cum_nouts'].append(cum_n_outs_onstep)
                    exp_dict['art%'].append(x['art_pct'])
                    exp_dict['hyb_clu'].append(x['id'])
                    exp_dict['f1_ba'].append(f1_ba)
                    # Primary channel signal for this cluster
                    trace = traces[:,x['best_c']]
                    n_est = (noise_est(trace))
                    
                    # Get spike waveforms
                    all_spikes = []
                    all_spikes.extend(x['hyb'][:])
                    all_spikes.extend(x['real'][:])
                    exp_dict['nspikes'].append(len(all_spikes))

                    waveform = m.get_waveforms(all_spikes,[x['best_c']])
                    mean_waveform = np.mean(waveform,axis=1)
                    s_est = np.max(np.abs(mean_waveform))
                    
                    SNR = (s_est/n_est)
                    
                    print('SNR: %f'%SNR)
                    exp_dict['snr'].append(SNR)

timestr_pdf = timestr+".pdf"
timestr_dict_npy = timestr+"_res.npy"

np.save(timestr_dict_npy,exp_dict)

pp = PdfPages(timestr_pdf)

f1_ba_arr = np.array(exp_dict['f1_ba'])
f1_diff_arr = f1_ba_arr[:,1]-f1_ba_arr[:,0]

# Plot before F1 vs after F1
fig1=plt.figure()
plt.scatter(f1_ba_arr[:,0],f1_ba_arr[:,1],s=.2)
for i in range(f1_ba_arr.shape[0]):
    plt.text(f1_ba_arr[i,0],f1_ba_arr[i,1]*1.0,'%d'%exp_dict['hyb_clu'][i],size='xx-small')
plt.plot(np.linspace(0,np.max(f1_ba_arr.flatten()),100),np.linspace(0,np.max(f1_ba_arr.flatten()),100),'k--',lw=.1)
plt.ylabel('F1 After')
plt.xlabel('F1 Before')
fig1.tight_layout()
plt.draw()
pp.savefig(plt.gcf())

# Plot SNR vs dF1
fig1=plt.figure()
plt.scatter(exp_dict['snr'],f1_diff_arr,s=.2)
for i in range(f1_ba_arr.shape[0]):
    plt.text(exp_dict['snr'][i]*1.002,f1_diff_arr[i]*1.002,'%d'%exp_dict['hyb_clu'][i],size='xx-small')
plt.ylabel('dF1')
plt.xlabel('SNR')
fig1.tight_layout()
plt.draw()
pp.savefig(plt.gcf())

# Plot nspikes vs dF1
fig1=plt.figure()
plt.scatter(exp_dict['nspikes'],f1_diff_arr,s=.2)
for i in range(f1_ba_arr.shape[0]):
    plt.text(exp_dict['nspikes'][i]*1.002,f1_diff_arr[i]*1.002,'%d'%exp_dict['hyb_clu'][i],size='xx-small')
plt.ylabel('dF1')
plt.xlabel('Number of spikes (count)')
fig1.tight_layout()
plt.draw()
pp.savefig(plt.gcf())

for i,clu in enumerate(exp_dict['hyb_clu']):
    max_idx = np.argmax(exp_dict['f1_onstep'][i])
    fig,axes=plt.subplots(nrows=1,ncols=2,figsize=(10,6),sharey=False,sharex=False)
    axes[0].plot(exp_dict['cum_nouts'][i],exp_dict['f1_onstep'][i])
    axes[0].plot(exp_dict['cum_nouts'][i][max_idx],exp_dict['f1_onstep'][i][max_idx],'ro')
    axes[0].set_xlabel('N spikes removed (count)')
    axes[0].set_ylabel('F1 score')
    axes[0].set_title('Unit %d'%(clu))
    axes[1].plot(exp_dict['cum_nouts'][i],exp_dict['prec_rem_onstep'][i])
    axes[1].plot(exp_dict['cum_nouts'][i][max_idx],exp_dict['prec_rem_onstep'][i][max_idx],'ro')
    axes[1].set_xlabel('N spikes removed (count)')
    axes[1].set_ylabel('Precision of removed spikes')
    axes[1].set_title('Unit %d'%(clu))
    fig.tight_layout()
    plt.draw()
    pp.savefig(plt.gcf())

pp.close()
sys.stdout.close()
sys.stdout = orig_stdout
import winsound
winsound.Beep(450,900)