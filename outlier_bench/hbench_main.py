from hyb_clu import hyb_clu
import numpy as np
import pandas as pd
import glob
import outlier
import time
import os
import importlib

# For development pipeline convenience
importlib.reload(outlier)

# Operates on existing splits, iterouts, outs_by_iter info
res_file = glob.glob('outlier_*_res.npy')[0]
exp_dict = np.load(res_file,allow_pickle=True).item()

# Check that we've done some outlier rejection
try:
	assert cluster_out
	assert outs_per_iter
except:
	print('Reject outliers first to benchmark performance.')

hybrid_clusters = exp_dict['hyb_clu']

# Check that it was on an artificial cluster.
try:
	assert cluster_out in hybrid_clusters
except:
	print('No hybrid data available for this cluster.')

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

hyb_clu_list = []
matches = []
if 'hyb_clu_list' not in dir():
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
			if x['id'] == cluster_out[0]:
				matches.append(x)
elif len(hyb_clu_list)==len(gt_clus):
	for i,clu in enumerate(gt_clus):
		for x in hyb_clu_list[i].exp_clusts:
			if x['id'] == cluster_out[0]:
				matches.append(x)
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
			if x['id'] == cluster_out[0]:
				matches.append(x)



assert len(matches)>0

for i,clu in enumerate(gt_clus):
	for x in hyb_clu_list[i].exp_clusts:
		print(x['id'],cluster_out)


# Check that we've matched the right number of previous results to the current cluster
idxs = np.nonzero([cluster_out==x for x in exp_dict['hyb_clu']])[0]
assert len(idxs)==len(matches)

for idx,i in enumerate(idxs):
	fig,axes=plt.subplots(nrows=2,ncols=2,figsize=(10,6),sharey=False,sharex=False)
	axes[0].plot(exp_dict['cum_nouts'][i],exp_dict['fdr_onstep'][i])
	ax2 = axes[0].twinx()
	ax2.plot(exp_dict['cum_nouts'][i],exp_dict['f1_onstep'][i],'r-')
	ax2.legend(['F1'],loc=1)
	ax2.set_ylabel('F1 Score')
	axes[0].legend(['FDR'],loc=2)
	for x,d in enumerate(exp_dict['fdr_idxs'][i]):
	    axes[0].plot(exp_dict['cum_nouts'][i][d],exp_dict['fdr_onstep'][i][d],c=cmap[x],marker='o')
	axes[0].set_xlabel('N spikes removed (count)')
	axes[0].set_ylabel('False discovery rate')
	axes[0].set_title('Unit %d (auto)'%(cluster_out))
	axes[1].plot(exp_dict['cum_nouts'][i],exp_dict['prec_rem_onstep'][i])
	for x,d in enumerate(exp_dict['fdr_idxs'][i]):
	    axes[1].plot(exp_dict['cum_nouts'][i][d],exp_dict['prec_rem_onstep'][i][d],c=cmap[x],marker='o')
	axes[1].set_xlabel('N spikes removed (count)')
	axes[1].set_ylabel('Precision of removed spikes')
	axes[1].set_title('Unit %d (auto)'%(cluster_out))
	fig,axes=plt.subplots(nrows=1,ncols=2,figsize=(10,6),sharey=False,sharex=False)
	prec_rem_onstep,fdr_onstep,cum_n_outs_onstep,fdr_ba,f1_onstep,fdr_min_idxs,f1_ba = outlier.human_bench(matches[idx],s,m,c,outs_per_iter)
	axes[2].plot(cum_n_outs,fdr_onstep[i])
	ax2 = axes[2].twinx()
	ax2.plot(cum_n_outs,f1_onstep,'r-')
	ax2.legend(['F1'],loc=1)
	ax2.set_ylabel('F1 Score')
	axes[2].legend(['FDR'],loc=2)
	for x,d in enumerate(fdr_min_idxs):
	    axes[2].plot(cum_n_outs,fdr_onstep,c=cmap[x],marker='o')
	axes[2].set_xlabel('N spikes removed (count)')
	axes[2].set_ylabel('False discovery rate')
	axes[2].set_title('Unit %d (human)'%(cluster_out))
	axes[3].plot(cum_n_outs,prec_rem_onstep)
	for x,d in enumerate(fdr_min_idxs):
	    axes[3].plot(cum_n_outs,prec_rem_onstep,c=cmap[x],marker='o')
	axes[3].set_xlabel('N spikes removed (count)')
	axes[3].set_ylabel('Precision of removed spikes')
	axes[3].set_title('Unit %d (human)'%(cluster_out))
	fig.tight_layout()
	plt.show()