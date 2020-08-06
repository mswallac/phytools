import scipy.cluster.vq as scc
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd
from gt_clu import gt_clu
from hyb_clu import hyb_clu
import numpy as np
import matplotlib.pyplot as plt

def run_exp_split(exp_clust,s,m,c,nclusts):
    cid = exp_clust['id']
    real_spks = exp_clust['real']
    hyb_spks = exp_clust['hyb']
    art_pct = exp_clust['art_pct']
    if 0.05 < art_pct < 0.90:
        cid,spikes,nspikes,chan,mstdict = get_spikes([cid],m,c)
        splits = cluster(mstdict,spikes,list(mstdict.keys())[2:11],nclusts)
        clust_precisions,f1s_merged,merged_clusts,f1_ba = merge_clusters(splits,real_spks,hyb_spks,spikes,nclusts)

        sts = []
        keys = np.array(list(splits.keys()))
        idxs = np.array(merged_clusts[np.argmax(f1s_merged)])
        for x in keys[idxs]:
            sts.extend(splits[x])
        #s.actions.split(sts)
        return art_pct,clust_precisions,f1s_merged,merged_clusts,f1_ba,sts
    else:
        return None,None,None,None,None,None

def get_spikes(cid,m,c):
    spikes = m.get_cluster_spikes(cid)
    try:
        channel = c.selection.channel_id
    except AttributeError:
        channel = c.get_best_channels(cid[0])[0]
    av=(c._get_amplitude_functions())['raw']
    data=av(spikes, channel_id=channel, channel_ids=np.array(channel), load_all=True, first_cluster=cid[0])
    nspikes = len(spikes)
    cid = cid[0]
    chan = c.get_best_channels(cid)[0:5]
    spike_times = m.spike_times[spikes]
    spike_amps = data
    features = m.get_features(spikes,chan)[:,:,:]
    temp_w=[]
    temp_a=[]
    temp_t=[]
    temp_f0=[]
    temp_f1=[]
    temp_f2=[]
    mstdict={}
    temp_a.extend(spike_amps[:])
    temp_t.extend(spike_times[:])
    temp_f0.extend(features[:,:,0])
    temp_f1.extend(features[:,:,1])
    temp_f2.extend(features[:,:,2])
    mstdict.update({'Time': np.array(temp_t)})
    for i,d in enumerate(chan):
        mstdict.update({"PC0_C"+str(d): np.array(temp_f0)[:,i]})
        mstdict.update({"PC1_C"+str(d): np.array(temp_f1)[:,i]})
        mstdict.update({"PC2_C"+str(d): np.array(temp_f2)[:,i]})
    return (cid,spikes,nspikes,chan,mstdict)


def cluster(mstdict,spikes,feat_keys,nclusts):
    features = []
    splits = {}
    for i,d in enumerate(feat_keys):
            features.append(mstdict[d][:])
    obs = scc.whiten(np.transpose(features))
    pd = ssd.pdist(obs,metric='euclidean')
    Z = sch.ward(pd)
    l = sch.fcluster(Z,t=nclusts,criterion='maxclust')
    for i in np.arange(1,nclusts+1):
        inter1=np.nonzero(np.in1d(l,i))[0]
        splits.update({(i): spikes[inter1]})
    return splits

def merge_clusters(splits,real_spikes,hyb_spikes,spikes,nclusts):
    # Figure out what % each cluster is composed of each given cluster
    keys = list(splits.keys())
    clust_precisions = []
    n_art = len(hyb_spikes)

    TP = sum(np.in1d(hyb_spikes,spikes))
    FP = sum(np.in1d(real_spikes,spikes))
    FN = n_art-TP
    assert (TP+FP) == len(spikes)
    assert FN == 0 

    before_F1 = TP/(TP+0.5*(FP+FN))

    # Using precision as metric to decide merge order
    for i,d in enumerate(keys):
        split_spikes = splits[d]
        TP = sum(np.in1d(hyb_spikes,split_spikes))
        FP = sum(np.in1d(real_spikes,split_spikes))
        assert (TP+FP) == len(split_spikes)
        FN = n_art-TP
        clust_precisions.append(TP/(TP+FP))

    clust_precisions=np.array(clust_precisions)
    sort_map = np.flip(np.argsort(clust_precisions))
    precs_sorted = clust_precisions[sort_map]
    f1s_merged = []
    merge_spikes = []
    merged_clusts = []

    # Merge clusters in order of precision and recompute F1 after each merge
    for i,d in enumerate(precs_sorted):
        merge_spikes.extend(splits[keys[sort_map[i]]])
        TP = sum(np.in1d(hyb_spikes,merge_spikes))
        FP = sum(np.in1d(real_spikes,merge_spikes))
        assert (TP+FP) == len(merge_spikes)
        FN = n_art-TP
        F1 = TP/(TP+0.5*(FP+FN))
        f1s_merged.append(F1)
        merged_clusts.append(sort_map[0:i])

    # Some code to find optimum F1 and return dict with the appropriate split
    after_F1 = np.max(f1s_merged)
    f1_ba = [before_F1,after_F1]
    return clust_precisions,f1s_merged,merged_clusts,f1_ba