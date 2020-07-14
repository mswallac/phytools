import scipy.cluster.vq as scc
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd
from gt_clu import gt_clu
from hyb_clu import hyb_clu
import numpy as np
import matplotlib.pyplot as plt

def run_exp_split(exp_clust,s,m,c):
    cid = exp_clust['id']
    real_spks = exp_clust['real']
    hyb_spks = exp_clust['hyb']
    art_pct = exp_clust['art_pct']
    nclusts = 20
    if 0.1 < art_pct < 0.90:
        cid,spikes,nspikes,chan,mstdict = get_spikes([cid],m,c)
        splits = cluster(mstdict,spikes,list(mstdict.keys())[2:6],nclusts)
        merged,clust_group = merge_clusters(splits,real_spks,hyb_spks,nclusts)
        real_res,hyb_res = check_res(merged,clust_group,real_spks,hyb_spks)
        #s.actions.split(merged['r'])
        return art_pct,real_res,hyb_res,merged
    else:
        return None,None,None,None

def check_res(merged,clust_group,real_spks,hyb_spks):
    real_in = np.in1d(real_spks,merged['r'])
    real_res = sum(real_in)/len(merged['r'])
    hyb_in = np.in1d(hyb_spks,merged['h'])
    hyb_res = sum(hyb_in)/len(merged['h'])
    return real_res,hyb_res

    

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
    #temp_f2.extend(features[:,:,2])
    mstdict.update({'Time': np.array(temp_t)})
    for i,d in enumerate(chan):
        mstdict.update({"PC0_C"+str(d): np.array(temp_f0)[:,i]})
        mstdict.update({"PC1_C"+str(d): np.array(temp_f1)[:,i]})
        #mstdict.update({"PC2_C"+str(d): np.array(temp_f2)[:,i]})
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

def merge_clusters(splits,gt1_spikes,gt2_spikes,nclusts):
    # Figure out what % each cluster is composed of each given cluster
    keys = list(splits.keys())
    both_spikes = []
    both_spikes.append(gt1_spikes)
    both_spikes.append(gt2_spikes)
    scores = np.zeros((2,len(keys)))
    for i in range(scores.shape[0]):
        clu_spikes = both_spikes[i]
        for j,d in enumerate(keys):
            split_spikes = splits[d]
            n_match = sum(np.in1d(clu_spikes,split_spikes))
            scores[i,j] = n_match/len(split_spikes)
    
    merged = {}
    merged.update({'r':[],'h':[]})

    prop_n_clusts = (len(gt1_spikes)/(len(gt1_spikes)+len(gt2_spikes)))*nclusts
    prop_n_clusts = int(prop_n_clusts)

    # Merge clusters w/ best agreement.
    clust_group = [scores[0,i] > scores[1,i] for i,d in enumerate(keys)]
    clust_score_diff = [scores[0,i] - scores[1,i] for i,d in enumerate(keys)]
    sort_score_diff = np.sort(clust_score_diff)
    clust_diff_group = np.in1d(clust_score_diff,sort_score_diff[0:prop_n_clusts])
    top_real_idxs = np.nonzero(clust_diff_group)[0]
    clust_score_diff = [scores[1,i] - scores[0,i] for i,d in enumerate(keys)]
    sort_score_diff = np.sort(clust_score_diff)
    clust_diff_group = np.in1d(clust_score_diff,sort_score_diff[0:(nclusts-prop_n_clusts)])
    top_hyb_idxs = np.nonzero(clust_diff_group)[0]

    if not (np.all(clust_group) or np.all(np.logical_not(clust_group))):
        for i,k in enumerate(keys):
            if clust_group[i]:
                merged['r'].extend(splits[k])
            else:
                merged['h'].extend(splits[k])
        # Return a dict of the new clusters
        return merged,clust_group
    else:
        for i,k in enumerate(keys):
            top_hyb = i in top_hyb_idxs
            top_real = i in top_real_idxs
            if top_hyb and top_real:
                print('huh?')
            elif top_real:
                merged['r'].extend(splits[k])
            elif top_hyb:
                merged['h'].extend(splits[k])
        # Return a dict of the new clusters
        return merged,clust_diff_group
