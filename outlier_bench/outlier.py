import scipy.cluster.vq as scc
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd
from hyb_clu import hyb_clu
import numpy as np
from scipy.spatial.distance import mahalanobis
from scipy.stats import zscore
import pandas as pd


nchans = 6
min_nspikes = 1000
max_ntimeseg = 10
wavewin = slice(0,82)

def get_nchunks():
    nchunks = 50
    return nchunks

def load_clust_data(cid,m,c):
    cid,spikes,nspikes,chan,mstdict,splits = get_spikes([cid],m,c)
    n_to_rem = int(np.round(nspikes*.50))
    feats_keys = list(mstdict.keys())[2:11]
    res = (cid,spikes,nspikes,chan,mstdict,splits)
    return res,n_to_rem,feats_keys

def run_exp_outlier(exp_clust,s,m,c):
    cid = exp_clust['id']
    real_spks = exp_clust['real']
    hyb_spks = exp_clust['hyb']
    art_pct = exp_clust['art_pct']
    outs = []
    outs_by_iter = []
    iters = 0
    if (0.40 < art_pct < 0.985):
        res,n_to_rem,feats_keys = load_clust_data(cid,m,c)
        cid,spikes,nspikes,chan,mstdict,splits = res

        while len(outs) < n_to_rem and iters<1500:
            splits,iterouts=find_out(n_to_rem,splits,outs,spikes,feats_keys,mstdict)
            iters+=1
            outs_by_iter.append(iterouts)

        print('Cluster %d ran for %d iterations, removing %d of %d spikes in search.'%(cid,iters,len(outs),nspikes))

        prec_rem_onstep,f1_onstep,cum_n_outs_onstep,f1_ba = bench_outlier(mstdict,outs,outs_by_iter,real_spks,hyb_spks,spikes)
        return prec_rem_onstep,f1_onstep,cum_n_outs_onstep,f1_ba
    else:
        return None,None,None,None

def bench_outlier(mstdict,outs,outs_each_iter,real_spks,hyb_spks,spikes):
    n_art = len(hyb_spks)
    # Calculate performance metrics on orig cluster
    TP = sum(np.in1d(spikes,hyb_spks))
    FP = sum(np.in1d(spikes,real_spks))
    FN = n_art-TP
    before_f1 = TP/(TP+(0.5*(FP+FN)))
    assert (TP+FP) == len(spikes)
    assert FN == 0 

    f1_onstep = []
    prec_rem_onstep = []
    cum_outs = []
    cum_outs_onstep = []

    n_outs_onstep = [len(x) for x in outs_each_iter]
    cum_n_outs_onstep = np.cumsum(n_outs_onstep)

    for i,out_onstep in enumerate(outs_each_iter):
        # Keeping track of which spikes we have excluded
        cum_outs.extend(out_onstep)
        cum_outs_onstep.append(cum_outs)
       
        # Calculate performance metrics. Since this is for REMOVED spikes--a true positive is removing
        # a spike that doesnt belong in this case, spikes which don't belong are spikes which were NOT 
        # inserted by hybridfactory, since hybridfactory clusters are the only ones we know for sure
        FP = sum(np.in1d(cum_outs,hyb_spks))
        TP = sum(np.in1d(cum_outs,real_spks))
        assert (TP+FP) == len(cum_outs)

        # Calculate what the precision was in removing spikes which are artifical
        removed_prec = (TP/(TP+FP))
        prec_rem_onstep.append(removed_prec)

        # Updated list of remaining spikes for other perf. metric
        remaining_spks = np.delete(spikes,np.nonzero(np.in1d(spikes,cum_outs))[0])

        # Calculate performance metrics
        TP = sum(np.in1d(hyb_spks,remaining_spks))
        FP = sum(np.in1d(real_spks,remaining_spks))
        FN = n_art-TP
        F1 = TP/(TP+(0.5*(FP+FN)))
        assert (TP+FP) == len(remaining_spks)
        f1_onstep.append(F1)

    after_f1 = np.max(f1_onstep)
    f1_ba = [before_f1,after_f1]

    return prec_rem_onstep,f1_onstep,cum_n_outs_onstep,f1_ba

def time_split(nchunks,splits,nspikes,spikes):
    splits = {}
    if nchunks!=1:
        for i in np.arange(nchunks):
            splits.update({i+1: spikes[(i*nspikes)//nchunks:((i+1)*nspikes)//nchunks]})
    else:
        splits.update({1: spikes})
    return splits


def get_spikes(cid,m,c):
    splits = {}
    spikes = m.get_cluster_spikes(cid)
    nspikes = len(spikes)

    splits = time_split(get_nchunks(),splits,nspikes,spikes)

    try:
        channel = c.selection.channel_id
    except AttributeError:
        channel = c.get_best_channels(cid[0])[0]

    av=(c._get_amplitude_functions())['raw']
    data=av(spikes, channel_id=channel, channel_ids=np.array(channel), load_all=True, first_cluster=cid[0])

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
    
    return (cid,spikes,nspikes,chan,mstdict,splits)


def find_out(n_to_rem,splits,outs,spikes,feats_keys,mstdict):
    keys = range(1,1+len(splits))
    n_excl_per_iter = int(n_to_rem/(50*len(keys)))
    n_excl_per_iter = 2 if n_excl_per_iter<=1 else n_excl_per_iter
    outs_each_iter = []
    for d in keys:
        which = []
        train = []
        cnk = splits[d]
        chunk_size = len(cnk)
        if chunk_size!=0:
            if chunk_size>=min_nspikes:
                which.append(cnk)
            else:
                which.extend(cnk)
                nchunks = (int(min_nspikes)-chunk_size)//chunk_size
                chunk_loc = d
                chunk_ind = np.nonzero(np.in1d(keys,chunk_loc))[0]
                if len(chunk_ind)==1:
                    chunk_ind = chunk_ind[0]
                    beg = (chunk_ind-nchunks//2)
                    end = (chunk_ind+nchunks//2)+1
                    if beg>0:
                        to_add = [x for x in keys[beg:end] if x != d]
                        if ((len(to_add)) < nchunks):
                            to_add.extend(keys[(beg-(nchunks-(len(to_add)))):beg])
                    else:
                        to_add = [x for x in keys[0:(end-beg)] if x != d]
                    for i,x in enumerate(to_add):
                        which.extend(splits[x])
            
            unique_which,unsortinds = np.unique(which,return_index=True)
            which = unique_which[np.argsort(unsortinds)] 
            
            # Checking spike / index mapping
            which_inds = (np.in1d(spikes,which))
            assert sum(which_inds)==len(which)
            assert np.array_equal(spikes[which_inds],which)
            
            # Checking spike / index mapping pt. 2
            which_inds = np.nonzero(which_inds)[0]
            assert np.array_equal(spikes[which_inds],which)

            # Create data train for use in outlier rejection algorithm
            for i,fk in enumerate(feats_keys):
                temp = np.array(mstdict[fk][which_inds])
                temp = np.reshape(temp,len(which_inds))
                train.append(temp)
            train = np.transpose(train)
            mpt = np.median(train, axis=0)
            cmat = np.transpose(train)
            d1,d2 = cmat.shape
            cmat = cmat.astype(np.float32)+0.00000001*np.random.rand(d1,d2)
            cmat = np.cov(cmat)
            cmati = np.linalg.inv(cmat)
            dist = []
            for i in np.arange(train.shape[0]):
                pt = train[i,:]
                dist.append(mahalanobis(pt,mpt,cmati))
            dist=np.abs(np.array(dist))

            cnk_spikes = np.in1d(which,cnk)
            to_rem_from = dist[cnk_spikes]
            inds_rem = which_inds[cnk_spikes]
            assert len(to_rem_from) == len(cnk)
            
            outinds = np.flip(np.argsort(to_rem_from))[0:n_excl_per_iter]
            outinds = spikes[inds_rem[outinds]]
            assert np.all(np.in1d(outinds,cnk))

            outs.extend(outinds)
            outs_each_iter.extend(outinds)
            
            
            a = (splits[d])
            spike_rem = np.nonzero(np.in1d(a,outinds))
            b = np.delete(a,spike_rem)
            splits.update({d: (b)})
            return splits,outs_each_iter