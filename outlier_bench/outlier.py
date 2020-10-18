import scipy.cluster.vq as scc
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd
from hyb_clu import hyb_clu
import numpy as np
from scipy.spatial.distance import mahalanobis
from scipy.stats import zscore
import pandas as pd


nchans = 6
min_nspikes = 300
max_ntimeseg = 5
wavewin = slice(0,82)

def load_clust_data(cid,m,c):
    cid,spikes,nspikes,chan,mstdict,splits = get_spikes([cid],m,c)
    n_to_rem = int(np.round(nspikes*(1/8)))
    use_keys = []
    #use_inds = np.array([2,3,4,5,8,11])
    use_inds = np.arange(2,12)
    feats_keys = list(mstdict.keys())
    for i in use_inds:
        use_keys.append(feats_keys[i])
    res = (cid,spikes,nspikes,chan,mstdict,splits)
    return res,n_to_rem,use_keys

def human_bench(exp_clust,s,m,c,hperf):
    cid = exp_clust['id']
    res,n_to_rem,feats_keys = load_clust_data(cid,m,c)
    cid,spikes,nspikes,chan,mstdict,splits = res
    real_spks = exp_clust['real']
    hyb_spks = exp_clust['hyb']
    art_pct = exp_clust['art_pct']
    outs_by_iter = hperf['outs_per_iter']
    outs = []
    for x in outs_by_iter:
        outs.extend(x)

    fdr_onstep,cum_n_outs_onstep,fdr_ba,f1_onstep,fdr_min_idxs,f1_ba = bench_outlier(mstdict,outs,outs_by_iter,real_spks,hyb_spks,spikes)
    return fdr_onstep,cum_n_outs_onstep,fdr_ba,f1_onstep,fdr_min_idxs,f1_ba

def isi_fdr_comp(exp_clust,s,m,c):
    cid = exp_clust['id']
    res,n_to_rem,feats_keys = load_clust_data(cid,m,c)
    cid,spikes,nspikes,chan,mstdict,splits = res
    outs_by_iter = np.load('hperf_%d.npy'%cid,allow_pickle=True).item()
    outs_by_iter = outs_by_iter['outs_per_iter']
    real_spks = exp_clust['real']
    hyb_spks = exp_clust['hyb']
    art_pct = exp_clust['art_pct']
    spike_times = mstdict['Time']

    isi_rem = []
    fp_rem = []
    tp_rem = []
    cum_outs = []
    isi_all_arr = []
    remaining_spks = spikes
    for i,x in enumerate(outs_by_iter):
        cum_outs.extend(x)
        out_idxs = np.nonzero(np.in1d(remaining_spks,cum_outs))[0]
        remaining_spks = np.delete(remaining_spks,out_idxs)
        spike_times = np.delete(spike_times,out_idxs)
        remaining_spk_idxs = np.nonzero(np.in1d(remaining_spks,spikes))[0]
        rem_spike_times = np.diff(spike_times[remaining_spk_idxs])
        isi_all_arr.append(rem_spike_times)
        hist,t = np.histogram(rem_spike_times,bins=51,range=(0.0,.220))
        hist_count = sum(hist)
        TP = sum(np.in1d(remaining_spks,hyb_spks))
        FP = sum(np.in1d(remaining_spks,real_spks))
        isi_rem.append(hist_count)
        fp_rem.append(FP)
        tp_rem.append(TP)
        assert TP+FP == len(remaining_spks)
        assert len(remaining_spks)+len(cum_outs) == len(spikes)

    return isi_rem,fp_rem,tp_rem,isi_all_arr

def run_exp_outlier(exp_clust,s,m,c):
    cid = exp_clust['id']
    real_spks = exp_clust['real']
    hyb_spks = exp_clust['hyb']
    art_pct = exp_clust['art_pct']
    iters = 0
    if 0.80 < art_pct < 0.998:
        res,n_to_rem,feats_keys = load_clust_data(cid,m,c)
        cid,spikes,nspikes,chan,mstdict,splits = res
        outs=[]
        outs_by_iter = []
        thresh = 5.6
        d_thr = 0.2
        dt_tmax = int(thresh/d_thr)
        dt_ticks = 0
        minspikes_iter = 5
        # Remove iteration on this layer and make it all happen within find_out
        while (len(outs) < n_to_rem) and (iters<250 or dt_ticks<dt_tmax):
            splits,obi=find_out(thresh,outs,n_to_rem,splits,spikes,feats_keys,mstdict)
            iters+=1
            if len(obi)<=minspikes_iter:
                thresh-=d_thr
                dt_ticks+=1
                outs_by_iter.extend(obi)
            else:
                outs_by_iter.extend(obi)
        print('Cluster %d ran for %d iterations, removing %d of %d spikes in search.'%(cid,iters,len(outs),nspikes))

        # split code
        
        fdr_onstep,cum_n_outs_onstep,fdr_ba,f1_onstep,f1_max_idx,f1_ba = bench_outlier(mstdict,outs,outs_by_iter,real_spks,hyb_spks,spikes)
        return fdr_onstep,cum_n_outs_onstep,fdr_ba,f1_onstep,f1_max_idx,f1_ba,obi
    else:
        return None,None,None,None,None,None,None

def rem_random(spikes,n_rem,real_spks,hyb_spks):
    n_art = len(hyb_spks)
    outs_by_iter = []
    outs = []
    while len(outs)<=n_rem:
        rem = np.random.choice(spikes,10)
        outs_by_iter.append(rem)
        outs.extend(rem)
        spike_rem = np.nonzero(np.in1d(spikes,rem))
        spikes = np.delete(spikes,spike_rem)
    # Calculate performance metrics
    TP = sum(np.in1d(hyb_spks,spikes))
    FP = sum(np.in1d(real_spks,spikes))
    FN = n_art-TP
    FDR = FP/(TP+FP)
    F1 = TP/(TP+0.5*(FP+FN))
    assert (TP+FP) == len(spikes)
    return F1,FDR

def bench_outlier(mstdict,outs,outs_each_iter,real_spks,hyb_spks,spikes):
    n_art = len(hyb_spks)
    # Calculate performance metrics on orig cluster
    TP = sum(np.in1d(spikes,hyb_spks))
    FP = sum(np.in1d(spikes,real_spks))
    FN = n_art-TP
    before_fdr = FP/(TP+FP)
    before_f1 = TP/(TP+0.5*(FP+FN))
    assert (TP+FP) == len(spikes)
    assert FN == 0 

    fdr_onstep = []
    f1_onstep = []
    cum_outs = []
    cum_outs_onstep = []

    n_outs_onstep = [len(x) for x in outs_each_iter]
    cum_n_outs_onstep = np.cumsum(n_outs_onstep)
    for i,out_onstep in enumerate(outs_each_iter):
        if (len(out_onstep)):
            # Keeping track of which spikes we have excluded
            cum_outs.extend(out_onstep)
            cum_outs_onstep.append(cum_outs)

            # Updated list of remaining spikes for other perf. metric
            remaining_spks = np.delete(spikes,np.nonzero(np.in1d(spikes,cum_outs))[0])

            # Calculate performance metrics
            TP = sum(np.in1d(hyb_spks,remaining_spks))
            FP = sum(np.in1d(real_spks,remaining_spks))
            FN = n_art-TP
            FDR = FP/(TP+FP)
            F1 = TP/(TP+0.5*(FP+FN))
            assert (TP+FP) == len(remaining_spks)
            fdr_onstep.append(FDR)
            f1_onstep.append(F1)

    fdr_ba = [before_fdr]
    f1_ba = [before_f1]
    fdr_min_idx = np.argmin(fdr_onstep)
    nspikes_removed = cum_n_outs_onstep[fdr_min_idx]
    fdr_ba.append(fdr_onstep[fdr_min_idx])
    f1_ba.append(f1_onstep[fdr_min_idx])
    rand_f1,rand_fdr = rem_random(spikes,nspikes_removed,real_spks,hyb_spks)
    fdr_ba.append(rand_fdr)
    f1_ba.append(rand_f1)
    
    # remove random spikes, choose point which corresponds to this idx to benchmark random with

        
    return fdr_onstep,cum_n_outs_onstep,fdr_ba,f1_onstep,fdr_min_idx,f1_ba

def time_split(splits,nspikes,spikes):
    if nspikes>=5000:
        nchunks = 50
    else:
        nchunks = 25

    splits = {}
    for i in np.arange(nchunks):
        splits.update({i+1: spikes[(i*nspikes)//nchunks:((i+1)*nspikes)//nchunks]})
    return splits


def get_spikes(cid,m,c):
    splits = {}
    spikes = m.get_cluster_spikes(cid)
    nspikes = len(spikes)
    splits = time_split(splits,nspikes,spikes)

    try:
        channel = c.selection.channel_id
    except AttributeError:
        channel = c.get_best_channels(cid[0])[0]

    
    cid = cid[0]
    chan = c.get_best_channels(cid)[0:5]
    spike_amps = c.get_spike_raw_amplitudes(spikes,chan[0])
    spike_template_amps = c.get_spike_template_amplitudes(spikes)
    spike_times = m.spike_times[spikes]
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
    mstdict.update({'Amplitude': spike_amps})
    mstdict.update({'Template': spike_template_amps})
    for i,d in enumerate(chan):
        mstdict.update({"PC0_C"+str(d): np.array(temp_f0)[:,i]})
        mstdict.update({"PC1_C"+str(d): np.array(temp_f1)[:,i]})
        mstdict.update({"PC2_C"+str(d): np.array(temp_f2)[:,i]})
    
    return (cid,spikes,nspikes,chan,mstdict,splits)


def find_out(thresh,outs,n_to_rem,splits,spikes,feats_keys,mstdict):
    nchunks=len(splits)
    keys = range(1,nchunks+1)
    outs_by_iter = []
    for d in keys:
        which = []
        train = []
        cnk = splits[d]
        chunk_size = len(cnk)
        if chunk_size!=0:
            if chunk_size>=min_nspikes:
                which.extend(cnk)
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

            which_inds = np.nonzero(np.in1d(spikes,which))[0]
            assert len(which_inds)==len(which)
            assert np.array_equal(np.sort(spikes[which_inds]),np.sort(which))

            # Create data train for use in outlier rejection algorithm
            for i,fk in enumerate(feats_keys):
                temp = np.array(mstdict[fk][which_inds])
                temp = np.reshape(temp,len(which_inds))
                train.append(temp)
            train = np.transpose(train)
            mpt = np.median(train, axis=0)
            cmat = np.transpose(train)
            d1,d2 = cmat.shape
            cmat = cmat.astype(np.float32)+0.0000000001*np.random.rand(d1,d2)
            cmat = np.cov(cmat)
            cmati = np.linalg.inv(cmat)
            dist = []
            for i in np.arange(train.shape[0]):
                pt = train[i,:]
                dist.append(mahalanobis(pt,mpt,cmati))
            dist=np.abs(np.array(dist))
            dist_bool = np.nonzero([x>thresh for x in dist])[0]
            out_spikes = spikes[which_inds[dist_bool]]
            out_in_cnk = np.nonzero(np.in1d(out_spikes,cnk))[0]
            if len(out_in_cnk)>=1:
                outs.extend(out_spikes[out_in_cnk])
                outs_by_iter.append(out_spikes[out_in_cnk])
                a = (splits[d])
                spike_rem = np.nonzero(np.in1d(a,out_spikes[out_in_cnk]))
                b = np.delete(a,spike_rem)
                splits.update({d: (b)})

    return splits,outs_by_iter