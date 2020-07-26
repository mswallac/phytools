import scipy.cluster.vq as scc
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd
from hyb_clu import hyb_clu
import numpy as np
from scipy.spatial.distance import mahalanobis
from scipy.stats import zscore

nchans = 6
nchunks = 50
min_nspikes = 1000
max_ntimeseg = 15
wavewin = slice(0,82)

def run_exp_outlier(exp_clust,s,m,c):
    cid = exp_clust['id']
    real_spks = exp_clust['real']
    hyb_spks = exp_clust['hyb']
    art_pct = exp_clust['art_pct']
    if art_pct < 0.15:
        cid,spikes,nspikes,chan,mstdict,splits = get_spikes([cid],m,c)
        print(sum(np.in1d(real_spks,spikes))+sum(np.in1d(hyb_spks,spikes)),len(spikes))
        outs = []
        # TODO : some ratio for how many spikes to remove: maybe:
        #            - the contaminating # of spikes
        #            - some % of total # of spikes (doing this for now)
        #            - according to ISI violation 
        #                (should this maybe be the criteria for using the outlier rejection plugin?)
        #
        n_to_rem = int(nspikes*.15)
        iters = 0
        outs_each_iter = []
        spikes_live = np.array(spikes)
        feats_keys = list(mstdict.keys())[2:8]
        while len(outs) < n_to_rem and iters<100:
            spikes_live=find_out(splits,outs,spikes,spikes_live,feats_keys,mstdict)
            print(len(spikes_live),len(outs), iters)
            outs_each_iter.append(outs)
            iters+=1

        final_prec,prec_onstep,cum_n_outs_onstep = bench_outlier(outs,outs_each_iter,real_spks,hyb_spks)

        return final_prec,prec_onstep,cum_n_outs_onstep
    else:
        return None,None,None

def bench_outlier(outs,outs_each_iter,real_spks,hyb_spks):
    # Calculate performance metrics
    TP = sum(np.in1d(hyb_spks,outs))
    FP = sum(np.in1d(real_spks,outs))
    print(TP,FP,TP+FP,len(outs))
    assert (TP+FP) == len(outs)
    final_prec = (TP/(TP+FP))
    prec_onstep = []
    n_outs_onstep = [len(x) for x in outs_each_iter]
    cum_n_outs_onstep = np.cumsum(n_outs_onstep)
    for out_onstep in outs_each_iter:
        TP = sum(np.in1d(hyb_spks,out_onstep))
        FP = sum(np.in1d(real_spks,out_onstep))
        assert (TP+FP) == len(out_onstep)
        prec = (TP/(TP+FP))
        prec_onstep.append(prec)

    return final_prec,prec_onstep,cum_n_outs_onstep

def get_spikes(cid,m,c):
    splits = {}
    spikes = m.get_cluster_spikes(cid)
    nspikes = len(spikes)

    if nchunks!=1:
        for i in np.arange(nchunks):
            splits.update({i+1: spikes[(i*nspikes)//nchunks:((i+1)*nspikes)//nchunks]})
    else:
        splits.update({1: spikes})
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
    mstdict.update({'Time': np.array(temp_t)})
    for i,d in enumerate(chan):
        mstdict.update({"PC0_C"+str(d): np.array(temp_f0)[:,i]})
        mstdict.update({"PC1_C"+str(d): np.array(temp_f1)[:,i]})
    
    return (cid,spikes,nspikes,chan,mstdict,splits)


def find_out(splits,outs,spikes,spikes_live,feats_keys,mstdict):
    keys = list(splits.keys())
    n_excl_per_iter = int(len(splits[keys[0]])*0.05)
    n_excl_per_iter = 100 if n_excl_per_iter==0 else n_excl_per_iter
    print(n_excl_per_iter)
    for i,d in enumerate(keys):
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
            which = np.unique(which)
            which_inds = np.nonzero(np.in1d(spikes,which))[0]
            for i,d in enumerate(feats_keys):
                temp = np.array(mstdict[d][which_inds])
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
            outinds = np.flip(np.argsort(dist))[0:n_excl_per_iter]
            outinds = which_inds[outinds]
            outinds = spikes[outinds]
            spikes_live = np.delete(spikes_live,np.in1d(spikes_live,outinds))
            outs.extend(outinds)
    return spikes_live