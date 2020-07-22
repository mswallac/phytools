import scipy.cluster.vq as scc
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd
from gt_clu import gt_clu
from hyb_clu import hyb_clu
import numpy as np
import matplotlib.pyplot as plt

nchans = 6
nchunks = 50
wavewin = slice(0,82)

def run_exp_outlier(exp_clust,s,m,c,nclusts):
    cid = exp_clust['id']
    real_spks = exp_clust['real']
    hyb_spks = exp_clust['hyb']
    art_pct = exp_clust['art_pct']
    if 0.80 < art_pct or art_pct < 0.20:
        cid,spikes,nspikes,chan,mstdict = get_spikes([cid],m,c)
        return None,None,None,None
    else:
        return None,None,None,None

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


def find_out():
    contam_f = 5
    which=[]
    train = []
    cnk = splits[d]
    chunk_size = len(cnk)
    if chunk_size!=0:
        spike_inc=np.nonzero(np.in1d((spikes),(cnk)))[0]
        if chunk_size>=int(screen.nse.text()):
            which.append(spike_inc)
        else:
            which.extend(spike_inc)
            nchunks = (int(screen.nse.text())-chunk_size)//chunk_size
            if nchunks>(int(screen.nce.text())):
                nchunks=int(screen.nce.text())
            fchunks_l = np.arange(ngs)+1
            chunk_loc = d
            temp_chunks=[int(x) for x in fchunks_l]
            chunk_ind = np.nonzero(np.in1d(temp_chunks,chunk_loc))[0][0]
            beg=(chunk_ind-nchunks//2)
            end=(chunk_ind+nchunks//2)+1
            if beg>0:
                to_add = [x for x in temp_chunks[beg:end] if x != d]
                if ((len(to_add)) < nchunks):
                    to_add.extend(temp_chunks[(beg-(nchunks-(len(to_add)))):beg])
            else:
                to_add = [x for x in temp_chunks[0:(end-beg)] if x != d]
            for i,d in enumerate(to_add):
                which.extend(np.nonzero(np.in1d((spikes),(splits[d])))[0])
        which=np.array(which)
        which=which[0] if len(which.shape)>1 else which 
        which=np.array([int(x) for x in which])
        for i,d in enumerate(lb_val1):
            temp=np.array(mstdict[d][which])
            temp=np.reshape(temp,len(which))
            train.append(temp)
        train=np.transpose(train)
        mpt = np.median(train,axis=0)
        cmat = np.transpose(train)
        d1,d2=cmat.shape
        cmat = cmat.astype(np.float32)+0.00000001*np.random.rand(d1,d2)
        cmat = np.cov(cmat)
        cmati = np.linalg.inv(cmat)
        dist=[]
        for i in np.arange(train.shape[0]):
            pt = train[i,:]
            dist.append(mahalanobis(pt,mpt,cmati))
        dist=np.abs(np.array(dist))
        out_pred = np.nonzero((dist>contam_f)[0:chunk_size-1])[0]
        for x in spike_inc[out_pred]:
            if x not in outs:
                outs.append(x)

        outs=np.array(outs)
        nouts=sum(np.array(outs)>0)
        somesum=len(spike_inds)


