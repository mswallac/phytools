import pandas as pd
import numpy as np

class gt_clu(object):
    
    def __init__(self, clu_id, raw_spk_ids, raw_spk_times):
        super(gt_clu, self).__init__()
        self.clu_id = clu_id
        self.raw_spike_idxs = raw_spk_ids
        self.raw_spike_times = raw_spk_times
        self.nspikes = len(self.raw_spike_idxs)
        self.clu_hid = -1
        self.clu_hids = []
    
    def __str__(self):
        return None

    def link_hybrid(self,hyb_spike_times,hyb_spike_clus):
        # Backend work of matching to a hybrid cluster, identifying  artificial spikes, matching them to new detection    
        self.diffs=np.zeros(self.raw_spike_times.shape)
        self.hyb_idxs=np.zeros(self.raw_spike_times.shape).astype(int)
        self.raw_assoc=np.zeros(self.raw_spike_times.shape).astype(int)
        last_idx = 0
        # Associate spikes from raw by time with hybrid data, saving the difference in time, and the spike_idx from hybrid
        for i,d in enumerate(self.raw_spike_times):
            st_sub = hyb_spike_times-d
            idx = np.argmin(np.abs(st_sub))
            self.hyb_idxs[i] = idx+last_idx
            self.diffs[i] = st_sub[idx]
            self.raw_assoc[i] = self.raw_spike_idxs[i]
            hyb_spike_times = hyb_spike_times[idx:]
            last_idx = self.hyb_idxs[i]
        # Keep all spikes detected within 2ms of original time of raw data in hyb.
        filt_hyb_spikes = np.where(self.diffs <= 50)[0]
        print('GT Cluster %d: %2.3f %% (%d/%d) of spikes matched within 2ms'%(self.clu_id,len(filt_hyb_spikes)/len(self.diffs)*100,len(filt_hyb_spikes),len(self.diffs)))
        self.hyb_idxs = self.hyb_idxs[filt_hyb_spikes]
        self.diffs = self.diffs[filt_hyb_spikes]
        self.raw_assoc = self.raw_assoc[filt_hyb_spikes]
        # Now grab best cluster (containing most of original cluster) keep only spikes within this one
        self.clu_counts = pd.Series(data=hyb_spike_clus[self.hyb_idxs][:,0])
        self.clu_counts = self.clu_counts.value_counts()[0:4]
        self.clu_hid = self.clu_counts.idxmax()
        hyb_clu_nspks = self.clu_counts[self.clu_hid]
        print('GT Cluster %d: Associated with hybrid cluster %d, contains %2.3f %% (%d/%d) of original spikes'%(self.clu_id,self.clu_hid,(hyb_clu_nspks/self.nspikes)*100,hyb_clu_nspks,self.nspikes))
        return
    