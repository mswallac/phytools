import pandas as pd
import numpy as np
from phy import *

class hyb_clu(object):
    
    def __init__(self, clu_id, spk_times,s,m,c,chan):
        super(hyb_clu, self).__init__()
        self.clu_id = clu_id
        self.spike_times = spk_times
        self.nspikes = len(self.spike_times)
        self.s, self.m, self.c = s, m, c
        self.chan = chan
        self.matched_clusts = []
    
    def __str__(self):
        return None

    def link_hybrid(self,hyb_spike_times,hyb_spike_clus):
        
        self.diffs=np.zeros(self.spike_times.shape)
        self.hyb_idxs=np.zeros(self.spike_times.shape).astype(int)
        last_idx = 0
        for i,d in enumerate(self.spike_times):
            st_sub = hyb_spike_times-d
            idx = np.argmin(np.abs(st_sub))
            self.hyb_idxs[i] = idx+last_idx
            self.diffs[i] = st_sub[idx]
            hyb_spike_times = hyb_spike_times[idx:]
            last_idx = self.hyb_idxs[i]
        
        # Keep all spikes detected within 2ms of original time of raw data in hyb.
        filt_hyb_spikes = np.where(self.diffs <= 50)[0]
        print('Artificial cluster (based on %d): %2.3f %% (%d/%d) of spikes matched within 2ms'%(self.clu_id,len(filt_hyb_spikes)/len(self.diffs)*100,len(filt_hyb_spikes),len(self.diffs)))
        self.hyb_idxs = self.hyb_idxs[filt_hyb_spikes]
        self.diffs = self.diffs[filt_hyb_spikes]
        
        # Now grab best cluster (containing most of original cluster) keep only spikes within this one
        self.matched_clus = hyb_spike_clus[self.hyb_idxs][:,0]
        self.clu_counts = pd.Series(data=self.matched_clus)
        self.clu_counts = self.clu_counts.value_counts()
        # TODO: Convert this into % of each cluster that is artificial, use this info to feed into splitter
        self.clu_hid = self.clu_counts.idxmax()
        hyb_clu_nspks = self.clu_counts[self.clu_hid]
        self.exp_clusts = []
        for x in self.clu_counts.index[0:5]:
	        c_clu_spikes = self.m.get_cluster_spikes(x)
        	if len(c_clu_spikes)>500:
	            ref_idxs = np.where(self.matched_clus==x)
	            art_clu_spikes = self.hyb_idxs[ref_idxs]
	            print('\tArtifical cluster (based on %d): Associated with hybrid cluster %d, contains %2.3f %% artificial spikes (%d/%d artificial)'%(self.clu_id,x,(self.clu_counts[x]/len(c_clu_spikes))*100,self.clu_counts[x],len(c_clu_spikes)))
	            matched_hyb = (np.in1d(c_clu_spikes,art_clu_spikes))
	            hyb_spikes = c_clu_spikes[matched_hyb]
	            real_spikes = c_clu_spikes[np.logical_not(matched_hyb)]
	            idx_match_pct = (sum(matched_hyb)/len(art_clu_spikes))
	            hyb_frac = (self.clu_counts[x]/len(c_clu_spikes))
	            #print('\t\t%2.3f %% (%d/%d) artificial spikes matched by index in hybrid (for confirmation)'%(idx_match_pct*100,sum(matched_hyb),len(art_clu_spikes)))
	            self.exp_clusts.append({'id':x,'hyb':hyb_spikes,'real':real_spikes,'art_pct':hyb_frac})
        return
    
    def merge_hybrid(self):
        #Prepare to merge all the small clusters which contain artificial spikes together
        return