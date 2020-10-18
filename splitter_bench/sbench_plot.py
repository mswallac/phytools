from matplotlib.backends.backend_pdf import PdfPages
# from phylib.io.traces import get_ephys_reader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import os
import importlib
from matplotlib import cm
import sys
from glob import glob

def generate_splitter_plots(exp_dict,pp,hperf_ids,hybclu_structs,s,m,c):
    # Prepare arrays for plotting
    f1_ba_arr = np.array(exp_dict['f1_ba'])
    f1_diff_arr = (f1_ba_arr[:,1]-f1_ba_arr[:,0])/(1-f1_ba_arr[:,0])

    # Plot presets
    xlocs = [1,2]
    xlabels = ['Before', 'After']

    # PLOTS TO SHOW CHANGES IN FDR / F1

    # F1
    fig1=plt.figure()
    for i,clu in enumerate(exp_dict['hyb_clu']):
        plt.plot(xlocs,exp_dict['f1_ba'][i]-exp_dict['f1_ba'][i][0],'-ok',alpha = .8)
    plt.title('F1 Before/After automated curation')
    plt.xticks(xlocs,xlabels,fontsize='x-large')
    plt.xlim(0,3)
    plt.ylabel('F1 Score')
    fig1.tight_layout()
    plt.draw()
    pp.savefig(plt.gcf())

    # Plot F1 score before vs after
    fig1=plt.figure()
    plt.scatter(f1_ba_arr[:,0],f1_ba_arr[:,1],s=.2)
    for i in range(f1_ba_arr.shape[0]):
        plt.text(f1_ba_arr[i,0]*1.002,f1_ba_arr[i,1]*1.002,'%d'%exp_dict['hyb_clu'][i],size='xx-small')
    plt.plot(np.linspace(0,np.max(f1_ba_arr.flatten()),100),np.linspace(0,np.max(f1_ba_arr.flatten()),100),'k--',lw=.1)
    plt.ylabel('F1 After')
    plt.xlabel('F1 Before')
    fig1.tight_layout()
    plt.draw()
    pp.savefig(plt.gcf())
    plt.close()

    # Plot SNR vs dF1
    fig1=plt.figure()
    plt.scatter(exp_dict['snr'],f1_diff_arr,s=.2)
    for i in range(f1_ba_arr.shape[0]):
        plt.text(exp_dict['snr'][i]*1.002,f1_diff_arr[i]*1.002,'%d'%exp_dict['hyb_clu'][i],size='xx-small')
    plt.axhline(y=0,c='k',ls='--')
    plt.ylabel('dF1 (fraction of possible +dF1)')
    plt.xlabel('SNR')
    fig1.tight_layout()
    plt.draw()
    pp.savefig(plt.gcf())

    # Plot nspikes vs dF1
    fig1=plt.figure()
    plt.scatter(exp_dict['nspikes'],f1_diff_arr,s=.2)
    for i in range(f1_ba_arr.shape[0]):
        plt.text(exp_dict['nspikes'][i]*1.002,f1_diff_arr[i]*1.002,'%d'%exp_dict['hyb_clu'][i],size='xx-small')
    plt.axhline(y=0,c='k',ls='--')
    plt.ylabel('dF1 (fraction of possible +dF1)')
    plt.xlabel('Number of spikes (count)')
    fig1.tight_layout()
    plt.draw()
    pp.savefig(plt.gcf())

    # Plot per-cluster representation of splitting process.
    for i,clu in enumerate(exp_dict['hyb_clu']):
        fig,axes=plt.subplots(nrows=1,ncols=2,figsize=(8,4),sharey=False,sharex=True)
        axes[0].bar(np.arange(1,nclusts+1),(np.flip(np.sort(exp_dict['clu_precision'][i]))*100))
        axes[0].set_xlabel('Sub-cluster (idx)')
        axes[0].set_ylabel('Precision (%)')
        axes[0].set_title('Unit %d'%(clu))
        f1_merged_scores = np.array(exp_dict['f1_scores'][i])
        best_idx = np.argmax(f1_merged_scores)
        axes[1].plot(np.arange(1,nclusts+1),(f1_merged_scores*100))
        axes[1].plot(np.arange(1,nclusts+1)[best_idx],(f1_merged_scores*100)[best_idx],'ro')
        axes[1].set_xlabel('Sub-clusters combined (count)')
        axes[1].set_ylabel('F1 Score (%) of combined clust.')
        axes[1].set_title('Unit %d'%(clu))
        fig.tight_layout()
        pp.savefig(plt.gcf())
return