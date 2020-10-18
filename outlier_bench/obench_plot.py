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
import outlier
def generate_outlier_plots(exp_dict,pp):

    xlocs = [1,2,3]
    xlabels = ['Init.','Min FDR','Control']
    colors = ['black','green','red']

    
    # PLOTS TO SHOW CHANGES IN FDR / F1

    fig,axes=plt.subplots(nrows=1,ncols=2,figsize=(10,6),sharey=False,sharex=True)

    # F1
    f1_arr = []
    for i,clu in enumerate(exp_dict['hyb_clu']):
        axes[0].scatter(xlocs,exp_dict['f1_ba'][i]-exp_dict['f1_ba'][i][0],alpha = 1,c=colors)
        f1_arr.append(exp_dict['f1_ba'][i]-exp_dict['f1_ba'][i][0])
        axes[0].plot(xlocs,exp_dict['f1_ba'][i]-exp_dict['f1_ba'][i][0],'k',alpha = .1)
    f1_arr = np.array(f1_arr)
    stddev_arr = np.std(f1_arr,axis=0)
    mean_arr = np.mean(f1_arr,axis=0)
    axes[0].scatter(xlocs,np.mean(f1_arr,axis=0),800,alpha =1,c=colors,marker='_',linewidth=1.6)
    for i,x in enumerate(xlocs):
        axes[0].plot([x,x],[-stddev_arr[i]+mean_arr[i],stddev_arr[i]+mean_arr[i]],'-_',color=colors[i],ms=16,lw=.8)
    axes[0].set_title('F1 Before/After automated curation')
    axes[0].set_ylabel(r'$\Delta$F1 ')
    axes[0].set_xlim(0.5,3.5)
    axes[0].set_xticks(xlocs)
    axes[0].set_xticklabels(xlabels)

    # FDR
    fdr_arr = []
    for i,clu in enumerate(exp_dict['hyb_clu']):
        axes[1].scatter(xlocs,exp_dict['fdr_ba'][i]-exp_dict['fdr_ba'][i][0],alpha = .8,c=colors)
        fdr_arr.append(exp_dict['fdr_ba'][i]-exp_dict['fdr_ba'][i][0])
        axes[1].plot(xlocs,exp_dict['fdr_ba'][i]-exp_dict['fdr_ba'][i][0],'k',alpha = .1)
    fdr_arr = np.array(fdr_arr)
    stddev_arr = np.std(fdr_arr,axis=0)
    mean_arr = np.mean(fdr_arr,axis=0)
    axes[1].scatter(xlocs,np.mean(fdr_arr,axis=0),800,alpha =1,c=colors,marker='_',linewidth=1.6)
    for i,x in enumerate(xlocs):
        axes[1].plot([x,x],[-stddev_arr[i]+mean_arr[i],stddev_arr[i]+mean_arr[i]],'-_',color=colors[i],ms=16,lw=.8)
    axes[1].set_title('FDR Before/After automated curation')
    axes[1].set_xticks(xlocs)
    axes[1].set_xticklabels(xlabels)
    axes[1].set_ylabel(r'$\Delta$FDR')
    fig.tight_layout()
    plt.draw()
    pp.savefig(plt.gcf())

    # PLOTS TO SHOW CHANGES IN FDR / F1 (w/o 0-baseline)

    fig,axes=plt.subplots(nrows=1,ncols=2,figsize=(10,6),sharey=False,sharex=True)

    # F1
    f1_arr = []
    for i,clu in enumerate(exp_dict['hyb_clu']):
        axes[0].scatter(xlocs,exp_dict['f1_ba'][i],alpha = 1,c=colors)
        f1_arr.append(exp_dict['f1_ba'][i])
        axes[0].plot(xlocs,exp_dict['f1_ba'][i],'k',alpha = .1)
    f1_arr = np.array(f1_arr)
    axes[0].set_title('F1 Before/After automated curation')
    axes[0].set_ylabel('F1')
    axes[0].set_xlim(0.5,3.5)
    axes[0].set_xticks(xlocs)
    axes[0].set_xticklabels(xlabels)

    # FDR
    fdr_arr = []
    for i,clu in enumerate(exp_dict['hyb_clu']):
        axes[1].scatter(xlocs,exp_dict['fdr_ba'][i],alpha = .8,c=colors)
        fdr_arr.append(exp_dict['fdr_ba'][i])
        axes[1].plot(xlocs,exp_dict['fdr_ba'][i],'k',alpha = .1)
    fdr_arr = np.array(fdr_arr)
    axes[1].set_title('FDR Before/After automated curation')
    axes[1].set_xticks(xlocs)
    axes[1].set_xticklabels(xlabels)
    axes[1].set_ylabel('FDR')
    fig.tight_layout()
    plt.draw()
    pp.savefig(plt.gcf())

    for i,clu in enumerate(exp_dict['hyb_clu']):
        f1max_idx = exp_dict['f1_max_idx'][i]
        fig,axes=plt.subplots(nrows=1,ncols=2,figsize=(10,6),sharey='col',sharex=True)
        axes[0].plot(exp_dict['cum_nouts'][i],exp_dict['fdr_onstep'][i])
        axes[0].plot(exp_dict['cum_nouts'][i][f1max_idx],exp_dict['fdr_onstep'][i][f1max_idx],'ro')
        axes[0].set_xlabel('N spikes removed (count)')
        axes[0].set_ylabel('False discovery rate')
        axes[0].set_title('Unit %d (auto.)'%(clu))
        axes[1].plot(exp_dict['cum_nouts'][i],exp_dict['f1_onstep'][i])
        axes[1].set_xlabel('N spikes removed (count)')
        axes[1].set_ylabel('F1 Score')
        axes[1].set_title('Unit %d (auto.)'%(clu))
        fig.tight_layout()
        plt.draw()
        pp.savefig(plt.gcf())
    return

def human_perf_plot(exp_dict,hc_struct,pp,s,m,c,hperf_ids):
    for i,clu in enumerate(exp_dict['hyb_clu']):
        if clu in hperf_ids:
            print(clu)
            fig,axes=plt.subplots(nrows=2,ncols=2,figsize=(10,6),sharey='row',sharex=True)
            f1max_idx = exp_dict['f1_max_idx'][i]
            axes[0][0].plot(exp_dict['cum_nouts'][i],exp_dict['fdr_onstep'][i])
            axes[0][0].plot(exp_dict['cum_nouts'][i][f1max_idx],exp_dict['fdr_onstep'][i][f1max_idx],'ro')
            axes[0][0].set_xlabel('N spikes removed (count)')
            axes[0][0].set_ylabel('False discovery rate')
            axes[0][0].set_title('Unit %d (auto.)'%(clu))
            
            axes[1][0].plot(exp_dict['cum_nouts'][i],exp_dict['f1_onstep'][i])
            axes[1][0].plot(exp_dict['cum_nouts'][i][f1max_idx],exp_dict['f1_onstep'][i][f1max_idx],'ro')
            axes[1][0].set_xlabel('N spikes removed (count)')
            axes[1][0].set_ylabel('F1 Score')
            axes[1][0].set_title('Unit %d (auto.)'%(clu))

            hperf = np.load('hperf_%d.npy'%clu,allow_pickle=True).item()
            fdr_onstep,cum_n_outs_onstep,fdr_ba,f1_onstep,fdr_min_idxs,f1_ba=outlier.human_bench(hc_struct[i],s,m,c,hperf)
            axes[0][1].plot(cum_n_outs_onstep,fdr_onstep)
            axes[0][1].set_xlabel('N spikes removed (count)')
            axes[0][1].set_ylabel('False discovery rate')
            axes[0][1].set_title('Unit %d (human)'%(clu))
            axes[1][1].plot(cum_n_outs_onstep,f1_onstep)
            axes[1][1].set_xlabel('N spikes removed (count)')
            axes[1][1].set_ylabel('F1 Score')
            axes[1][1].set_title('Unit %d (human)'%(clu))
            fig.tight_layout()
            plt.draw()
            pp.savefig(plt.gcf())
    return

def isi_fdr_comparison(exp_dict,hc_struct,pp,s,m,c):
    for i,clu in enumerate(exp_dict['hyb_clu']):
        isi_arr,fp_arr,tp_arr,isi_all_arr = outlier.isi_fdr_comp(hc_struct[i],s,m,c)
        plt.figure()
        plt.plot(isi_arr/max(isi_arr))
        plt.plot(fp_arr/max(fp_arr))
        plt.plot(tp_arr/max(tp_arr))
        plt.legend(['ISI','FP','TP'])
        plt.xlabel('Iteration #')
        plt.ylabel('Fraction of Events Remaining by Type')
        plt.title('O.R. Process: Unit %d'%clu)
        plt.draw()
        pp.savefig(plt.gcf())
        hist_win = .220
        fig,axes=plt.subplots(nrows=1,ncols=2,figsize=(10,6),sharey='row',sharex=True)
        axes[0].hist(isi_all_arr[0],bins=np.linspace(0.0,hist_win,51),color='k')
        axes[0].set_title('ISI Hist. Before Curation: Unit %d'%clu)
        axes[1].hist(isi_all_arr[len(isi_all_arr)-1],bins=np.linspace(0.0,hist_win,51),color='k')
        axes[1].set_title('ISI Hist. After Curation: Unit %d'%clu)
        plt.draw()
        pp.savefig(plt.gcf())
    return
