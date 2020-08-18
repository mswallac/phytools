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

def generate_outlier_plots(exp_dict,pp):
    f1_ba_arr = np.array(exp_dict['f1_ba'])
    fdr_ba_arr = np.array(exp_dict['fdr_ba'])
    pdf1_arr = np.zeros(f1_ba_arr.shape)
    
    for i in range(f1_ba_arr.shape[0]):
        temp_f1_arr = f1_ba_arr[i,:]
        pdf1_arr[i,:] = np.array([(x-temp_f1_arr[0])/(1-temp_f1_arr[0]) for x in temp_f1_arr])

    # Get cmap
    cmap = plt.get_cmap('plasma').colors
    cmap = [cmap[x] for x in np.linspace(0,252,n_minima).astype(int)]
    cmap.reverse()
    labels = ['']
    ext_lbls = [str(x+1) for x in range(n_minima)]
    labels.extend(ext_lbls)

    # Plot before F1 vs after F1
    fig1=plt.figure()
    plt.plot(np.linspace(0,np.max(f1_ba_arr.flatten()),100),np.linspace(0,np.max(f1_ba_arr.flatten()),100),'k--',lw=.15)
    for x in range(n_minima): 
        plt.scatter(f1_ba_arr[:,0],f1_ba_arr[:,x],c=cmap[x],s=7)
    for i in range(f1_ba_arr.shape[0]):
        plt.text(f1_ba_arr[i,0]*.998,f1_ba_arr[i,0]*1.002,'%d'%exp_dict['hyb_clu'][i],size='xx-small')
    plt.ylabel('F1 After')
    plt.xlabel('F1 Before')
    plt.legend(labels)
    fig1.tight_layout()
    plt.draw()
    pp.savefig(plt.gcf())

    # Plot before FDR vs after FDR
    fig1=plt.figure()
    plt.plot(np.linspace(0,np.max(fdr_ba_arr.flatten()),100),np.linspace(0,np.max(fdr_ba_arr.flatten()),100),'k--',lw=.15)
    for x in range(n_minima): 
        plt.scatter(fdr_ba_arr[:,0],fdr_ba_arr[:,x],c=cmap[x],s=7)
    for i in range(fdr_ba_arr.shape[0]):
        plt.text(fdr_ba_arr[i,0]*.998,fdr_ba_arr[i,0]*1.002,'%d'%exp_dict['hyb_clu'][i],size='xx-small')
    plt.ylabel('FDR After')
    plt.xlabel('FDR Before')
    plt.legend(labels)
    fig1.tight_layout()
    plt.draw()
    pp.savefig(plt.gcf())

    # Plot SNR vs dF1
    fig1=plt.figure()
    for x in range(n_minima): 
        plt.scatter(exp_dict['snr'],pdf1_arr[:,x],c=cmap[x],s=7)
    for i in range(pdf1_arr.shape[0]):
        plt.text(exp_dict['snr'][i]*.998,pdf1_arr[i,0]*1.002,'%d'%exp_dict['hyb_clu'][i],size='xx-small')

    plt.axhline(y=0,c='k',ls='--')
    plt.ylabel('dF1 (fraction of possible +dF1)')
    plt.xlabel('SNR')
    plt.legend(labels)
    fig1.tight_layout()
    plt.draw()
    pp.savefig(plt.gcf())

    # Plot nspikes vs dF1
    fig1=plt.figure()
    for x in range(n_minima): 
        plt.scatter(exp_dict['nspikes'],pdf1_arr[:,x],c=cmap[x],s=7)
    for i in range(pdf1_arr.shape[0]):
        plt.text(exp_dict['nspikes'][i]*.998,pdf1_arr[i,0]*1.002,'%d'%exp_dict['hyb_clu'][i],size='xx-small')

    plt.axhline(y=0,c='k',ls='--')
    plt.ylabel('dF1 (fraction of possible +dF1)')
    plt.xlabel('Number of spikes (count)')
    plt.legend(labels)
    fig1.tight_layout()
    plt.draw()
    pp.savefig(plt.gcf())

    for i,clu in enumerate(exp_dict['hyb_clu']):
        fig,axes=plt.subplots(nrows=1,ncols=2,figsize=(10,6),sharey=False,sharex=False)
        axes[0].plot(exp_dict['cum_nouts'][i],exp_dict['fdr_onstep'][i])
        ax2 = axes[0].twinx()
        ax2.plot(exp_dict['cum_nouts'][i],exp_dict['f1_onstep'][i],'r-')
        ax2.legend(['F1'],loc=1)
        ax2.set_ylabel('F1 Score')
        axes[0].legend(['FDR'],loc=2)
        for x,d in enumerate(exp_dict['fdr_idxs'][i]):
            axes[0].plot(exp_dict['cum_nouts'][i][d],exp_dict['fdr_onstep'][i][d],c=cmap[x],marker='o')
        axes[0].set_xlabel('N spikes removed (count)')
        axes[0].set_ylabel('False discovery rate')
        axes[0].set_title('Unit %d'%(clu))
        axes[1].plot(exp_dict['cum_nouts'][i],exp_dict['prec_rem_onstep'][i])
        for x,d in enumerate(exp_dict['fdr_idxs'][i]):
            axes[1].plot(exp_dict['cum_nouts'][i][d],exp_dict['prec_rem_onstep'][i][d],c=cmap[x],marker='o')
        axes[1].set_xlabel('N spikes removed (count)')
        axes[1].set_ylabel('Precision of removed spikes')
        axes[1].set_title('Unit %d'%(clu))
        fig.tight_layout()
        plt.draw()
        pp.savefig(plt.gcf())
        
def generate_splitter_plots(exp_dict,pp):
    # Prepare arrays for plotting
    f1_ba_arr = np.array(exp_dict['f1_ba'])
    f1_diff_arr = (f1_ba_arr[:,1]-f1_ba_arr[:,0])/(1-f1_ba_arr[:,0])

    nclusts=15
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

maindir = 'C:\\Users\\black\\Desktop\\*\\'

dirs = glob(maindir)
filt = np.nonzero(['eel6' in x for x in dirs])[0]
datasets = [dirs[x] for x in filt]
hyb_data_dirs = []
for directory in datasets:
    this_dir = directory+'*\\'
    this_glob = glob(this_dir)
    good_dirs = ['_bu']
    filt = np.nonzero([np.any([x in d for x in good_dirs]) for d in this_glob])[0]
    hyb_datasets = [this_glob[x] for x in filt]
    hyb_data_dirs.extend(hyb_datasets)

split_res = []
out_res = []
cont = False
for directory in hyb_data_dirs:  
    if 'bu' in directory:
        files = os.listdir(directory)
        filts = np.nonzero(['_res.npy' in x for x in files])[0]
        res_files = [files[x] for x in filts]
        print('Found %s in dir: %s'%(res_files[0],directory))
        split_res.append((np.load(directory+res_files[0],allow_pickle=True,))[()])
        print(split_res)

split_dict = {}
for result in split_res:
    for key in list(result.keys()):
        try:
            split_dict[key].extend(result[key])
        except:
            split_dict.update({key:[]})
            split_dict[key].extend(result[key])
            
pp = PdfPages('summary.pdf')

generate_splitter_plots(split_dict,pp)


pp.close()