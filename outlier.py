from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from PyQt5 import QtCore,QtGui,QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from scipy.spatial.distance import mahalanobis
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd
import scipy.cluster.vq as scc
from scipy.stats import zscore
from matplotlib.figure import Figure
from matplotlib import colors
import matplotlib
import numpy as np
import warnings
import sys

warnings.filterwarnings("ignore")
matplotlib.pyplot.style.use('fivethirtyeight')

a1=0
b1=0
c1=0
rem_spike = []
out_ext = []
spike_ref=[]
spike_ref_subw=[]
spike_ref_subs=[]
spike_inds=[]
mstdict={}
splits={}

#update display for chunk selection and outliers
def onselect(event=None):
    global colors
    global spike_ref
    global spike_ref_subs
    global spike_ref_subw
    global a1,b1,c1,splits
    lb_ind = screen.list2.selectedItems()
    lb_val = [screen.list2.indexFromItem(x).row()+1 for x in lb_ind]
    if lb_ind:
        ax1.clear()
        ax2.clear()
        ax3.clear()
        colors=[]
        spike_ref.clear()
        spike_ref_subw.clear()
        spike_ref_subs.clear()
        spike_inds.clear()
        
        for i,d in enumerate(k):
            if d in lb_val:
                spike_inds.extend(np.nonzero(np.in1d((spikes),(splits[d])))[0])
        clean_sp = spike_inds[:]
        np.random.shuffle(spike_inds) 
        spike_ref=spike_inds[:]
        #waveforms
        if len(spike_ref)<=200:
            sub_slice = slice(0,len(spike_ref),1)
        else:
            sub_slice = slice(0,len(spike_ref),int(np.around(len(spike_ref)/200)))
        #scatter
        if len(spike_ref)<=3000:
            sub_slice1 = slice(0,len(spike_ref),1)
        else:
            sub_slice1 = slice(0,len(spike_ref),int(np.around(len(spike_ref)/3000)))
        
        sub_indw = np.array(spike_inds[sub_slice])
        spike_ref_subw.extend(sub_indw)
        sub_inds = np.array(spike_inds[sub_slice1])
        spike_ref_subs.extend(sub_inds)
        
        colors=get_colors(screen.xcb.currentText(),screen.ycb.currentText(),screen.zcb.currentText(),sub_inds)
        colors1=get_colors(screen.xcb.currentText(),screen.ycb.currentText(),screen.zcb.currentText(),sub_indw)
        
        ax1.set_prop_cycle(color=colors1)
        p1=ax1.plot(mstdict['w'][:,sub_indw],linewidth=.1)
        p2=ax2.scatter(mstdict[screen.xcb.currentText()][sub_inds],
            mstdict[screen.ycb.currentText()][sub_inds],s=0.7,c=colors)
        ax3.axvline(x=0.002,color='red',alpha=.5)
        ax3.hist(np.diff(mstdict['Time'][clean_sp]),bins=np.linspace(0.0,.025,51),color='k')
        
        pcs.update({1: p1,2: p2})
        szs = np.ones(len(spike_ref_subs))
        for i,d in enumerate(out_ext):
            if (d in spike_ref_subs):
                isubs = int((np.nonzero(np.in1d(spike_ref_subs,d))[0]))
                colors[isubs]=(0,0,0,1)
                szs[isubs]+=1.5
        pcs[2].set_color(colors)
        pcs[2].set_sizes(szs)

        if np.array(out_ext).any():
            ct = len(spike_ref_subw)

            if sum(out_ext>0)<=100:
                sub_slice2 = slice(0,len(out_ext),1)
            else:
                sub_slice2 = slice(0,len(out_ext),int(np.around(len(out_ext)/100)))
            
            for i,d in enumerate(out_ext[sub_slice2]):

                if spikes[d] not in rem_spike:
                    rem_spike.append(spikes[d])
                
                ax2.scatter(np.array(mstdict[screen.xcb.currentText()])[d],
                    np.array(mstdict[screen.ycb.currentText()])[d],s=2.5,c='k')
                
                if (spikes[d] in spike_ref_subw):
                    isubw = int((np.nonzero(np.in1d(spike_ref_subw,spikes[d]))[0]))
                    pcs[1][isubw].set_lw(.2)
                    pcs[1][isubw].set_zorder(ct)
                    pcs[1][isubw].set_color('k')
                else:
                    ax1.plot(mstdict['w'][:,d],linewidth=.2,color='k',zorder=ct)
                ct-=1
        ypos=(ax1.get_ylim()[0])*.95
        for i in np.arange(len(chan)):
            ax1.axvline(x=i*wavewin,color='black',alpha=1)
            ax1.text(x=((i)*wavewin)+wavewin/2.4,y=ypos,s=("Ch. "+str(chan[i])),size='small',alpha=1.,c='black')

        
        screen.canvas1.draw()
        screen.canvas2.draw()
        screen.canvas3.draw()
        a1=len(spike_inds)
        b1=len(out_ext)
        c1=a1-b1
        screen.msg.setText("Number of outliers: "+str(b1)+
            "\nNumber of inliers: "+str(c1)+
            "\nTotal: " + str(a1))

# to find outliers
def find_out():
    global out_ext
    global rem_spike
    global colors
    global a1,b1,c1
    onselect(None)
    if np.array(out_ext).any():
        outs = list(out_ext)
    else:
        outs=[]
    
    contam_f = float(screen.ten.text())
    lb_ind = screen.list2.selectedItems()
    lb_val = [screen.list2.indexFromItem(x).row()+1 for x in lb_ind]
    lb_ind1 = screen.list1.selectedItems()
    lb_val1 = [x.text() for x in lb_ind1]
    if len(lb_val1)>1:
        for i,d in enumerate(lb_val):
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
        screen.msg.setText("Number of outliers: "+str(nouts)+
            "\nNumber of inliers: "+str(somesum-nouts)+
            "\nTotal: " + str(somesum))
        szs = np.ones(len(spike_ref_subs))
        for i,d in enumerate(outs):
            if (d in spike_ref_subs):
                isubs = int((np.nonzero(np.in1d(spike_ref_subs,d))[0]))
                colors[isubs]=(0,0,0,1)
                szs[isubs]+=1.5
        pcs[2].set_color(colors)
        pcs[2].set_sizes(szs)

        if outs.any():
            ct = len(outs)
            
            if sum(outs>0)<=100:
                sub_slice1 = slice(0,len(outs),1)
            else:
                sub_slice1 = slice(0,len(outs),int(np.around(len(outs)/100)))
            
            for i,d in enumerate(outs[sub_slice1]):
                
                if spikes[d] not in rem_spike:
                    rem_spike.append(spikes[d])
                
                ax2.scatter(np.array(mstdict[screen.xcb.currentText()])[d],
                    np.array(mstdict[screen.ycb.currentText()])[d],s=2.5,c='k')
                
                if (spikes[d] in spike_ref_subw):
                    isubw = int((np.nonzero(np.in1d(spike_ref_subw,spikes[d]))[0]))
                    pcs[1][isubw].set_lw(.2)
                    pcs[1][isubw].set_zorder(ct)
                    pcs[1][isubw].set_color('k')
                else:
                    ax1.plot(mstdict['w'][:,d],linewidth=.2,color='k',zorder=ct)
                ct-=1

            out_ext=list(out_ext)
            for x in outs:
                if x not in out_ext:
                    out_ext.append(x)
            out_ext=np.array(out_ext)

            screen.canvas1.draw()
            screen.canvas2.draw()

        else:
            screen.msg.setText("Number of outliers: "+str(b1)+
                "\nNumber of inliers: "+str(c1)+
                "\nTotal: " + str(a1))
    else:
        screen.msg.setText("Select at least \ntwo features for O.D.!")

# get color based on axis limits for display
def get_colors(x,y,z,inds):
    global mstdict
    xmin=np.percentile(mstdict[x][inds],5)
    xmax=np.percentile(mstdict[x][inds],95)
    ymin=np.percentile(mstdict[y][inds],5)
    ymax=np.percentile(mstdict[y][inds],95)
    zmin=np.percentile(mstdict[z][inds],5)
    zmax=np.percentile(mstdict[z][inds],95)
    cs=[]
    cs.append((-mstdict[screen.xcb.currentText()][inds]+xmax)/(xmax-xmin))
    cs.append((mstdict[screen.ycb.currentText()][inds]-ymin)/(ymax-ymin))
    cs.append((mstdict[screen.zcb.currentText()][inds]-zmin)/(zmax-zmin))
    cs.append([1 for x in np.arange(len(inds))])
    cs=(np.transpose(cs))
    cs[(cs<0)]=0
    cs[(cs>1)]=1
    return cs

def _quit():
    screen.close()

def _split():
    lb_ind = screen.list2.selectedItems()
    lb_val = [screen.list2.indexFromItem(x).row()+1 for x in lb_ind]
    spikes_tsplit = []
    for i,d in enumerate(k):
        if d in lb_val:
            spikes_tsplit.extend(splits[d])
    s.actions.split(spikes_tsplit)
    screen.close()

def fix_axis(ax,textsize):
    ax.tick_params(axis='x', labelsize=textsize)
    ax.tick_params(axis='y', labelsize=textsize)
    ax.xaxis.label.set_color('black')
    ax.tick_params(axis='x', colors='black')
    ax.tick_params(axis='y', colors='black')
    ax.axis('on')

def get_spikes(result):
    nchans,ngs,wslice = result
    cid = s.selected
    data_raw = c._amplitude_getter(cid, name='raw',load_all=True)[0]
    data_template = c._amplitude_getter(cid, name='template',load_all=True)[0]
    spikes=data_raw.spike_ids
    nspikes = len(spikes)
    if ngs!=1:
        for i in np.arange(ngs):
            splits.update({i+1: spikes[(i*nspikes)//ngs:((i+1)*nspikes)//ngs]})
    else:
        splits.update({1: spikes})

    spike_times = m.spike_times[spikes]
    spike_amps = data_raw.amplitudes
    spike_temps = data_template.amplitudes
    
    cid = cid[0]
    
    chan = c.get_best_channels(cid)[0:nchans]
    waves = np.asarray(c._get_waveforms_with_n_spikes(cid,nspikes,1).data)
    wavewin = waves.shape[1]
    wslice = wslice if wslice.stop<wavewin else slice(wslice.start,wavewin,1)
    wave=waves[:,wslice,0:nchans]
    waves_g = []
    
    for i in np.arange(nspikes):
        temp = waves[i,:,:]
        temp2 = []
        for j in np.arange(len(chan)):
            temp2.extend(temp[:,j])
        waves_g.append(temp2)
    
    waves_g=np.transpose(np.array(waves_g))
    features = m.get_features(spikes,chan)[:,0:nchans,0:3]
    
    (mstdict.update({'Time': np.array(spike_times),'Amplitude': np.array(spike_amps),
        'w' : waves_g ,'Template' : np.array(spike_temps)}))
    for i,d in enumerate(chan):
        mstdict.update({"PC0 Ch. "+str(d): np.array(features[:,:,0])[:,i]})
        mstdict.update({"PC1 Ch. "+str(d): np.array(features[:,:,1])[:,i]})
        mstdict.update({"PC2 Ch. "+str(d): np.array(features[:,:,2])[:,i]})
    return (cid,spikes,nspikes,ngs,chan,splits,mstdict,wavewin)

# functions to reset outlier selection 
def exc_out(spikes_rem=None):
    global out_ext
    global rem_spike
    sr=[]
    if spikes_rem is not None: 
        sr=spikes_rem
    elif len(out_ext)>=len(rem_spike):
        sr=out_ext
    elif len(out_ext)<=len(rem_spike):
        sr=rem_spike2
    sr=np.array(sr)
    lb_ind = screen.list2.selectedItems()
    lb_val = [screen.list2.indexFromItem(x).row()+1 for x in lb_ind]
    for i,d in enumerate(k):
        a = (splits[d])
        spike_ind_rem=np.nonzero(np.in1d((a),spikes[sr]))
        b = np.delete(a,spike_ind_rem)
        splits.update({d: (b)})
    rem_spike.clear()
    out_ext = np.delete(out_ext,slice(0,len(out_ext),1))
    onselect(None)

# and to remove the outliers from data
def clear_out():
    global out_ext
    global rem_spike
    rem_spike.clear()
    out_ext = np.delete(out_ext,slice(0,len(out_ext),1))
    onselect(None)

class suDialog(QDialog):
    def __init__(self):
        QDialog.__init__(self)
        self.setWindowTitle('Outlier Rejector: Cluster '+str(s.selected[0]))
        layout = QGridLayout()
        self.l1 = QLabel('# of channels:')
        self.e1 = QLineEdit('6')
        self.l2 = QLabel('# of chunks:')
        self.e2 = QLineEdit('50')
        self.l3 = QLabel('Waveform window:')
        self.e3 = QLineEdit('0')
        self.e4 = QLineEdit('82')
        self.b1 = QPushButton('OK', parent=self)
        self.b2 = QPushButton('Cancel', parent=self)
        layout.addWidget(self.l1,0,0)
        layout.addWidget(self.e1,0,1,1,2)
        layout.addWidget(self.l2,1,0)
        layout.addWidget(self.e2,1,1,1,2)
        layout.addWidget(self.l3,2,0)
        layout.addWidget(self.e3,2,1)
        layout.addWidget(self.e4,2,2)
        layout.addWidget(self.b1,3,1)
        layout.addWidget(self.b2,3,2)
        self.b1.clicked.connect(lambda: self.ok())
        self.b2.clicked.connect(lambda: self.cancel())
        self.setLayout(layout)
        self.canceled=True

    def cancel(self):
        self.canceled=True
        self.close()
    
    def ok(self):
        self.canceled=False
        self.close()
    
    def get_entry(self):
        nchans=int(self.e1.text())
        nchunks=int(self.e2.text())
        strt=int(self.e3.text())
        stp=int(self.e4.text())
        if strt>=81:
            strt=81
        if stp>=82:
            stp=82
        wslice = slice(strt,stp,1)
        return nchans,nchunks,wslice

class Window(QWidget):
    def __init__(self):
        QWidget.__init__(self)
        self.setWindowTitle('Outlier Rejector: Cluster '+str(s.selected[0]))
        self.font1 = QFont('Helvetica',11)
        self.font = QFont('Helvetica',9)
        self.figure1 = Figure(figsize=(2,5),dpi=110)
        self.canvas1 = FigureCanvas(self.figure1)
        self.figure2 = Figure(figsize=(5,1),dpi=150)
        self.canvas2 = FigureCanvas(self.figure2)
        self.figure3 = Figure(figsize=(5,1),dpi=150)
        self.canvas3 = FigureCanvas(self.figure3)
        self.ax1=self.figure1.add_subplot(111)
        self.ax2=self.figure2.add_subplot(111)
        self.ax3=self.figure3.add_subplot(111)
        self.figure2.subplots_adjust(top=.94,bottom=0.1)
        self.figure3.subplots_adjust(top=.94,bottom=0.1)
        layout1 = QGridLayout()
        layout1.setRowStretch(0,1.2)
        layout1.setRowStretch(1,1)
        layout1.setRowStretch(2,.3)
        layout1.addWidget(self.canvas1,0,0,2,3)
        layout1.addWidget(self.canvas2,2,0,1,1)
        layout1.addWidget(self.canvas3,2,2,1,1)
        layout2 = QGridLayout()
        layout2.setColumnStretch(0,1.3)
        layout2.setColumnStretch(1,0)
        layout2.setColumnStretch(2,1.1)
        layout1.addLayout(layout2,2,1,1,1)
        self.list1 = QListWidget(parent=self)
        self.list2 = QListWidget(parent=self)
        self.l1label = QLabel('Features for outlier rejection: ')
        self.l1label.setFont(self.font)
        self.l2label = QLabel('Time Segments: ')
        self.l2label.setFont(self.font)
        self.b1 = QPushButton('Select all', parent=self)
        self.b1.clicked.connect(lambda: self.select_all1())
        self.b2 = QPushButton('Deselect all', parent=self)
        self.b2.clicked.connect(lambda: self.deselect_all1())
        self.b3 = QPushButton('Select all', parent=self)
        self.b3.clicked.connect(lambda: self.select_all2())
        self.b4 = QPushButton('Deselect all', parent=self)
        self.b4.clicked.connect(lambda: self.deselect_all2())
        self.b5 = QPushButton('Split', parent=self)
        self.b5.clicked.connect(lambda: _split())
        self.b6 = QPushButton('Quit', parent=self)
        self.b6.clicked.connect(lambda : _quit())
        layout2.addWidget(self.l1label,0,1)
        layout2.addWidget(self.list1,1,1)
        layout2.addWidget(self.l2label,0,2)
        layout2.addWidget(self.list2,1,2)
        layout2.addWidget(self.b1,3,1)
        layout2.addWidget(self.b2,4,1)
        layout2.addWidget(self.b3,3,2)
        layout2.addWidget(self.b4,4,2)
        layout2.addWidget(self.b5,5,1)
        layout2.addWidget(self.b6,5,2)
        layout3 = QGridLayout()
        layout2.addLayout(layout3,1,0,4,1)
        self.otitle = QLabel("Outlier detection: ")
        self.otitle.setFont(self.font)
        self.mlb = QLabel("Method: ")
        self.mcb = QComboBox()
        self.mcb.setEditable(False)
        self.mcb.addItem('Mahalanobis')
        self.mcb.setCurrentIndex(0)
        self.xcb = QComboBox()
        self.xcb.setEditable(False)
        self.ycb = QComboBox()
        self.ycb.setEditable(False)
        self.zcb = QComboBox()
        self.zcb.setEditable(False)
        self.tlb = QLabel("Threshold: ")
        self.ten = QLineEdit('5')
        self.adlb = QLabel("Axes Display:")
        self.adlb.setAlignment(QtCore.Qt.AlignCenter)
        self.xalb = QLabel("X-axis (red):")
        self.yalb = QLabel("Y-axis (green):")
        self.calb = QLabel("Color (blue):")
        self.b7 = QPushButton('Find outliers', parent=self)
        self.b7.clicked.connect(lambda: find_out())
        self.b8 = QPushButton('Reset selection', parent=self)
        self.b8.clicked.connect(lambda: clear_out())
        self.b9 = QPushButton('Delete outliers', parent=self)
        self.b9.clicked.connect(lambda : exc_out())
        layout2.addWidget(self.otitle,0,0)
        layout3.addWidget(self.mlb,1,0)
        layout3.addWidget(self.tlb,2,0)
        layout3.addWidget(self.mcb,1,1)
        layout3.addWidget(self.ten,2,1)
        layout3.addWidget(self.b7,3,0,1,2)
        layout3.addWidget(self.b8,4,0,1,2)
        layout3.addWidget(self.b9,5,0,1,2)
        layout3.addWidget(self.adlb,6,0,1,2)
        layout3.addWidget(self.xalb,7,0)
        layout3.addWidget(self.yalb,8,0)
        layout3.addWidget(self.calb,9,0)
        layout3.addWidget(self.xcb,7,1)
        layout3.addWidget(self.ycb,8,1)
        layout3.addWidget(self.zcb,9,1)
        for i in np.arange(0,2):
            layout3.setColumnStretch(i,0)
        self.msg = QLabel("")
        self.msg1 = QLabel("")
        self.msg.setFont(self.font)
        self.msg.setAlignment(QtCore.Qt.AlignCenter)
        self.nslb = QLabel("Min. # spk.: ")
        self.nse = QLineEdit('1000')
        self.nclb = QLabel("Max # time seg.: ")
        self.nce = QLineEdit('10')
        layout3.addWidget(self.nslb,10,0,1,2)
        layout3.addWidget(self.nse,10,1,1,2)
        layout3.addWidget(self.nclb,11,0,1,2)
        layout3.addWidget(self.nce,11,1,1,2)
        layout3.addWidget(self.msg1,12,0,1,2)
        layout3.addWidget(self.msg,13,0,1,2)
        self.setLayout(layout1)

        self.list1.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.list2.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.list2.itemClicked.connect(lambda: onselect())
        self.list2.itemSelectionChanged.connect(lambda: onselect())
        self.list2.currentRowChanged.connect(lambda: onselect())
        self.xcb.currentTextChanged.connect(lambda: onselect())
        self.ycb.currentTextChanged.connect(lambda: onselect())
        self.zcb.currentTextChanged.connect(lambda: onselect())

    def select_all1(self):
        nitems=self.list1.count()
        self.list1.blockSignals(True)
        for i in np.arange(nitems):
            if i==nitems-1:
                self.list1.blockSignals(False)
            self.list1.item(i).setSelected(True)

    def select_all2(self):
        nitems=self.list2.count()
        self.list2.blockSignals(True)
        for i in np.arange(nitems):
            if i==nitems-1:
                self.list2.blockSignals(False)
            self.list2.item(i).setSelected(True)

    def deselect_all1(self):
        nitems=self.list1.count()
        self.list1.blockSignals(True)
        for i in np.arange(nitems):
            if i==nitems-1:
                self.list1.blockSignals(False)
            self.list1.item(i).setSelected(False)

    def deselect_all2(self):
        nitems=self.list2.count()
        self.list2.blockSignals(True)
        for i in np.arange(nitems):
            if i==nitems-1:
                self.list2.blockSignals(False)
            self.list2.item(i).setSelected(False)

# Check if a previous instance of app is lingering -> means they cant be run simultaneously
if 'screen' in dir():
    _quit()

# Start-up
dialog = suDialog()
dialog.exec()


if not dialog.canceled:
    # Get parameters
    res = dialog.get_entry()
    (cid,spikes,nspikes,ngs,chan,splits,mstdict,wavewin) = get_spikes(res)
    colors = []

    k = splits.keys()
    pcs = {}

    screen = Window()
    screen.show()
    ax1=screen.ax1
    ax2=screen.ax2
    ax3=screen.ax3
    fix_axis(ax1,12)
    fix_axis(ax2,8)
    fix_axis(ax3,8)

    plot_k = [*mstdict]
    plot_k.remove('w')
    chunks = [str(i+1) for i in np.arange(ngs)]
    screen.list1.addItems(plot_k)
    screen.list2.addItems(chunks)
    screen.xcb.addItems(plot_k)
    screen.ycb.addItems(plot_k)
    screen.zcb.addItems(plot_k)
    screen.xcb.setCurrentIndex(0)
    screen.ycb.setCurrentIndex(1)
    screen.zcb.setCurrentIndex(2)
    screen.list2.blockSignals(True)
    for i in np.arange(ngs):
        if i==ngs-1:
            screen.list2.blockSignals(False)
        screen.list2.item(i).setSelected(True)
    screen.show()
else:
    sys.exit(0)
