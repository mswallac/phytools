from PyQt5.QtWidgets import *
from PyQt5 import QtCore,QtGui,QtWidgets
from PyQt5.QtGui import *
import sys,os
import matplotlib
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import scipy.cluster.vq as scc
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd
import math,time
import numpy as np
import warnings

# Variable to keep mode from log search
mode = ''

# Get data path in correct format
path = str(m.dir_path)
path = str(np.core.defchararray.replace(np.array(path),"\\","/"))

# Uses phy.log to check which amplitude type is being used
try:
    t=time.time()
    f = open(path+'/phy.log','r')
    contents = np.array(f.read().splitlines())
    contents_mode = (np.nonzero([('amplitudes type:' in line) for i,line in enumerate(contents)])[0])
    try:
        contents_mode = contents_mode[-1]
    except IndexError:
        print('Hit \'A\' once in order to use the splitter plugin!')
        sys.exit(0)
    mode=contents[contents_mode].split()
    mode=mode[mode.index('type:')+1].split('.')[0]
except OSError:       
    sys.stderr.write("splitter.py: phy.log not found in: "+str(path)+"\n")
    sys.exit(0)

warnings.filterwarnings("ignore")
matplotlib.pyplot.style.use('fivethirtyeight')
cluster_done = False

def fix_axis(ax,textsize):
    ax.tick_params(axis='x', labelsize=textsize)
    ax.tick_params(axis='y', labelsize=textsize)
    ax.xaxis.label.set_color('black')
    ax.tick_params(axis='x', colors='black')
    ax.tick_params(axis='y', colors='black')
    ax.axis('on')

def _quit():
    screen.close()

# Split current selection into two clusters (highlighted/not highlighted)
def _split():
    lb_ind = screen.list2.selectedItems()
    lb_val = [screen.list2.indexFromItem(x).row()+1 for x in lb_ind]
    spikes_tsplit = []
    for i,d in enumerate(k):
        if d in lb_val:
            spikes_tsplit.extend(splits[d])
    s.actions.split(spikes_tsplit)
    screen.close()

def _reset():
    ax1.clear()
    ax2.clear()
    ax3.clear()
    screen.list2.clear()
    k.clear()
    pcs.clear()
    splits.clear()
    colors.clear()
    screen.canvas1.draw()
    screen.canvas2.draw()
    screen.canvas3.draw()

def _cluster():
    global wavewin,cluster_done
    if not cluster_done:
        cluster_done=True
    else:
        _reset()
    ypos=0
    if k or colors or splits:
        k.clear()
        splits.clear()
    lb_ind = screen.list1.selectedItems()
    lb_val = [x.text() for x in lb_ind]
    if len(lb_val)<1:
        screen.msg.setText("WARNING: Select at least one feature!")
    else:
        screen.msg.setText("")
        features=[]
        for i,d in enumerate(lb_val):
            features.append(mstdict[d][:])
        obs = scc.whiten(np.transpose(features))
        pd = ssd.pdist(obs,metric='euclidean')
        nclusts = int(screen.ncfield.text())
        Z = sch.ward(pd)
        l = sch.fcluster(Z,t=nclusts,criterion='maxclust')
        colors.extend([cmap1(i) for i in np.linspace(0, .99, nclusts//2)])
        colors.extend([cmap2(i) for i in np.linspace(0, .99, 1+nclusts//2)])
        for i in np.arange(1,nclusts+1):
            inter1=np.nonzero(np.in1d(l,i))[0]
            splits.update({(i): spikes[inter1]})
            screen.list2.addItem(str(i)+": ("+str(len(splits[i]))+")")
        k.extend(splits.keys())
        for i,d in enumerate(k):
            colr=[]
            temp=(np.array(colors[i])*255)
            for bb,d in enumerate(temp):
                if bb<3:
                    colr.append(int(d))
            hexcol=('#%02x%02x%02x' % tuple(colr))
            screen.list2.item(i).setForeground(QColor(hexcol))
        ct=2*len(chan)+len(k)
        for i,d in enumerate(k):
            spike_inds=np.nonzero(np.in1d((sub_spikes),(splits[d])))[0]
            p1=ax1.scatter((mstdict['Time'][spike_inds]),(mstdict[mode][spike_inds]),s=2.8,color=colors[i])
            p1.set_alpha(.15)
            ws=mstdict['w'][:,spike_inds]
            mws=np.mean(ws,axis=1)
            meanw_min = np.amin(mws)
            if meanw_min<ypos:
                ypos = meanw_min 
            p2=ax3.plot(mws,color=colors[i],linewidth=1.5,zorder=ct)
            ct-=1
            pcs.append({1: p1,2: p2})
        leg=ax1.legend(k,prop={'size':8},ncol=len(k)//3,fancybox=True)
        for text in leg.get_texts():
            matplotlib.pyplot.setp(text,color='k')
        for lh in leg.legendHandles: 
            lh.set_alpha(1.)
        for i in np.arange(len(chan)):
            ax3.axvline(x=i*wavewin,color='black',alpha=.8,zorder=ct)
            ct-=1
            ax3.text(x=((i)*wavewin)+wavewin//2.4,y=ypos,s=("Ch. "+str(chan[i])),size='small',alpha=1.,zorder=ct,c='k')
            ct-=1
        screen.canvas1.draw()
        screen.canvas2.draw()
        screen.canvas3.draw()

def onselect():
    global wavewin
    lb_ind = screen.list2.selectedItems()
    lb_val = [screen.list2.indexFromItem(x).row()+1 for x in lb_ind]
    ax2.clear()
    ax2.axvline(x=0.002,color='red',alpha=.5)
    histdata=[]        
    ct=len(k)+2*len(chan)
    for i,d in enumerate(k):
        if d in lb_val:
            spike_inds=np.nonzero(np.in1d((spikes),(splits[d])))[0]
            histdata.extend(np.diff(mstdict['Time'][spike_inds]))
            (pcs[i][1]).set_alpha(0.96)
            (pcs[i][2][0]).set_color(colors[i])
            (pcs[i][2][0]).set_linewidth(2)
            (pcs[i][2][0]).set_zorder(ct)
            ct-=1
            colr=[]
            temp=(np.array(colors[i])*255)
            for bb,d in enumerate(temp):
                if bb<3:
                    colr.append(int(d))
            hexcol=('#%02x%02x%02x' % tuple(colr))
            screen.list2.item(i).setForeground(QColor(hexcol))
        else:
            (pcs[i][1]).set_alpha(.1)
            (pcs[i][2][0]).set_color([0.05,0.05,0.05,0.9])
            (pcs[i][2][0]).set_linewidth(1.5)
            colr=[]
            temp=(np.array(colors[i])*255)
            for bb,d in enumerate(temp):
                if bb<3:
                    colr.append(int(d))
            hexcol=('#%02x%02x%02x' % tuple(colr))
            screen.list2.item(i).setForeground(QColor(hexcol))
    for i in np.arange(len(chan)):
        ax3.axvline(x=i*wavewin,color='black',alpha=.8,zorder=ct)
        ct-=1
    ax2.hist(histdata,bins=np.linspace(0,.025,51),color='black')
    screen.canvas1.draw()
    screen.canvas2.draw()
    screen.canvas3.draw()

def get_spikes():
    global wavewin,mode,ch
    cid = s.selected[0] if type(s.selected)=='list' else s.selected
    spikes = m.get_cluster_spikes(cid)
    try:
        channel = c.selection.channel_id
    except AttributeError:
        channel = c.get_best_channels(cid[0])[0]
    av=(c._get_amplitude_functions())[mode]
    data=av(spikes, channel_id=channel, channel_ids=np.array(channel), load_all=True, first_cluster=cid[0])
    mode = mode+' - Ch. '+str(channel)
    nspikes = len(spikes)
    if nspikes>12000:
        sub_spikes = spikes[0::int(nspikes/12000)]
    else:
        sub_spikes = spikes
    sub_spikes1=(np.nonzero(np.in1d(spikes,sub_spikes)))[0]
    cid = cid[0]
    chan = c.get_best_channels(cid)[0:5]
    spike_times = m.spike_times[sub_spikes1]
    spike_amps = data[sub_spikes1]
    spike_times_f = m.spike_times[spikes]
    spike_amps_f = data
    waves = np.asarray(c._get_waveforms_with_n_spikes(cid,nspikes,1).data)[:,:,0:len(chan)]
    wavewin = waves.shape[1]
    waves_g = []
    for i in np.arange(nspikes):
        temp = waves[i,:,:]
        temp2 = []
        for j in np.arange(len(chan)):
            temp2.extend(temp[:,j])
        waves_g.append(temp2)
    waves_g=np.transpose(np.array(waves_g))
    features = m.get_features(spikes,chan)[:,:,0:3]
    temp_w=[]
    temp_a=[]
    temp_t=[]
    temp_f0=[]
    temp_f1=[]
    temp_f2=[]
    mstdict={}
    temp_a.extend(spike_amps_f[:])
    temp_t.extend(spike_times_f[:])
    temp_f0.extend(features[:,:,0])
    temp_f1.extend(features[:,:,1])
    temp_f2.extend(features[:,:,2])
    mstdict.update({'Time': np.array(temp_t),mode: np.array(temp_a),'w' : waves_g})
    for i,d in enumerate(chan):
        mstdict.update({"PC0 - Ch. "+str(d): np.array(temp_f0)[:,i]})
        mstdict.update({"PC1 - Ch. "+str(d): np.array(temp_f1)[:,i]})
        mstdict.update({"PC2 - Ch. "+str(d): np.array(temp_f2)[:,i]})
    return (cid,spikes,nspikes,chan,mstdict,sub_spikes)

class Window(QWidget):
    def __init__(self):
        QWidget.__init__(self)
        self.setWindowTitle('Splitter: Cluster '+str(s.selected[0]))
        self.font = QFont('Helvetica',16)
        self.figure1 = Figure(figsize=(6,10),dpi=130)
        self.canvas1 = FigureCanvas(self.figure1)
        self.figure2 = Figure(figsize=(6,8),dpi=150)
        self.canvas2 = FigureCanvas(self.figure2)
        self.figure3 = Figure(figsize=(4,5),dpi=125)
        self.canvas3 = FigureCanvas(self.figure3)
        self.ax1=self.figure1.add_subplot(111)
        self.ax2=self.figure2.add_subplot(111)
        self.ax3=self.figure3.add_subplot(111)
        self.figure2.subplots_adjust(top=.94,bottom=0.1)
        self.figure3.subplots_adjust(top=.96,bottom=0.12)
        layout1 = QGridLayout()
        layout1.setColumnStretch(0,4)
        layout1.setColumnStretch(1,0)
        layout1.addWidget(self.canvas1,0,0,2,2)
        layout1.addWidget(self.canvas2,2,0,2,1)
        layout1.addWidget(self.canvas3,2,1,1,1)
        layout2 = QGridLayout()
        layout1.addLayout(layout2,3,1,1,1)
        self.list1 = QListWidget(self)
        self.list1.setFixedHeight(130)
        self.list2 = QListWidget(self)
        self.list2.setFixedHeight(130)
        self.l1label = QLabel('Features for Clustering:')
        self.l2label = QLabel('Generated Clusters:')
        self.b1 = QPushButton('Select all', parent=self)
        self.b1.clicked.connect(lambda: self.select_all1())
        self.b2 = QPushButton('Deselect all', parent=self)
        self.b2.clicked.connect(lambda: self.deselect_all1())
        self.b3 = QPushButton('Select all', parent=self)
        self.b3.clicked.connect(lambda: self.select_all2())
        self.b4 = QPushButton('Deselect all', parent=self)
        self.b4.clicked.connect(lambda: self.deselect_all2())
        self.b6 = QPushButton('Split', parent=self)
        self.b6.clicked.connect(lambda: _split())
        self.b7 = QPushButton('Quit', parent=self)
        self.b7.clicked.connect(lambda : _quit())
        layout2.addWidget(self.l1label,0,0)
        layout2.addWidget(self.list1,1,0)
        layout2.addWidget(self.l2label,0,1)
        layout2.addWidget(self.list2,1,1)
        layout2.addWidget(self.b1,3,0)
        layout2.addWidget(self.b2,4,0)
        layout2.addWidget(self.b3,3,1)
        layout2.addWidget(self.b4,4,1)
        layout3 = QGridLayout()
        layout2.addLayout(layout3,2,1,1,1)
        layout31 = QGridLayout()
        layout2.addLayout(layout31,2,0,1,1)
        self.b5 = QPushButton('Cluster!', parent=self)
        self.b5.clicked.connect(lambda : _cluster())
        self.nclabel = QLabel('# of clusters to generate: ')
        self.ncfield = QLineEdit('10',parent=self)
        layout3.addWidget(self.b5,0,0,1,1)
        layout31.addWidget(self.nclabel,0,0,1,1)
        layout31.addWidget(self.ncfield,0,1,1,1)
        layout4 = QGridLayout()
        layout1.addLayout(layout4,5,1,1,1)
        layout4.addWidget(self.b6,0,0)
        layout4.addWidget(self.b7,0,1)
        self.msg = QLabel("")
        self.msg.setFont(self.font)
        layout1.addWidget(self.msg,5,0,1,1)
        self.setLayout(layout1)

        self.list1.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.list2.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.list2.itemClicked.connect(lambda: onselect())
        self.list2.itemSelectionChanged.connect(lambda: onselect())
        self.list2.currentRowChanged.connect(lambda: onselect())

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
        onselect()

# Check if a previous instance of app is lingering -> means they cant be run simultaneously
if 'screen' in dir():
    _quit()


(cid,spikes,nspikes,chan,mstdict,sub_spikes) = get_spikes()
splits={}
k=[]

if cid:
    screen = Window()
    screen.show()
    ax3=screen.ax1
    ax1=screen.ax2
    ax2=screen.ax3
    fix_axis(ax1,8)
    fix_axis(ax2,7)
    fix_axis(ax3,8)

    fkeys = [*mstdict]
    fkeys.remove('w')
    fkeys.remove('Time')
    for i,d in enumerate(fkeys):
        screen.list1.addItem(d)
        if i<4:
            screen.list1.item(i).setSelected(True)

    cmap1 = matplotlib.pyplot.get_cmap("tab20")
    cmap2 = matplotlib.pyplot.get_cmap("Paired")
    colors = []
    pcs = []
    screen.show()