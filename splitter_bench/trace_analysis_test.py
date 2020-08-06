from phylib.io.traces import get_ephys_reader
import numpy as np
import os
chan = 49
args = ['sample_rate', 'n_channels_dat', 'dtype', 'offset']
vals = [[25000],[64],[np.int16],[0]]
kwargs = {args[x]: vals[x][0] for x in range(len(args))}
data_dir = os.getcwd()+r'\ConcatenatedData_Probe1.bin'

traces = get_ephys_reader(data_dir, **kwargs)
pbounds = slice(traces.part_bounds[0],traces.part_bounds[1])
trace = traces._get_part(pbounds,0)
trace = trace[:,chan]
plt.plot(trace[0:100000])
plt.show()