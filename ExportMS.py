import numpy as np

# .ms file name
oms_path = 'test_data'
oms_name = 'spw12_avg'

# Use CASA table tools to get columns of UVW, DATA, WEIGHT, etc.
tb.open(oms_path+'/'+oms_name+'.ms')
data   = tb.getcol("DATA")
flag   = tb.getcol("FLAG")
uvw    = tb.getcol("UVW")
weight = tb.getcol("WEIGHT")
tb.close()

# Get the frequency information
tb.open(oms_path+'/'+oms_name+'.ms/SPECTRAL_WINDOW')
freq = tb.getcell("CHAN_FREQ")
tb.close()

# Get rid of any flagged columns 
flagged   = np.all(flag, axis=(0, 1))
unflagged = np.squeeze(np.where(flagged == False))
data   = data[:,:,unflagged]
weight = weight[:,unflagged]
uvw    = uvw[:,unflagged]

# Break out the u, v spatial frequencies (in meter units)
u = uvw[0,:]
v = uvw[1,:]

# Assign uniform spectral-dependence to the weights (pending CASA improvements)
sp_wgt = np.zeros_like(data.real)
for i in range(len(freq)): sp_wgt[:,i,:] = weight

# (weighted) average the polarizations
Re  = np.sum(data.real*sp_wgt, axis=0) / np.sum(sp_wgt, axis=0)
Im  = np.sum(data.imag*sp_wgt, axis=0) / np.sum(sp_wgt, axis=0)
Vis = np.squeeze(Re + 1j*Im)
Wgt = np.squeeze(np.sum(sp_wgt, axis=0))
#if (len(freq) == 1): freq = freq[0]

# output to numpy file
os.system('rm -rf '+oms_name+'.vis.npz')
np.savez(oms_name+'.vis', u=u, v=v, Vis=Vis, Wgt=Wgt, freq=freq[0])
