import numpy as np

# .ms file name
oms_path = 'test_data'
oms_name = 'testcont'

# Use CASA table tools to get columns of UVW, DATA, WEIGHT, etc.
tb.open(oms_path+'/'+oms_name+'.ms')
data   = tb.getcol("DATA")
flag   = np.invert(tb.getcol("FLAG"))	# want flagged data == FALSE (avoid)
uvw    = tb.getcol("UVW")
weight = tb.getcol("WEIGHT")
spw    = tb.getcol("DATA_DESC_ID")
tb.close()

# Get the frequency information
freqs=[]
tb.open(oms_path+'/'+oms_name+'.ms/SPECTRAL_WINDOW')
nchan = tb.getcol("NUM_CHAN")
for i in range(len(nchan)):
    chanfreq = tb.getcell("CHAN_FREQ", i)
    freqs.append(chanfreq)
tb.close()
print(freqs)

# break out the u, v spatial frequencies (in meter units)
u = uvw[0,:]
v = uvw[1,:]

# until CASA records spectral-dependence in its weights, assign the same 
# weight to each spectral channel
sp_wgt = np.zeros_like(data.real)
for i in range(nchan): sp_wgt[:,i,:] = weight

# (weighted) average the polarizations
# note that including the boolean flag here removes the contribution of flagged
# data from the averages; leaves flagged data in with zero weight (sort of 
# wasteful, but it should work generically); squeeze for continuum data
Re = np.squeeze(np.sum(flag*data.real*sp_wgt, axis=0) / np.sum(sp_wgt, axis=0))
Im = np.squeeze(np.sum(flag*data.imag*sp_wgt, axis=0) / np.sum(sp_wgt, axis=0))
Wt = np.squeeze(np.sum(flag*sp_wgt, axis=0))

# output to numpy file
os.system('rm -rf '+oms_name+'.vis.npz')
np.savez(oms_name+'.vis', u=u, v=v, Re=Re, Im=Im, Wt=Wt, freq=freqs)
