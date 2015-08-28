import numpy as np

# .ms file name
oms_name = 'testspec'

# Use CASA table tools to get columns of UVW, DATA, WEIGHT, etc.
tb.open('test_data/'+oms_name+'.ms')
data    = tb.getcol("DATA")
flag    = tb.getcol("FLAG")
uvw     = tb.getcol("UVW")
weight  = tb.getcol("WEIGHT")
spw     = tb.getcol("DATA_DESC_ID")
field   = tb.getcol("FIELD_ID")
time_st = tb.getcol("TIME")
ant1    = tb.getcol("ANTENNA1")
ant2    = tb.getcol("ANTENNA2")
tb.close()

# break out the u, v spatial frequencies
u = uvw[0,:]
v = uvw[1,:]

# identify number of channels
nchan = np.shape(data)[1]

# until CASA records spectral-dependence in its weights, assign the same 
# weight to each spectral channel
sp_wgt = np.zeros_like(data.real)
for i in range(nchan): sp_wgt[:,i,:] = weight

print(np.shape(sp_wgt), np.shape(data), np.shape(flag))

print(flag[0,:,40])
print(sp_wgt[0,:,40])
print(data.real[0,:,40])

# (weighted) average the polarizations
Re = np.sum(data.real*sp_wgt, axis=0) / np.sum(sp_wgt, axis=0)
Im = np.sum(data.imag*sp_wgt, axis=0) / np.sum(sp_wgt, axis=0)
Wt = np.sum(sp_wgt, axis=0) 

# toss out the autocorrelation placeholders
#xc = np.where(ant1 != ant2)

# output to numpy file
os.system('rm -rf '+oms_name+'.vis.npz')
np.savez(oms_name+'.vis', u=u, v=v, Re=Re, Im=Im, Wt=Wt)
#np.savez(oms_name+'.vis', u=u[xc], v=v[xc], Re=Re[xc], Im=Im[xc], Wt=Wt[xc])
