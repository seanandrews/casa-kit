import numpy as np

# .ms file name
oms_name = 'testspec'

# Use CASA table tools to get columns of UVW, DATA, WEIGHT, etc.
tb.open(oms_name+'.ms')
data    = tb.getcol("DATA")
uvw     = tb.getcol("UVW")
weight  = tb.getcol("WEIGHT")
spw     = tb.getcol("DATA_DESC_ID")
field   = tb.getcol("FIELD_ID")
time_st = tb.getcol("TIME")
ant1    = tb.getcol("ANTENNA1")
ant2    = tb.getcol("ANTENNA2")
tb.close()
data = np.squeeze(data)

# break out the u, v spatial frequencies
u = uvw[0,:]
v = uvw[1,:]

# polarization averaging
Re_xx = data[0,:].real
Re_yy = data[1,:].real
Im_xx = data[0,:].imag
Im_yy = data[1,:].imag
wei_xx = weight[0,:]
wei_yy = weight[1,:]
# - weighted averages
Re = (Re_xx*wei_xx + Re_yy*wei_yy) / (wei_xx + wei_yy)
Im = (Im_xx*wei_xx + Im_yy*wei_yy) / (wei_xx + wei_yy)
Wt = (wei_xx + wei_yy)

# toss out the autocorrelation placeholders
#xc = np.where(ant1 != ant2)

# output to numpy file
os.system('rm -rf '+oms_name+'.vis.npz')
np.savez(oms_name+'.vis', u=u, v=v, Re=Re, Im=Im, Wt=Wt)
#np.savez(oms_name+'.vis', u=u[xc], v=v[xc], Re=Re[xc], Im=Im[xc], Wt=Wt[xc])
