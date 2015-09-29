import numpy as np
import os
import time
import sys
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

# load some data
fname = 'spw12_avg'
visdata = np.load(fname+'.vis.npz')
freq = visdata['freq']
u = 1e-3*visdata['u']*freq/2.9979e8
v = 1e-3*visdata['v']*freq/2.9979e8
Vis = visdata['Vis']
Re = Vis.real
Im = Vis.imag
Wt = visdata['Wgt']
nvis = len(Re)

# compute the phase-center distances
Ruv = np.sqrt(u**2+v**2)

# sort based on phase-center distances
Ruv_sort_indices = np.argsort(Ruv)
Res = Re[Ruv_sort_indices]
Ims = Im[Ruv_sort_indices]
Wts = Wt[Ruv_sort_indices]
rho = Ruv[Ruv_sort_indices]

# compute a rolling mean from the visibility data 
window = 1000
rmean_Re = pd.rolling_mean(Res, window, center=True)
rmean_Im = pd.rolling_mean(Ims, window, center=True)

# extrapolate to edges using simple linear model
# short spacings
xfit  = rho[0.5*window:2.5*window]
Refit = rmean_Re[0.5*window:2.5*window]
Imfit = rmean_Im[0.5*window:2.5*window]
bb, aa, rval, pval, serr = stats.linregress(xfit,Refit)
rmean_Re[:0.5*window] = aa + bb*rho[:0.5*window]
bb, aa, rval, pval, serr = stats.linregress(xfit,Imfit)
rmean_Im[:0.5*window] = aa + bb*rho[:0.5*window]
# long spacings
xfit  = rho[nvis-2.5*window:nvis-0.5*window]
Refit = rmean_Re[nvis-2.5*window:nvis-0.5*window]
Imfit = rmean_Im[nvis-2.5*window:nvis-0.5*window]
bb, aa, rval, pval, serr = stats.linregress(xfit,Refit)
rmean_Re[nvis-0.5*window+1:] = aa + bb*rho[nvis-0.5*window+1:]
bb, aa, rval, pval, serr = stats.linregress(xfit,Imfit)
rmean_Im[nvis-0.5*window+1:] = aa + bb*rho[nvis-0.5*window+1:]

# subtract the rolling mean models (only scatter remains)
rm_Re = Res - rmean_Re
rm_Im = Ims - rmean_Im

# loop through each visibility and calculate the RMS scatter from neighboring
# visibilities (a purely empirical noise estimate; captures non-thermal noise)
nclump = 100		# number of neighboring visibilities used to get sigma
trunc_sep = 50.		# in kilolambda
sigma_scat = np.zeros_like(Ruv)
ri_scat = np.zeros_like(Ruv)
for i in np.arange(len(rm_Re)):
    tic = time.time()
    # calculate distances from this (u,v) point
    uvsep = ((u-u[i])**2 + (v-v[i])**2)
    # truncate list to minimize sorting overheads
    uvsep = uvsep[uvsep < trunc_sep**2]
    suvsep = np.sqrt((uvsep[np.argsort(uvsep)])[:nclump])
    # sort; select nclump nearest visibilities
    srt_Re = (rm_Re[np.argsort(uvsep)])[:nclump]
    srt_Im = (rm_Im[np.argsort(uvsep)])[:nclump]
    #srt_Vis = np.concatenate((srt_Re, srt_Im))
    #sRuv = np.concatenate((suvsep, suvsep))
    # calculate and store the standard deviation
    #mu  = np.average(srt_Vis, weights=np.exp(-0.5*(sRuv/12.)**2))
    #var = np.average((srt_Vis-mu)**2, weights=np.exp(-0.5*(sRuv/12.)**2))
    sigma_scat[i] = np.std(srt_Re)
    ri_scat[i] = np.std(np.concatenate((srt_Re, srt_Im)))
    #print(sigma_scat[i], np.sqrt(var), np.std(np.concatenate((srt_Re, srt_Im))))
    #plt.hist(np.concatenate((srt_Re, srt_Im)), bins=20, color='g')
    #plt.hist(srt_Re, color='b', alpha=0.2)
    #plt.hist(srt_Im, color='r', alpha=0.2)
    #plt.show()
    #plt.plot(suvsep, srt_Re, '.k')
    #plt.plot(suvsep, srt_Im, '.b')
    #plt.show()

# now reverse the sort to compare the derived noise with the CASA noise
est_sigma  = sigma_scat[np.argsort(Ruv_sort_indices)]
ri_sigma = ri_scat[np.argsort(Ruv_sort_indices)]
casa_sigma = 1./np.sqrt(Wt)

# output the updated weights
np.savez(fname+'.rwtd.vis', u=u, v=v, Re=Re, Im=Im, WtR=1./est_sigma**2, WtRI=1./ri_sigma**2)
