import numpy as np
import os
import time
import sys
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

tic = time.time()

# load some data
visdata = np.load('test_data/blind2_fo.340GHz.vis.npz')
freq = 340e9
u = 1e-3*visdata['u']*freq/2.9979e8
v = 1e-3*visdata['v']*freq/2.9979e8
Re = visdata['Re']
Im = visdata['Im']
Wt = visdata['Wt']
nvis = len(Re)

# compute the phase-center distances
Ruv = np.sqrt(u**2+v**2)

# sort based on phase-center distances
Res = Re[np.argsort(Ruv)]
Ims = Im[np.argsort(Ruv)]
Wts = Wt[np.argsort(Ruv)]
rho = Ruv[np.argsort(Ruv)]
print(np.mean(rho),np.min(rho),np.max(rho))

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

# convert weights into uncertainties
sigma = 1./np.sqrt(Wts)

# loop through each visibility and calculate the RMS scatter from neighboring
# visibilities (a purely empirical noise estimate; captures non-thermal noise)
nclump = 100
sigma_scat = np.zeros_like(sigma)
tic = time.time()
for i in np.arange(10):
    # calculate distances from this (u,v) point
    uvsep = ((u-u[i])**2 + (v-v[i])**2)
    # truncate list to minimize sorting overheads
    uvsep = uvsep[uvsep < 50.**2]
    # sort; select nclump nearest visibilities
    srt_Re = (rm_Re[np.argsort(uvsep)])[1:nclump]
    srt_Im = (rm_Im[np.argsort(uvsep)])[1:nclump]
    # calculate and store the standard deviation
    sigma_scat[i] = np.std(srt_Re)
    print(sigma_scat[i], sigma[i])

toc = time.time()
#n, bins, patches = plt.hist(sig, 10, normed=1, facecolor='g', alpha=.8)
#n, bins, patches = plt.hist(Re_scat, 100, normed=1, facecolor='r', alpha=.1)
#n, bins, patches = plt.hist(Im_scat, 100, normed=1, facecolor='b', alpha=.1)
#plt.axis([0.0, 0.01, 0.0, 1000.])
plt.savefig('test.png')
plt.clf()

#toc = time.time()

print(nvis*(toc-tic)/60./10.)
