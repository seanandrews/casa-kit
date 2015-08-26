import numpy as np
import os
import time
import sys
import matplotlib.pyplot as plt
import pandas as pd

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

# compute a running mean of the visibility profile
runmean = pd.rolling_mean(Res, 1000, center=True)

# convert weights into uncertainties
sig = 1./np.sqrt(Wts)

plt.plot(rho, Res-runmean, '.r', rho, np.zeros_like(rho), '--b')
#plt.plot(rho, runmean, 'b')

nclump = 100
Re_scat = np.zeros_like(Re)
Im_scat = np.zeros_like(Im)
tic = time.time()
for i in np.arange(10):
    uvsep = ((u-u[i])**2 + (v-v[i])**2)
    uvsep = uvsep[uvsep < 100.]
    Re_scat[i] = np.std((Re[np.argsort(uvsep)])[1:nclump])
    Im_scat[i] = np.std((Im[np.argsort(uvsep)])[1:nclump])

toc = time.time()
#n, bins, patches = plt.hist(sig, 10, normed=1, facecolor='g', alpha=.8)
#n, bins, patches = plt.hist(Re_scat, 100, normed=1, facecolor='r', alpha=.1)
#n, bins, patches = plt.hist(Im_scat, 100, normed=1, facecolor='b', alpha=.1)
#plt.axis([0.0, 0.01, 0.0, 1000.])
plt.savefig('test.png')
plt.clf()

#toc = time.time()

print(nvis*(toc-tic)/3600.)
