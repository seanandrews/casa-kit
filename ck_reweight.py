import numpy as np
import os
import time
import sys
import matplotlib.pyplot as plt


# load some data
visdata = np.load('test_data/blind2_fo.340GHz.vis.npz')
freq = 340e9
u = 1e-3*visdata['u']*freq/2.9979e8
v = 1e-3*visdata['v']*freq/2.9979e8
Re = visdata['Re']
Im = visdata['Im']
Wt = visdata['Wt']
nvis = len(Re)

# convert weights into uncertainties
sig = 1./np.sqrt(Wt)

nclump = 1000
for i in np.arange(1):
    sort_ix = np.argsort(np.sqrt((u-u[i])**2 + (v-v[i])**2))
    Re_clump = (Re[sort_ix])[:nclump]
    Im_clump = (Im[sort_ix])[:nclump]
    n, bins, patches = plt.hist(Re_clump, 50, facecolor='green', alpha=0.75)

plt.axis([0.0, 0.15, 0.0, 100.])
plt.savefig('test.png')
plt.clf()

