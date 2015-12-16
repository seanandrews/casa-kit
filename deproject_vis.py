import numpy as np

def deproject_vis(data, bins=np.array([0.]), incl=0., PA=0., offx=0., offy=0., 
                  errtype='mean'):

    # - read in, parse data
    u, v, vis, wgt = data

    # - convert keywords into relevant units
    inclr = np.radians(incl)
    PAr = 0.5*np.pi-np.radians(PA)
    offx *= -np.pi/(180.*3600.)
    offy *= -np.pi/(180.*3600.)

    # - change to a deprojected, rotated coordinate system
    uprime = (u*np.cos(PAr) + v*np.sin(PAr)) 
    vprime = (-u*np.sin(PAr) + v*np.cos(PAr)) * np.cos(inclr)
    rhop = np.sqrt(uprime**2 + vprime**2)

    # - phase shifts to account for offsets
    shifts = np.exp(-2.*np.pi*1.0j*(u*-offx + v*-offy))
    visp = vis*shifts
    realp = visp.real
    imagp = visp.imag

    # - if requested, return a binned (averaged) representation
    if (bins.size > 1.):
        bins *= 1e3	# scale to lambda units (input in klambda)
        bwid = 0.5*(bins[1]-bins[0])	# only for evenly-space linear bins
        bvis = np.zeros_like(bins, dtype='complex')
        berr = np.zeros_like(bins, dtype='complex')
        for ib in np.arange(len(bins)):
            inb = np.where((rhop >= bins[ib]-bwid) & (rhop < bins[ib]+bwid))
            if (len(inb[0]) >= 5):
                bRe, eRemu = np.average(realp[inb], weights=wgt[inb], 
                                        returned=True)
                eRese = np.std(realp[inb])
                bIm, eImmu = np.average(imagp[inb], weights=wgt[inb], 
                                        returned=True)
                eImse = np.std(imagp[inb])
                bvis[ib] = bRe+1j*bIm
                if (errtype == 'scat'):
                    berr[ib] = eRese+1j*eImse
                else: berr[ib] = 1./np.sqrt(eRemu)+1j/np.sqrt(eImmu)
            else:
                bvis[ib] = 0+1j*0
                berr[ib] = 0+1j*0
            parser = np.where(berr.real != 0)
            output = bins[parser], bvis[parser], berr[parser]
        return output       
        
    # - if not, returned the unbinned representation
    output = rhop, realp+1j*imagp, 1./np.sqrt(wgt)

    return output
