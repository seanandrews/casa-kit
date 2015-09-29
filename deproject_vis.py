import numpy as np

def deproject_vis(data, bins=np.array([0.]), incl=0., PA=0., offx=0., offy=0., 
                  errtype='mean'):

    # - read in, parse data
    u, v, vis, wgt = data

    # - convert keywords into relevant units
    inclr = np.radians(incl)
    PAr = np.radians(PA)
    offx *= -np.pi/(180.*3600.)
    offy *= -np.pi/(180.*3600.)

    # - change to a deprojected, cylindrical coordinate system
    thet = np.arctan2(v, u)
    dmaj = -np.sqrt(u**2+v**2) * np.cos(thet+PAr) * np.cos(inclr)
    dmin = -np.sqrt(u**2+v**2) * np.sin(thet+PAr)
    rhop = np.sqrt(dmaj**2 + dmin**2)

    # - phase shifts to account for offsets
    amp = np.sqrt(vis.real**2 + vis.imag**2)
    pha = np.arctan2(vis.imag, vis.real)
    realp = amp*np.cos(pha+2.*np.pi*(offx*u+offy*v))
    imagp = amp*np.sin(pha+2.*np.pi*(offx*u+offy*v))

    # - if requested, return a binned (averaged) representation
    if (bins.size > 1.):
        bins *= 1e3	# scale to lambda units (input in klambda)
        bwid = 0.5*(bins[1]-bins[0])	# only for evenly-space linear bins
        bvis = np.zeros_like(bins, dtype='complex')
        berr = np.zeros_like(bins, dtype='complex')
        for ib in np.arange(len(bins)):
            inb = np.where((rhop >= bins[ib]-bwid) & (rhop < bins[ib]+bwid))
            bRe, eRemu = np.average(realp[inb], weights=wgt[inb], returned=True)
            eRese = np.std(realp[inb])
            bIm, eImmu = np.average(imagp[inb], weights=wgt[inb], returned=True)
            eImse = np.std(imagp[inb])
            bvis[ib] = bRe+1j*bIm
            if (errtype == 'scat'):
                berr[ib] = eRese+1j*eImse
            else: berr[ib] = 1./np.sqrt(eRemu)+1j/np.sqrt(eImmu)
        output = bins, bvis, berr
        return output       
        
    # - if not, returned the unbinned representation
    output = rhop, realp+1j*imagp, 1./np.sqrt(wgt)

    return output
