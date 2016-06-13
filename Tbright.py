# 
# Convert standard radio surface brightness (Jy per beam) into brightness 
# temperature units (K).
#

import numpy as np

def Tbright(sb, beam, freq):

    # unpack beam parameters (major and minor FWHM in arcseconds)
    bmaj, bmin = beam

    # constants
    hh = 6.626e-27
    kk = 1.381e-16
    cc = 2.9979e10

    # calculate beam area (steradians)
    omega_beam = (np.pi*bmaj*bmin / (4.*np.log(2.))) * (np.pi/180.)**2

    # convert surface brightness into cgs units 
    I_cgs = sb * 1e-23 / omega_beam

    # now calculate brightness temperature from inverted Planck function
    Tb = (hh*freq/kk) / np.log((2*hh*freq**3/(I_cgs*cc**2))+1.)

    return(Tb)
