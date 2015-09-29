import numpy as np

# original .ms file name
oms_path = '../DR/ALMA_B6/2013.1.00226.S/science_goal.uid___A001_X122_X1f3/group.uid___A001_X122_X1f4/member.uid___A001_X122_X1f5/final'
oms_name = 'continuum'
mdl_name = oms_name+'.model'
mkres = True
res_name = oms_name+'.resid'

# NAMING CONVENTIONS: If your data file is named 'name.ms', then you will 
# export it as 'name.vis.npz' and your corresponding model and residual files
# will be 'name.model.ms' / 'name.model.vis.npz' and 'name.resid.ms' / 
# 'name.resid.vis.npz'.  You can change this with 'mdl_name' and 'res_name'.

# copy the data file into a model 
os.system('rm -rf '+mdl_name+'.ms')
os.system('cp -r '+oms_path+'/'+oms_name+'.ms '+mdl_name+'.ms')

# load the data
tb.open(mdl_name+'.ms')
data = tb.getcol("DATA")
flag = tb.getcol("FLAG")
tb.close()

# Note the flagged columns
flagged = np.all(flag, axis=(0, 1))
unflagged = np.squeeze(np.where(flagged == False))

# load the model file (presume this is just an array of complex numbers, with 
# the appropriate sorting/ordering in original .ms file; also assume that the 
# polarizations have been averaged, and that the model is unpolarized)
mdl = (np.load(mdl_name+'.vis.npz'))['vis']

# replace the original data with the model
data[:,:,unflagged] = mdl

# now re-pack those back into the .ms
tb.open(mdl_name+'.ms', nomodify=False)
tb.putcol("DATA", data)
tb.flush()
tb.close()


# now repeat this for the residual visibilities if you want
if (mkres == True):
    os.system('rm -rf '+res_name+'.ms')
    os.system('cp -r '+oms_path+'/'+oms_name+'.ms '+res_name+'.ms')

    tb.open(res_name+'.ms')
    data = tb.getcol("DATA")
    flag = tb.getcol("FLAG")
    tb.close()

    flagged = np.all(flag, axis=(0, 1))
    unflagged = np.squeeze(np.where(flagged == False))

    mdl = (np.load(mdl_name+'.vis.npz'))['vis']
    data[:,:,unflagged] -= mdl

    tb.open(res_name+'.ms', nomodify=False)
    tb.putcol("DATA", data)
    tb.flush()
    tb.close()
