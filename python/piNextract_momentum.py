# This file extracts a diagram for the piN interacting system 
#
# The steps followed are :
# - Specify the momentum pi2,pf1,pf2 with 9 component
# - Specify the diagram possible choices B1,B2,W1,W2,W3,W4,Z1,Z2,Z3,Z4
# - Specify the output hdf5 file
# - Specify the input hdf5 files
#
# Run this as 'python3 piNextract_momentum.py 0 0 0 1 1 1 2 2 2 B1 filtered.h5 Diagramm0000_B.h5'
# 0 0 0 is pi2 momentum
# 1 1 1 is pf1 momentum
# 2 2 2 is pf2 momentum
# B1 diagram
# filtered.h5 output file
# Diagramm0000_B.h5 input file
# Authors: F. Pittler


#*************Imports*********************************


import numpy as np
import sys
import h5py
import os

pi2x = sys.argv[1]
pi2y = sys.argv[2]
pi2z = sys.argv[3]
pf1x = sys.argv[4]
pf1y = sys.argv[5]
pf1z = sys.argv[6]
pf2x = sys.argv[7]
pf2y = sys.argv[8]
pf2z = sys.argv[9]
diagramindex = sys.argv[10]
output = sys.argv[11]
with h5py.File(output, "a") as fo:
    for _file in sys.argv[12:]:
        print("Opening "+_file)
        with h5py.File(_file, "r") as fp:
            for src in fp.keys():
                 print(src)
                 grpname = "/"+src+"/pi2="+pi2x+"_"+pi2y+"_"+pi2z+"/"
                 print(grpname)
                 data2 = fp[grpname+"mvec"][:,:]
                 tmp = np.where((data2 == (int(pi2x), int(pi2y), int(pi2z), int(pf1x), int(pf1y), int(pf1z), int(pf2x), int(pf2y), int(pf2z))).all(axis=1))
                 print(tmp)
                 data = fp[grpname+diagramindex][:,tmp,:,:,:]
                 grp = fo.require_group("/"+src+"/pi2="+pi2x+"_"+pi2y+"_"+pi2z+"/")
                 grp.create_dataset(diagramindex, data.shape, dtype = data.dtype, data = data)
