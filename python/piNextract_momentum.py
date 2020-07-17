# This file extracts a diagram for the piN interacting system 
#
# The steps followed are :
# - Specify the momentum pi2,pf1,pf2 with 9 component
# - Specify the type of the gamma structure, as detailed in the attributes of the dataset
# - Specify the diagram possible choices B1,B2,W1,W2,W3,W4,Z1,Z2,Z3,Z4
# - Specify the output hdf5 file
# - Specify the input hdf5 files
#
# Run this as 'python3 piNextract_momentum.py 0 0 0 1 1 1 2 2 2 1 1 Cg5 g5 Cg5 g5 B1 filtered.h5 Diagramm0000_B.h5'
# 0 0 0 is pi2 momentum
# 1 1 1 is pf1 momentum
# 2 2 2 is pf2 momentum
# 1 external gamma_i
# 1 external gamma_f
# Cg5 gamma_i1
# g5  gamma_i2
# Cg5 gamma_f1
# g5  gamma_f2
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
gamma_exti_input=sys.argv[10]
gamma_extf_input=sys.argv[11]
gamma_i1_input=sys.argv[12]
gamma_i2_input=sys.argv[13]
gamma_f1_input=sys.argv[14]
gamma_f2_input=sys.argv[15]
diagramindex = sys.argv[16]
output = sys.argv[17]
with h5py.File(output, "a") as fo:
    for _file in sys.argv[18:]:
        print("Opening "+_file)
        with h5py.File(_file, "r") as fp:
            _dir = _file.split("Diagramm")[-1].split("_")[0]
            for src in fp.keys():
                 print(src)
                 grpname = "/"+src+"/pi2="+pi2x+"_"+pi2y+"_"+pi2z+"/"
                 print(grpname)
                 if ((grpname+"mvec") in fp):
                   data2 = fp[grpname+"mvec"][:,:]
                   index_momentum = np.where((data2 == (int(pi2x), int(pi2y), int(pi2z), int(pf1x), int(pf1y), int(pf1z), int(pf2x), int(pf2y), int(pf2z))).all(axis=1))
                   pstr=fp[grpname+diagramindex].attrs["description"]
                   lpstr=pstr.split()
                   llpstr=lpstr[1]
                   lllpstr=llpstr.split('/'.encode())
                   gamma_text= (lllpstr[3])
                   gamma_text_separate=gamma_text.split('},{'.encode())
                   texttmp=gamma_text_separate[0].split('{'.encode())
                   gamma_exti_s=texttmp[1].split(','.encode())
                   gamma_exti_len  =len(gamma_exti_s)
                   gamma_exti_index=gamma_exti_s.index(gamma_exti_input.encode())
                   gamma_extf= (gamma_text_separate[1])
                   gamma_extf_s=gamma_extf.split(','.encode());
                   gamma_extf_len  =len(gamma_extf_s)
                   gamma_extf_index=gamma_extf_s.index(gamma_extf_input.encode())
                   gamma_i1=   (gamma_text_separate[2])
                   gamma_i1_s=gamma_i1.split(','.encode());
                   gamma_i1_len  =len(gamma_i1_s)
                   gamma_i1_index=gamma_i1_s.index(gamma_i1_input.encode())
                   gamma_i2=   (gamma_text_separate[3])
                   gamma_i2_s=gamma_i2.split(','.encode());
                   gamma_i2_len  =len(gamma_i2_s)
                   gamma_i2_index=gamma_i2_s.index(gamma_i2_input.encode())
                   gamma_f1=   (gamma_text_separate[4])
                   gamma_f1_s=gamma_f1.split(','.encode());
                   gamma_f1_len  =len(gamma_f1_s)
                   gamma_f1_index=gamma_f1_s.index(gamma_f1_input.encode())
                   texttmp=gamma_text_separate[5].split('}'.encode())
                   gamma_f2=   (texttmp[0])
                   gamma_f2_s=gamma_f2.split(','.encode());
                   gamma_f2_len  =len(gamma_f2_s)
                   gamma_f2_index=gamma_f2_s.index(gamma_f2_input.encode())
                   index_gamma=gamma_exti_index*gamma_extf_len*gamma_i1_len*gamma_i2_len*gamma_f1_len*gamma_f2_len+gamma_extf_index*gamma_i1_len*gamma_i2_len*gamma_f1_len*gamma_f2_len+gamma_i1_index*gamma_i2_len*gamma_f1_len*gamma_f2_len+gamma_i2_index*gamma_f1_len*gamma_f2_len+gamma_f1_index*gamma_f2_len+gamma_f2_index
                   print(index_gamma)
                   print(index_momentum)
                   data = fp[grpname+diagramindex][:,index_momentum,index_gamma,:]
                   grp = fo.require_group("/"+_dir+"/"+src+"/pi2="+pi2x+"_"+pi2y+"_"+pi2z+"/")
                   grp.create_dataset(diagramindex, data.shape, dtype = data.dtype, data = data)
