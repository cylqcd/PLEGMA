# This file converts D diagramm like data from 
# scattering contractions to sum of correlation
# functions used to be computed by Calc2pt
# Specifially the following four fields will be computed
# as output:
#	OmegaMn/Pp_Cgi_Cgi
#	OmegaMn/Pm_Cgi_Cgi
#       OmegaMn/Ppgigj-Cgi-Cgj
#       OmegaMn/Pmgigj-Cgi-Cgj
# The hdf5 tag will look like the following:
# /XXX_SS_gN25a4p_aN50a0p5/sxXXsyYYszZZstTTTbaryons_u[+2.5e-03]d[-2.5e-03]s[+1.5e-02]/OmegaMn
# where SS means that the source and sink are smeared
#       gN25 25 steps of Gaussian smearing was used
#       a4p  with alpha parameter 4
#       aN50 gauge field smeared using ape smearing 50 times
#       a0p5 with alpa_ape=0.5
# Usage
# python convert_Calc2pt_Ddiagrammscattering.py destfile.h5 `cat sourcefilelist`
#*************Imports*********************************
import numpy as np
import sys
import h5py
import os

output = sys.argv[1]
with h5py.File(output, "a") as fo:
    for _file in sys.argv[2:]:
        print("Opening "+_file)
        with h5py.File(_file, "r") as fp:
            _dir = _file.split("Diagramm")[-1].split("_")[0]
            print(_dir)
            for src in fp.keys():
                #if "twop/"+_dir+"/"+src in fo: continue
                print(src)
                if src+"/Oms[+1.5e-02]" in fp:
                    grpname = src+"/Oms[+1.5e-02]"
                    #cgx-cgx
                    data1 = fp[grpname][:,13,0,0,:]
                    data2 = fp[grpname][:,13,0,5,:]
                    data3 = fp[grpname][:,13,0,10,:]
                    data4 = fp[grpname][:,13,0,15,:]
                    #cgy-cgy
                    data5 = fp[grpname][:,13,7,0,:]
                    data6 = fp[grpname][:,13,7,5,:]
                    data7 = fp[grpname][:,13,7,10,:]
                    data8 = fp[grpname][:,13,7,15,:]
                    #cgz-cgz
                    data9 = fp[grpname][:,13,14,0,:]
                    data10 = fp[grpname][:,13,14,5,:]
                    data11 = fp[grpname][:,13,14,10,:]
                    data12 = fp[grpname][:,13,14,15,:]
                    datapp = -1*data1+-1*data2 + data5+ data6+-1*data9 +-1*data10
                    datamm = -1*data3+-1*data4 + data7+ data8+-1*data11+-1*data12
                    grp = fo.require_group("/"+_dir+"_SS_gN25a4p_aN50a0p5/"+src+"/baryons_u[+2.5e-03]d[-2.5e-03]s[+1.5e-02]")
                    dset=grp.create_dataset("OmegaMn/Pp_Cgi_Cgi", datapp.shape, dtype = datapp.dtype, data = datapp)
                    #dset.attrs['description']="shape: /time/moms/{1},{1},{cg1,cg2,cg3,cg1g4,cg2g4,cg3g4}{cg1,cg2,cg3,cg1g4,cg2g4,cg3g4}/S1/S2//re-im"
                    dset=grp.create_dataset("OmegaMn/Pm_Cgi_Cgi", datamm.shape, dtype = datamm.dtype, data = datamm)
                    #dset.attrs['description']="shape: /time/moms/{1},{1},{cg1,cg2,cg3,cg1g4,cg2g4,cg3g4}{cg1,cg2,cg3,cg1g4,cg2g4,cg3g4}/S1/S2//re-im"
                    #gx*gy*cgx-ggy
                    data1 = fp[grpname][:,13,1,0,:]
                    data1[:,0]=-1*fp[grpname][:,13,1,0,1]
                    data1[:,1]=+1*fp[grpname][:,13,1,0,0]
                    data2 = fp[grpname][:,13,1,5,:]
                    data2[:,0]=+1*fp[grpname][:,13,1,5,1]
                    data2[:,1]=-1*fp[grpname][:,13,1,5,0]
                    data3 = fp[grpname][:,13,1,10,:]
                    data3[:,0]=-1*fp[grpname][:,13,1,10,1]
                    data3[:,1]=+1*fp[grpname][:,13,1,10,0]
                    data4 = fp[grpname][:,13,1,15,:]
                    data4[:,0]=+1*fp[grpname][:,13,1,15,1]
                    data4[:,1]=-1*fp[grpname][:,13,1,15,0]
                    #gy*gz*Cgy-Cgz
                    data5 = fp[grpname][:,13,8,4,:]
                    data5[:,0]=-1*fp[grpname][:,13,8,4,1]
                    data5[:,1]=+1*fp[grpname][:,13,8,4,0]
                    data6 = fp[grpname][:,13,8,1,:]
                    data6[:,0]=-1*fp[grpname][:,13,8,1,1]
                    data6[:,1]=+1*fp[grpname][:,13,8,1,0]
                    data7 = fp[grpname][:,13,8,14,:]
                    data7[:,0]=-1*fp[grpname][:,13,8,14,1]
                    data7[:,1]=+1*fp[grpname][:,13,8,14,0]
                    data8 = fp[grpname][:,13,8,11,:]
                    data8[:,0]=-1*fp[grpname][:,13,8,11,1]
                    data8[:,1]=+1*fp[grpname][:,13,8,11,0]
                    #gxgz*Cgx-Cgz
                    data9 = fp[grpname][:,13,2,4,:]
                    data9[:,0]=-1*fp[grpname][:,13,2,4,0]
                    data9[:,1]=-1*fp[grpname][:,13,2,4,1]
                    data10 = fp[grpname][:,13,2,1,:]
                    data10[:,0]=+1*fp[grpname][:,13,2,1,0]
                    data10[:,1]=+1*fp[grpname][:,13,2,1,1]
                    data11 = fp[grpname][:,13,2,14,:]
                    data11[:,0]=-1*fp[grpname][:,13,2,14,0]
                    data11[:,1]=-1*fp[grpname][:,13,2,14,1]
                    data12 = fp[grpname][:,13,2,11,:]
                    data12[:,0]=+1*fp[grpname][:,13,2,11,0]
                    data12[:,1]=+1*fp[grpname][:,13,2,11,1]
                    datapp = +1*data1+1*data2-1*data5-1*data6-1*data9-1*data10
                    datamm = +1*data3+1*data4-1*data7-1*data8-1*data11-1*data12
                    dset=grp.create_dataset("OmegaMn/Ppgigj-Cgi-Cgj", datapp.shape, dtype = datapp.dtype, data = datapp)
                    dset=grp.create_dataset("OmegaMn/Pmgigj-Cgi-Cgj", datamm.shape, dtype = datamm.dtype, data = datamm)
                if src+"/Oms[+1.8e-02]" in fp:
                    grpname = src+"/Oms[+1.8e-02]"
                    #cgx-cgx
                    data1 = fp[grpname][:,13,0,0,:]
                    data2 = fp[grpname][:,13,0,5,:]
                    data3 = fp[grpname][:,13,0,10,:]
                    data4 = fp[grpname][:,13,0,15,:]
                    #cgy-cgy
                    data5 = fp[grpname][:,13,7,0,:]
                    data6 = fp[grpname][:,13,7,5,:]
                    data7 = fp[grpname][:,13,7,10,:]
                    data8 = fp[grpname][:,13,7,15,:]
                    #cgz-cgz
                    data9 = fp[grpname][:,13,14,0,:]
                    data10 = fp[grpname][:,13,14,5,:]
                    data11 = fp[grpname][:,13,14,10,:]
                    data12 = fp[grpname][:,13,14,15,:]
                    datapp = -1*data1+-1*data2 + data5+ data6+-1*data9 +-1*data10
                    datamm = -1*data3+-1*data4 + data7+ data8+-1*data11+-1*data12
                    grp = fo.require_group("/"+_dir+"_SS_gN25a4p_aN50a0p5/"+src+"/baryons_u[+2.5e-03]d[-2.5e-03]s[+1.8e-02]_only-s")
                    dset=grp.create_dataset("OmegaMn/Pp_Cgi_Cgi", datapp.shape, dtype = datapp.dtype, data = datapp)
                    #dset.attrs['description']="shape: /time/moms/{1},{1},{cg1,cg2,cg3,cg1g4,cg2g4,cg3g4}{cg1,cg2,cg3,cg1g4,cg2g4,cg3g4}/S1/S2//re-im"
                    dset=grp.create_dataset("OmegaMn/Pm_Cgi_Cgi", datamm.shape, dtype = datamm.dtype, data = datamm)
                    #dset.attrs['description']="shape: /time/moms/{1},{1},{cg1,cg2,cg3,cg1g4,cg2g4,cg3g4}{cg1,cg2,cg3,cg1g4,cg2g4,cg3g4}/S1/S2//re-im"
                    #gx*gy*cgx-ggy
                    data1 = fp[grpname][:,13,1,0,:]
                    data1[:,0]=-1*fp[grpname][:,13,1,0,1]
                    data1[:,1]=+1*fp[grpname][:,13,1,0,0]
                    data2 = fp[grpname][:,13,1,5,:]
                    data2[:,0]=+1*fp[grpname][:,13,1,5,1]
                    data2[:,1]=-1*fp[grpname][:,13,1,5,0]
                    data3 = fp[grpname][:,13,1,10,:]
                    data3[:,0]=-1*fp[grpname][:,13,1,10,1]
                    data3[:,1]=+1*fp[grpname][:,13,1,10,0]
                    data4 = fp[grpname][:,13,1,15,:]
                    data4[:,0]=+1*fp[grpname][:,13,1,15,1]
                    data4[:,1]=-1*fp[grpname][:,13,1,15,0]
                    #gy*gz*Cgy-Cgz
                    data5 = fp[grpname][:,13,8,4,:]
                    data5[:,0]=-1*fp[grpname][:,13,8,4,1]
                    data5[:,1]=+1*fp[grpname][:,13,8,4,0]
                    data6 = fp[grpname][:,13,8,1,:]
                    data6[:,0]=-1*fp[grpname][:,13,8,1,1]
                    data6[:,1]=+1*fp[grpname][:,13,8,1,0]
                    data7 = fp[grpname][:,13,8,14,:]
                    data7[:,0]=-1*fp[grpname][:,13,8,14,1]
                    data7[:,1]=+1*fp[grpname][:,13,8,14,0]
                    data8 = fp[grpname][:,13,8,11,:]
                    data8[:,0]=-1*fp[grpname][:,13,8,11,1]
                    data8[:,1]=+1*fp[grpname][:,13,8,11,0]
                    #gxgz*Cgx-Cgz
                    data9 = fp[grpname][:,13,2,4,:]
                    data9[:,0]=-1*fp[grpname][:,13,2,4,0]
                    data9[:,1]=-1*fp[grpname][:,13,2,4,1]
                    data10 = fp[grpname][:,13,2,1,:]
                    data10[:,0]=+1*fp[grpname][:,13,2,1,0]
                    data10[:,1]=+1*fp[grpname][:,13,2,1,1]
                    data11 = fp[grpname][:,13,2,14,:]
                    data11[:,0]=-1*fp[grpname][:,13,2,14,0]
                    data11[:,1]=-1*fp[grpname][:,13,2,14,1]
                    data12 = fp[grpname][:,13,2,11,:]
                    data12[:,0]=+1*fp[grpname][:,13,2,11,0]
                    data12[:,1]=+1*fp[grpname][:,13,2,11,1]
                    datapp = +1*data1+1*data2-1*data5-1*data6-1*data9-1*data10
                    datamm = +1*data3+1*data4-1*data7-1*data8-1*data11-1*data12
                    dset=grp.create_dataset("OmegaMn/Ppgigj-Cgi-Cgj", datapp.shape, dtype = datapp.dtype, data = datapp)
                    dset=grp.create_dataset("OmegaMn/Pmgigj-Cgi-Cgj", datamm.shape, dtype = datamm.dtype, data = datamm)
                if src+"/Oms[+2.2e-02]" in fp:
                    grpname = src+"/Oms[+2.2e-02]"
                    #cgx-cgx
                    data1 = fp[grpname][:,13,0,0,:]
                    data2 = fp[grpname][:,13,0,5,:]
                    data3 = fp[grpname][:,13,0,10,:]
                    data4 = fp[grpname][:,13,0,15,:]
                    #cgy-cgy
                    data5 = fp[grpname][:,13,7,0,:]
                    data6 = fp[grpname][:,13,7,5,:]
                    data7 = fp[grpname][:,13,7,10,:]
                    data8 = fp[grpname][:,13,7,15,:]
                    #cgz-cgz
                    data9 = fp[grpname][:,13,14,0,:]
                    data10 = fp[grpname][:,13,14,5,:]
                    data11 = fp[grpname][:,13,14,10,:]
                    data12 = fp[grpname][:,13,14,15,:]
                    datapp = -1*data1+-1*data2 + data5+ data6+-1*data9 +-1*data10
                    datamm = -1*data3+-1*data4 + data7+ data8+-1*data11+-1*data12
                    grp = fo.require_group("/"+_dir+"_SS_gN25a4p_aN50a0p5/"+src+"/baryons_u[+2.5e-03]d[-2.5e-03]s[+2.2e-02]_only-s")
                    dset=grp.create_dataset("OmegaMn/Pp_Cgi_Cgi", datapp.shape, dtype = datapp.dtype, data = datapp)
                    #dset.attrs['description']="shape: /time/moms/{1},{1},{cg1,cg2,cg3,cg1g4,cg2g4,cg3g4}{cg1,cg2,cg3,cg1g4,cg2g4,cg3g4}/S1/S2//re-im"
                    dset=grp.create_dataset("OmegaMn/Pm_Cgi_Cgi", datamm.shape, dtype = datamm.dtype, data = datamm)
                    #dset.attrs['description']="shape: /time/moms/{1},{1},{cg1,cg2,cg3,cg1g4,cg2g4,cg3g4}{cg1,cg2,cg3,cg1g4,cg2g4,cg3g4}/S1/S2//re-im"
                    #gx*gy*cgx-ggy
                    data1 = fp[grpname][:,13,1,0,:]
                    data1[:,0]=-1*fp[grpname][:,13,1,0,1]
                    data1[:,1]=+1*fp[grpname][:,13,1,0,0]
                    data2 = fp[grpname][:,13,1,5,:]
                    data2[:,0]=+1*fp[grpname][:,13,1,5,1]
                    data2[:,1]=-1*fp[grpname][:,13,1,5,0]
                    data3 = fp[grpname][:,13,1,10,:]
                    data3[:,0]=-1*fp[grpname][:,13,1,10,1]
                    data3[:,1]=+1*fp[grpname][:,13,1,10,0]
                    data4 = fp[grpname][:,13,1,15,:]
                    data4[:,0]=+1*fp[grpname][:,13,1,15,1]
                    data4[:,1]=-1*fp[grpname][:,13,1,15,0]
                    #gy*gz*Cgy-Cgz
                    data5 = fp[grpname][:,13,8,4,:]
                    data5[:,0]=-1*fp[grpname][:,13,8,4,1]
                    data5[:,1]=+1*fp[grpname][:,13,8,4,0]
                    data6 = fp[grpname][:,13,8,1,:]
                    data6[:,0]=-1*fp[grpname][:,13,8,1,1]
                    data6[:,1]=+1*fp[grpname][:,13,8,1,0]
                    data7 = fp[grpname][:,13,8,14,:]
                    data7[:,0]=-1*fp[grpname][:,13,8,14,1]
                    data7[:,1]=+1*fp[grpname][:,13,8,14,0]
                    data8 = fp[grpname][:,13,8,11,:]
                    data8[:,0]=-1*fp[grpname][:,13,8,11,1]
                    data8[:,1]=+1*fp[grpname][:,13,8,11,0]
                    #gxgz*Cgx-Cgz
                    data9 = fp[grpname][:,13,2,4,:]
                    data9[:,0]=-1*fp[grpname][:,13,2,4,0]
                    data9[:,1]=-1*fp[grpname][:,13,2,4,1]
                    data10 = fp[grpname][:,13,2,1,:]
                    data10[:,0]=+1*fp[grpname][:,13,2,1,0]
                    data10[:,1]=+1*fp[grpname][:,13,2,1,1]
                    data11 = fp[grpname][:,13,2,14,:]
                    data11[:,0]=-1*fp[grpname][:,13,2,14,0]
                    data11[:,1]=-1*fp[grpname][:,13,2,14,1]
                    data12 = fp[grpname][:,13,2,11,:]
                    data12[:,0]=+1*fp[grpname][:,13,2,11,0]
                    data12[:,1]=+1*fp[grpname][:,13,2,11,1]
                    datapp = +1*data1+1*data2-1*data5-1*data6-1*data9-1*data10
                    datamm = +1*data3+1*data4-1*data7-1*data8-1*data11-1*data12
                    dset=grp.create_dataset("OmegaMn/Ppgigj-Cgi-Cgj", datapp.shape, dtype = datapp.dtype, data = datapp)
                    dset=grp.create_dataset("OmegaMn/Pmgigj-Cgi-Cgj", datamm.shape, dtype = datamm.dtype, data = datamm)
