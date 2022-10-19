# This file converts output of omegamass O correlations functions to 
# the format of Calc_2pt.cpp 
# 
# Input parameters  the outputfile name
#                   the list of input files
# Output: Pp_Cgi_Cgi, Pm_Cgi_Cgi and the projected continuum spin 32
#         correlation functions
#
#*************Imports*********************************


import numpy as np
import sys
import h5py
import os


output = sys.argv[1]
with h5py.File(output, "a") as fo:
    for _file in sys.argv[2:]:
        #twopfiles = [f for f in os.listdir(_dir) if f.startswith("two")]
        #for _file in twopfiles:
        print("Opening "+_file)
        with h5py.File(_file, "r") as fp:
            _dir = _file.split("Diagramm")[-1].split("_")[0]
            print(_dir)
            for src in fp.keys():
                #if "twop/"+_dir+"/"+src in fo: continue
                print(src)
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
                    datapp = data1+data2 + data5+ data6 +data9 +data10
                    datamm = data3+data4 + data7+ data8 +data11+data12
                    grp = fo.require_group("/"+_dir+"_SS_gN25a4p_aN50a0p5/"+src+"/baryons_u[+4.0e-03]d[-4.0e-03]s[+1.8e-02]")
                    dset=grp.create_dataset("OmegaMn/Pp_Cgi_Cgi", datapp.shape, dtype = datapp.dtype, data = datapp)
                    dset=grp.create_dataset("OmegaMn/Pm_Cgi_Cgi", datamm.shape, dtype = datamm.dtype, data = datamm)
                    #gy*gx*cgy-cgx
                    #Ppgygx -i 0 0 0
                    #        0 i 0 0
                    #        0 0 0 0
                    #        0 0 0 0
                    #Selecting cgy sink cgx source
                    data1 = fp[grpname][:,13,1,0,:]
                    data1[:,0]=+1*fp[grpname][:,13,1,0,1]
                    data1[:,1]=-1*fp[grpname][:,13,1,0,0]
                    data2 = fp[grpname][:,13,1,5,:]
                    data2[:,0]=-1*fp[grpname][:,13,1,5,1]
                    data2[:,1]=+1*fp[grpname][:,13,1,5,0]
                    #Pmgygx  0 0  0  0
                    #        0 0  0  0
                    #        0 0 -i  0
                    #        0 0  0 +i
                    data3 = fp[grpname][:,13,1,10,:]
                    data3[:,0]=+1*fp[grpname][:,13,1,10,1]
                    data3[:,1]=-1*fp[grpname][:,13,1,10,0]
                    data4 = fp[grpname][:,13,1,15,:]
                    data4[:,0]=-1*fp[grpname][:,13,1,15,1]
                    data4[:,1]=+1*fp[grpname][:,13,1,15,0]
                    #gx*gy*cgx-cgy
                    #Ppgxgy  i  0 0 0
                    #        0 -i 0 0
                    #        0  0 0 0
                    #        0  0 0 0
                    #Selecting cgx sink cgy source
                    data5 = fp[grpname][:,13,6,0,:]
                    data5[:,0]=-1*fp[grpname][:,13,6,0,1]
                    data5[:,1]=+1*fp[grpname][:,13,6,0,0]
                    data6 = fp[grpname][:,13,1,5,:]
                    data6[:,0]=+1*fp[grpname][:,13,6,5,1]
                    data6[:,1]=-1*fp[grpname][:,13,6,5,0]
                    #Pmgxgy  0 0  0  0
                    #        0 0  0  0
                    #        0 0 +i  0
                    #        0 0  0 -i
                    data7 = fp[grpname][:,13,1,10,:]
                    data7[:,0]=-1*fp[grpname][:,13,6,10,1]
                    data7[:,1]=+1*fp[grpname][:,13,6,10,0]
                    data8 = fp[grpname][:,13,1,15,:]
                    data8[:,0]=+1*fp[grpname][:,13,6,15,1]
                    data8[:,1]=-1*fp[grpname][:,13,6,15,0]
                    #gz*gy*cgz-cgy
                    #Ppgzgy  0 -i 0 0
                    #       -i  0 0 0
                    #        0  0 0 0
                    #        0  0 0 0
                    #Selecting cgz sink cgy source
                    #gz*gy*Cgz-Cgy
                    data9 = fp[grpname][:,13,8,4,:]
                    data9[:,0]=+1*fp[grpname][:,13,8,4,1]
                    data9[:,1]=-1*fp[grpname][:,13,8,4,0]
                    data10 = fp[grpname][:,13,8,1,:]
                    data10[:,0]=+1*fp[grpname][:,13,8,1,1]
                    data10[:,1]=-1*fp[grpname][:,13,8,1,0]
                    #Pmgzgy  0  0  0  0
                    #        0  0  0  0
                    #        0  0  0 -i
                    #        0  0 -i  0
                    data11 = fp[grpname][:,13,8,14,:]
                    data11[:,0]=+1*fp[grpname][:,13,8,14,1]
                    data11[:,1]=-1*fp[grpname][:,13,8,14,0]
                    data12 = fp[grpname][:,13,8,11,:]
                    data12[:,0]=+1*fp[grpname][:,13,8,11,1]
                    data12[:,1]=-1*fp[grpname][:,13,8,11,0]
                    #gy*gz*cgy-cgz
                    #Ppgygz  0  i 0 0
                    #        i  0 0 0
                    #        0  0 0 0
                    #        0  0 0 0
                    #Selecting cgz sink cgy source
                    #gy*gz*Cgy-Cgz
                    data13 = fp[grpname][:,13,13,4,:]
                    data13[:,0]=-1*fp[grpname][:,13,13,4,1]
                    data13[:,1]=+1*fp[grpname][:,13,13,4,0]
                    data14 = fp[grpname][:,13,13,1,:]
                    data14[:,0]=-1*fp[grpname][:,13,13,1,1]
                    data14[:,1]=+1*fp[grpname][:,13,13,1,0]
                    #Pmgygz  0  0 0 0
                    #        0  0 0 0
                    #        0  0 0 i
                    #        0  0 i 0
                    data15 = fp[grpname][:,13,13,14,:]
                    data15[:,0]=-1*fp[grpname][:,13,13,14,1]
                    data15[:,1]=+1*fp[grpname][:,13,13,14,0]
                    data16 = fp[grpname][:,13,13,11,:]
                    data16[:,0]=-1*fp[grpname][:,13,13,11,1]
                    data16[:,1]=+1*fp[grpname][:,13,13,11,0]
                    #gz*gx*cgz-cgx
                    #Ppgzgx   0  1 0 0
                    #        -1  0 0 0
                    #         0  0 0 0
                    #         0  0 0 0
                    #Selecting cgz sink cgx source
                    #gzgx*Cgz-Cgx
                    data17 = fp[grpname][:,13,2,4,:]
                    data17[:,0]=+1*fp[grpname][:,13,2,4,0]
                    data17[:,1]=+1*fp[grpname][:,13,2,4,1]
                    data18 = fp[grpname][:,13,2,1,:]
                    data18[:,0]=-1*fp[grpname][:,13,2,1,0]
                    data18[:,1]=-1*fp[grpname][:,13,2,1,1]
                    #Pmgzgx   0  0  0 0
                    #         0  0  0 0
                    #         0  0  0 1
                    #         0  0 -1 0
                    data19 = fp[grpname][:,13,2,14,:]
                    data19[:,0]=+1*fp[grpname][:,13,2,14,0]
                    data19[:,1]=+1*fp[grpname][:,13,2,14,1]
                    data20 = fp[grpname][:,13,2,11,:]
                    data20[:,0]=-1*fp[grpname][:,13,2,11,0]
                    data20[:,1]=-1*fp[grpname][:,13,2,11,1]
                    #gx*gz*cgx-cgz
                    #Ppgxgz   0 -1 0 0
                    #        +1  0 0 0
                    #         0  0 0 0
                    #         0  0 0 0
                    #Selecting cgx sink cgz source
                    #gzgx*Cgz-Cgx
                    data21 = fp[grpname][:,13,12,4,:]
                    data21[:,0]=-1*fp[grpname][:,13,12,4,0]
                    data21[:,1]=-1*fp[grpname][:,13,12,4,1]
                    data22 = fp[grpname][:,13,12,1,:]
                    data22[:,0]=+1*fp[grpname][:,13,12,1,0]
                    data22[:,1]=+1*fp[grpname][:,13,12,1,1]
                    data23 = fp[grpname][:,13,12,14,:]
                    data23[:,0]=-1*fp[grpname][:,13,12,14,0]
                    data23[:,1]=-1*fp[grpname][:,13,12,14,1]
                    data24 = fp[grpname][:,13,12,11,:]
                    data24[:,0]=+1*fp[grpname][:,13,12,11,0]
                    data24[:,1]=+1*fp[grpname][:,13,12,11,1]
                    datapp = +1*data1+1*data2+1*data5+1*data6+1*data9 +1*data10+1*data13+1*data14+1*data17+1*data18+1*data21+1*data22
                    datamm = +1*data3+1*data4+1*data7+1*data8+1*data11+1*data12+1*data15+1*data16+1*data19+1*data20+1*data23+1*data24
                    dset=grp.create_dataset("OmegaMn/Ppgigj-Cgi-Cgj", datapp.shape, dtype = datapp.dtype, data = datapp)
                    dset=grp.create_dataset("OmegaMn/Pmgigj-Cgi-Cgj", datamm.shape, dtype = datamm.dtype, data = datamm)
                if src+"/Oms[+2.3e-02]" in fp:
                    grpname = src+"/Oms[+2.3e-02]"
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
                    grp = fo.require_group("/"+_dir+"_SS_gN25a4p_aN50a0p5/"+src+"/baryons_u[+4.0e-03]d[-4.0e-03]s[+2.3e-02]_only-s")
                    dset=grp.create_dataset("OmegaMn/Pp_Cgi_Cgi", datapp.shape, dtype = datapp.dtype, data = datapp)
                    #dset.attrs['description']="shape: /time/moms/{1},{1},{cg1,cg2,cg3,cg1g4,cg2g4,cg3g4}{cg1,cg2,cg3,cg1g4,cg2g4,cg3g4}/S1/S2//re-im"
                    dset=grp.create_dataset("OmegaMn/Pm_Cgi_Cgi", datamm.shape, dtype = datamm.dtype, data = datamm)
                    #dset.attrs['description']="shape: /time/moms/{1},{1},{cg1,cg2,cg3,cg1g4,cg2g4,cg3g4}{cg1,cg2,cg3,cg1g4,cg2g4,cg3g4}/S1/S2//re-im"
		    #gy*gx*cgy-cgx
                    #Ppgygx -i 0 0 0
                    #        0 i 0 0
                    #        0 0 0 0
                    #        0 0 0 0
                    #Selecting cgy sink cgx source
                    data1 = fp[grpname][:,13,1,0,:]
                    data1[:,0]=+1*fp[grpname][:,13,1,0,1]
                    data1[:,1]=-1*fp[grpname][:,13,1,0,0]
                    data2 = fp[grpname][:,13,1,5,:]
                    data2[:,0]=-1*fp[grpname][:,13,1,5,1]
                    data2[:,1]=+1*fp[grpname][:,13,1,5,0]
                    #Pmgygx  0 0  0  0
                    #        0 0  0  0
                    #        0 0 -i  0
                    #        0 0  0 +i
                    data3 = fp[grpname][:,13,1,10,:]
                    data3[:,0]=+1*fp[grpname][:,13,1,10,1]
                    data3[:,1]=-1*fp[grpname][:,13,1,10,0]
                    data4 = fp[grpname][:,13,1,15,:]
                    data4[:,0]=-1*fp[grpname][:,13,1,15,1]
                    data4[:,1]=+1*fp[grpname][:,13,1,15,0]
                    #gx*gy*cgx-cgy
                    #Ppgxgy  i  0 0 0
                    #        0 -i 0 0
                    #        0  0 0 0
                    #        0  0 0 0
                    #Selecting cgx sink cgy source
                    data5 = fp[grpname][:,13,6,0,:]
                    data5[:,0]=-1*fp[grpname][:,13,6,0,1]
                    data5[:,1]=+1*fp[grpname][:,13,6,0,0]
                    data6 = fp[grpname][:,13,1,5,:]
                    data6[:,0]=+1*fp[grpname][:,13,6,5,1]
                    data6[:,1]=-1*fp[grpname][:,13,6,5,0]
                    #Pmgxgy  0 0  0  0
                    #        0 0  0  0
                    #        0 0 +i  0
                    #        0 0  0 -i
                    data7 = fp[grpname][:,13,1,10,:]
                    data7[:,0]=-1*fp[grpname][:,13,6,10,1]
                    data7[:,1]=+1*fp[grpname][:,13,6,10,0]
                    data8 = fp[grpname][:,13,1,15,:]
                    data8[:,0]=+1*fp[grpname][:,13,6,15,1]
                    data8[:,1]=-1*fp[grpname][:,13,6,15,0]
                    #gz*gy*cgz-cgy
                    #Ppgzgy  0 -i 0 0
                    #       -i  0 0 0
                    #        0  0 0 0
                    #        0  0 0 0
                    #Selecting cgz sink cgy source
                    #gz*gy*Cgz-Cgy
                    data9 = fp[grpname][:,13,8,4,:]
                    data9[:,0]=+1*fp[grpname][:,13,8,4,1]
                    data9[:,1]=-1*fp[grpname][:,13,8,4,0]
                    data10 = fp[grpname][:,13,8,1,:]
                    data10[:,0]=+1*fp[grpname][:,13,8,1,1]
                    data10[:,1]=-1*fp[grpname][:,13,8,1,0]
                    #Pmgzgy  0  0  0  0
                    #        0  0  0  0
                    #        0  0  0 -i
                    #        0  0 -i  0
                    data11 = fp[grpname][:,13,8,14,:]
                    data11[:,0]=+1*fp[grpname][:,13,8,14,1]
                    data11[:,1]=-1*fp[grpname][:,13,8,14,0]
                    data12 = fp[grpname][:,13,8,11,:]
                    data12[:,0]=+1*fp[grpname][:,13,8,11,1]
                    data12[:,1]=-1*fp[grpname][:,13,8,11,0]
                    #gy*gz*cgy-cgz
                    #Ppgygz  0  i 0 0
                    #        i  0 0 0
                    #        0  0 0 0
                    #        0  0 0 0
                    #Selecting cgz sink cgy source
                    #gy*gz*Cgy-Cgz
                    data13 = fp[grpname][:,13,13,4,:]
                    data13[:,0]=-1*fp[grpname][:,13,13,4,1]
                    data13[:,1]=+1*fp[grpname][:,13,13,4,0]
                    data14 = fp[grpname][:,13,13,1,:]
                    data14[:,0]=-1*fp[grpname][:,13,13,1,1]
                    data14[:,1]=+1*fp[grpname][:,13,13,1,0]
                    #Pmgygz  0  0 0 0
                    #        0  0 0 0
                    #        0  0 0 i
                    #        0  0 i 0
                    data15 = fp[grpname][:,13,13,14,:]
                    data15[:,0]=-1*fp[grpname][:,13,13,14,1]
                    data15[:,1]=+1*fp[grpname][:,13,13,14,0]
                    data16 = fp[grpname][:,13,13,11,:]
                    data16[:,0]=-1*fp[grpname][:,13,13,11,1]
                    data16[:,1]=+1*fp[grpname][:,13,13,11,0]
                    #gz*gx*cgz-cgx
                    #Ppgzgx   0  1 0 0
                    #        -1  0 0 0
                    #         0  0 0 0
                    #         0  0 0 0
                    #Selecting cgz sink cgx source
                    #gzgx*Cgz-Cgx
                    data17 = fp[grpname][:,13,2,4,:]
                    data17[:,0]=+1*fp[grpname][:,13,2,4,0]
                    data17[:,1]=+1*fp[grpname][:,13,2,4,1]
                    data18 = fp[grpname][:,13,2,1,:]
                    data18[:,0]=-1*fp[grpname][:,13,2,1,0]
                    data18[:,1]=-1*fp[grpname][:,13,2,1,1]
                    #Pmgzgx   0  0  0 0
                    #         0  0  0 0
                    #         0  0  0 1
                    #         0  0 -1 0
                    data19 = fp[grpname][:,13,2,14,:]
                    data19[:,0]=+1*fp[grpname][:,13,2,14,0]
                    data19[:,1]=+1*fp[grpname][:,13,2,14,1]
                    data20 = fp[grpname][:,13,2,11,:]
                    data20[:,0]=-1*fp[grpname][:,13,2,11,0]
                    data20[:,1]=-1*fp[grpname][:,13,2,11,1]
                    #gx*gz*cgx-cgz
                    #Ppgxgz   0 -1 0 0
                    #        +1  0 0 0
                    #         0  0 0 0
                    #         0  0 0 0
                    #Selecting cgx sink cgz source
                    #gzgx*Cgz-Cgx
                    data21 = fp[grpname][:,13,12,4,:]
                    data21[:,0]=+1*fp[grpname][:,13,12,4,0]
                    data21[:,1]=+1*fp[grpname][:,13,12,4,1]
                    data22 = fp[grpname][:,13,12,1,:]
                    data22[:,0]=-1*fp[grpname][:,13,12,1,0]
                    data22[:,1]=-1*fp[grpname][:,13,12,1,1]
                    data23 = fp[grpname][:,13,12,14,:]
                    data23[:,0]=+1*fp[grpname][:,13,12,14,0]
                    data23[:,1]=+1*fp[grpname][:,13,12,14,1]
                    data24 = fp[grpname][:,13,12,11,:]
                    data24[:,0]=-1*fp[grpname][:,13,12,11,0]
                    data24[:,1]=-1*fp[grpname][:,13,12,11,1]
                    datapp = +1*data1+1*data2+1*data5+1*data6+1*data9 +1*data10+1*data13+1*data14+1*data17+1*data18+1*data21+1*data22
                    datamm = +1*data3+1*data4+1*data7+1*data8+1*data11+1*data12+1*data15+1*data16+1*data19+1*data20+1*data23+1*data24
                    dset=grp.create_dataset("OmegaMn/Ppgigj-Cgi-Cgj", datapp.shape, dtype = datapp.dtype, data = datapp)
                    dset=grp.create_dataset("OmegaMn/Pmgigj-Cgi-Cgj", datamm.shape, dtype = datamm.dtype, data = datamm)
                if src+"/Oms[+2.7e-02]" in fp:
                    grpname = src+"/Oms[+2.7e-02]"
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
                    grp = fo.require_group("/"+_dir+"_SS_gN25a4p_aN50a0p5/"+src+"/baryons_u[+4.0e-03]d[-4.0e-03]s[+2.7e-02]_only-s")
                    dset=grp.create_dataset("OmegaMn/Pp_Cgi_Cgi", datapp.shape, dtype = datapp.dtype, data = datapp)
                    #dset.attrs['description']="shape: /time/moms/{1},{1},{cg1,cg2,cg3,cg1g4,cg2g4,cg3g4}{cg1,cg2,cg3,cg1g4,cg2g4,cg3g4}/S1/S2//re-im"
                    dset=grp.create_dataset("OmegaMn/Pm_Cgi_Cgi", datamm.shape, dtype = datamm.dtype, data = datamm)
                    #dset.attrs['description']="shape: /time/moms/{1},{1},{cg1,cg2,cg3,cg1g4,cg2g4,cg3g4}{cg1,cg2,cg3,cg1g4,cg2g4,cg3g4}/S1/S2//re-im"
                    #gy*gx*cgy-cgx
                    #Ppgygx -i 0 0 0
                    #        0 i 0 0
                    #        0 0 0 0
                    #        0 0 0 0
                    #Selecting cgy sink cgx source
                    data1 = fp[grpname][:,13,1,0,:]
                    data1[:,0]=+1*fp[grpname][:,13,1,0,1]
                    data1[:,1]=-1*fp[grpname][:,13,1,0,0]
                    data2 = fp[grpname][:,13,1,5,:]
                    data2[:,0]=-1*fp[grpname][:,13,1,5,1]
                    data2[:,1]=+1*fp[grpname][:,13,1,5,0]
                    #Pmgygx  0 0  0  0
                    #        0 0  0  0
                    #        0 0 -i  0
                    #        0 0  0 +i
                    data3 = fp[grpname][:,13,1,10,:]
                    data3[:,0]=+1*fp[grpname][:,13,1,10,1]
                    data3[:,1]=-1*fp[grpname][:,13,1,10,0]
                    data4 = fp[grpname][:,13,1,15,:]
                    data4[:,0]=-1*fp[grpname][:,13,1,15,1]
                    data4[:,1]=+1*fp[grpname][:,13,1,15,0]
                    #gx*gy*cgx-cgy
                    #Ppgxgy  i  0 0 0
                    #        0 -i 0 0
                    #        0  0 0 0
                    #        0  0 0 0
                    #Selecting cgx sink cgy source
                    data5 = fp[grpname][:,13,6,0,:]
                    data5[:,0]=-1*fp[grpname][:,13,6,0,1]
                    data5[:,1]=+1*fp[grpname][:,13,6,0,0]
                    data6 = fp[grpname][:,13,1,5,:]
                    data6[:,0]=+1*fp[grpname][:,13,6,5,1]
                    data6[:,1]=-1*fp[grpname][:,13,6,5,0]
                    #Pmgxgy  0 0  0  0
                    #        0 0  0  0
                    #        0 0 +i  0
                    #        0 0  0 -i
                    data7 = fp[grpname][:,13,1,10,:]
                    data7[:,0]=-1*fp[grpname][:,13,6,10,1]
                    data7[:,1]=+1*fp[grpname][:,13,6,10,0]
                    data8 = fp[grpname][:,13,1,15,:]
                    data8[:,0]=+1*fp[grpname][:,13,6,15,1]
                    data8[:,1]=-1*fp[grpname][:,13,6,15,0]
                    #gz*gy*cgz-cgy
                    #Ppgzgy  0 -i 0 0
                    #       -i  0 0 0
                    #        0  0 0 0
                    #        0  0 0 0
                    #Selecting cgz sink cgy source
                    #gz*gy*Cgz-Cgy
                    data9 = fp[grpname][:,13,8,4,:]
                    data9[:,0]=+1*fp[grpname][:,13,8,4,1]
                    data9[:,1]=-1*fp[grpname][:,13,8,4,0]
                    data10 = fp[grpname][:,13,8,1,:]
                    data10[:,0]=+1*fp[grpname][:,13,8,1,1]
                    data10[:,1]=-1*fp[grpname][:,13,8,1,0]
                    #Pmgzgy  0  0  0  0
                    #        0  0  0  0
                    #        0  0  0 -i
                    #        0  0 -i  0
                    data11 = fp[grpname][:,13,8,14,:]
                    data11[:,0]=+1*fp[grpname][:,13,8,14,1]
                    data11[:,1]=-1*fp[grpname][:,13,8,14,0]
                    data12 = fp[grpname][:,13,8,11,:]
                    data12[:,0]=+1*fp[grpname][:,13,8,11,1]
                    data12[:,1]=-1*fp[grpname][:,13,8,11,0]
                    #gy*gz*cgy-cgz
                    #Ppgygz  0  i 0 0
                    #        i  0 0 0
                    #        0  0 0 0
                    #        0  0 0 0
                    #Selecting cgz sink cgy source
                    #gy*gz*Cgy-Cgz
                    data13 = fp[grpname][:,13,13,4,:]
                    data13[:,0]=-1*fp[grpname][:,13,13,4,1]
                    data13[:,1]=+1*fp[grpname][:,13,13,4,0]
                    data14 = fp[grpname][:,13,13,1,:]
                    data14[:,0]=-1*fp[grpname][:,13,13,1,1]
                    data14[:,1]=+1*fp[grpname][:,13,13,1,0]
                    #Pmgygz  0  0 0 0
                    #        0  0 0 0
                    #        0  0 0 i
                    #        0  0 i 0
                    data15 = fp[grpname][:,13,13,14,:]
                    data15[:,0]=-1*fp[grpname][:,13,13,14,1]
                    data15[:,1]=+1*fp[grpname][:,13,13,14,0]
                    data16 = fp[grpname][:,13,13,11,:]
                    data16[:,0]=-1*fp[grpname][:,13,13,11,1]
                    data16[:,1]=+1*fp[grpname][:,13,13,11,0]
                    #gz*gx*cgz-cgx
                    #Ppgzgx   0  1 0 0
                    #        -1  0 0 0
                    #         0  0 0 0
                    #         0  0 0 0
                    #Selecting cgz sink cgx source
                    #gzgx*Cgz-Cgx
                    data17 = fp[grpname][:,13,2,4,:]
                    data17[:,0]=+1*fp[grpname][:,13,2,4,0]
                    data17[:,1]=+1*fp[grpname][:,13,2,4,1]
                    data18 = fp[grpname][:,13,2,1,:]
                    data18[:,0]=-1*fp[grpname][:,13,2,1,0]
                    data18[:,1]=-1*fp[grpname][:,13,2,1,1]
                    #Pmgzgx   0  0  0 0
                    #         0  0  0 0
                    #         0  0  0 1
                    #         0  0 -1 0
                    data19 = fp[grpname][:,13,2,14,:]
                    data19[:,0]=+1*fp[grpname][:,13,2,14,0]
                    data19[:,1]=+1*fp[grpname][:,13,2,14,1]
                    data20 = fp[grpname][:,13,2,11,:]
                    data20[:,0]=-1*fp[grpname][:,13,2,11,0]
                    data20[:,1]=-1*fp[grpname][:,13,2,11,1]
                    #gx*gz*cgx-cgz
                    #Ppgxgz   0 -1 0 0
                    #        +1  0 0 0
                    #         0  0 0 0
                    #         0  0 0 0
                    #Selecting cgx sink cgz source
                    #sign of -1 from epsilon
                    #gzgx*Cgz-Cgx
                    data21 = fp[grpname][:,13,12,4,:]
                    data21[:,0]=-1*fp[grpname][:,13,12,4,0]
                    data21[:,1]=-1*fp[grpname][:,13,12,4,1]
                    data22 = fp[grpname][:,13,12,1,:]
                    data22[:,0]=+1*fp[grpname][:,13,12,1,0]
                    data22[:,1]=+1*fp[grpname][:,13,12,1,1]
                    data23 = fp[grpname][:,13,12,14,:]
                    data23[:,0]=-1*fp[grpname][:,13,12,14,0]
                    data23[:,1]=-1*fp[grpname][:,13,12,14,1]
                    data24 = fp[grpname][:,13,12,11,:]
                    data24[:,0]=+1*fp[grpname][:,13,12,11,0]
                    data24[:,1]=+1*fp[grpname][:,13,12,11,1]
                    datapp = +1*data1+1*data2+1*data5+1*data6+1*data9 +1*data10+1*data13+1*data14+1*data17+1*data18+1*data21+1*data22
                    datamm = +1*data3+1*data4+1*data7+1*data8+1*data11+1*data12+1*data15+1*data16+1*data19+1*data20+1*data23+1*data24
                    dset=grp.create_dataset("OmegaMn/Ppgigj-Cgi-Cgj", datapp.shape, dtype = datapp.dtype, data = datapp)
                    dset=grp.create_dataset("OmegaMn/Pmgigj-Cgi-Cgj", datamm.shape, dtype = datamm.dtype, data = datamm)
