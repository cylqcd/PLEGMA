import numpy as np
import sys
import h5py

NS=4
I = complex(0,1)

gx = np.array([[ 0, 0, 0, I],
	       [ 0, 0, I, 0],
               [ 0,-I, 0, 0],
               [-I, 0, 0, 0]])

gy = np.array([[ 0, 0, 0, 1],
               [ 0, 0,-1, 0],
               [ 0,-1, 0, 0],
               [+1, 0, 0, 0]])

gz = np.array([[ 0, 0, I, 0],
               [ 0, 0, 0,-I],
               [-I, 0, 0, 0],
               [0,+I, 0, 0]])

gt = np.array([[ 1, 0, 0, 0],
               [ 0, 1, 0, 0],
               [ 0, 0,-1, 0],
               [ 0, 0, 0,-1]])

one = np.eye(NS, dtype=complex)
    
C = I*gy.dot(gt)

g5 = gx.dot(gy.dot(gz.dot(gt)))

Cg5 = C.dot(g5)

igtCg5= I*gt.dot(Cg5)

Pminus= (1/2)*(one - gt)
Pplus= (1/2)*(one + gt)

fname = sys.argv[1]

def load_group(groups, group):
    arr=[]
    if type(group) is list:
        for g in group:
            arr.append(load_group(groups, g))
        arr=np.array(arr).sum(axis=0)
    else:
        if group not in groups:
            print("Error: group", group, "not found.")
            return arr
        arr = np.array(groups[group])
    return arr

def check_baryons(groups, group1, group2, proj):
    bar1 = load_group(groups, group1)
    bar2 = load_group(groups, group2)

    bar1 = bar1[...,0]+I*bar1[...,1]
    bar2 = bar2[...,0]+I*bar2[...,1]

    bar2_shape=bar2.shape
    bar2 = np.trace(bar2.reshape(-1,NS,NS).dot(proj), axis1=-2, axis2=-1).reshape(bar2_shape[:-1])

    if bar1.shape != bar2.shape:
        print("Error: shapes", bar1.shape, bar2.shape, "do not match.")
        return

    ratio=(bar1-bar2)/(bar1+bar2)
    check=np.arange(int(ratio.shape[0]/4))-int(ratio.shape[0]/8)
    non_zero = np.isclose(ratio,np.zeros_like(ratio),atol=1e-3)
    
    if np.all(non_zero[check]):
        print("PASSED: groups", group1, group2, "match.")
    else:
        print("FAILED: groups", group1, group2, "do not match.")
        print("The following components do no pass the test: (first 10)")
        non_zero_comp = np.array(np.where(non_zero==False)).transpose()[:10]
        for i in non_zero_comp:
            j=tuple(i)
            print(j, bar1[j], bar2[j], np.abs(bar1[j]-bar2[j])/np.abs(bar1[j]+bar2[j]))

with h5py.File(fname, "r") as fp:
    for src, groups in fp.items():
        print("Running for src: ", src)
        if "baryons" not in groups:
            print("Error: baryons not found.")
            break

        check=None
        for key in groups:
            if "proton" in groups[key]:
                check=key
                break
        if check is None:
            print("Error: key for baryons_udsc not found.")
            break

        check_baryons(groups, key+"/proton/Pp_Cg5_Cg5", "baryons/nucl_nucl/twop_baryon_1", Pplus)
        check_baryons(groups, key+"/proton/Pm_Cg5_Cg5", "baryons/nucl_nucl/twop_baryon_1", Pminus)
        check_baryons(groups, key+"/neutron/Pp_Cg5_Cg5", "baryons/nucl_nucl/twop_baryon_2", Pplus)
        check_baryons(groups, key+"/neutron/Pm_Cg5_Cg5", "baryons/nucl_nucl/twop_baryon_2", Pminus)
        check_baryons(groups, key+"/proton/g5Ppg5_C_C", "baryons/nucl2_nucl2/twop_baryon_1", Pplus)
        check_baryons(groups, key+"/proton/g5Pmg5_C_C", "baryons/nucl2_nucl2/twop_baryon_1", Pminus)
        check_baryons(groups, key+"/neutron/g5Ppg5_C_C", "baryons/nucl2_nucl2/twop_baryon_2", Pplus)
        check_baryons(groups, key+"/neutron/g5Pmg5_C_C", "baryons/nucl2_nucl2/twop_baryon_2", Pminus)
        check_baryons(groups, key+"/DeltaPl/Pp_Cgi_Cgi", ["baryons/deltap_deltaz_11/twop_baryon_1",
                                                          "baryons/deltap_deltaz_22/twop_baryon_1",
                                                          "baryons/deltap_deltaz_33/twop_baryon_1"], -Pplus)
        check_baryons(groups, key+"/DeltaPl/Pm_Cgi_Cgi", ["baryons/deltap_deltaz_11/twop_baryon_1",
                                                          "baryons/deltap_deltaz_22/twop_baryon_1",
                                                          "baryons/deltap_deltaz_33/twop_baryon_1"], -Pminus)
        check_baryons(groups, key+"/Delta0/Pp_Cgi_Cgi", ["baryons/deltap_deltaz_11/twop_baryon_2",
                                                         "baryons/deltap_deltaz_22/twop_baryon_2",
                                                         "baryons/deltap_deltaz_33/twop_baryon_2"], -Pplus)
        check_baryons(groups, key+"/Delta0/Pm_Cgi_Cgi", ["baryons/deltap_deltaz_11/twop_baryon_2",
                                                         "baryons/deltap_deltaz_22/twop_baryon_2",
                                                         "baryons/deltap_deltaz_33/twop_baryon_2"], -Pminus)
        check_baryons(groups, key+"/DeltaPlPl/Pp_Cgi_Cgi", ["baryons/deltapp_deltamm_11/twop_baryon_1",
                                                            "baryons/deltapp_deltamm_22/twop_baryon_1",
                                                            "baryons/deltapp_deltamm_33/twop_baryon_1"], -Pplus)
        check_baryons(groups, key+"/DeltaPlPl/Pm_Cgi_Cgi", ["baryons/deltapp_deltamm_11/twop_baryon_1",
                                                            "baryons/deltapp_deltamm_22/twop_baryon_1",
                                                            "baryons/deltapp_deltamm_33/twop_baryon_1"], -Pminus)
        check_baryons(groups, key+"/DeltaMn/Pp_Cgi_Cgi", ["baryons/deltapp_deltamm_11/twop_baryon_2",
                                                          "baryons/deltapp_deltamm_22/twop_baryon_2",
                                                          "baryons/deltapp_deltamm_33/twop_baryon_2"], -Pplus)
        check_baryons(groups, key+"/DeltaMn/Pm_Cgi_Cgi", ["baryons/deltapp_deltamm_11/twop_baryon_2",
                                                          "baryons/deltapp_deltamm_22/twop_baryon_2",
                                                          "baryons/deltapp_deltamm_33/twop_baryon_2"], -Pminus)
