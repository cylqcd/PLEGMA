# This file produce the contractions for all the interpolating field provided.
#
# The steps followed are :
#
# - Construct <O_bar O>
# - Contracting the propagator products (Wick rule)
# - Change of indices in prop. products and gamma
# - Take care of the gammas sign
# - Return the list of gammas indices and its sizes
#   for each flavor combination in the appropriate c++ code format.
#
# Run this as 'python baryons.py file1 file2'
# file1 will be the header file
# file2 will be a compilable file
# Authors: S.bacchio, E. Papadiofantous


# *************Imports*********************************

import copy
import numpy as np
import collections
from collections import defaultdict
import os
import sys
import re

sys.path.append(os.path.realpath("."))

# *************************Constants**********************************

NS = 4
I = complex(0, 1)

SORT_ORDER = {"up": 0, "dn": 1, "sr": 2, "ch": 3, "bt": 4}

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

#C = I * gy.dot(gt)
C = gy.dot(gt)

g5 = gx.dot(gy.dot(gz.dot(gt)))


mg5 = (-1)*g5
mgx = (-1)*gx
mgy = (-1)*gy
mgz = (-1)*gz

Cg5 = C.dot(g5)
mCg5=(-1)*Cg5
igtCg5 = I * gt.dot(Cg5)

Pminus = (1 / 2) * (one - gt)
Pplus = (1 / 2) * (one + gt)

Cgx = C.dot(gx)
Cgy = C.dot(gy)
Cgz = C.dot(gz)

g5C = g5.dot(C)
gxC = gx.dot(C)
gyC = gy.dot(C)
gzC = gz.dot(C)

g0_g5CDag_g0 = gt.dot( (g5C.conj().T).dot(gt))
g0_gxCDag_g0 = gt.dot( (gxC.conj().T).dot(gt))
g0_gyCDag_g0 = gt.dot( (gyC.conj().T).dot(gt))
g0_gzCDag_g0 = gt.dot( (gzC.conj().T).dot(gt))


gxgy = gx.dot(gy)
gygx = gy.dot(gx)
gygz = gy.dot(gz)
gzgy = gz.dot(gy)
gxgz = gx.dot(gz)
gzgx = gz.dot(gx)

all_gammas = {}

# negative sign for odd permutations of the epsilon tensor is
# taken into account by using mg5 instead of g5
# or in the last one via mgi in the first gamma matrix

all_gammas[0] = {
    "meson-meson" : {
    "g5-g5-g5-g5_ll" : {
            "gammaMatrix" :[
        (g5, g5, g5.conj().T, g5.conj().T)
    ],            
        },
    "g5-g5-gj-gj_ll" : {
            "gammaMatrix" :[
        (g5, g5, gx.conj().T, gx.conj().T),
        (g5, g5, gy.conj().T, gy.conj().T),
        (g5, g5, gz.conj().T, gz.conj().T)
    ],
        },
    "gj-gj-g5-g5_ll" : {
            "gammaMatrix" :[
        (gx, gx, g5.conj().T, g5.conj().T),
        (gy, gy, g5.conj().T, g5.conj().T),
        (gz, gz, g5.conj().T, g5.conj().T)
    ],
        },
    "gj-gj-gk-gk_ll" : {
            "gammaMatrix" :[
        (gx, gx, gx.conj().T, gx.conj().T),
        (gy, gy, gx.conj().T, gx.conj().T),
        (gz, gz, gx.conj().T, gx.conj().T),
        (gx, gx, gy.conj().T, gy.conj().T),
        (gy, gy, gy.conj().T, gy.conj().T),
        (gz, gz, gy.conj().T, gy.conj().T),
        (gx, gx, gz.conj().T, gz.conj().T),
        (gy, gy, gz.conj().T, gz.conj().T),
        (gz, gz, gz.conj().T, gz.conj().T)
    ],
        },
    },

    "meson-diquark" : {
     "g5-g5-g5C-Cg5_ll" : {
            "gammaMatrix" :[
        (g5, g5, g0_g5CDag_g0, mCg5)
    ],
        },
    "gj-gj-g5C-Cg5_ll" : {
            "gammaMatrix" :[
        (gx, gx, g0_g5CDag_g0, mCg5),
        (gy, gy, g0_g5CDag_g0, mCg5),
        (gz, gz, g0_g5CDag_g0, mCg5)
    ],
        },
            
    },

    "diquark-meson" : {
     "g5C-Cg5-g5-g5_ll": {
            "gammaMatrix" :[
        (g5C, Cg5, g5.conj().T, g5.conj().T)
    ],
        },
    "g5C-Cg5-gj-gj_ll": {
            "gammaMatrix" :[
        (g5C, Cg5, gx.conj().T, gx.conj().T),
        (g5C, Cg5, gy.conj().T, gy.conj().T),
        (g5C, Cg5, gz.conj().T, gz.conj().T)
    ],
        },
    },

    "diquark-diquark" : {
        "g5C-Cg5-g5C-Cg5_ll": {
            "gammaMatrix" :[
        (g5C, Cg5, g0_g5CDag_g0, mCg5)
        ],
        },
    },
}
            
            
            
#**************************************************            


all_gammas[1] = {
    "meson-meson" : {
    "g5-gj-g5-gj_ll" : {
            "gammaMatrix" :[
        (g5, gx, g5.conj().T, gx.conj().T),
        (g5, gy, g5.conj().T, gy.conj().T),
        (g5, gz, g5.conj().T, gz.conj().T)
    ],
        },
    "g5-gj-g5-gj_ll_12" : {
            "gammaMatrix" :[
        (g5, gx, g5.conj().T, gy.conj().T),
        (g5, gy, g5.conj().T, gx.conj().T)
    ],
        },
    "g5-g3-g5-g3_ll_z" : {
            "gammaMatrix" :[
        (g5, gz, g5.conj().T, gz.conj().T)
    ],
        },
    "g5-gj-gj-g5_ll" : {
            "gammaMatrix" :[
        (g5, gx, gx.conj().T, g5.conj().T),
        (g5, gy, gy.conj().T, g5.conj().T),
        (g5, gz, gz.conj().T, g5.conj().T)
    ],
        },
    "g5-gj-gj-g5_ll_12" : {
            "gammaMatrix" :[
        (g5, gx, gy.conj().T, g5.conj().T),
        (g5, gy, gx.conj().T, g5.conj().T)
    ],
        },
    "g5-g3-g3-g5_ll_z" : {
            "gammaMatrix" :[
        (g5, gz, gz.conj().T, g5.conj().T)
    ],
        },
    "gj-g5-g5-gj_ll" : {
            "gammaMatrix" :[
        (gx, g5, g5.conj().T, gx.conj().T),
        (gy, g5, g5.conj().T, gy.conj().T),
        (gz, g5, g5.conj().T, gz.conj().T)
    ],
        },
    "gj-g5-g5-gj_ll_12" : {
            "gammaMatrix" :[
        (gx, g5, g5.conj().T, gy.conj().T),
        (gy, g5, g5.conj().T, gx.conj().T)
    ],
        },
    "g3-g5-g5-g3_ll_z" : {
            "gammaMatrix" :[
        (gz, g5, g5.conj().T, gz.conj().T)
    ],
        },
    "gj-g5-gj-g5_ll" : {
            "gammaMatrix" :[
        (gx, g5, gx.conj().T, g5.conj().T),
        (gy, g5, gy.conj().T, g5.conj().T),
        (gz, g5, gz.conj().T, g5.conj().T)
    ],
        },
    "gj-g5-gj-g5_ll_12" : {
            "gammaMatrix" :[
        (gx, g5, gy.conj().T, g5.conj().T),
        (gy, g5, gx.conj().T, g5.conj().T)
    ],
        },
    "g3-g5-g3-g5_ll_z" : {
            "gammaMatrix" :[
        (gz, g5, gz.conj().T, g5.conj().T)
    ],
        },
    "gi-gj-g5-gk_ll" : {
            "gammaMatrix" :[
        (gy, gz, g5.conj().T, gx.conj().T),
        (gz, gy,mg5.conj().T, gx.conj().T),
        (gx, gz,mg5.conj().T, gy.conj().T),
        (gz, gx, g5.conj().T, gy.conj().T),
        (gx, gy, g5.conj().T, gz.conj().T),
        (gy, gx,mg5.conj().T, gz.conj().T)
    ],
        },
    "gi-gj-g5-gk_ll_12" : {
            "gammaMatrix" :[
        (gy, gz, g5.conj().T, gy.conj().T),
        (gz, gy,mg5.conj().T, gy.conj().T),
        (gx, gz,mg5.conj().T, gx.conj().T),
        (gz, gx, g5.conj().T, gx.conj().T)
    ],
        },
    "gi-gj-g5-gk_ll_z" : {
            "gammaMatrix" :[
        (gx, gy, g5.conj().T, gz.conj().T),
        (gy, gx,mg5.conj().T, gz.conj().T)
    ],
        },
    "gi-gj-gk-g5_ll" : {
            "gammaMatrix" :[
        (gy, gz, gx.conj().T, g5.conj().T),
        (gz, gy, gx.conj().T,mg5.conj().T),
        (gx, gz, gy.conj().T,mg5.conj().T),
        (gz, gx, gy.conj().T, g5.conj().T),
        (gx, gy, gz.conj().T, g5.conj().T),
        (gy, gx, gz.conj().T,mg5.conj().T)
    ],
        },
    "gi-gj-gk-g5_ll_12" : {
            "gammaMatrix" :[
        (gy, gz, gy.conj().T, g5.conj().T),
        (gz, gy, gy.conj().T,mg5.conj().T),
        (gx, gz, gx.conj().T,mg5.conj().T),
        (gz, gx, gx.conj().T, g5.conj().T)
    ],
        },
    "gi-gj-gk-g5_ll_z" : {
            "gammaMatrix" :[
        (gx, gy, gz.conj().T, g5.conj().T),
        (gy, gx, gz.conj().T,mg5.conj().T)
    ],
        },
    "g5-gi-gj-gk_ll" : {
            "gammaMatrix" :[
        ( g5, gx, gy.conj().T, gz.conj().T),
        (mg5, gx, gz.conj().T, gy.conj().T),
        (mg5, gy, gx.conj().T, gz.conj().T),
        ( g5, gy, gz.conj().T, gx.conj().T),
        ( g5, gz, gx.conj().T, gy.conj().T),
        (mg5, gz, gy.conj().T, gx.conj().T)
    ],
        },
    "g5-gi-gj-gk_ll_12" : {
            "gammaMatrix" :[
        (mg5, gx, gx.conj().T, gz.conj().T),
        ( g5, gx, gz.conj().T, gx.conj().T),
        ( g5, gy, gy.conj().T, gz.conj().T),
        (mg5, gy, gz.conj().T, gy.conj().T)
    ],
        },
    "g5-gi-gj-gk_ll_z" : {
            "gammaMatrix" :[
        ( g5, gz, gx.conj().T, gy.conj().T),
        (mg5, gz, gy.conj().T, gx.conj().T)
    ],
        },
    "gi-g5-gj-gk_ll" : {
            "gammaMatrix" :[
        (gx, g5, gy.conj().T, gz.conj().T),
        (gx,mg5, gz.conj().T, gy.conj().T),
        (gy,mg5, gx.conj().T, gz.conj().T),
        (gy, g5, gz.conj().T, gx.conj().T),
        (gz, g5, gx.conj().T, gy.conj().T),
        (gz,mg5, gy.conj().T, gx.conj().T)
    ],
        },
    "gi-g5-gj-gk_ll_12" : {
            "gammaMatrix" :[
        (gx,mg5, gx.conj().T, gz.conj().T),
        (gx, g5, gz.conj().T, gx.conj().T),
        (gy, g5, gy.conj().T, gz.conj().T),
        (gy,mg5, gz.conj().T, gy.conj().T)
    ],
        },
    "gi-g5-gj-gk_ll_z" : {
            "gammaMatrix" :[
        (gz, g5, gx.conj().T, gy.conj().T),
        (gz,mg5, gy.conj().T, gx.conj().T)
    ],
        },
    "gi-gj-gi-gj_ll" : {
            "gammaMatrix" :[
        ( gy, gz, gy.conj().T, gz.conj().T),
        (mgy, gz, gz.conj().T, gy.conj().T),
        ( gz, gy, gz.conj().T, gy.conj().T),
        (mgz, gy, gy.conj().T, gz.conj().T),
        ( gx, gz, gx.conj().T, gz.conj().T),
        (mgx, gz, gz.conj().T, gx.conj().T),
        ( gz, gx, gz.conj().T, gx.conj().T),
        (mgz, gx, gx.conj().T, gz.conj().T),
        ( gx, gy, gx.conj().T, gy.conj().T),
        (mgx, gy, gy.conj().T, gx.conj().T),
        ( gy, gx, gy.conj().T, gx.conj().T),
        (mgy, gx, gx.conj().T, gy.conj().T)
    ],
        },
    "gi-gj-gi-gj_ll_12" : {
            "gammaMatrix" :[
        ( gy, gz, gz.conj().T, gx.conj().T),
        (mgy, gz, gx.conj().T, gz.conj().T),
        (mgz, gy, gz.conj().T, gx.conj().T),
        ( gz, gy, gx.conj().T, gz.conj().T),
        ( gz, gx, gy.conj().T, gz.conj().T),
        (mgz, gx, gz.conj().T, gy.conj().T),
        (mgx, gz, gy.conj().T, gz.conj().T),
        ( gx, gz, gz.conj().T, gy.conj().T)
    ],
        },
    "gi-gj-gi-gj_ll_z" : {
            "gammaMatrix" :[
        ( gx, gy, gx.conj().T, gy.conj().T),
        (mgx, gy, gy.conj().T, gx.conj().T),
        ( gy, gx, gy.conj().T, gx.conj().T),
        (mgy, gx, gx.conj().T, gy.conj().T)
    ],
        },
            
    },

    "meson-diquark" : {
    "g5-gj-gjC-Cg5_ll" : {
            "gammaMatrix" :[
        (g5, gx, g0_gxCDag_g0, mCg5),
        (g5, gy, g0_gyCDag_g0, mCg5),
        (g5, gz, g0_gzCDag_g0, mCg5)
    ],
        },
    "g5-gj-gjC-Cg5_ll_12" : {
            "gammaMatrix" :[
        (g5, gx, g0_gyCDag_g0, mCg5),
        (g5, gy, g0_gxCDag_g0, mCg5)
    ],
        },
    "g5-g3-g3C-Cg5_ll_z" : {
            "gammaMatrix" :[
        (g5, gz, g0_gzCDag_g0, mCg5)
    ],
        },
    "gj-g5-gjC-Cg5_ll" : {
            "gammaMatrix" :[
        (gx, g5, g0_gxCDag_g0, mCg5),
        (gy, g5, g0_gyCDag_g0, mCg5),
        (gz, g5, g0_gzCDag_g0, mCg5)
    ],
        },
    "gj-g5-gjC-Cg5_ll_12" : {
            "gammaMatrix" :[
        (gx, g5, g0_gyCDag_g0, mCg5),
        (gy, g5, g0_gxCDag_g0, mCg5)
    ],
        },
    "g3-g5-g3C-Cg5_ll_z" : {
            "gammaMatrix" :[
        (gz, g5, g0_gzCDag_g0, mCg5)
    ],
        },
    "gi-gj-gkC-Cg5_ll" : {
            "gammaMatrix" :[
        (gy, gz, g0_gxCDag_g0, mCg5),
        (gz,mgy, g0_gxCDag_g0, mCg5),
        (gx,mgz, g0_gyCDag_g0, mCg5),
        (gz, gx, g0_gyCDag_g0, mCg5),
        (gx, gy, g0_gzCDag_g0, mCg5),
        (gy,mgx, g0_gzCDag_g0, mCg5)
    ],
        },
    "gi-gj-gkC-Cg5_ll_12" : {
            "gammaMatrix" :[
        (gy, gz, g0_gyCDag_g0, mCg5),
        (gz,mgy, g0_gyCDag_g0, mCg5),
        (gx,mgz, g0_gxCDag_g0, mCg5),
        (gz, gx, g0_gxCDag_g0, mCg5)
    ],
        },
    "gi-gj-gkC-Cg5_ll_z" : {
            "gammaMatrix" :[
        (gx, gy, g0_gzCDag_g0, mCg5),
        (gy,mgx, g0_gzCDag_g0, mCg5)
    ],
        },
    },

    "diquark-meson" : {
     "gjC-Cg5-g5-gj_ll": {
            "gammaMatrix" :[
        (gxC, Cg5, g5.conj().T, gx.conj().T),
        (gyC, Cg5, g5.conj().T, gy.conj().T),
        (gzC, Cg5, g5.conj().T, gz.conj().T)
    ],
        },
     "gjC-Cg5-g5-gj_ll_12": {
            "gammaMatrix" :[
        (gxC, Cg5, g5.conj().T, gy.conj().T),
        (gyC, Cg5, g5.conj().T, gx.conj().T)
    ],
        },
     "g3C-Cg5-g5-g3_ll_z": {
            "gammaMatrix" :[
        (gzC, Cg5, g5.conj().T, gz.conj().T)
    ],
        },
    "gjC-Cg5-gj-g5_ll": {
            "gammaMatrix" :[
        (gxC, Cg5, gx.conj().T, g5.conj().T),
        (gyC, Cg5, gy.conj().T, g5.conj().T),
        (gzC, Cg5, gz.conj().T, g5.conj().T)
    ],
        },
    "gjC-Cg5-gj-g5_ll_12": {
            "gammaMatrix" :[
        (gxC, Cg5, gy.conj().T, g5.conj().T),
        (gyC, Cg5, gx.conj().T, g5.conj().T)
    ],
        },
    "g3C-Cg5-g3-g5_ll_z": {
            "gammaMatrix" :[
        (gzC, Cg5, gz.conj().T, g5.conj().T)
    ],
        },


    "giC-Cg5-gj-gk_ll": {
            "gammaMatrix" :[
        (gxC, Cg5, gy.conj().T, gz.conj().T),
        (gxC, Cg5,mgz.conj().T, gy.conj().T),
        (gyC, Cg5,mgx.conj().T, gz.conj().T),
        (gyC, Cg5, gz.conj().T, gx.conj().T),
        (gzC, Cg5, gx.conj().T, gy.conj().T),
        (gzC, Cg5,mgy.conj().T, gx.conj().T)
    ],
        },
    "giC-Cg5-gj-gk_ll_12": {
            "gammaMatrix" :[
        (gxC, Cg5, gz.conj().T, gx.conj().T),
        (gxC, Cg5,mgx.conj().T, gz.conj().T),
        (gyC, Cg5, gy.conj().T, gz.conj().T),
        (gyC, Cg5,mgz.conj().T, gy.conj().T)
    ],
        },
    "giC-Cg5-gj-gk_ll_z": {
            "gammaMatrix" :[
        (gzC, Cg5, gx.conj().T, gy.conj().T),
        (gzC, Cg5,mgy.conj().T, gx.conj().T)
    ],
        },
    },

    "diquark-diquark" : {
        "gjC-Cg5-gjC-Cg5_ll": {
            "gammaMatrix" :[
        (gxC, Cg5, g0_gxCDag_g0, mCg5),
        (gyC, Cg5, g0_gyCDag_g0, mCg5),
        (gzC, Cg5, g0_gzCDag_g0, mCg5)
        ],
        },
        "gjC-Cg5-gjC-Cg5_ll_12": {
            "gammaMatrix" :[
        (gxC, Cg5, g0_gyCDag_g0, mCg5),
        (gyC, Cg5, g0_gxCDag_g0, mCg5)
        ],
        },
        "g3C-Cg5-g3C-Cg5_ll_z": {
            "gammaMatrix" :[
        (gzC, Cg5, g0_gzCDag_g0, mCg5)
        ],
        },
    },
}



   
    
tetraquarks = {
      "bcud_J1" :  {"flavs"  : ["bt^-up-ch^-dn","bt^-dn-ch^-up"], #["bt^-up-bt^-dn", "bt^-dn-bt^-up"],
                    "coeffs" : [1,-1], #[1,-1],
                    "spin"   : 1
                  },
      "bcud_J0" :  {"flavs"  : ["bt^-up-ch^-dn","bt^-dn-ch^-up"], #["bt^-up-bt^-dn", "bt^-dn-bt^-up"],
                    "coeffs" : [1,-1], #[1,-1],
                    "spin"   : 0
                  }
}


# **********************Interpolation field*************************************
# The methods return the interpolating field for the given baryon
class quark(object):
    """
    Quark class. Store information about color and spin indices, flavor and conjugation.
    """

    def __init__(self, fl, anti=False, s=0, c=0):
        self.fl = fl
        self.c = c
        self.s = s
        self.anti = anti

    def __repr__(self):
        if self.anti:
            return "\\bar{%s}_{c_%d, s_%d}" % (self.fl, self.c, self.s)
        else:
            return "%s_{c_%d, s_%d}" % (self.fl, self.c, self.s)


class interpolating_field(object):
    """
    Interpolating field class. List of quarks and multiplicative coefficient
    """

    def __init__(self, flavs, coeff=1):
        self.quarks = []
        self.coeff = coeff
        for i, fl in enumerate(flavs.split("-")):
            if fl[-1] == "^":
                self.quarks.append(quark(fl[:-1], True, i, i))
            else:
                self.quarks.append(quark(fl, False, i, i))

    def __repr__(self):
        s = ""
        if self.coeff != 1:
            s += repr(self.coeff) + "*"
        s += "(" + ",".join([repr(q) for q in self.quarks]) + ")"
        return s
    
 
 
 
class propagator(object):
    """
    Propagator class. Store information about in/out color and spin indices and flavor
    """

    def __init__(self, quark1, quark2):
        if quark1.fl != quark2.fl:
            error()
        if not quark1.anti and quark2.anti:
            self.fl = quark1.fl
            self.c = (quark1.c, quark2.c)
            self.s = (quark1.s, quark2.s)
        else:
            error()

    def __repr__(self):
        return "<%s\\bar{%s}>_{c_{%d,%d}, s_{%d,%d}}" % (
            self.fl,
            self.fl,
            self.c[0],
            self.c[1],
            self.s[0],
            self.s[1],
        )
    
    
def incr_s_c(quarks):
    """
    Set increasing value for spin and color indices in list of quarks
    """
    for i, q in enumerate(quarks):
        q.s = i
        q.c = i
    return quarks


def wick(quarks):
    """
    Perform wick contraction of list of quarks returning list of propagators and signs
    """
    q0 = quarks.pop(0)
    match = []
    others = []
    signs = []
    for i, q in enumerate(quarks):
        
        if q.fl == q0.fl and q.anti != q0.anti:
            match.append(propagator(q0, q) if q.anti else propagator(q, q0))
            others.append(quarks[:i] + quarks[i + 1 :])
            signs.append((+1 if i % 2 == 0 else -1) * (+1 if q.anti else -1))

    if match == []:
        error()

    result_props = []
    result_signs = []
    for i, qs in enumerate(others):
        if qs == []:
            result_props.append(
                [
                    match[i],
                ]
            )
            result_signs.append(signs[i])
        else:
            w_props, w_signs = wick(qs)
            for w_p, w_s in zip(w_props, w_signs):
                result_props.append(
                    [
                        match[i],
                    ]
                    + w_p
                )
                result_signs.append(signs[i] * w_s)

    return result_props, result_signs




class expectation_value(object):
    """
    Expectation value class. List of propagators with coefficients
    """

    def __init__(self, J, J_bar):
        self.propagators, signs = wick(incr_s_c(J.quarks + J_bar.quarks))
        coeff = J.coeff * J_bar.coeff
        self.coeffs = []
        for sign in signs:
            self.coeffs.append(coeff * sign)

    def default_view(self):
        """
        Gets the default view of the propagators in the class,
        giving a specific order for flavor, spin and color indeces
        """
        fls = [p.fl for p in self.propagators[0]]
        fls.sort(key=lambda val: SORT_ORDER[val])
        default = []
        for i, fl in enumerate(fls):
            q1 = quark(fl, False, 2 * i, i)
            q2 = quark(fl, True, 2 * i + 1, len(fls) + i)
            default.append(propagator(q1, q2))
        return default

    def map_to_default_view(self):
        """
        Returns the map of spin and color indeces to
        obtain the default view of the class
        """
        map_c = []
        map_s = []
        for props in self.propagators:
            print (props)
            my_map_c = []
            my_map_s = []
            default = self.default_view()
            print ("default   ", default)
            for prop in props:
                fl = prop.fl
                #print ("prop:  ", prop)
                #print ("fl:    ", fl)
                dprop = None
                for i, p in enumerate(default):
                    #print ("i,p:  ", i,p)
                    if p.fl == fl:
                        dprop = default.pop(i)
                        break
                #print ("dprop:  ", dprop)
                if dprop is None:
                    error()
                my_map_c += list(zip(prop.c, dprop.c))
                my_map_s += list(zip(prop.s, dprop.s))
            map_c.append(my_map_c)
            map_s.append(my_map_s)
        return map_c, map_s
    
    
    
    
    def get_color_indices(self):
        c_index = []
        for props in self.propagators:
            c_index_prop= []
            
            for prop in props:
                c_index_prop += list(prop.c)
                
            c_index.append(c_index_prop)
            
        return c_index

        
        
        

    def __repr__(self):
        ls = []
        for coeff, props in zip(self.coeffs, self.propagators):
            s = ""
            if coeff != 1:
                s = repr(coeff) + " "
            s += " * ".join([repr(p) for p in props])
            ls.append(s)
        return " + ".join(ls)
    
    


def baryon_bar(flavs):
    """
    Rule to bar the string of a baryon interpolating field
    """
    flavs = flavs.split("-")
    if len(flavs) != 3:
        error()
    new = [
        flavs[2],
    ] +     [
        flavs[0],
    ] +     [
        flavs[1],
    ]
    #flavs[:2]
    for i, fl in enumerate(new):
        new[i] += "^"
    return "-".join(new)






def meson_bar(flavs):
    """
    Rule to bar the string of a meson interpolating field
    """
    flavs = flavs.split("-")
    if len(flavs) != 2:
        error()
    new = [
        flavs[1],
    ] + flavs[:1]
    for i, fl in enumerate(new):
        if ("^" in new[i]):
            new[i]= new[i].replace("^","")
        else:
            new[i] += "^"
    return "-".join(new)



def tetraquark_bar(flavs):
    """
    Rule to bar the string of a tetraquark interpolating field
    """
    flavs = flavs.split("-")
    if len(flavs) != 4:
        error()
    new = [
        flavs[1],
    ] + [
        flavs[0],
    ] + [
        flavs[3],
    ] + [
        flavs[2],
    ]

    for i, fl in enumerate(new):
        if ("^" in new[i]):
            new[i]= new[i].replace("^","")
        else:
            new[i] += "^"
    return "-".join(new)



def tetraquark_diquark(flavs):
    """
    Rule to get the string of a tetraquark diquark interpolating field
    """
    flavs = flavs.split("-")
    if len(flavs) != 4:
        error()
    new = [
        flavs[0],
    ] + [
        flavs[2],
    ] + [
        flavs[1],
    ] + [
        flavs[3],
    ]

    return "-".join(new)


def perm_parity(lst):
    """
    Given a permutation of the digits 0..N in order as a list,
    returns its parity (or sign): +1 for even parity; -1 for odd.
    """
    parity = 1
    for i in range(0, len(lst) - 1):
        if lst[i] != i:
            parity *= -1
            mn = min(range(i, len(lst)), key=lst.__getitem__)
            lst[i], lst[mn] = lst[mn], lst[i]
    return parity



def sign_from_map_c(map_c):
    """
    Takes into account the sign produced by remapping of color indeces in the epsilon
    """
    e = [0] * len(map_c)
    for m in map_c:
        e[m[0]] = m[1]
    return perm_parity(e)


def color_contractions(indices_c, op_type, mapes_c):
    """
    compute color contractions based on operator structure
    """
    
    ordering_c=[(0,0),(4,1),(1,2),(5,3),(2,4),(6,5),(3,6),(7,7)]
    
    op_type_arr = op_type.split("-")
    contracted_indices=[]
    
    for index_c,map_c in zip(indices_c, mapes_c):
        contracted_c_index = [0] * len(index_c)
        
        i = 0
        for ic in index_c:
            
            contracted_c=0
            
            if (op_type_arr[0] == "meson" and ic < 4):
                contracted_c = (int)(ic / 2)
                
            elif (op_type_arr[0] == "diquark" and ic < 4 ):
                contracted_c = ic % 2
            
            if (op_type_arr[1] == "meson" and ic >= 4):
                contracted_c = (int)(ic / 2)

            elif (op_type_arr[1] == "diquark" and ic >= 4):
                 contracted_c = ic % 2 + 2
             
             
            new_pos = 0
            for mc in map_c:
               
                if (mc[0] == ic):
                    new_pos = mc[1]
                    
            for oc in ordering_c:
                if(oc[0] == new_pos):
                   contracted_c_index[oc[1]] = contracted_c
            
        #print (contracted_c_index)
        contracted_indices += [ contracted_c_index ,]
    
    return contracted_indices





# note, that this is only valid for mesonic structures.
# however, this is not a problem, as diquarks always have a local structure.
def reorder_positions(indices_c, quarkPos, mapes_c):
    """
    compute color contractions based on operator structure
    """
    
    ordering_c=[(0,0),(4,1),(1,2),(5,3),(2,4),(6,5),(3,6),(7,7)]
    
    op_type_arr = op_type.split("-")
    reorderedQuarkPos=[]
    
    for index_c,map_c in zip(indices_c, mapes_c):
        newQuarkPos = [0] * len(index_c)
        
        i = 0
        for ic in index_c:
            
            position = quarkPos[ic]
      
             
            new_pos = 0
            for mc in map_c:
               
                if (mc[0] == ic):
                    new_pos = mc[1]
                    
            for oc in ordering_c:
                if(oc[0] == new_pos):
                   newQuarkPos[oc[1]] = position
            
        #print (contracted_c_index)
        reorderedQuarkPos += [ newQuarkPos ,]
    
    return reorderedQuarkPos



def convertPositionToIndex(reorderedQuarkPos):
    
    posIndices=[]
    
    for quarkPos in reorderedQuarkPos:
        #print (quarkPos)
        NProps= int(len(quarkPos)/2)
        #print (NProps)
        posIndex=[]
        
        
        
        for i in range (NProps):
            if (i in posIndex):
                #print ("skip")
                continue
            
            for j in range (i+1,NProps):
                #print (i,j)
                #print (quarkPos[2*i], quarkPos[2*i+1] , quarkPos[2*j] ,quarkPos[2*j+1] )
                
                if ( ( (quarkPos[2*i] == quarkPos[2*j]) and (quarkPos[2*i+1] == quarkPos[2*j+1]) ) or 
                     ( (quarkPos[2*i] == quarkPos[2*j+1]) and (quarkPos[2*i+1] == quarkPos[2*j]) )):
                    #print(i,j)
                    posIndex.append(i)
                    posIndex.append(j)
                    #print (posIndex)
                    break
                
        posIndices  += [   posIndex,]
        
    return posIndices
                


    
    

def non_zero_combinations(matrices):
    """
    Return the non-zero indeces of the outer product of matrices
    """
    all_s1 = [[]]
    all_s2 = [[]]
    all_val = [1]
    for m in matrices:
        #print ("m:  ",m)
        s1, s2 = np.nonzero(m)
        #print ("s1:  ", s1)
        #print ("s2:  ", s2)
        val = m[np.nonzero(m)]
        tmp_s1 = []
        tmp_s2 = []
        tmp_val = []
        for i1, i2, i in zip(all_s1, all_s2, all_val):
            #print( "i1:  ", i1)
            #print( "i2:  ", i2)
            #print( "i:   ", i)
            for j1, j2, j in zip(s1, s2, val):
                #print( "j1:  ", j1)
                #print( "j2:  ", j2)
                #print( "j:   ", j)
                tmp_s1.append(
                    i1
                    + [
                        j1,
                    ]
                )
                tmp_s2.append(
                    i2
                    + [
                        j2,
                    ]
                )
                tmp_val.append(i * j)
            
            #print ("tmp_s1:  ", tmp_s1)
            #print ("tmp_s2:  ", tmp_s2)
            #print ("tmp_val:  ", tmp_val)
            
        all_s1 = tmp_s1
        all_s2 = tmp_s2
        all_val = tmp_val
        
        #print ("all_s1:  ", all_s1)
        #print ("all_s2:  ", all_s2)
        #print ("all_val: ", all_val)
        
        #print ("------------")
            
    arr = np.array(list(zip(all_s1, all_s2))).transpose(0, 2, 1).reshape((-1, 8))
    return arr, all_val


def remap(arr, map_s):
    """
    Remap the list of indeces in arr with the map in map_s
    """
    new = []
    for a in arr:
        n = [0] * len(a)
        for ms in map_s:
            n[ms[1]] = a[ms[0]]
        new.append(n)
    return new




#-----------------------------------  
        
operator_types=["meson-meson","meson-diquark","diquark-meson","diquark-diquark"]
#operator_types=["meson-meson"]



#print ("Gammas:\n", all_gammas)


#print (all_gammas[1]['meson-meson'])



prop_prods = defaultdict(list)   
prop_infos = defaultdict(list)   
for tetraquark, info in tetraquarks.items(): 
    
    print("#####################################")
    print(tetraquark, info)
    print("#####################################")
    
    
    final_list = {}
    #final_infos = {}
    for op_type in operator_types:
        
        print (op_type)
        op_type_arr = op_type.split("-")
        print (op_type_arr)
        
    
        if (op_type_arr[0] == "meson"):
            J = [
                interpolating_field(flavs, coeff)
                for flavs, coeff in zip(info["flavs"], info["coeffs"])
                ]
        elif (op_type_arr[0] == "diquark"):
            J = [
                interpolating_field(tetraquark_diquark(flavs), coeff)
                for flavs, coeff in zip(info["flavs"], info["coeffs"])
            ] 
            
            
        if (op_type_arr[1] == "meson"):
            J_bar = [
                interpolating_field(tetraquark_bar(flavs), coeff)
                for flavs, coeff in zip(info["flavs"], info["coeffs"])
                ]
        elif (op_type_arr[1] == "diquark"):
            J_bar = [
                interpolating_field(tetraquark_bar(tetraquark_diquark(flavs)), coeff)
                for flavs, coeff in zip(info["flavs"], info["coeffs"])
            ]
        
        print ("J: ", J)
        print ("Jbar: " ,J_bar)

    
        exps = [expectation_value(j, j_bar) for j in J for j_bar in J_bar]
        
        
        
        print("\n\n------ Exps -------\n")
        print (exps)
        i=0
        for exp in exps:
            print (i)
            i+=1
            print (exp)
            print(exp.coeffs)
            print ("Color Indices:", exp.get_color_indices())
        print("\n----------------------\n\n") 
        ordered = exps[0].default_view()
        print("ordered: ", ordered)
        coeffs = []
        index_c = []
        map_c = []
        map_s = []
        for exp in exps:
            coeffs += exp.coeffs
            mc, ms = exp.map_to_default_view()
            map_c += mc
            map_s += ms
            index_c += exp.get_color_indices()
            

        
        print (map_c)
        print (map_s)
        #print(index_c)
        contracted_c_indices =color_contractions(index_c, op_type, map_c)
        print ("ColInd",contracted_c_indices)
        
        ## I think this is not needed as we have no baryons
        ##for i, mc in enumerate(map_c):
        ##    coeffs[i] *= sign_from_map_c(mc)
            
        #isZero=0
        #isDouble=0
        
    
        for gamma, matrices in all_gammas[info["spin"]][op_type].items():
            
            #newPositions= reorder_positions(index_c,matrices["quarkPositions"]  , map_c)
            #posIndices = convertPositionToIndex(newPositions)
            
            #print (posIndices)
            
            final_list[gamma] = {}
            #final_infos[gamma] = {}
            for gammas in matrices["gammaMatrix"]:
                non_zero, vals = non_zero_combinations([gammas[0], gammas[1], gammas[2],gammas[3]])
                #print(non_zero[0:20])
                #print ("length gammas:", len(non_zero))
                #print(vals[0:20])
                #print (map_s) 
                for i, ms in enumerate(map_s):
                    #print ("i: ", i, "  ms: ",ms)
                    #print (contracted_c_indices[i])
                    mapped = remap(non_zero, ms)
                    for m, v in zip(mapped, vals):
                        #m = repr(m)
                        m = (repr( m )+ repr("--") + repr( contracted_c_indices[i]))
                        
                        #print (m)
                        #print (v)
                        if m in final_list[gamma]:
                            final_list[gamma][m] += v * coeffs[i]
                            #isDouble+=1
                        else:
                            final_list[gamma][m] = v * coeffs[i]
            for m, v in list(final_list[gamma].items()):
                if np.isclose(v, 0):
                    #isZero+=1
                    del final_list[gamma][m] 
                    

        #print (final_list['g5-gj-g5-gj']) 
        
        #print ( "isZero:  ", isZero)
        #print ( "isDouble:  ", isDouble)

    
    flv_key = "-".join([p.fl for p in ordered])
    
    #print ("flv_key: ", flv_key)
    
    prop_prods[flv_key].append([tetraquark, final_list])




            
        


    
# ************************************Tools**************************************************************
def write_gamma_line(line):
    string = ""
    for elem in line:
        if elem == 0:
            string += "  0,"
        if elem == 1:
            string += "  1,"
        if elem == -1:
            string += " -1,"
        if elem == 1j:
            string += "  I,"
        if elem == -1j:
            string += " -I,"
    return string


def remove_symbols(name):
    name = name.replace("+", "Pl")
    name = name.replace("-", "Mn")
    name = name.replace("*", "St")
    name = name.replace("'", "Pr")
    return name



file1 = open(sys.argv[1], "w")
file2 = open(sys.argv[2], "w")

# Printing the header
for f in file1, file2:
    f.write("/* This file had been generated by python/tetraquarks.py\n")
    f.write(" * \n")
    f.write(" * The executed command has been:\n")
    f.write(" * python " + " ".join(sys.argv) + "\n")
    f.write(" * \n")
    f.write(" * The gamma used are :\n")
    f.write(" * gx = [ " + write_gamma_line(gx[0]) + "\n")
    f.write(" *        " + write_gamma_line(gx[1]) + "\n")
    f.write(" *        " + write_gamma_line(gx[2]) + "\n")
    f.write(" *        " + write_gamma_line(gx[3]) + " ] \n")
    f.write(" * gy = [ " + write_gamma_line(gy[0]) + "\n")
    f.write(" *        " + write_gamma_line(gy[1]) + "\n")
    f.write(" *        " + write_gamma_line(gy[2]) + "\n")
    f.write(" *        " + write_gamma_line(gy[3]) + " ] \n")
    f.write(" * gz = [ " + write_gamma_line(gz[0]) + "\n")
    f.write(" *        " + write_gamma_line(gz[1]) + "\n")
    f.write(" *        " + write_gamma_line(gz[2]) + "\n")
    f.write(" *        " + write_gamma_line(gz[3]) + " ] \n")
    f.write(" * gt = [ " + write_gamma_line(gt[0]) + "\n")
    f.write(" *        " + write_gamma_line(gt[1]) + "\n")
    f.write(" *        " + write_gamma_line(gt[2]) + "\n")
    f.write(" *        " + write_gamma_line(gt[3]) + " ] \n")
    f.write(" *\n")
    for fl_keys in tetraquarks.keys():
        f.write(" * " + fl_keys + " : " + repr(tetraquarks[fl_keys]) + "\n")
    f.write(" */\n")

# Printing the data
def vec(s):
    return "std::vector<" + s + ">"


file1.write("#pragma once\n")
file2.write("#include<PLEGMA.h>\n")
file2.write("#include<" + sys.argv[1] + ">\n")

for prop in prop_prods.values():
    for tetraquark in prop:
        for gamma, vals in tetraquark[1].items():
            string = (
                "float2 "
                + remove_symbols(tetraquark[0])
                + "_"
                + gamma.replace("-", "_")
                + "_vals"
                + "["
                + repr(len(vals))
                + "] = { "
            )
            for x in vals.values():
                string += "{" + repr(x.real) + ", " + repr(x.imag) + "}, "
            string += "};\n"
            file2.write(string)
            file1.write(
                "extern float2 "
                + remove_symbols(tetraquark[0])
                + "_"
                + gamma.replace("-", "_")
                + "_vals"
                + "["
                + repr(len(vals))
                + "];\n"
            )
            string = (
                "short "
                + remove_symbols(tetraquark[0])
                + "_"
                + gamma.replace("-", "_")
                + "_idxs"
                + "["
                + repr(len(vals))
                + "*8] = { "
            )
            for x in vals.keys():
                x_splitted=x.split("'--'")
                string += repr(x_splitted[0]).replace("'[", "").replace("]'", ", ")
            string += "};\n"
            #for x in vals.keys():
                #string += repr(x).replace("'[", "").replace("]'", ", ")
            #string += "};\n"
            file2.write(string)
            file1.write(
                "extern short "
                + remove_symbols(tetraquark[0])
                + "_"
                + gamma.replace("-", "_")
                + "_idxs"
                + "["
                + repr(len(vals))
                + "*8];\n"
            )
  
  
            string = (
                "short "
                + remove_symbols(tetraquark[0])
                + "_"
                + gamma.replace("-", "_")
                + "_col_contr"
                + "["
                + repr(len(vals))
                + "*8] = { "
            )
            for x in vals.keys():
                x_splitted=x.split("'--'")
                string += repr(x_splitted[1]).replace("'[", "").replace("]'", ", ")
            string += "};\n"

            file2.write(string)
            file1.write(
                "extern short "
                + remove_symbols(tetraquark[0])
                + "_"
                + gamma.replace("-", "_")
                + "_col_contr"
                + "["
                + repr(len(vals))
                + "*8];\n"
            )            

props = [
    '"' + "".join([p[0] for p in key_prop.split("-")]) + '"'
    for key_prop in prop_prods.keys()
]
string = (
    "const " + vec("std::string") + " TETRA_bcud_prop_prods = {" + ", ".join(props) + "};\n"
)
file1.write(string)

datasets = [
    "{"
    + ", ".join(
        [
            '"' + remove_symbols(tetraquark[0]) + "/" + gamma.replace("-", "_") + '"'
            for tetraquark in prop
            for gamma in tetraquark[1].keys()
        ]
    )
    + "}"
    for prop in prop_prods.values()
]
string = (
    "const "
    + vec(vec("std::string"))
    + " TETRA_bcud_prop_prods_names = {"
    + ", ".join(datasets)
    + "};\n"
)
file1.write(string)

datasets = [
    "{"
    + ", ".join([repr(len(gamma)) for tetraquark in prop for gamma in tetraquark[1].values()])
    + "}"
    for prop in prop_prods.values()
]
string = (
    "const "
    + vec(vec("short"))
    + " TETRA_bcud_prop_prods_count = {"
    + ", ".join(datasets)
    + "};\n"
)
file1.write(string)

datasets = [
    "{"
    + ", ".join(
        [
            remove_symbols(tetraquark[0]) + "_" + gamma.replace("-", "_") + "_idxs"
            for tetraquark in prop
            for gamma in tetraquark[1].keys()
        ]
    )
    + "}"
    for prop in prop_prods.values()
]
string = (
    "const "
    + vec(vec("short *"))
    + " TETRA_bcud_prop_prods_idxs = {"
    + ", ".join(datasets)
    + "};\n"
)
file1.write(string)

datasets = [
    "{"
    + ", ".join(
        [
            remove_symbols(tetraquark[0]) + "_" + gamma.replace("-", "_") + "_vals"
            for tetraquark in prop
            for gamma in tetraquark[1].keys()
        ]
    )
    + "}"
    for prop in prop_prods.values()
]
string = (
    "const "
    + vec(vec("float2 *"))
    + " TETRA_bcud_prop_prods_vals = {"
    + ", ".join(datasets)
    + "};\n"
)
file1.write(string)


datasets = [
    "{"
    + ", ".join(
        [
            remove_symbols(tetraquark[0]) + "_" + gamma.replace("-", "_") + "_col_contr"
            for tetraquark in prop
            for gamma in tetraquark[1].keys()
        ]
    )
    + "}"
    for prop in prop_prods.values()
]
string = (
    "const "
    + vec(vec("short *"))
    + " TETRA_bcud_prop_prods_col_contr = {"
    + ", ".join(datasets)
    + "};\n"
)
file1.write(string)






