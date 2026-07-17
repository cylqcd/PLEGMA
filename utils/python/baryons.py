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


#*************Imports*********************************

import copy
import numpy as np
import collections
from collections import defaultdict
import os
import sys
import re
sys.path.append(os.path.realpath('.'))

#*************************Constants**********************************

NS = 4
I = complex(0,1)

SORT_ORDER = {"up": 0, "dn": 1, "sr": 2, "ch":3}

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

Cgx= C.dot(gx)
Cgy= C.dot(gy)
Cgz= C.dot(gz)

gxgy = gx.dot(gy)
gygx = gy.dot(gx)
gygz = gy.dot(gz)
gzgy = gz.dot(gy)
gxgz = gx.dot(gz)
gzgx = gz.dot(gx)

all_gammas={}

all_gammas[1/2] = {"Pp-Cg5-Cg5": [(Pplus,Cg5,Cg5)],
                   "Pm-Cg5-Cg5": [(Pminus,Cg5,Cg5)],
                   "Pp-igtCg5-igtCg5": [(Pplus,igtCg5,igtCg5)],
                   "Pm-igtCg5-igtCg5": [(Pminus,igtCg5,igtCg5)],
                   "g5Ppg5-C-C": [(g5.dot(Pplus).dot(g5),C,C)],
                   "g5Pmg5-C-C": [(g5.dot(Pminus).dot(g5),C,C)]}
all_gammas[3/2] = {"Pp-Cgi-Cgi": [(Pplus,Cgx,Cgx), (Pplus,Cgy,Cgy), (Pplus,Cgz,Cgz)],
                   "Pm-Cgi-Cgi": [(Pminus,Cgx,Cgx), (Pminus,Cgy,Cgy), (Pminus,Cgz,Cgz)],
                   "Ppgigj-Cgi-Cgj": [(Pplus.dot(gxgy),Cgx,Cgy),(Pplus.dot(gxgz),Cgx,Cgz),(Pplus.dot(gygz),Cgy,Cgz),
                                      (Pplus.dot(gygx),Cgy,Cgx),(Pplus.dot(gzgx),Cgz,Cgx),(Pplus.dot(gzgy),Cgz,Cgy)],
                   "Pmgigj-Cgi-Cgj": [(Pminus.dot(gxgy),Cgx,Cgy),(Pminus.dot(gxgz),Cgx,Cgz),(Pminus.dot(gygz),Cgy,Cgz),
                                      (Pminus.dot(gygx),Cgy,Cgx),(Pminus.dot(gzgx),Cgz,Cgx),(Pminus.dot(gzgy),Cgz,Cgy)]}

#***********************Baryon list***************************************
# List of interpolating fields.
# The quark flavors (flavs) "f1-f2-f3" are used as
#    J = ε^abc (f1_a GAMMA f2_b) f3_c
#                                _
# The expectation value O = <J(x)J(y)> is constructed as
#                                                       __       __           __
#  O = ε^abc ε^def <(f1^m_a gB_mn f2^n_b) (f3^m_c gA_mn f3^n_d) (f1^m_e gC_mn f2^n_f)>

baryons = {
      "Xicc++" :  {"flavs"  : ["ch-up-ch"],
                   "coeffs" : [1],
                   "spin"   : 1/2
                      },
       "Xicc+" :  {"flavs"  : ["ch-dn-ch"],
                   "coeffs" : [1],
                   "spin"   : 1/2
                      },
    "Omegacc+" :  {"flavs"  : ["ch-sr-ch"],
                   "coeffs" : [1],
                   "spin"   : 1/2
                      },
    "Sigmac++" :  {"flavs"  : ["up-ch-up"],
                   "coeffs" : [1],
                   "spin"   : 1/2
                      },
    "Sigmac+"  :  {"flavs"  : ["up-ch-dn","dn-ch-up"],
                   "coeffs" : [1,1],
                   "spin"   : 1/2,
                     },
    "Sigmac0"  :  {"flavs"  : ["dn-ch-dn"],
                   "coeffs" : [1],
                   "spin"   : 1/2
                     },
       "Xi'c+" :  {"flavs"  : ["up-ch-sr","sr-ch-up"],
                   "coeffs" : [1,1],
                   "spin"   : 1/2
                     },
       "Xi'c0" :  {"flavs"  : ["dn-ch-sr","sr-ch-dn"],
                   "coeffs" : [1,1],
                   "spin"   : 1/2
                     },
    "Omegac0"  :  {"flavs"  : ["sr-ch-sr"],
                   "coeffs" : [1],
                   "spin"   : 1/2
                     },
   "Lambdac+"  :  {"flavs"  : ["up-dn-ch","up-ch-dn","dn-ch-up"],
                   "coeffs" : [2,1,-1],
                   "spin"   : 1/2
                     },
       "Xic+"  :  {"flavs"  : ["sr-up-ch","sr-ch-up","up-ch-sr"],
                   "coeffs" : [2,1,-1],
                   "spin"   : 1/2
                     },
       "Xic0"  :  {"flavs"  : ["sr-dn-ch","sr-ch-dn","dn-ch-sr"],
                   "coeffs" : [2,1,-1],
                   "spin"   : 1/2
                     },
      "proton" :  {"flavs"  : ["up-dn-up"],
                   "coeffs" : [1],
                   "spin"   : 1/2
                     },
     "neutron" :  {"flavs"  : ["dn-up-dn"],
                   "coeffs" : [1],
                   "spin"   : 1/2
                     },
     "Lambda"  :  {"flavs"  : ["up-dn-sr","up-sr-dn","dn-sr-up"],
                   "coeffs" : [2,1,-1],
                   "spin"   : 1/2
                     },
    "Sigma+"   :  {"flavs"  : ["up-sr-up"],
                   "coeffs" : [1],
                   "spin"   : 1/2
                     },
    "Sigma0"   :  {"flavs"  : ["up-sr-dn","dn-sr-up"],
                   "coeffs" : [1,1],
                   "spin"   : 1/2
                     },
     "Sigma-"  :  {"flavs"  : ["dn-sr-dn"],
                   "coeffs" : [1],
                   "spin"   : 1/2
                     },
       "Xi0"   :  {"flavs"  : ["sr-up-sr"],
                   "coeffs" : [1],
                   "spin"   : 1/2
                     },
       "Xi-"   :  {"flavs"  : ["sr-dn-sr"],
                   "coeffs" : [1],
                   "spin"   : 1/2
                     },
   "Omegaccc++":  {"flavs"  : ["ch-ch-ch"],
                   "coeffs" : [1],
                   "spin"   : 3/2
                     },
     "Xi*cc++" :  {"flavs"  : ["ch-up-ch","ch-ch-up"],
                   "coeffs" : [2,1],
                   "spin"   : 3/2
                     },
     "Xi*cc+"  :  {"flavs"  : ["ch-dn-ch","ch-ch-dn"],
                   "coeffs" : [2,1],
                   "spin"   : 3/2
                     },
    "Omega*cc+":  {"flavs"  : ["ch-sr-ch","ch-ch-sr"],
                   "coeffs" : [2,1],
                   "spin"   : 3/2
                     },
    "Sigma*c++":  {"flavs"  : ["up-up-ch","ch-up-up"],
                   "coeffs" : [1,2],
                   "spin"   : 3/2
                     },
     "Sigma*c+":  {"flavs"  : ["up-dn-ch","dn-ch-up","ch-up-dn"],
                   "coeffs" : [1,1,1],
                   "spin"   : 3/2
                     },
     "Sigma*c0":  {"flavs"  : ["dn-dn-ch","ch-dn-dn"],
                   "coeffs" : [1,2],
                   "spin"   : 3/2
                     },
      "Xi*c+"  :  {"flavs"  : ["up-sr-ch","sr-ch-up","ch-up-sr"],
                   "coeffs" : [1,1,1],
                   "spin"   : 3/2
                     },
      "Xi*c0"  :  {"flavs"  : ["dn-sr-ch","sr-ch-dn","ch-dn-sr"],
                   "coeffs" : [1,1,1],
                   "spin"   : 3/2
                     },
     "Omega*c0":  {"flavs"  : ["sr-ch-sr","sr-sr-ch"],
                   "coeffs" : [2,1],
                   "spin"   : 3/2
                     },
      "Delta++":  {"flavs"  : ["up-up-up"],
                   "coeffs" : [1],
                   "spin"   : 3/2
                     },
      "Delta+" :  {"flavs"  : ["up-dn-up","up-up-dn"],
                   "coeffs" : [2,1],
                   "spin"   : 3/2
                     },
      "Delta0" :  {"flavs"  : ["dn-up-dn","dn-dn-up"],
                   "coeffs" : [2,1],
                   "spin"   : 3/2
                     },
      "Delta-" :  {"flavs"  : ["dn-dn-dn"],
                   "coeffs" : [1],
                   "spin"   : 3/2
                     },
      "Sigma*+":  {"flavs"  : ["up-up-sr","sr-up-up"],
                   "coeffs" : [1,2],
                   "spin"   : 3/2
                     },
      "Sigma*0":  {"flavs"  : ["up-dn-sr","dn-sr-up","sr-up-dn"],
                   "coeffs" : [1,1,1],
                   "spin"   : 3/2
                     },
      "Sigma*-":  {"flavs"  : ["dn-dn-sr","sr-dn-dn"],
                   "coeffs" : [1,2],
                   "spin"   : 3/2
                     },
      "Xi*0"   :  {"flavs"  : ["sr-up-sr","sr-sr-up"],
                   "coeffs" : [2,1],
                   "spin"   : 3/2
                     },
      "Xi*-"   :  {"flavs"  : ["sr-dn-sr","sr-sr-dn"],
                   "coeffs" : [2,1],
                   "spin"   : 3/2
                     },
      "Omega-" :  {"flavs"  : ["sr-sr-sr"],
                   "coeffs" : [1],
                   "spin"   : 3/2
                     }
    }

#**********************Interpolation field*************************************
# The methods return the interpolating field for the given baryon
class quark(object):
  '''
  Quark class. Store information about color and spin indeces, flavor and conjugation.
  '''
  def __init__(self,fl,anti=False,s=0,c=0):
    self.fl = fl
    self.c = c
    self.s = s
    self.anti = anti
  def __repr__(self) :
    if self.anti:
      return "\\bar{%s}_{c_%d, s_%d}" % (self.fl,self.c,self.s)
    else:
      return "%s_{c_%d, s_%d}" % (self.fl,self.c,self.s)

class interpolating_field(object):
  '''
  Interpolating field class. List of quarks and multiplicative coefficient
  '''
  def __init__(self,flavs,coeff=1):
    self.quarks=[]
    self.coeff=coeff
    for i,fl in enumerate(flavs.split("-")):
      if fl[-1] == "^":
        self.quarks.append(quark(fl[:-1],True,i,i))
      else:
        self.quarks.append(quark(fl,False,i,i))
  def __repr__(self) :
    s=""
    if self.coeff != 1:
      s+=repr(self.coeff)+" "
    s+=" ".join([repr(q) for q in self.quarks])
    return s

class propagator(object):
  '''
  Propagator class. Store infarmation about in/out color and spin indeces and flavor
  '''
  def __init__(self,quark1,quark2):
    if quark1.fl != quark2.fl:
      error()
    if not quark1.anti and quark2.anti:
      self.fl = quark1.fl
      self.c = (quark1.c, quark2.c)
      self.s = (quark1.s, quark2.s)
    else:
      error()
  def __repr__(self) :
    return "<%s\\bar{%s}>_{c_{%d,%d}, s_{%d,%d}}" % (self.fl,self.fl,self.c[0],self.c[1],self.s[0],self.s[1])

def incr_s_c(quarks):
  '''
  Set increasing value for spin and color indeces in list of quarks
  '''
  for i,q in enumerate(quarks):
    q.s=i
    q.c=i
  return quarks
  
def wick(quarks):
  '''
  Perform wick contraction of list of quarks returning list of propagators and signs
  '''
  q0=quarks.pop(0)
  match=[]
  others=[]
  signs=[]
  for i,q in enumerate(quarks):
    if q.fl == q0.fl and q.anti != q0.anti:
      match.append(propagator(q0,q) if q.anti else propagator(q,q0))
      others.append(quarks[:i]+quarks[i+1:])
      signs.append((+1 if i%2==0 else -1) * (+1 if q.anti else -1))
  
  if match == []:
    error()
    
  result_props=[]
  result_signs=[]
  for i,qs in enumerate(others):
    if qs == []:
      result_props.append([match[i],])
      result_signs.append(signs[i])
    else:
      w_props, w_signs = wick(qs)
      for w_p,w_s in zip(w_props, w_signs):
        result_props.append([match[i],]+w_p)
        result_signs.append(signs[i]*w_s)
        
  return result_props, result_signs

  
class expectation_value(object):
  '''
  Expectation value class. List of propagators with coefficients
  '''
  def __init__(self,J,J_bar):
    self.propagators, signs = wick(incr_s_c(J.quarks + J_bar.quarks))
    coeff = J.coeff*J_bar.coeff
    self.coeffs = []
    for sign in signs:
      self.coeffs.append(coeff*sign)

  def default_view(self):
    '''
    Gets the default view of the propagators in the class,
    giving a specific order for flavor, spin and color indeces
    '''
    fls=[p.fl for p in self.propagators[0]]
    fls.sort(key= lambda val : SORT_ORDER[val])
    default=[]
    for i,fl in enumerate(fls):
      q1=quark(fl,False,2*i,i)
      q2=quark(fl,True,2*i+1,len(fls)+i)
      default.append(propagator(q1,q2))
    return default

  def map_to_default_view(self):
    '''
    Returns the map of spin and color indeces to
    obtain the default view of the class
    '''
    map_c = []
    map_s = []
    for props in self.propagators:
      my_map_c=[]
      my_map_s=[]
      default=self.default_view()
      for prop in props:
        fl = prop.fl
        dprop = None
        for i,p in enumerate(default):
          if p.fl==fl:
            dprop = default.pop(i)
            break
        if dprop is None:
          error()
        my_map_c += list(zip(prop.c,dprop.c))
        my_map_s += list(zip(prop.s,dprop.s))
      map_c.append(my_map_c)
      map_s.append(my_map_s)
    return map_c,map_s

  def __repr__(self) :
    ls=[]
    for coeff, props in zip(self.coeffs, self.propagators):
      s=""
      if coeff != 1:
        s=repr(coeff)+" "
      s+=" * ".join([repr(p) for p in props])
      ls.append(s)
    return " + ".join(ls)

def baryon_bar(flavs):
  '''
  Rule to bar the string of a baryon interpolating field
  '''
  flavs=flavs.split("-")
  if len(flavs)!=3:
    error()
  new=[flavs[2],] + flavs[:2]
  for i,fl in enumerate(new):
    new[i]+="^"
  return "-".join(new)

def perm_parity(lst):
  '''
  Given a permutation of the digits 0..N in order as a list, 
  returns its parity (or sign): +1 for even parity; -1 for odd.
  '''
  parity = 1
  for i in range(0,len(lst)-1):
    if lst[i] != i:
      parity *= -1
      mn = min(range(i,len(lst)), key=lst.__getitem__)
      lst[i],lst[mn] = lst[mn],lst[i]
  return parity
  
def sign_from_map_c(map_c):
  '''
  Takes into account the sign produced by remapping of color indeces in the epsilon
  '''
  e=[0]*len(map_c)
  for m in map_c:
    e[m[0]] = m[1]
  return perm_parity(e)

def non_zero_combinations(matrices):
  '''
  Return the non-zero indeces of the outer product of matrices
  '''
  all_s1=[[]]
  all_s2=[[]]
  all_val=[1]
  for m in matrices:
    s1,s2=np.nonzero(m)
    val=m[np.nonzero(m)]
    tmp_s1=[]
    tmp_s2=[]
    tmp_val=[]
    for i1,i2,i in zip(all_s1,all_s2,all_val):
      for j1,j2,j in zip(s1,s2,val):
        tmp_s1.append(i1+[j1,])
        tmp_s2.append(i2+[j2,])
        tmp_val.append(i*j)
    all_s1=tmp_s1
    all_s2=tmp_s2
    all_val=tmp_val
  arr=np.array(list(zip(all_s1,all_s2))).transpose(0,2,1).reshape((-1,6))
  return arr, all_val

def remap(arr,map_s):
  '''
  Remap the list of indeces in arr with the map in map_s
  '''
  new=[]
  for a in arr:
    n=[0]*len(a)
    for ms in map_s:
      n[ms[1]]=a[ms[0]]
    new.append(n)
  return new

# Here we perform the contractions
prop_prods =  defaultdict(list)
for baryon,info in baryons.items():
  if info["spin"] == 3/2:
    # To match normalization of spin baryons as in table VI
    # of https://arxiv.org/pdf/1704.02647.pdf
    # SB: I didn't understand the origin of those coefficients
    # NOTE: here norm is coeff^2/3
    if len(info['coeffs']) == 3:
      norm = 1/2
    elif len(info['coeffs']) == 2:
      norm = 1
    else:
      norm = 1/3
  else:
    norm = (np.array(info['coeffs'])**2).sum()
  J=[interpolating_field(flavs, coeff) for flavs, coeff in zip(info['flavs'],info['coeffs'])]
  J_bar=[interpolating_field(baryon_bar(flavs), coeff) for flavs, coeff in zip(info['flavs'],info['coeffs'])]
  exps=[expectation_value(j,j_bar) for j in J for j_bar in J_bar]
  ordered = exps[0].default_view()
  coeffs = []
  map_c = []
  map_s = []
  for exp in exps:
    coeffs += exp.coeffs
    mc,ms = exp.map_to_default_view()
    map_c += mc
    map_s += ms
  for i,mc in enumerate(map_c):
    coeffs[i]*=sign_from_map_c(mc)

  final_list={}
  for gamma,matrices in all_gammas[info["spin"]].items():
    final_list[gamma]={}
    for gammas in matrices:
      non_zero,vals = non_zero_combinations([gammas[1],gammas[0],gammas[2]])
      for i,ms in enumerate(map_s):
        mapped=remap(non_zero,ms)
        for m,v in zip(mapped,vals):
          m=repr(m)
          if m in final_list[gamma]:
            final_list[gamma][m]+=v*coeffs[i]
          else:
            final_list[gamma][m]=v*coeffs[i]
    for m,v in list(final_list[gamma].items()):
      if np.isclose(v,0):
        del final_list[gamma][m]
      else:
        final_list[gamma][m]/=norm
    

  flv_key = "-".join([p.fl for p in ordered])
  prop_prods[flv_key].append([baryon,final_list])



#************************************Tools**************************************************************
def write_gamma_line(line):
  string=''
  for elem in line:
    if elem == 0:
      string += '  0,'
    if elem == 1:
      string += '  1,'
    if elem == -1:
      string += ' -1,'
    if elem == 1j:
      string += '  I,'
    if elem == -1j:
      string += ' -I,'
  return string

def remove_symbols(name):
  name = name.replace('+','Pl')
  name = name.replace('-','Mn')
  name = name.replace('*','St')
  name = name.replace('\'','Pr')
  return name


file1 = open(sys.argv[1],"w")
file2 = open(sys.argv[2],"w")

# Printing the header
for f in file1,file2:
  f.write("/* This file had been generated by python/baryons.py\n")
  f.write(" * \n")
  f.write(" * The executed command has been:\n")
  f.write(" * python " + ' '.join(sys.argv) + "\n")
  f.write(" * \n")  
  f.write(" * The gamma used are :\n")
  f.write(" * gx = [ " + write_gamma_line(gx[0]) + "\n")
  f.write(" *        " +  write_gamma_line(gx[1]) + "\n")
  f.write(" *        " +  write_gamma_line(gx[2]) + "\n")
  f.write(" *        " +  write_gamma_line(gx[3]) + " ] \n")
  f.write(" * gy = [ " +  write_gamma_line(gy[0]) + "\n")
  f.write(" *        " +  write_gamma_line(gy[1]) + "\n")
  f.write(" *        " +  write_gamma_line(gy[2]) + "\n")
  f.write(" *        " +  write_gamma_line(gy[3]) + " ] \n")
  f.write(" * gz = [ " +  write_gamma_line(gz[0]) + "\n")
  f.write(" *        " +  write_gamma_line(gz[1]) + "\n")
  f.write(" *        " +  write_gamma_line(gz[2]) + "\n")
  f.write(" *        " +  write_gamma_line(gz[3]) + " ] \n")
  f.write(" * gt = [ " +  write_gamma_line(gt[0]) + "\n")
  f.write(" *        " +  write_gamma_line(gt[1]) + "\n")
  f.write(" *        " +  write_gamma_line(gt[2]) + "\n")
  f.write(" *        " +  write_gamma_line(gt[3]) + " ] \n")
  f.write(" *\n")
  for fl_keys in baryons.keys(): 
    f.write(" * " + fl_keys + " : " + repr(baryons[fl_keys]) + "\n")
  f.write(" */\n")
  
# Printing the data
def vec(s):
  return "std::vector<"+s+">"

file1.write("#pragma once\n")
file2.write("#include<PLEGMA.h>\n")
file2.write("#include<"+sys.argv[1]+">\n")

for prop in prop_prods.values():
  for baryon in prop:
    for gamma,vals in baryon[1].items():
        string = "float2 "+ remove_symbols(baryon[0])+"_"+gamma.replace('-','_')+"_vals" + "[" + repr(len(vals)) + "] = { "
        for x in  vals.values():
          string += "{"+repr(x.real)+", "+repr(x.imag)+"}, "
        string += "};\n"
        file2.write(string)
        file1.write("extern float2 "+ remove_symbols(baryon[0])+"_"+gamma.replace('-','_')+"_vals" + "[" + repr(len(vals)) + "];\n")
        string = "short "+ remove_symbols(baryon[0])+"_"+gamma.replace('-','_')+"_idxs" + "[" + repr(len(vals)) + "*6] = { "
        for x in  vals.keys():
          string += repr(x).replace('\'[','').replace(']\'',', ')
        string += "};\n"
        file2.write(string)
        file1.write("extern short "+ remove_symbols(baryon[0])+"_"+gamma.replace('-','_')+"_idxs" + "[" + repr(len(vals)) + "*6];\n")

props=["\""+"".join([p[0] for p in key_prop.split("-")])+"\"" for key_prop in prop_prods.keys()]
string = "const "+vec("std::string")+" BP_prop_prods = {" + ", ".join(props) + "};\n"
file1.write(string)

datasets=["{" + ", ".join(["\""+remove_symbols(baryon[0])+"/"+gamma.replace('-','_')+"\"" for baryon in prop for gamma in baryon[1].keys()])+"}" for prop in prop_prods.values()]
string = "const "+vec(vec("std::string"))+" BP_prop_prods_names = {" + ", ".join(datasets) + "};\n"
file1.write(string)

datasets=["{" + ", ".join([repr(len(gamma)) for baryon in prop for gamma in baryon[1].values()])+"}" for prop in prop_prods.values()]
string = "const "+vec(vec("short"))+" BP_prop_prods_count = {" + ", ".join(datasets) + "};\n"
file1.write(string)

datasets=["{" + ", ".join([remove_symbols(baryon[0])+"_"+gamma.replace('-','_')+"_idxs" for baryon in prop for gamma in baryon[1].keys()])+"}" for prop in prop_prods.values()]
string = "const "+vec(vec("short *"))+" BP_prop_prods_idxs = {" + ", ".join(datasets) + "};\n"
file1.write(string)

datasets=["{" + ", ".join([remove_symbols(baryon[0])+"_"+gamma.replace('-','_')+"_vals" for baryon in prop for gamma in baryon[1].keys()])+"}" for prop in prop_prods.values()]
string = "const "+vec(vec("float2 *"))+" BP_prop_prods_vals = {" + ", ".join(datasets) + "};\n"
file1.write(string)
