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
# Run is as 'python baryons.py > FILENAME.h'
# Authors: E. Papadiofantous, S. Bacchio


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

all_gammas[1/2] = {"Pp-Cg5-Cg5": (Pplus,Cg5,Cg5),
                   "Pm-Cg5-Cg5":(Pminus,Cg5,Cg5),
                   "Pp-igtCg5-igtCg5": (Pplus,igtCg5,igtCg5),
                   "Pm-igtCg5-igtCg5":(Pminus,igtCg5,igtCg5),
                   "Pp-C-C": (Pplus,C,C),
                   "Pm-C-C":(Pminus,C,C)}
all_gammas[3/2] = {"Pp-Cgi-Cgi":[(Pplus,Cgx,Cgx), (Pplus,Cgy,Cgy), (Pplus,Cgz,Cgz)],
                   "Pm-Cgi-Cgi":[(Pminus,Cgx,Cgx), (Pminus,Cgy,Cgy), (Pminus,Cgz,Cgz)],
                   "Ppgigj-Cgi-Cgj":[(Pplus.dot(gxgy),Cgx,Cgy),(Pplus.dot(gxgz),Cgx,Cgz),(Pplus.dot(gygz),Cgy,Cgz),
                                     (Pplus.dot(gygx),Cgy,Cgx),(Pplus.dot(gzgx),Cgz,Cgx),(Pplus.dot(gzgy),Cgz,Cgy)],
                   "Pmgigj-Cgi-Cgj":[(Pminus.dot(gxgy),Cgx,Cgy),(Pminus.dot(gxgz),Cgx,Cgz),(Pminus.dot(gygz),Cgy,Cgz),
                                     (Pminus.dot(gygx),Cgy,Cgx),(Pminus.dot(gzgx),Cgz,Cgx),(Pminus.dot(gzgy),Cgz,Cgy)]}

#***********************Baryon list***************************************

flav_dict = {
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
# The methods return the selected baryon flavs, spin
#and the coefficient of the interpoalting field terms

class Construct_interpolation_field(object):
  def __init__(self,baryon,flav_dict):
    self.baryon = baryon
    self.flav_dict = flav_dict
    
  def build_baryon(self):
    flavors = self.flav_dict[self.baryon]['flavs']
    spin = self.flav_dict[self.baryon]['spin']
    self.coeffs = flav_dict[self.baryon]["coeffs"]
    return flavors, spin
  
  def term_coeff(self):
    #they are the coefficients inside the interpolating field parenthesis
    self.build_baryon()
    return self.coeffs
  
  def baryon_interpolating_field_coeff(self):
    #they are the coefficients inside the interpolating field parenthesis
    self.term_coeff()
    coeffs = self.coeffs
    term_sum = 0
    
    for coeff in coeffs:
      term_sum += coeff
    term = 1/np.sqrt(term_sum)
    if len(coeffs) == 3:
      if coeffs[2] == -1 :
        term *= 1/np.sqrt(3)
      if coeffs[2] == 1 :
        term *= np.sqrt(2)
    return term

#*******************************************************************************************************
#Returns O and O_bar lists numbered
 
class Construct_o_o_bar_list(object):
  def __init__(self,baryon):
    self.flavors, self.spin = Construct_interpolation_field(baryon,flav_dict).build_baryon()
    
  def produce_o_lists(self):
    flavors =self.flavors
    O = []
    O_bar = []
    for num,flv in enumerate(flavors):
      O_bar.append(flv.split('-'))
     
    O = copy.deepcopy(O_bar)
    for element in O:
      element.insert(0,element.pop())
    
    return O,O_bar
  
  def num_o_obar(self,O,O_bar):
    for num,item in enumerate(O_bar) :
      if not item[-1].isnumeric() :
        if O_bar[num][-1] == '^':
          O_bar[num] = O_bar[num][:-1] + str(num)
        else:
          O_bar[num] += '^'+str(num)
      if not O[num][-1].isnumeric():
        O[num] += str(num+3)

    return O,O_bar

#***************************Frequency count*******************************************************

#Returns the max and min appearence frequency and the flavors related to these freqs.

class Count_appear_freq(object):
  def __init__(self,alist):
    counter = collections.Counter(alist)
    self.counter_values = list(counter.values())
    self.counter_keys = list(counter.keys())
    
  def min_freq_method(self):
    counter_values = self.counter_values
    counter_keys = self.counter_keys
    index_min = np.argmin(counter_values)
    min_freq = counter_values[index_min]
    min_freq_key = counter_keys[index_min]
    return min_freq,min_freq_key
  
  def max_freq_method(self):
    counter_values = self.counter_values
    counter_keys = self.counter_keys
    index_max =np.argmax(counter_values)
    max_freq = counter_values[index_max]
    max_freq_key = counter_keys[index_max]
    return max_freq,max_freq_key

#*****************Propagator products*************************************************************

#Returns thr propagator products

class Make_propagator_products(object):
  def __init__(self,O,O_bar):
    self.O = O
    self.O_bar = O_bar
    self.templist = []
    for k,itm in enumerate(O):
      self.templist.append(itm[0])
    self.min_freq, self.min_freq_key = Count_appear_freq(self.templist).min_freq_method()
    self.max_freq, self.max_freq_key = Count_appear_freq(self.templist).max_freq_method()
    
  def products(self):
    O = self.O
    O_bar = self.O_bar
    min_freq, min_freq_key = self.min_freq, self.min_freq_key
    max_freq, max_freq_key = self.max_freq, self.max_freq_key
    prop_list = []
    newlist= [[]]
    templist2= []
    dd = []
    sum_indx_O = 0
    sum_indx_O_bar = 0
    sum1 = 0
    sum2 = 0
    
    for item in O_bar:
      sum_indx_O_bar += int(item[-1])
      
    for item in O:
      sum_indx_O += int(item[-1])
    
    for i,item2 in enumerate(O_bar):
      for j,item in enumerate(O):
        if item[0] == item2[0]:
          prop_list.append([item2,item])
          if min_freq_key == item[0:1] and max_freq ==2:
            dd = [item2,item]
            
    start = 0
    end = 3 
    while(len(prop_list) != 0):
      sum1 =0
      sum2 =0
      for item in prop_list[start:end]:
        templist2.append(item)
      if len(templist2) == 4:
        if templist2[0][0][-1] == templist2[1][0][-1] or templist2[0][1][-1] == templist2[1][1][-1]: 
          templist2.remove(templist2[1])
        if len(templist2) == 2:
          continue
        if templist2[0][0][-1] == templist2[2][0][-1] or templist2[1][0][-1] == templist2[2][0][-1]:
          templist2.remove(templist2[2])
        if len(templist2) == 2:
          continue
        if templist2[0][1][-1] == templist2[2][1][-1] or templist2[1][1][-1] == templist2[2][1][-1]:
          templist2.remove(templist2[2])
        if len(templist2) == 2:
          continue
          
      if len(prop_list)== 2:
        newlist.append(prop_list)
        newlist[-1].append(dd)
        prop_list = []
        continue
        
      if sum1 != sum_indx_O_bar :
        for item2 in templist2:
          sum1 += int(item2[0][-1])
      if sum2 != sum_indx_O :
        for item2 in templist2:
          sum2 += int(item2[1][-1])
          
      if sum1 == sum_indx_O_bar and sum2 == sum_indx_O :
        templist2.sort(key= lambda x: x[0][-1])
        for i in templist2:
          prop_list.remove(i)
        if templist2 not in newlist:
          newlist.append(templist2)
        
        if max_freq == 2 or max_freq ==1 :
          continue
        else:
          templist3 = copy.deepcopy(templist2)
          save_var_temp = templist3[1][1][-1]
          templist3[1][1] = templist3[1][1].replace(templist3[1][1][-1],templist3[2][1][-1])
          templist3[2][1] = templist3[2][1].replace(templist3[2][1][-1],save_var_temp)
          newlist.append(templist3) 
     
        templist2 = []
        start =0 
        end=3
        sum1 = 0
        sum2 = 0
      start = end
      end += 1 
      if end ==len(prop_list)+1:
        start = 0
        end = 1
    newlist.remove(newlist[0])
    return newlist

#*******************************************************************************************
#Returns the correct sign, taking into account the prop prods formation
#and levi-civita permutations

class Get_sign_for_single_o_obar(object):
  def __init__(self,prop_prod_num,items,sign_from_indx_list):
    self.items = items
    self.sign_from_indx_list = sign_from_indx_list
    self.prop_prod_num = prop_prod_num
    
  def get_sign(self):
    items = self.items
    sign = 1
    
    for enumi,item in enumerate(items[:2]):  
      diff = abs(int(item[1][-1]) - int(item[0][-1])-1)
      if enumi == 0:
        tmp_save = item[1][-1]
      if tmp_save < item[1][-1]:
        for q in range(enumi):
          diff -= 1
      mod = np.mod(diff,2) 
      if mod != 0 :
        sign *= -1
    return sign
  
  def get_correct_sign(self):
    sign =self.get_sign()   
    diff = 0
    sign_from_indx_list = self.sign_from_indx_list
    
    for num ,ilist in enumerate(sign_from_indx_list):
      if num == 0:
        tmp = int(ilist)
      diff += tmp - int(ilist)
      tmp = int(ilist)

    if diff ==-1 or diff ==2:
      sign *= -1
    return sign

#*************************Change indices**************************************************************
#Returns a list with the correct order of gamma indices

class Change_indx(object):
  def __init__(self,indx_list1,o,item):
    self.indx_list1 = indx_list1
    self.o = o
    self.item = item
    
  def change_indx_to_gamma(self):
    indx_list1 = self.indx_list1
    o = self.o
    tmp = []
    hold_exchange = []
    item = self.item

    for num,it in enumerate(item):
      if it[1][-1] != o[num][-1] :
        hold_exchange.append([it[1][-1],o[num][-1]])
 
    if not hold_exchange :
      return indx_list1
    for num,it in enumerate(indx_list1):
      for tmp in hold_exchange:
        if it == tmp[0]:
          indx_list1[num] = tmp[1]
    return indx_list1

#***********************Gamma coefs and sign********************************************************************************
#This is where we call some of the previous classes
#Combine them and as a result we have
#the gamma coeff and sign for each o-o_bar combination.

class Return_gamma_coeff_and_gamma_sign(object):
  def __init__(self,O,O_bar,baryon):
    self.O = O
    self.O_bar = O_bar
    self.baryon = baryon
    
  def combine_all(self):
    O = self.O
    O_bar = self.O_bar 
    baryon =self.baryon
    sign_o_o_bar_gamma_ilist = []
    interpol_sign_o_o_bar = []
    sign = []
    new_o = []
    new_o_bar = []
    o_o_bar_gamma_ilist = []
    prop_prod_general = []
    
    indxap = 'ap'
    indxbp = 'bp'
    indxcp = 'cp'

    interp_field_sign = Construct_interpolation_field(baryon,flav_dict).term_coeff()
    
    for i,o in enumerate(O):
      for j,o_bar in enumerate(O_bar):
        new_o,new_o_bar = Construct_o_o_bar_list(baryon).num_o_obar(o,o_bar)
       # print("***************NEW O - O_bar combination*************************")
        
        interpolatingf_sign_o = interp_field_sign[i]
        interpolatingf_sign_o_bar = interp_field_sign[j]
        interpol_sign_o_o_bar = interpolatingf_sign_o * interpolatingf_sign_o_bar
  
        propagator_prod = Make_propagator_products(new_o,new_o_bar).products()
        for item in propagator_prod:
          item.sort(key= lambda x: x[0][-1]) 
      
        #print("Gamma indices:")
        for enumer,item in enumerate(propagator_prod):
          indx_list = ['3','4','5']
          indxa,indxb,indxc = Change_indx(indx_list,new_o,item).change_indx_to_gamma()
          sign_o_o_bar_gamma_ilist = [indxa,indxb,indxc]
          
          index_dict = {'3':'a','4':'b','5':'c'}
          indxa = index_dict[indxa]
          indxb = index_dict[indxb]
          indxc = index_dict[indxc]
          
          sign_value = Get_sign_for_single_o_obar(enumer,item,sign_o_o_bar_gamma_ilist).get_correct_sign()
          sign.append(sign_value*interpol_sign_o_o_bar)
                    
          o_o_bar_gamma_ilist.append([indxcp,indxa,indxap,indxbp,indxb,indxc])
          
          item.sort(key=lambda val: SORT_ORDER[val[0][:2]])
        prop_prod_general.append(propagator_prod)  
   
    return prop_prod_general,sign,o_o_bar_gamma_ilist
  
#*************************************Call Save Indices**************************************************
#Returns  the non zero gamma elements
class Non_zero_gamma(object):
  def __init__(self,baryon,gammas,sign_gamma_final,o_o_bar_gamma_indexlist):
    self.gammas = gammas
    self.sign_gamma_final = sign_gamma_final
    self.o_o_bar_gamma_indexlist = o_o_bar_gamma_indexlist
    self.baryon = baryon
    
  def give_non_zero_gamma(self):
    gammas = self.gammas
    sign_gamma_final = self.sign_gamma_final
    o_o_bar_gamma_indexlist = self.o_o_bar_gamma_indexlist

    #To find the interpolating field coefficient
    i_field_general_coeff = Construct_interpolation_field(self.baryon,flav_dict).baryon_interpolating_field_coeff()
    i_field_general_coeff = i_field_general_coeff * i_field_general_coeff
    
    final_list={}
    for name in gammas.keys():
      save_indices = {}
      for (sign,indices) in zip(sign_gamma_final,o_o_bar_gamma_indexlist):
        for ap in range(4):
          for a in range(4):
            for bp in range(4):
              for b in range(4):
                for cp in range(4):
                  for c in range(4):
                    di = {'ap':ap,'bp':bp,'cp':cp,'a':a,'b':b,'c':c}
                    indxcp,indxa,indxap,indxbp,indxb,indxc = indices
                    if type(gammas[name]) is list:
                      for g in gammas[name]:
                        gammaA,gammaB,gammaC = g
                        quantity = sign * gammaA[di[indxcp],di[indxa]]*gammaB[di[indxap],di[indxbp]]*gammaC[di[indxb],di[indxc]]
                        if quantity != complex(0) :
                          key=(di[indxap],di[indxa],di[indxbp],di[indxb],di[indxcp],di[indxc])
                          if key in save_indices.keys():
                            save_indices[key] += quantity
                          else:
                            save_indices[key] = quantity
                    else:
                      gammaA,gammaB,gammaC = gammas[name]
                      quantity = sign * gammaA[di[indxcp],di[indxa]]*gammaB[di[indxap],di[indxbp]]*gammaC[di[indxb],di[indxc]]
                      if quantity != complex(0) :
                        key=(di[indxap],di[indxa],di[indxbp],di[indxb],di[indxcp],di[indxc])
                        if key in save_indices.keys():
                          save_indices[key] += quantity
                        else:
                          save_indices[key] = quantity
      for key,value in list(save_indices.items()):
        if value == complex(0):
          save_indices.pop(key)

      final_list[name] = save_indices
    
    return final_list

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

#************************************Baryons saved all together*************************************************
    
prop_prods =  defaultdict(list)
for enumb,baryon in enumerate(flav_dict.keys()):
  flavor ,spin = Construct_interpolation_field(baryon,flav_dict).build_baryon()
  gammas = all_gammas[spin]  
  
  flv_list1,flv_list2 = Construct_o_o_bar_list(baryon).produce_o_lists()
  O = copy.deepcopy(flv_list1)
  O_bar = copy.deepcopy(flv_list2)
  flv_list1[0].sort(key= lambda val : SORT_ORDER[val])

  prop_prod_general,sign_gamma_final,o_o_bar_gamma_indexlist = Return_gamma_coeff_and_gamma_sign(O,O_bar,baryon).combine_all()
  final_list = Non_zero_gamma(baryon,gammas,sign_gamma_final,o_o_bar_gamma_indexlist).give_non_zero_gamma()
  
  flv_key = "-".join(flv_list1[0])
  prop_prods[flv_key].append([baryon,final_list])


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
  for fl_keys in flav_dict.keys(): 
    f.write(" * " + fl_keys + " : " + repr(flav_dict[fl_keys]) + "\n")
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
        string = "int "+ remove_symbols(baryon[0])+"_"+gamma.replace('-','_')+"_idxs" + "[" + repr(len(vals)) + "*6] = { "
        for x in  vals.keys():
          string += repr(x).replace('(','').replace(')',', ')
        string += "};\n"
        file2.write(string)
        file1.write("extern int "+ remove_symbols(baryon[0])+"_"+gamma.replace('-','_')+"_idxs" + "[" + repr(len(vals)) + "*6];\n")

props=["\""+"".join([p[0] for p in key_prop.split("-")])+"\"" for key_prop in prop_prods.keys()]
string = "const "+vec("std::string")+" BP_prop_prods = {" + ", ".join(props) + "};\n"
file1.write(string)

datasets=["{" + ", ".join(["\""+remove_symbols(baryon[0])+"/"+gamma.replace('-','_')+"\"" for baryon in prop for gamma in baryon[1].keys()])+"}" for prop in prop_prods.values()]
string = "const "+vec(vec("std::string"))+" BP_prop_prods_names = {" + ", ".join(datasets) + "};\n"
file1.write(string)

datasets=["{" + ", ".join([repr(len(gamma)) for baryon in prop for gamma in baryon[1].values()])+"}" for prop in prop_prods.values()]
string = "const "+vec(vec("int"))+" BP_prop_prods_count = {" + ", ".join(datasets) + "};\n"
file1.write(string)

datasets=["{" + ", ".join([remove_symbols(baryon[0])+"_"+gamma.replace('-','_')+"_idxs" for baryon in prop for gamma in baryon[1].keys()])+"}" for prop in prop_prods.values()]
string = "const "+vec(vec("int *"))+" BP_prop_prods_idxs = {" + ", ".join(datasets) + "};\n"
file1.write(string)

datasets=["{" + ", ".join([remove_symbols(baryon[0])+"_"+gamma.replace('-','_')+"_vals" for baryon in prop for gamma in baryon[1].keys()])+"}" for prop in prop_prods.values()]
string = "const "+vec(vec("float2 *"))+" BP_prop_prods_vals = {" + ", ".join(datasets) + "};\n"
file1.write(string)
