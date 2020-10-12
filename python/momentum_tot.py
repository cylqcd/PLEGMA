# This file extracts all the momentum combinations of 
# p_i1, p_i2, p_f1, p_f2
# for which |p_ij| ^2 <= individual_mom^2
#           |p_i1+p_i2| <= total_momentum^2 
# Input parameters p_imax^2
#                  p_totmax^2
# Output: all the possible combinations
#         Note output is written to the standard output
#
#*************Imports*********************************


import sys
import os

pimax2   = int(sys.argv[1])
ptotmax2 = int(sys.argv[2])
for pi2x in range(-pimax2, pimax2):
     for pi2y in range(-pimax2, pimax2):
          for pi2z in range(-pimax2, pimax2):
               for pf2x in range(-pimax2, pimax2):
                    for pf2y in range(-pimax2, pimax2):
                         for pf2z in range(-pimax2, pimax2):
                              for pf1x in range(-pimax2, pimax2):
                                   for pf1y in range(-pimax2, pimax2):
                                        for pf1z in range(-pimax2, pimax2):
                                             if  ((pi2x*pi2x+pi2y*pi2y+pi2z*pi2z)>pimax2) or (pf2x*pf2x+pf2y*pf2y+pf2z*pf2z >pimax2) or (pf1x*pf1x+pf1y*pf1y+pf1z*pf1z >pimax2) or (((pf1x+pf2x)*(pf1x+pf2x)+(pf1y+pf2y)*(pf1y+pf2y)+(pf1z+pf2z)*(pf1z+pf2z)) >ptotmax2):
                                                   continue
                                             print("%d %d %d %d %d %d %d %d %d" %(pi2x,pi2y,pi2z,pf1x,pf1y,pf1z,pf2x,pf2y,pf2z))
