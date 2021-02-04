import numpy as np
import h5py as h5
import sys
import argparse

parser = argparse.ArgumentParser(prog='postProcZfac.py', description='Transform the conventions from the PLEGMA Zfacs to the old Zfacs')
parser.add_argument('--inputPath', help='The path where I can find the input of the files', default = '/cyclamen/home/khadjiyiannakou/runs_Zfac_b1.726_3D/')
parser.add_argument('--outputPath', help='The path where I can write the output files', default='/cyclamen/home/khadjiyiannakou/runs_Zfac_b1.726_3D/')
parser.add_argument('--momentum', help='Give the momentum in order PX,PY,PZ,PT', default='2,2,2,2')
parser.add_argument('--confID', help='The ID of the configuration', default='0700')

args= vars(parser.parse_args())
args['momentum']=args['momentum'].split(',')
mom=[ int(x) for x in args['momentum']]
momT=[mom[3],mom[0],mom[1],mom[2]]

matr=np.matrix
I=np.complex(0.,1.)
U=np.array([[1, 0, -1, 0],[0, 1, 0, -1],[1, 0, 1, 0],[0, 1, 0, 1]]) / np.sqrt(2.)
g5=np.array([[0,0,1,0],[0,0,0,1],[1,0,0,0],[0,1,0,0]])
g4=np.array([[1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,-1]])
U=matr(U)*matr(g4)
#print(U)
L=24
T=48
V=L**3*T

Gdprop=(np.loadtxt('%s/GpropMom_px%dpy%dpz%dpt%d_conf_%s.dat'%(tuple([args['inputPath']]+mom+[args['confID']])),usecols=(5,6))).reshape(4,4,9,2)
Gdprop=Gdprop[...,0]+I*Gdprop[...,1]
GdpropRot = np.transpose(np.array([matr(U.T) * matr(Gdprop[:,:,i]) * matr(U) for i in range(9)]),(1,2,0))

Guprop=(np.transpose(np.conj(Gdprop.reshape(4,4,3,3)), (1,0,3,2))).reshape(4,4,9)
GupropRot=np.transpose(np.array([matr(U.T) * matr(g5)*matr(Guprop[:,:,i])*matr(g5)*matr(U) for i in range(9)]), (1,2,0))

fl=['d','u']
G={'d': GdpropRot,'u': GupropRot}

###########################  Propagator in momentum space #########################

with open('%s/p_propagator_p%d_%d_%d_%d.%s.dat'%(tuple([args['outputPath']]+momT+[args['confID']])),'w') as fp:
    for mu in range(4):
        for nu in range(4):
            for c1 in range(3):
                for c2 in range(3):
                    for f in fl:
                        fp.write('%s %d %d %d %d %e %e\n' % (f,mu,nu,c1,c2,G[f][mu,nu,c1*3+c2].real/V, G[f][mu,nu,c1*3+c2].imag/V))
        
####################################################################################

######################### Vertex functions for ultralocal operators ################

with h5.File('%s/Vloc_px%dpy%dpz%dpt%d_conf_%s.h5'%(tuple([args['inputPath']]+mom+[args['confID']])),'r') as fp:
    A=(fp['/sx00sy00sz00st00/Local/threep'][()]).squeeze()
    A=A.flatten()
    A=A.reshape(T,4,4,3,3,16,2)
    #A=A.reshape(4,4,3,3,T,16,2)
    Vloc=np.sum(A,axis=0)
    Vloc=Vloc[...,0]+I*Vloc[...,1]
VlocM={}
VlocM['s']=Vloc[...,0]
VlocM['p']=Vloc[...,5]
VlocM['v']=[-Vloc[...,4], Vloc[...,1], Vloc[...,2], Vloc[...,3]]
VlocM['a']=[-Vloc[...,9], Vloc[...,6], Vloc[...,7], Vloc[...,8]]
#S12,S13,S23,S41,S42,S43
# 0   1   2   3   4   5
VlocM['t']=[Vloc[...,0]*I, -Vloc[...,10+2]*I, Vloc[...,10+1]*I, -Vloc[...,10+0]*I,
            +Vloc[...,10+2]*I, Vloc[...,0]*I, Vloc[...,10+5]*I,  -Vloc[...,10+4]*I,
            -Vloc[...,10+1]*I, -Vloc[...,10+5]*I,Vloc[...,0]*I, Vloc[...,10+3]*I,
            +Vloc[...,10+0]*I, +Vloc[...,10+4]*I, -Vloc[...,10+3]*I,Vloc[...,0]*I]
            
for ff in ['s','p']:
    Ver=np.transpose(np.array([matr(U.T) * matr( (VlocM[ff].reshape(4,4,9))[:,:,i] ) *matr(U) for i in range(9)]), (1,2,0))
    with open('%s/Vertex_%s_p%d_%d_%d_%d.%s.dat'%(tuple([args['outputPath']]+[ff]+momT+[args['confID']])),'w') as fp:
        for mu in range(4):
            for nu in range(4):
                for c1 in range(3):
                    for c2 in range(3):
                        fp.write('%d %d %d %d %+e %+e\n' % (mu,nu,c1,c2,Ver[mu,nu,c1*3+c2].real/V, Ver[mu,nu,c1*3+c2].imag/V))

for ff in ['v','a']:
    with open('%s/Vertex_%s_p%d_%d_%d_%d.%s.dat'%(tuple([args['outputPath']]+[ff]+momT+[args['confID']])),'w') as fp:
        for j in range(4):
            Ver=np.transpose(np.array([matr(U.T) * matr( (VlocM[ff][j].reshape(4,4,9))[:,:,i] ) *matr(U) for i in range(9)]), (1,2,0))
            for mu in range(4):
                for nu in range(4):
                    for c1 in range(3):
                        for c2 in range(3):
                            fp.write('%d %d %d %d %d %+e %+e\n' % (j,mu,nu,c1,c2,Ver[mu,nu,c1*3+c2].real/V, Ver[mu,nu,c1*3+c2].imag/V))

with open('%s/Vertex_%s_p%d_%d_%d_%d.%s.dat'%(tuple([args['outputPath']]+['t']+momT+[args['confID']])),'w') as fp:
    for j in range(4):
        for k in range(4):
            if j != k:
                Ver=np.transpose(np.array([matr(U.T) * matr( (VlocM['t'][j*4+k].reshape(4,4,9))[:,:,i] ) *matr(U) for i in range(9)]), (1,2,0))
                for mu in range(4):
                    for nu in range(4):
                        for c1 in range(3):
                            for c2 in range(3):
                                fp.write('%d %d  %d %d %d %d %+e %+e\n' % (j,k,mu,nu,c1,c2,Ver[mu,nu,c1*3+c2].real/V, Ver[mu,nu,c1*3+c2].imag/V))

###############################################################
####################### Vertex functions for the one derivative operators #####################
with h5.File('%s/VoneD_px%dpy%dpz%dpt%d_conf_%s.h5'%(tuple([args['inputPath']] + mom + [args['confID']])),'r') as fp:
    A=(fp['/sx00sy00sz00st00/OneD/threep'][()]).squeeze()
    A=A.flatten() # {48, 1, 4, 4, 3, 3, 4, 16, 2}
    A=A.reshape(T,4,4,3,3,4,16,2)
    #    A=A.reshape(4,4,3,3,T,4,16,2)
    VOneD=np.sum(A,axis=0)
    VOneD=VOneD[...,0]+I*VOneD[...,1]
VOneDM={}

VOneDM['v']=[-VOneD[...,4], VOneD[...,1], VOneD[...,2], VOneD[...,3]]
VOneDM['a']=[-VOneD[...,9], VOneD[...,6], VOneD[...,7], VOneD[...,8]]

#S12,S13,S23,S41,S42,S43
# 0   1   2   3   4   5
zeroD=np.zeros(VOneD[...,0].shape)+I*np.zeros(VOneD[...,0].shape)
VOneDM['t']=[zeroD, -VOneD[...,10+2]*I, VOneD[...,10+1]*I, -VOneD[...,10+0]*I,
            +VOneD[...,10+2]*I, zeroD, VOneD[...,10+5]*I,  -VOneD[...,10+4]*I,
            -VOneD[...,10+1]*I, -VOneD[...,10+5]*I, zeroD, VOneD[...,10+3]*I,
            +VOneD[...,10+0]*I, +VOneD[...,10+4]*I, -VOneD[...,10+3]*I,zeroD]


for ff in ['v','a']:
    with open('%s/Vertex_%sD_p%d_%d_%d_%d.%s.dat'%(tuple([args['outputPath']]+[ff]+momT+[args['confID']])),'w') as fp:
        for j in range(4):
            for k in range(j+1):
                Ver1=np.transpose(np.array([matr(U.T) * matr( (VOneDM[ff][j][...,(k+3)%4]).reshape(4,4,9)[:,:,i] ) *matr(U) for i in range(9)]), (1,2,0))
                Ver2=np.transpose(np.array([matr(U.T) * matr( (VOneDM[ff][k][...,(j+3)%4]).reshape(4,4,9)[:,:,i] ) *matr(U) for i in range(9)]), (1,2,0))
                for mu in range(4):
                    for nu in range(4):
                        for c1 in range(3):
                            for c2 in range(3):
                                fp.write('%d %d %d %d %d %d %+e %+e\n' % (j,k,mu,nu,c1,c2,
                                                                          (Ver1[mu,nu,c1*3+c2].real + Ver2[mu,nu,c1*3+c2].real)/(8*V),
                                                                          (Ver1[mu,nu,c1*3+c2].imag + Ver2[mu,nu,c1*3+c2].imag)/(8*V)))

with open('%s/Vertex_%sD_p%d_%d_%d_%d.%s.dat'%(tuple([args['outputPath']]+['t']+momT+[args['confID']])),'w') as fp:
        for j in range(4):
            for k in range(4):
                for l in range(4):
                    Ver1=np.transpose(np.array([matr(U.T) * matr( (VOneDM['t'][j*4+k][...,(l+3)%4]).reshape(4,4,9)[:,:,i] ) *matr(U) for i in range(9)]), (1,2,0))
                    Ver2=np.transpose(np.array([matr(U.T) * matr( (VOneDM['t'][j*4+l][...,(k+3)%4]).reshape(4,4,9)[:,:,i] ) *matr(U) for i in range(9)]), (1,2,0))
                    for mu in range(4):
                        for nu in range(4):
                            for c1 in range(3):
                                for c2 in range(3):
                                    fp.write('%d %d %d %d %d %d %d %+e %+e\n' % (j,k,l,mu,nu,c1,c2,
                                                                              (Ver1[mu,nu,c1*3+c2].real + Ver2[mu,nu,c1*3+c2].real)/(8*V),
                                                                              (Ver1[mu,nu,c1*3+c2].imag + Ver2[mu,nu,c1*3+c2].imag)/(8*V)))

###############################################################################################

################### Vertex with two derivatives ##################
with h5.File('%s/VtwoD_px%dpy%dpz%dpt%d_conf_%s.h5'%(tuple([args['inputPath']] + mom + [args['confID']])),'r') as fp:
    A=(fp['/sx00sy00sz00st00/TwoD/threep'][()]).squeeze()
    A=A.flatten() # {48, 1, 4, 4, 3, 3, 12, 16, 2}
    A=A.reshape(T,4,4,3,3,12,16,2)
    #    A=A.reshape(4,4,3,3,T,4,16,2)
    VTwoD=np.sum(A,axis=0)
    VTwoD=VTwoD[...,0]+I*VTwoD[...,1]
VTwoDM={}
VTwoDM['v']=[-VTwoD[...,4], VTwoD[...,1], VTwoD[...,2], VTwoD[...,3]] # gt,gx,gy,gz
VTwoDM['a']=[-VTwoD[...,9], VTwoD[...,6], VTwoD[...,7], VTwoD[...,8]]
# PLEGMA directions: (0,1,2,3) -> (x,y,z,t)
# qcd conventions: (0,1,2,3) -> (t,x,y,z)
mapDirs={}
for et in range(12):
    dir1 = int(et/(4-1)) % 4
    dir2 = et % (4-1)
    if dir2 >= dir1: dir2+=1
    mapDirs[(dir1,dir2)]=et

for ff in ['v','a']:
    with open('%s/Vertex_%sDD_p%d_%d_%d_%d.%s.dat'%(tuple([args['outputPath']]+[ff]+momT+[args['confID']])),'w') as fp:
        for mu in range(4):
            for nu in range(4):
                for tau in range(4):
                    if mu != nu and mu != tau and nu != tau:
                        Ver=np.transpose(np.array([matr(U.T) * matr( (VTwoDM[ff][mu][...,mapDirs[((nu+3)%4,(tau+3)%4)]]).reshape(4,4,9)[:,:,i] ) *matr(U) for i in range(9)]), (1,2,0))
                        for mu1 in range(4):
                            for nu1 in range(4):
                                for c1 in range(3):
                                    for c2 in range(3):
                                        fp.write('%d %d %d %d %d %d %d %+e %+e\n' % (mu,nu,tau,mu1,nu1,c1,c2,
                                                                              Ver[mu1,nu1,c1*3+c2].real/(8*4*V),
                                                                              Ver[mu1,nu1,c1*3+c2].imag/(8*4*V)))

################### Vertex with two derivatives ##################
with h5.File('%s/VthreeD_px%dpy%dpz%dpt%d_conf_%s.h5'%(tuple([args['inputPath']] + mom + [args['confID']])),'r') as fp:
    A=(fp['/sx00sy00sz00st00/ThreeD/threep'][()]).squeeze()
    A=A.flatten() # {48, 1, 4, 4, 3, 3, 12, 16, 2}
    A=A.reshape(T,4,4,3,3,24,16,2)
    #    A=A.reshape(4,4,3,3,T,4,16,2)
    VThreeD=np.sum(A,axis=0)
    VThreeD=VThreeD[...,0]+I*VThreeD[...,1]
VThreeDM={}
VThreeDM['v']=[-VThreeD[...,4], VThreeD[...,1], VThreeD[...,2], VThreeD[...,3]] # gt,gx,gy,gz
VThreeDM['a']=[-VThreeD[...,9], VThreeD[...,6], VThreeD[...,7], VThreeD[...,8]]
# PLEGMA directions: (0,1,2,3) -> (x,y,z,t)
# qcd conventions: (0,1,2,3) -> (t,x,y,z)
mapDirsx={}
for et in range(24):
    dir1 = int(et/(4-1)/(4-2)) % 4
    dir2 = int(et/(4-2)) % (4-1);
    dir3 = et % (4-2);
    if dir2>=dir1: dir2+=1
    if dir3>=dir1:
        dir3+=1
        if dir3>=dir2:
            dir3+=1
    elif dir3>=dir2:
        dir3+=1
        if dir3>=dir1:
            dir3+=1
    mapDirsx[(dir1,dir2,dir3)]=et


for ff in ['v','a']:
    with open('%s/Vertex_%sDDD_p%d_%d_%d_%d.%s.dat'%(tuple([args['outputPath']]+[ff]+momT+[args['confID']])),'w') as fp:
        for mu in range(4):
            for nu in range(4):
                for tau in range(4):
                    for rho in range(4):
                        if mu not in [nu,tau,rho] and nu not in [mu,tau,rho] and tau not in [mu,nu,rho] and rho not in [mu,nu,tau]:
                            Ver=np.transpose(np.array([matr(U.T) * matr( (VThreeDM[ff][mu][...,mapDirsx[((nu+3)%4,(tau+3)%4,(rho+3)%4)]]).reshape(4,4,9)[:,:,i] ) *matr(U) for i in range(9)]), (1,2,0))
                            for mu1 in range(4):
                                for nu1 in range(4):
                                    for c1 in range(3):
                                        for c2 in range(3):
                                            fp.write('%d %d %d %d %d %d %d %d %+e %+e\n' % (mu,nu,tau,rho,mu1,nu1,c1,c2,
                                                                                  Ver[mu1,nu1,c1*3+c2].real/V,
                                                                                         Ver[mu1,nu1,c1*3+c2].imag/V))
