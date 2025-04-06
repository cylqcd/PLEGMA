
'''
cat runAux/cfgs_runt | xargs -I @ -P 10 python3 -u run2out.py -c @ > log/run2out.out & 
'''
import os, click, h5py, re, pickle
import numpy as np

lat_L,lat_T=48,96

gtCj={'id':1,'gx':-1,'gy':-1,'gz':-1,'gt':1,'g5':-1,'g5gx':-1,'g5gy':-1,'g5gz':-1,'g5gt':1,'sgmxy':-1,'sgmyz':-1,'sgmzx':-1,'sgmtx':1,'sgmty':1,'sgmtz':1}

Gfs=['id','gx','gy','gz','gt','g5','g5gx','g5gy','g5gz','g5gt','sgmyz','sgmzx','sgmxy','sgmtx','sgmty','sgmtz']
Gis=['id','g5','g5gt']

@click.command()
@click.option('-c','--cfg')
def run(cfg):
    inpath=f'run/{cfg}/'
    outpath=f'out/{cfg}/'
    os.makedirs(outpath,exist_ok=True)

    infile=inpath+f'Diagram_{cfg[1:]}_jPP.h5'
    outfile=outpath+'jPP.h5'
    outfile_flag=outfile+'_flag'
    if (not os.path.isfile(outfile)) or os.path.isfile(outfile_flag):
        with open(outfile_flag,'w') as f:
            pass
        with h5py.File(outfile,'w') as fw, h5py.File(infile) as f:
            st='sx00sy00sz00st00'
            pfs=f[st]['PhiPhi']['mvec'][:]
            pis=[[int(i) for i in pi[3:].split('_')] for pi in f[st]['PhiPhi'].keys() if pi!='mvec']
            pis.sort()
            str_pis=['pi='+'_'.join([str(i) for i in pi]) for pi in pis]
            moms_new=[list(pi)+list(pf) for pi in pis for pf in pfs]
            moms_new.sort()
            
            dic={}
            for i,pi in enumerate(pis):
                dic[tuple(pi)]=i
            map_pi=[dic[tuple(mom[:3])] for mom in moms_new]
            dic={}
            for i,pf in enumerate(pfs):
                dic[tuple(pf)]=i
            map_pf=[dic[tuple(mom[3:])] for mom in moms_new]
            
            fw.create_dataset('moms',data=moms_new)
            fw.create_dataset('tf,mom,Gf,Gi',data=[])
            fw.create_dataset('Gfs',data=Gfs)
            fw.create_dataset('Gis',data=Gis)
            
            sign_gi=np.array([1,1,1]) # id,g5,g5gt
            sign_gf=np.array([1, 1,1,1,1, 1, 1,1,1,1, 1j,-1j,1j, 1j,1j,1j])
            sign_gi*= [gtCj[gm] for gm in ['id','g5','g5gt']]
            gis=['i_gi={}'.format(i) for i in range(len(sign_gi))]
            
            for i_st in range(lat_T):
                st='sx00sy00sz00st{:02d}'.format(i_st)
                st_new='st{:03d}'.format(i_st)
                for fl in ['uu','ud','du','dd']:
                    fla='q1q2='+fl
                    for sme in ['LL','SL','SS']:
                        smear='sisf='+sme
                        opab=fl[0]+'B'+fl[1]+sme[1]+'_'+fl[0]+'B'+fl[1]+sme[0]
                        
                        t=np.array([[f[st]['PhiPhi'][pi][fla][smear][gi][:] for pi in str_pis] for gi in gis])
                        t=t[...,0]+1j*t[...,1]
                        t=t[:,map_pi,:,map_pf]
                        t=np.transpose(t,[2,0,3,1]) # tf,mom,Gf,Gi
                        t*=-1 # (-1) for the (-1) in front of PhiPhi
                        t*=sign_gi[None,None,None,:]
                        t*=sign_gf[None,None,:,None]
                        fw.create_dataset('data/'+st_new+'/'+opab,data=t.astype('complex64'))
        os.remove(outfile_flag)
        
    print('flag_cfg_done: '+cfg)
    
run()