'''
cat runAux/cfgs_runt | xargs -I @ -P 10 python3 -u run2out.py -c @ > log/run2out.out & 
'''
import os, click, h5py, re, pickle
import numpy as np

lat_L,lat_T=48,96
SorL='S'

Gs=['id','gx','gy','gz','gt','g5','g5gx','g5gy','g5gz','g5gt','sgmyz','sgmzx','sgmxy','sgmtx','sgmty','sgmtz']

@click.command()
@click.option('-c','--cfg')
def run(cfg):
    inpath=f'run/{cfg}/'
    outpath=f'out/{cfg}/'
    os.makedirs(outpath,exist_ok=True)

    infile=inpath+f'Diagram_{cfg[1:]}_loop{SorL}.h5'
    outfile=outpath+f'loop{SorL}.h5'
    outfile_flag=outfile+'_flag'
    if (not os.path.isfile(outfile)) or os.path.isfile(outfile_flag):
        with open(outfile_flag,'w') as f:
            pass
        with h5py.File(outfile,'w') as fw, h5py.File(infile) as f:
            st='sx00sy00sz00st00'
            ps=f[st]['PhiPhi']['mvec'][:]
        
            fw.create_dataset('moms',data=ps)
            fw.create_dataset('t,mom,G',data=[])
            fw.create_dataset('Gs',data=Gs)
            
            sign_g=np.array([1, 1,1,1,1, 1, 1,1,1,1, 1j,-1j,1j, 1j,1j,1j])
            
            for seed in f[st]['PhiPhi'].keys():
                if seed == 'mvec':
                    continue
                for id in f[st]['PhiPhi'][seed].keys():
                    t=f[st]['PhiPhi'][seed][id]['up'][:]
                    t=t[...,0]+1j*t[...,1]
                    t=np.complex128(t)
                    t*=-1
                    fw.create_dataset('data/'+seed+'/'+id+'/up',data=t)
                    
        os.remove(outfile_flag)
        
    print('flag_cfg_done: '+cfg)
    
run()