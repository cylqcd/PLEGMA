#include <PLEGMA.h>
#include <PLEGMA_utils.h>

void createMom(int *Nmom, int momElem[][3], int Q_sq){
  int counter = 0;

  for(int iQ = 0 ; iQ <= Q_sq ; iQ++){
    for(int nx = iQ ; nx >= -iQ ; nx--)
      for(int ny = iQ ; ny >= -iQ ; ny--)
        for(int nz = iQ ; nz >= -iQ ; nz--){
          if( nx*nx + ny*ny + nz*nz == iQ ){
            momElem[counter][0] = nx;
            momElem[counter][1] = ny;
            momElem[counter][2] = nz;
            counter++;
          }
        }
  }
  *Nmom = counter;
}

void get_coords(int id, int *position){

  float temp = id/GK_localVolume;
  for(int i = N_DIMS-1; i >=0; --i) {

    temp = temp * GK_localL[i];
    position[i] = (int)temp % GK_localL[i];
    position[i] += comm_coords(default_topo)[i] * GK_localL[i]; 
  }
}
