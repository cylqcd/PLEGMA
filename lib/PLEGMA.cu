// We must put PLEGMA_global.h before PLEGMA.h so it's allocated here.
#define ALLOCATE 
#include <PLEGMA_global.h> 
#undef ALLOCATE
#include <PLEGMA.h>
#include <errno.h>
#include <limits>
#include <string.h>

//#define TIMING_REPORT
using namespace plegma;

//////////////////////////////////////////////////  
static void createMomenta(int Q_sq){
  int counter=0;
  for(int iQ = 0 ; iQ <= Q_sq ; iQ++){
    for(int nx = iQ ; nx >= -iQ ; nx--){
      for(int ny = iQ ; ny >= -iQ ; ny--){
        for(int nz = iQ ; nz >= -iQ ; nz--){
          if( nx*nx + ny*ny + nz*nz == iQ ){
            GK_moms[counter][0] = nx;
            GK_moms[counter][1] = ny;
            GK_moms[counter][2] = nz;
            counter++;
          }
        }
      }
    }
  }
  if(counter > MAX_NMOMENTA)errorQuda("Error exceeded max number of momenta\n");
  GK_Nmoms=counter;
}

void plegma::PLEGMA_init(){
  
  if(HGC_init_PLEGMA_flag == false){
    
    for(int i = 0 ; i < N_DIMS ; i++)
      HGC_nProc[i] = comm_dim(i);
    
    for(int i = 0 ; i < N_DIMS ; i++){   // take local and total lattice
      HGC_localL[i] = params->lL[i];
      HGC_totalL[i] = HGC_nProc[i] * HGC_localL[i];
    }
    
    for(int i = 0 ; i < N_DIMS ; i++){
      if( HGC_localL[i] < HGC_totalL[i])
	HGC_dimBreak[i] = true;
      else
	HGC_dimBreak[i] = false;
    }
    
    HGC_localVolume = 1;
    HGC_totalVolume = 1;
    for(int i = 0 ; i < N_DIMS ; i++){
      HGC_localVolume *= HGC_localL[i];
      HGC_totalVolume *= HGC_totalL[i];
    }

    HGC_strideFull = HGC_localVolume;

    for (int i=0; i<N_DIMS; i++) {
      if(HGC_dimBreak[i]) {
	HGC_surface3D[i] = 1;
	for (int j=0; j<N_DIMS; j++) {
	  if (i==j) continue;
	  HGC_surface3D[i] *= HGC_localL[j];
	}
      } else {
	HGC_surface3D[i] = 0;
      }
    }

    for(int i=0; i<N_DIMS; i++){
      for(int j=0; j<N_DIMS; j++){
	if(i!=j && HGC_dimBreak[i] && HGC_dimBreak[j]){
	  HGC_surface2D[i][j] = 1;
	  for(int k=0; k<N_DIMS; k++)
	    HGC_surface2D[i][j] *= (k!=i && k!=j) ? HGC_localL[k] : 1;
	}
	else HGC_surface2D[i][j] = 0;
      }
    }
    
    for(int i = 0 ; i < 2*N_DIMS ; i++){
      HGC_sideGhost[i] = 0;
    }

    for(int i=0; i<2*N_DIMS; i++){
      for(int j=0; j<2*N_DIMS; j++){
	HGC_cornerGhost[i][j] = 0;
      }
    }
    
#ifdef MULTI_GPU
    size_t lastIndex = HGC_localVolume;
    
    for(int i = 0 ; i < 2*N_DIMS ; i++)
      if( HGC_dimBreak[i%N_DIMS] ){
	HGC_sideGhost[i] = lastIndex ;
	lastIndex += HGC_surface3D[i%N_DIMS];
      }

    for(int i=0; i<2*N_DIMS; i++){
      for(int j=i+1; j<2*N_DIMS; j++){
	if( (i%N_DIMS != j%N_DIMS ) && HGC_dimBreak[i%N_DIMS] && HGC_dimBreak[j%N_DIMS] ){
	  HGC_cornerGhost[i][j] = lastIndex;
	  HGC_cornerGhost[j][i] = lastIndex;
	  lastIndex += HGC_surface2D[i%N_DIMS][j%N_DIMS];
	}
      }
    }
#endif

    for(int i= 0 ; i < N_DIMS ; i++)
      HGC_procPosition[i] = comm_coords(default_topo)[i];

    // copying globals to device
    globals.copyToDevice();

    // create groups of process to use mpi reduce only on spatial points
    MPI_Comm_group(MPI_COMM_WORLD, &HGC_fullGroup);
    MPI_Group_rank(HGC_fullGroup,&HGC_fullRank);
    MPI_Group_size(HGC_fullGroup,&HGC_fullSize);

    int space3D_proc;
    space3D_proc = HGC_nProc[0] * HGC_nProc[1] * HGC_nProc[2];
    int *ranks = (int*) malloc(space3D_proc*sizeof(int));

    for(int i= 0 ; i < space3D_proc ; i++)
      ranks[i] = comm_coords(default_topo)[3] + HGC_nProc[3]*i;

    MPI_Group_incl(HGC_fullGroup,space3D_proc,ranks,&HGC_spaceGroup);
    MPI_Group_rank(HGC_spaceGroup,&HGC_spaceRank);
    MPI_Group_size(HGC_spaceGroup,&HGC_spaceSize);
    MPI_Comm_create(MPI_COMM_WORLD, HGC_spaceGroup , &HGC_spaceComm);

    // create group of process to use mpi gather
    int *ranksTime = (int*) malloc(HGC_nProc[3]*sizeof(int));

    for(int i=0 ; i < HGC_nProc[3] ; i++)
      ranksTime[i] = i;
    
    MPI_Group_incl(HGC_fullGroup,HGC_nProc[3], ranksTime, &HGC_timeGroup);
    MPI_Group_rank(HGC_timeGroup, &HGC_timeRank);
    MPI_Group_size(HGC_timeGroup, &HGC_timeSize);
    MPI_Comm_create(MPI_COMM_WORLD, HGC_timeGroup, &HGC_timeComm);

    //////////////////////////////////////////////////////////////////////////////
    free(ranks);
    free(ranksTime);

    cublasStatus_t error = cublasCreate(&cublas_handle);
    if (error != CUBLAS_STATUS_SUCCESS) errorQuda("cublasCreate failed with error %d", error);
    
    HGC_init_PLEGMA_flag = true;
    printfQuda("PLEGMA has been initialized\n");
  }  
  else{
    printfQuda("PLEGMA already initialized. Doing nothing.\n");
    return;
  }
  
}

void plegma::print_status(){

  if(HGC_init_PLEGMA_flag == false) errorQuda("You must initialize init_PLEGMA first");
  printfQuda("Number of colors is %d\n",N_COLS);
  printfQuda("Number of spins is %d\n",N_SPINS);
  printfQuda("Number of dimensions is %d\n",N_DIMS);
  globals.print();
}

void plegma::PLEGMA_end() {
  // TODO: here we should destroy everything is created in init.
  cublasStatus_t error = cublasDestroy(cublas_handle);
  if (error != CUBLAS_STATUS_SUCCESS) errorQuda("\nError indestroying cublas context, error code = %d\n", error);
}
