// We must put PLEGMA_global.h before PLEGMA.h so it's allocated here.
#define ALLOCATE 
#include <PLEGMA_global.h> 
#undef ALLOCATE
#include <PLEGMA.h>
#include <errno.h>
#include <limits>
#include <string.h>
#include <PLEGMA_io.h>
#include <comm_quda.h>
#include <communicator_quda.h>

//#define TIMING_REPORT
using namespace plegma;
using namespace quda;
//extern Topology *default_topo;
std::vector<std::string> HDF5::open_files;
Communicator &get_current_communicator();

//Communicator default_t;
//extern Topology *default_topo;


void plegma::PLEGMA_init(int localL[4], int nProcs[4], int verbosity){
  HGC_hold_exit = false;
  HGC_options->checkErrors();

#define ADD_TO_GLOBAL
#include<global/PLEGMA_global_constants.h>
#undef ADD_TO_GLOBAL 
  if(HGC_init_PLEGMA_flag == false){
    for(int i = 0 ; i < N_DIMS ; i++)
      HGC_localL[i] = localL[i];

//    HGC_default_topo = get_current_communicator().comm_default_topology();//default_tcomm_default_topology();//get_current_communicator().default_topo;//default_topo;// get_current_communicator().default_topo;
    HGC_verbosity = verbosity;
    for(int i = 0 ; i < N_DIMS ; i++) {
      HGC_nProc[i] = nProcs[i];
      if(HGC_nProc[i] != comm_dim(i))
	PLEGMA_error("nProcs and comm_dim do not match for dim %d",i);
    }
    
    for(int i = 0 ; i < N_DIMS ; i++){   // take local and total lattice
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
    HGC_localVolume3D = HGC_localVolume/HGC_localL[3];
    
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

    for(int i=1; i<N_DIMS; i++){
      for(int j=0; j<i; j++){
	if(HGC_dimBreak[i] && HGC_dimBreak[j]){
	  HGC_surface2D[OFF2(i,j)] = 1;
	  for(int k=0; k<N_DIMS; k++)
	    HGC_surface2D[OFF2(i,j)] *= (k!=i && k!=j) ? HGC_localL[k] : 1;
	}
	else {
	  HGC_surface2D[OFF2(i,j)] = 0;
	}
      }
    }
    
    for(int i=2; i<N_DIMS; i++)
      for(int j=1; j<i; j++)
	for(int k=0; k<j; k++) {
	  if(HGC_dimBreak[i] && HGC_dimBreak[j] && HGC_dimBreak[k]){
	    HGC_surface1D[OFF3(i,j,k)] = 1;
	    for(int l=0; l<N_DIMS; l++)
	      HGC_surface1D[OFF3(i,j,k)] *= (l!=i && l!=j && l!=k) ? HGC_localL[l] : 1;
	  }
	  else {
	    HGC_surface1D[OFF3(i,j,k)] = 0;
	  }
	}
    
    for(int i = 0 ; i < N_DIMS ; i++)
      for(int dir = 0; dir < DIR_BOTH; dir++)
	HGC_sideGhost[i][dir] = 0;
    HGC_sideGhostVolume=0;
    HGC_sideGhostVolume3D=0;    

    for(int i=1; i<N_DIMS; i++)
      for(int j=0; j<i; j++)
	for(int dir1 = 0; dir1 < DIR_BOTH; dir1++)
	  for(int dir2 = 0; dir2 < DIR_BOTH; dir2++)
	    HGC_cornerGhost[OFF2SIGN(i,j,dir1,dir2)] = 0;
    HGC_cornerGhostVolume=0; 
    HGC_cornerGhostVolume3D=0;
   
    for(int i=2; i<N_DIMS; i++)
      for(int j=1; j<i; j++)
	for(int k=0; k<j; k++)
	  for(int dir1 = 0; dir1 < DIR_BOTH; dir1++)
	    for(int dir2 = 0; dir2 < DIR_BOTH; dir2++)
	      for(int dir3 = 0; dir3 < DIR_BOTH; dir3++)
		HGC_vertexGhost[OFF3SIGN(i,j,k,dir1,dir2,dir3)] = 0;
    HGC_vertexGhostVolume=0; 
    HGC_vertexGhostVolume3D=0;
   
#ifdef MULTI_GPU
    size_t lastIndex = 0;
    for(int i = 0 ; i < N_DIMS ; i++)
      for(int dir = 0; dir < DIR_BOTH; dir++) {
	HGC_sideGhost[i][dir] = lastIndex;
	if( HGC_dimBreak[i] ){
	  lastIndex += HGC_surface3D[i];
	}
      }
    HGC_sideGhostVolume = lastIndex;
    HGC_sideGhostVolume3D = HGC_sideGhost[DIM_T][0]/HGC_localL[DIM_T];
    
    lastIndex = 0;
    for(int i=1; i<N_DIMS; i++)
      for(int j=0; j<i; j++)
	for(int dir1 = 0; dir1 < DIR_BOTH; dir1++)
	  for(int dir2 = 0; dir2 < DIR_BOTH; dir2++) {
	    HGC_cornerGhost[OFF2SIGN(i,j,dir1,dir2)] = lastIndex;
	    if( HGC_dimBreak[i] && HGC_dimBreak[j] ) {
	      lastIndex += HGC_surface2D[OFF2(i,j)];
	    }
	  }
    HGC_cornerGhostVolume = lastIndex;
    HGC_cornerGhostVolume3D = HGC_cornerGhost[OFF2SIGN(DIM_T,0,0,0)]/HGC_localL[DIM_T];

    lastIndex = 0;
    for(int i=2; i<N_DIMS; i++)
      for(int j=1; j<i; j++)
	for(int k=0; k<j; k++)
	  for(int dir1 = 0; dir1 < DIR_BOTH; dir1++)
	    for(int dir2 = 0; dir2 < DIR_BOTH; dir2++)
	      for(int dir3 = 0; dir3 < DIR_BOTH; dir3++) {
		HGC_vertexGhost[OFF3SIGN(i,j,k,dir1,dir2,dir3)] = lastIndex;
		if( HGC_dimBreak[i] && HGC_dimBreak[j] && HGC_dimBreak[k] ) {
		  lastIndex += HGC_surface1D[OFF3(i,j,k)];
		}
	      }
    HGC_vertexGhostVolume = lastIndex;
    HGC_vertexGhostVolume3D = HGC_vertexGhost[OFF3SIGN(DIM_T,0,0,0,0,0)]/HGC_localL[DIM_T];
#endif

    for(int i= 0 ; i < N_DIMS ; i++)
      HGC_procPosition[i] = comm_coord(i);

    // copying globals to device
    printf("Local lattice volume %d \n",   HGC_localVolume);
    HGC_global_vars.copyToDevice();

    // create groups of process to use mpi reduce only on spatial points
    MPI_Comm_dup(MPI_COMM_WORLD, &HGC_fullComm);
    MPI_Comm_group(HGC_fullComm, &HGC_fullGroup);
    MPI_Group_rank(HGC_fullGroup,&HGC_fullRank);
    MPI_Group_size(HGC_fullGroup,&HGC_fullSize);

    int space3D_proc;
    space3D_proc = HGC_nProc[0] * HGC_nProc[1] * HGC_nProc[2];
    int ranks[space3D_proc];

    for(int i= 0 ; i < space3D_proc ; i++)
      ranks[i] = HGC_procPosition[3] + HGC_nProc[3]*i;

    MPI_Group_incl(HGC_fullGroup,space3D_proc,ranks,&HGC_spaceGroup);
    MPI_Group_rank(HGC_spaceGroup,&HGC_spaceRank);
    MPI_Group_size(HGC_spaceGroup,&HGC_spaceSize);
    MPI_Comm_create(HGC_fullComm, HGC_spaceGroup , &HGC_spaceComm);

    // create group of process to use mpi gather
    int ranksTime[HGC_nProc[3]];

    int spaceId = (HGC_procPosition[0] * HGC_nProc[1] + HGC_procPosition[1]) * HGC_nProc[2] + HGC_procPosition[2];
    for(int i=0 ; i < HGC_nProc[3] ; i++)
      ranksTime[i] = spaceId*HGC_nProc[3]+i;
    
    MPI_Group_incl(HGC_fullGroup,HGC_nProc[3], ranksTime, &HGC_timeGroup);
    MPI_Group_rank(HGC_timeGroup, &HGC_timeRank);
    MPI_Group_size(HGC_timeGroup, &HGC_timeSize);
    MPI_Comm_create(HGC_fullComm, HGC_timeGroup, &HGC_timeComm);

    int tmp;
    MPI_Comm_rank(HGC_timeComm,&tmp);
    assert(tmp==HGC_timeRank);
#ifdef __HIP__
    hipblasStatus_t error = hipblasCreate(&HGC_hipblas_handle);
    if (error != HIPBLAS_STATUS_SUCCESS) PLEGMA_error("hipblasCreate failed with error %d", error);
#else
    cublasStatus_t error = cublasCreate(&HGC_cublas_handle);
    if (error != CUBLAS_STATUS_SUCCESS) PLEGMA_error("cublasCreate failed with error %d", error);
#endif
    
    HGC_init_PLEGMA_flag = true;
    PLEGMA_printf("PLEGMA has been initialized\n");
  }  
  else{
    PLEGMA_printf("PLEGMA already initialized. Doing nothing.\n");
    return;
  }
  
}

void plegma::PLEGMA_status(){
  if(HGC_init_PLEGMA_flag == false) PLEGMA_error("You must initialize init_PLEGMA first");
  if(HGC_global_vars.check() == false) PLEGMA_error("Global variables do not match between host and device.\n");
  if(HGC_verbosity > 2) {
    PLEGMA_printf("Number of colors is %d\n",N_COLS);
    PLEGMA_printf("Number of spins is %d\n",N_SPINS);
    PLEGMA_printf("Number of dimensions is %d\n",N_DIMS);

    HGC_global_vars.print();
  }
}

void plegma::PLEGMA_end() {
  // TODO: here we should destroy everything is created in init.
  //cublasStatus_t error = cublasDestroy(HGC_cublas_handle);
  //if (error != CUBLAS_STATUS_SUCCESS) PLEGMA_error("\nError indestroying cublas context, error code = %d\n", error);
  if(HDF5::isWriting()) {
    PLEGMA_printf("Waiting for HDF5 to finish the writing\n");
    while(HDF5::isWriting()) sleep(0.001);
  }
  if(HGC_fullComm != MPI_COMM_NULL) MPI_Comm_free(&HGC_fullComm);
  if(HGC_spaceComm != MPI_COMM_NULL) MPI_Comm_free(&HGC_spaceComm);
  if(HGC_timeComm != MPI_COMM_NULL) MPI_Comm_free(&HGC_timeComm);
}
