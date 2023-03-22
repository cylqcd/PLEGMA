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
  HGC.hold_exit = false;
  HGC.options->checkErrors();

#define ADD_TO_GLOBAL
//#include<global/PLEGMA_global_constants.h>
#undef ADD_TO_GLOBAL
  
  if(HGC.init_PLEGMA_flag == false){
    for(int i = 0 ; i < N_DIMS ; i++)
      HGC.localL[i] = localL[i];

//    HGC.default_topo = get_current_communicator().comm_default_topology();//default_tcomm_default_topology();//get_current_communicator().default_topo;//default_topo;// get_current_communicator().default_topo;
    HGC.verbosity = verbosity;
    for(int i = 0 ; i < N_DIMS ; i++) {
      HGC.nProc[i] = nProcs[i];
      if(HGC.nProc[i] != comm_dim(i))
	PLEGMA_error("nProcs and comm_dim do not match for dim %d",i);
    }
    
    for(int i = 0 ; i < N_DIMS ; i++){   // take local and total lattice
      HGC.totalL[i] = HGC.nProc[i] * HGC.localL[i];
    }
    
    for(int i = 0 ; i < N_DIMS ; i++){
      if( HGC.localL[i] < HGC.totalL[i])
	HGC.dimBreak[i] = true;
      else
	HGC.dimBreak[i] = false;
    }
    
    HGC.localVolume = 1;
    HGC.totalVolume = 1;
    for(int i = 0 ; i < N_DIMS ; i++){
      HGC.localVolume *= HGC.localL[i];
      HGC.totalVolume *= HGC.totalL[i];
    }
    HGC.localVolume3D = HGC.localVolume/HGC.localL[3];
    
    for (int i=0; i<N_DIMS; i++) {
      if(HGC.dimBreak[i]) {
	HGC.surface3D[i] = 1;
	for (int j=0; j<N_DIMS; j++) {
	  if (i==j) continue;
	  HGC.surface3D[i] *= HGC.localL[j];
	}
      } else {
	HGC.surface3D[i] = 0;
      }
    }

    for(int i=1; i<N_DIMS; i++){
      for(int j=0; j<i; j++){
	if(HGC.dimBreak[i] && HGC.dimBreak[j]){
	  HGC.surface2D[OFF2(i,j)] = 1;
	  for(int k=0; k<N_DIMS; k++)
	    HGC.surface2D[OFF2(i,j)] *= (k!=i && k!=j) ? HGC.localL[k] : 1;
	}
	else {
	  HGC.surface2D[OFF2(i,j)] = 0;
	}
      }
    }
    
    for(int i=2; i<N_DIMS; i++)
      for(int j=1; j<i; j++)
	for(int k=0; k<j; k++) {
	  if(HGC.dimBreak[i] && HGC.dimBreak[j] && HGC.dimBreak[k]){
	    HGC.surface1D[OFF3(i,j,k)] = 1;
	    for(int l=0; l<N_DIMS; l++)
	      HGC.surface1D[OFF3(i,j,k)] *= (l!=i && l!=j && l!=k) ? HGC.localL[l] : 1;
	  }
	  else {
	    HGC.surface1D[OFF3(i,j,k)] = 0;
	  }
	}
    
    for(int i = 0 ; i < N_DIMS ; i++)
      for(int dir = 0; dir < DIR_BOTH; dir++)
	HGC.sideGhost[i][dir] = 0;
    HGC.sideGhostVolume=0;
    HGC.sideGhostVolume3D=0;    

    for(int i=1; i<N_DIMS; i++)
      for(int j=0; j<i; j++)
	for(int dir1 = 0; dir1 < DIR_BOTH; dir1++)
	  for(int dir2 = 0; dir2 < DIR_BOTH; dir2++)
	    HGC.cornerGhost[OFF2SIGN(i,j,dir1,dir2)] = 0;
    HGC.cornerGhostVolume=0; 
    HGC.cornerGhostVolume3D=0;
   
    for(int i=2; i<N_DIMS; i++)
      for(int j=1; j<i; j++)
	for(int k=0; k<j; k++)
	  for(int dir1 = 0; dir1 < DIR_BOTH; dir1++)
	    for(int dir2 = 0; dir2 < DIR_BOTH; dir2++)
	      for(int dir3 = 0; dir3 < DIR_BOTH; dir3++)
		HGC.vertexGhost[OFF3SIGN(i,j,k,dir1,dir2,dir3)] = 0;
    HGC.vertexGhostVolume=0; 
    HGC.vertexGhostVolume3D=0;
   
#ifdef MULTI_GPU
    size_t lastIndex = 0;
    for(int i = 0 ; i < N_DIMS ; i++)
      for(int dir = 0; dir < DIR_BOTH; dir++) {
	HGC.sideGhost[i][dir] = lastIndex;
	if( HGC.dimBreak[i] ){
	  lastIndex += HGC.surface3D[i];
	}
      }
    HGC.sideGhostVolume = lastIndex;
    HGC.sideGhostVolume3D = HGC.sideGhost[DIM_T][0]/HGC.localL[DIM_T];
    
    lastIndex = 0;
    for(int i=1; i<N_DIMS; i++)
      for(int j=0; j<i; j++)
	for(int dir1 = 0; dir1 < DIR_BOTH; dir1++)
	  for(int dir2 = 0; dir2 < DIR_BOTH; dir2++) {
	    HGC.cornerGhost[OFF2SIGN(i,j,dir1,dir2)] = lastIndex;
	    if( HGC.dimBreak[i] && HGC.dimBreak[j] ) {
	      lastIndex += HGC.surface2D[OFF2(i,j)];
	    }
	  }
    HGC.cornerGhostVolume = lastIndex;
    HGC.cornerGhostVolume3D = HGC.cornerGhost[OFF2SIGN(DIM_T,0,0,0)]/HGC.localL[DIM_T];

    lastIndex = 0;
    for(int i=2; i<N_DIMS; i++)
      for(int j=1; j<i; j++)
	for(int k=0; k<j; k++)
	  for(int dir1 = 0; dir1 < DIR_BOTH; dir1++)
	    for(int dir2 = 0; dir2 < DIR_BOTH; dir2++)
	      for(int dir3 = 0; dir3 < DIR_BOTH; dir3++) {
		HGC.vertexGhost[OFF3SIGN(i,j,k,dir1,dir2,dir3)] = lastIndex;
		if( HGC.dimBreak[i] && HGC.dimBreak[j] && HGC.dimBreak[k] ) {
		  lastIndex += HGC.surface1D[OFF3(i,j,k)];
		}
	      }
    HGC.vertexGhostVolume = lastIndex;
    HGC.vertexGhostVolume3D = HGC.vertexGhost[OFF3SIGN(DIM_T,0,0,0,0,0)]/HGC.localL[DIM_T];
#endif

    for(int i= 0 ; i < N_DIMS ; i++)
      HGC.procPosition[i] = comm_coord(i);

    // copying globals to device
#if defined (__NVCC__)
    cudaError_t cudaStatus = cudaGetSymbolAddress((void **)&DGC_ptr, DGC_const);
    if (cudaStatus != cudaSuccess) {
        PLEGMA_error("cudaGetSymbolAddress (dev_N) failed: %s\n", cudaGetErrorString(cudaStatus));
    }
#elif defined (__HIP__)
    hipError_t hipStatus = hipGetSymbolAddress((void **)&DGC_ptr, DGC_const);
    if (hipStatus != hipSuccess) {
        PLEGMA_error("hipGetSymbolAddress (dev_N) failed: %s\n", hipGetErrorString(cudaStatus));
    }
#endif

    qudaMemcpy(DGC_ptr, &HGC, sizeof(global_vars_both), qudaMemcpyHostToDevice);

    //HGC.global_vars.copyToDevice();

    // create groups of process to use mpi reduce only on spatial points
    MPI_Comm_dup(MPI_COMM_WORLD, &HGC.fullComm);
    MPI_Comm_group(HGC.fullComm, &HGC.fullGroup);
    MPI_Group_rank(HGC.fullGroup,&HGC.fullRank);
    MPI_Group_size(HGC.fullGroup,&HGC.fullSize);

    int space3D_proc;
    space3D_proc = HGC.nProc[0] * HGC.nProc[1] * HGC.nProc[2];
    int ranks[space3D_proc];

    for(int i= 0 ; i < space3D_proc ; i++)
      ranks[i] = HGC.procPosition[3] + HGC.nProc[3]*i;

    MPI_Group_incl(HGC.fullGroup,space3D_proc,ranks,&HGC.spaceGroup);
    MPI_Group_rank(HGC.spaceGroup,&HGC.spaceRank);
    MPI_Group_size(HGC.spaceGroup,&HGC.spaceSize);
    MPI_Comm_create(HGC.fullComm, HGC.spaceGroup , &HGC.spaceComm);

    // create group of process to use mpi gather
    int ranksTime[HGC.nProc[3]];

    int spaceId = (HGC.procPosition[0] * HGC.nProc[1] + HGC.procPosition[1]) * HGC.nProc[2] + HGC.procPosition[2];
    for(int i=0 ; i < HGC.nProc[3] ; i++)
      ranksTime[i] = spaceId*HGC.nProc[3]+i;
    
    MPI_Group_incl(HGC.fullGroup,HGC.nProc[3], ranksTime, &HGC.timeGroup);
    MPI_Group_rank(HGC.timeGroup, &HGC.timeRank);
    MPI_Group_size(HGC.timeGroup, &HGC.timeSize);
    MPI_Comm_create(HGC.fullComm, HGC.timeGroup, &HGC.timeComm);

    int tmp;
    MPI_Comm_rank(HGC.timeComm,&tmp);
    assert(tmp==HGC.timeRank);

    //cublasStatus_t error = cublasCreate(&HGC.cublas_handle);
    //if (error != CUBLAS_STATUS_SUCCESS) PLEGMA_error("cublasCreate failed with error %d", error);
    
    HGC.init_PLEGMA_flag = true;
    PLEGMA_printf("PLEGMA has been initialized\n");
  }  
  else{
    PLEGMA_printf("PLEGMA already initialized. Doing nothing.\n");
    return;
  }
  
}

void plegma::PLEGMA_status(){
  if(HGC.init_PLEGMA_flag == false) PLEGMA_error("You must initialize init_PLEGMA first");
//  if(HGC.global_vars.check() == false) PLEGMA_error("Global variables do not match between host and device.\n");
  if(HGC.verbosity > 2) {
    PLEGMA_printf("Number of colors is %d\n",N_COLS);
    PLEGMA_printf("Number of spins is %d\n",N_SPINS);
    PLEGMA_printf("Number of dimensions is %d\n",N_DIMS);

//    HGC.global_vars.print();
  }
}

void plegma::PLEGMA_end() {
  // TODO: here we should destroy everything is created in init.
  //cublasStatus_t error = cublasDestroy(HGC.cublas_handle);
  //if (error != CUBLAS_STATUS_SUCCESS) PLEGMA_error("\nError indestroying cublas context, error code = %d\n", error);
  if(HDF5::isWriting()) {
    PLEGMA_printf("Waiting for HDF5 to finish the writing\n");
    while(HDF5::isWriting()) sleep(0.001);
  }
  if(HGC.fullComm != MPI_COMM_NULL) MPI_Comm_free(&HGC.fullComm);
  if(HGC.spaceComm != MPI_COMM_NULL) MPI_Comm_free(&HGC.spaceComm);
  if(HGC.timeComm != MPI_COMM_NULL) MPI_Comm_free(&HGC.timeComm);
}
