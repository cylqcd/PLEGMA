#ifndef PLEGMA_KERNEL_EXTERN_CUH
#define PLEGMA_KERNEL_EXTERN_CUH

/* block for device constants */
extern __constant__ bool c_dimBreak[4];
extern __constant__ int c_localL[4];
extern __constant__ int c_plusGhost[4];
extern __constant__ int c_minusGhost[4];
extern __constant__ int c_cornerGhost[8][8];
extern __constant__ int c_stride;
extern __constant__ int c_stride_spatial;
extern __constant__ int c_surface[4];
extern __constant__ int c_surface2D[8][8];
extern __constant__ double c_alphaAPE;
extern __constant__ double c_alphaGauss;
extern __constant__ int c_threads;
extern __constant__ int c_procPosition[4];
extern __constant__ int c_totalL[4];
extern __constant__ int c_Nmoms;
extern __constant__ short int c_moms[MAX_NMOMENTA][3];
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////                                                                                                                                                    
/* Block for global variables */
extern float GK_deviceMemory;
extern int GK_strideFull;
extern double GK_alphaAPE;
extern double GK_alphaGauss;
extern int GK_localVolume;
extern int GK_totalVolume;
extern int GK_nsmearAPE;
extern int GK_nsmearGauss;
extern bool GK_dimBreak[N_DIMS];
extern int GK_localL[N_DIMS];
extern int GK_totalL[N_DIMS];
extern int GK_nProc[N_DIMS];
extern int GK_plusGhost[N_DIMS];
extern int GK_minusGhost[N_DIMS];
extern int GK_surface3D[N_DIMS];
extern bool GK_init_PLEGMA_flag;
extern int GK_Nsources;
extern int GK_sourcePosition[MAX_NSOURCES][N_DIMS];
extern int GK_Nmoms;
extern short int GK_moms[MAX_NMOMENTA][3];
// for mpi use global  variables
extern MPI_Group GK_fullGroup , GK_spaceGroup , GK_timeGroup;
extern MPI_Comm GK_spaceComm , GK_timeComm;
extern int GK_localRank;
extern int GK_localSize;
extern int GK_timeRank;
extern int GK_timeSize;

static const __device__ int eps[6][3]= {{0,1,2},
					{2,0,1},
					{1,2,0},
					{2,1,0},
					{0,2,1},
					{1,0,2}};
    
static const __device__ int sgn_eps[6]= { +1,+1,+1,-1,-1,-1 };


#endif
