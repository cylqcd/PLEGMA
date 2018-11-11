#ifndef PLEGMA_KERNEL_EXTERN_CUH
#define PLEGMA_KERNEL_EXTERN_CUH

/* block for device constants */
extern __constant__ bool c_dimBreak[4];
extern __constant__ int c_localL[4];
extern __constant__ int c_plusGhost[4];
extern __constant__ int c_minusGhost[4];
extern __constant__ int c_stride;
extern __constant__ int c_stride_spatial;
extern __constant__ int c_surface[4];
extern __constant__ double c_alphaAPE;
extern __constant__ double c_alphaGauss;
extern __constant__ int c_threads;
extern __constant__ int c_eps[6][3];
extern __constant__ int c_sgn_eps[6];
extern __constant__ int c_procPosition[4];
extern __constant__ int c_totalL[4];
extern __constant__ int c_Nmoms;
extern __constant__ short int c_moms[MAX_NMOMENTA][3];
extern __constant__ short int c_mesons_indices[10][16][4];
extern __constant__ short int c_NTN_indices[16][4];
extern __constant__ short int c_NTR_indices[64][6];
extern __constant__ short int c_RTN_indices[64][6];
extern __constant__ short int c_RTR_indices[256][8];
extern __constant__ short int c_Delta_indices[3][16][4];
extern __constant__ float c_mesons_values[10][16];
extern __constant__ float c_NTN_values[16];
extern __constant__ float c_NTR_values[64];
extern __constant__ float c_RTN_values[64];
extern __constant__ float c_RTR_values[256];
extern __constant__ float c_Delta_values[3][16];
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
extern short int GK_mesons_indices[10][16][4];
extern float GK_mesons_values[10][16];
extern short int GK_NTN_indices[16][4];
extern float GK_NTN_values[16];
extern short int GK_NTR_indices[64][6];
extern float GK_NTR_values[64];
extern short int GK_RTN_indices[64][6];
extern float GK_RTN_values[64];
extern short int GK_RTR_indices[256][8];
extern float GK_RTR_values[256];
extern short int GK_Delta_indices[3][16][4];
extern float GK_Delta_values[3][16];
// for mpi use global  variables
extern MPI_Group GK_fullGroup , GK_spaceGroup , GK_timeGroup;
extern MPI_Comm GK_spaceComm , GK_timeComm;
extern int GK_localRank;
extern int GK_localSize;
extern int GK_timeRank;
extern int GK_timeSize;
#endif
