#ifndef PLEGMA_KERNEL_EXTERN_CUH
#define PLEGMA_KERNEL_EXTERN_CUH

/* block for device constants */
extern __constant__ bool c_dimBreak[4];
extern __constant__ int c_localL[4];
extern __constant__ int c_sideGhost[2*N_DIMS];
extern __constant__ int c_cornerGhost[2*N_DIMS][2*N_DIMS];
extern __constant__ int c_stride;
extern __constant__ int c_stride_spatial;
extern __constant__ int c_surface3D[4];
extern __constant__ int c_surface2D[N_DIMS][N_DIMS];
extern __constant__ double c_alphaAPE;
extern __constant__ double c_alphaGauss;
extern __constant__ int c_threads;
extern __constant__ int c_procPosition[4];
extern __constant__ int c_totalL[4];
extern __constant__ int c_Nmoms;
extern __constant__ short int c_moms[MAX_NMOMENTA][3];
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////                                                                                                                                                    
static const __device__ int eps[6][3]= {{0,1,2},
					{2,0,1},
					{1,2,0},
					{2,1,0},
					{0,2,1},
					{1,0,2}};
    
static const __device__ int sgn_eps[6]= { +1,+1,+1,-1,-1,-1 };


#endif
