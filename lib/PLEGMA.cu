#include <PLEGMA.h>
#include <errno.h>
#include <mpi.h>  
#include <limits>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <typeinfo>

//#define TIMING_REPORT
using namespace plegma;

/* block for device constants */
__constant__ bool c_dimBreak[N_DIMS];
__constant__ int c_localL[N_DIMS];
__constant__ int c_plusGhost[N_DIMS];
__constant__ int c_minusGhost[N_DIMS];
__constant__ int c_cornerGhost[2*N_DIMS][2*N_DIMS];
__constant__ int c_stride;
__constant__ int c_stride_spatial;
__constant__ int c_surface[N_DIMS];
__constant__ double c_alphaAPE;
__constant__ double c_alphaGauss;
__constant__ int c_threads;
__constant__ int c_eps[6][3];
__constant__ int c_sgn_eps[6];
__constant__ int c_procPosition[N_DIMS];
__constant__ int c_totalL[N_DIMS];
__constant__ int c_Nmoms;
__constant__ short int c_moms[MAX_NMOMENTA][3];
__constant__ short int c_mesons_indices[10][16][N_DIMS];
__constant__ short int c_NTN_indices[16][N_DIMS];
__constant__ short int c_NTR_indices[64][6];
__constant__ short int c_RTN_indices[64][6];
__constant__ short int c_RTR_indices[256][8];
__constant__ short int c_Delta_indices[3][16][N_DIMS];
__constant__ float c_mesons_values[10][16];
__constant__ float c_NTN_values[16];
__constant__ float c_NTR_values[64];
__constant__ float c_RTN_values[64];
__constant__ float c_RTR_values[256];
__constant__ float c_Delta_values[3][16];
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////                                                                                                                                                    
/* Block for global variables */
float GK_deviceMemory = 0.;
int GK_strideFull;
double GK_alphaAPE;
double GK_alphaGauss;
int GK_localVolume;
int GK_totalVolume;
int GK_nsmearAPE;
int GK_nsmearGauss;
bool GK_dimBreak[N_DIMS];
int GK_localL[N_DIMS];
int GK_totalL[N_DIMS];
int GK_nProc[N_DIMS];
int GK_plusGhost[N_DIMS];
int GK_minusGhost[N_DIMS];
int GK_surface3D[N_DIMS];
int GK_surface2D[N_DIMS][N_DIMS];
int GK_cornerGhost[2*N_DIMS][2*N_DIMS];
bool GK_init_PLEGMA_flag = false;
int GK_Nsources;
int GK_sourcePosition[MAX_NSOURCES][N_DIMS];
int GK_Nmoms;
short int GK_moms[MAX_NMOMENTA][3];
short int GK_mesons_indices[10][16][4] = {0,0,0,0,0,0,1,1,0,0,2,2,0,0,3,3,1,1,0,0,1,1,1,1,1,1,2,2,1,1,3,3,2,2,0,0,2,2,1,1,2,2,2,2,2,2,3,3,3,3,0,0,3,3,1,1,3,3,2,2,3,3,3,3,0,2,0,2,0,2,1,3,0,2,2,0,0,2,3,1,1,3,0,2,1,3,1,3,1,3,2,0,1,3,3,1,2,0,0,2,2,0,1,3,2,0,2,0,2,0,3,1,3,1,0,2,3,1,1,3,3,1,2,0,3,1,3,1,0,3,0,3,0,3,1,2,0,3,2,1,0,3,3,0,1,2,0,3,1,2,1,2,1,2,2,1,1,2,3,0,2,1,0,3,2,1,1,2,2,1,2,1,2,1,3,0,3,0,0,3,3,0,1,2,3,0,2,1,3,0,3,0,0,3,0,3,0,3,1,2,0,3,2,1,0,3,3,0,1,2,0,3,1,2,1,2,1,2,2,1,1,2,3,0,2,1,0,3,2,1,1,2,2,1,2,1,2,1,3,0,3,0,0,3,3,0,1,2,3,0,2,1,3,0,3,0,0,2,0,2,0,2,1,3,0,2,2,0,0,2,3,1,1,3,0,2,1,3,1,3,1,3,2,0,1,3,3,1,2,0,0,2,2,0,1,3,2,0,2,0,2,0,3,1,3,1,0,2,3,1,1,3,3,1,2,0,3,1,3,1,0,0,0,0,0,0,1,1,0,0,2,2,0,0,3,3,1,1,0,0,1,1,1,1,1,1,2,2,1,1,3,3,2,2,0,0,2,2,1,1,2,2,2,2,2,2,3,3,3,3,0,0,3,3,1,1,3,3,2,2,3,3,3,3,0,1,0,1,0,1,1,0,0,1,2,3,0,1,3,2,1,0,0,1,1,0,1,0,1,0,2,3,1,0,3,2,2,3,0,1,2,3,1,0,2,3,2,3,2,3,3,2,3,2,0,1,3,2,1,0,3,2,2,3,3,2,3,2,0,1,0,1,0,1,1,0,0,1,2,3,0,1,3,2,1,0,0,1,1,0,1,0,1,0,2,3,1,0,3,2,2,3,0,1,2,3,1,0,2,3,2,3,2,3,3,2,3,2,0,1,3,2,1,0,3,2,2,3,3,2,3,2,0,0,0,0,0,0,1,1,0,0,2,2,0,0,3,3,1,1,0,0,1,1,1,1,1,1,2,2,1,1,3,3,2,2,0,0,2,2,1,1,2,2,2,2,2,2,3,3,3,3,0,0,3,3,1,1,3,3,2,2,3,3,3,3,0,2,0,2,0,2,1,3,0,2,2,0,0,2,3,1,1,3,0,2,1,3,1,3,1,3,2,0,1,3,3,1,2,0,0,2,2,0,1,3,2,0,2,0,2,0,3,1,3,1,0,2,3,1,1,3,3,1,2,0,3,1,3,1};
float GK_mesons_values[10][16] = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,-1,-1,1,1,-1,-1,1,1,1,1,-1,-1,1,1,-1,-1,1,-1,-1,1,-1,1,1,-1,-1,1,1,-1,1,-1,-1,1,-1,1,1,-1,1,-1,-1,1,1,-1,-1,1,-1,1,1,-1,1,1,-1,-1,1,1,-1,-1,-1,-1,1,1,-1,-1,1,1,-1,-1,1,1,-1,-1,1,1,1,1,-1,-1,1,1,-1,-1,1,-1,-1,1,-1,1,1,-1,-1,1,1,-1,1,-1,-1,1,-1,1,1,-1,1,-1,-1,1,1,-1,-1,1,-1,1,1,-1,1,1,-1,-1,1,1,-1,-1,-1,-1,1,1,-1,-1,1,1};
short int GK_NTN_indices[16][4] = {0,1,0,1,0,1,1,0,0,1,2,3,0,1,3,2,1,0,0,1,1,0,1,0,1,0,2,3,1,0,3,2,2,3,0,1,2,3,1,0,2,3,2,3,2,3,3,2,3,2,0,1,3,2,1,0,3,2,2,3,3,2,3,2};
float GK_NTN_values[16] = {-1,1,-1,1,1,-1,1,-1,-1,1,-1,1,1,-1,1,-1};
short int GK_NTR_indices[64][6] = {0,1,0,3,0,2,0,1,0,3,1,3,0,1,0,3,2,0,0,1,0,3,3,1,0,1,1,2,0,2,0,1,1,2,1,3,0,1,1,2,2,0,0,1,1,2,3,1,0,1,2,1,0,2,0,1,2,1,1,3,0,1,2,1,2,0,0,1,2,1,3,1,0,1,3,0,0,2,0,1,3,0,1,3,0,1,3,0,2,0,0,1,3,0,3,1,1,0,0,3,0,2,1,0,0,3,1,3,1,0,0,3,2,0,1,0,0,3,3,1,1,0,1,2,0,2,1,0,1,2,1,3,1,0,1,2,2,0,1,0,1,2,3,1,1,0,2,1,0,2,1,0,2,1,1,3,1,0,2,1,2,0,1,0,2,1,3,1,1,0,3,0,0,2,1,0,3,0,1,3,1,0,3,0,2,0,1,0,3,0,3,1,2,3,0,3,0,2,2,3,0,3,1,3,2,3,0,3,2,0,2,3,0,3,3,1,2,3,1,2,0,2,2,3,1,2,1,3,2,3,1,2,2,0,2,3,1,2,3,1,2,3,2,1,0,2,2,3,2,1,1,3,2,3,2,1,2,0,2,3,2,1,3,1,2,3,3,0,0,2,2,3,3,0,1,3,2,3,3,0,2,0,2,3,3,0,3,1,3,2,0,3,0,2,3,2,0,3,1,3,3,2,0,3,2,0,3,2,0,3,3,1,3,2,1,2,0,2,3,2,1,2,1,3,3,2,1,2,2,0,3,2,1,2,3,1,3,2,2,1,0,2,3,2,2,1,1,3,3,2,2,1,2,0,3,2,2,1,3,1,3,2,3,0,0,2,3,2,3,0,1,3,3,2,3,0,2,0,3,2,3,0,3,1};
float GK_NTR_values[64] = {1,1,1,1,-1,-1,-1,-1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,1,1,1,1,-1,-1,-1,-1,1,1,1,1,1,1,1,1,-1,-1,-1,-1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,1,1,1,1,-1,-1,-1,-1,1,1,1,1};
short int GK_RTN_indices[64][6] = {0,3,0,1,0,2,0,3,0,1,1,3,0,3,0,1,2,0,0,3,0,1,3,1,0,3,1,0,0,2,0,3,1,0,1,3,0,3,1,0,2,0,0,3,1,0,3,1,0,3,2,3,0,2,0,3,2,3,1,3,0,3,2,3,2,0,0,3,2,3,3,1,0,3,3,2,0,2,0,3,3,2,1,3,0,3,3,2,2,0,0,3,3,2,3,1,1,2,0,1,0,2,1,2,0,1,1,3,1,2,0,1,2,0,1,2,0,1,3,1,1,2,1,0,0,2,1,2,1,0,1,3,1,2,1,0,2,0,1,2,1,0,3,1,1,2,2,3,0,2,1,2,2,3,1,3,1,2,2,3,2,0,1,2,2,3,3,1,1,2,3,2,0,2,1,2,3,2,1,3,1,2,3,2,2,0,1,2,3,2,3,1,2,1,0,1,0,2,2,1,0,1,1,3,2,1,0,1,2,0,2,1,0,1,3,1,2,1,1,0,0,2,2,1,1,0,1,3,2,1,1,0,2,0,2,1,1,0,3,1,2,1,2,3,0,2,2,1,2,3,1,3,2,1,2,3,2,0,2,1,2,3,3,1,2,1,3,2,0,2,2,1,3,2,1,3,2,1,3,2,2,0,2,1,3,2,3,1,3,0,0,1,0,2,3,0,0,1,1,3,3,0,0,1,2,0,3,0,0,1,3,1,3,0,1,0,0,2,3,0,1,0,1,3,3,0,1,0,2,0,3,0,1,0,3,1,3,0,2,3,0,2,3,0,2,3,1,3,3,0,2,3,2,0,3,0,2,3,3,1,3,0,3,2,0,2,3,0,3,2,1,3,3,0,3,2,2,0,3,0,3,2,3,1};
float GK_RTN_values[64] = {-1,-1,-1,-1,1,1,1,1,-1,-1,-1,-1,1,1,1,1,1,1,1,1,-1,-1,-1,-1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,1,1,1,1,-1,-1,-1,-1,1,1,1,1,1,1,1,1,-1,-1,-1,-1,1,1,1,1,-1,-1,-1,-1};
short int GK_RTR_indices[256][8] = {0,3,0,3,0,2,0,2,0,3,0,3,0,2,1,3,0,3,0,3,0,2,2,0,0,3,0,3,0,2,3,1,0,3,0,3,1,3,0,2,0,3,0,3,1,3,1,3,0,3,0,3,1,3,2,0,0,3,0,3,1,3,3,1,0,3,0,3,2,0,0,2,0,3,0,3,2,0,1,3,0,3,0,3,2,0,2,0,0,3,0,3,2,0,3,1,0,3,0,3,3,1,0,2,0,3,0,3,3,1,1,3,0,3,0,3,3,1,2,0,0,3,0,3,3,1,3,1,0,3,1,2,0,2,0,2,0,3,1,2,0,2,1,3,0,3,1,2,0,2,2,0,0,3,1,2,0,2,3,1,0,3,1,2,1,3,0,2,0,3,1,2,1,3,1,3,0,3,1,2,1,3,2,0,0,3,1,2,1,3,3,1,0,3,1,2,2,0,0,2,0,3,1,2,2,0,1,3,0,3,1,2,2,0,2,0,0,3,1,2,2,0,3,1,0,3,1,2,3,1,0,2,0,3,1,2,3,1,1,3,0,3,1,2,3,1,2,0,0,3,1,2,3,1,3,1,0,3,2,1,0,2,0,2,0,3,2,1,0,2,1,3,0,3,2,1,0,2,2,0,0,3,2,1,0,2,3,1,0,3,2,1,1,3,0,2,0,3,2,1,1,3,1,3,0,3,2,1,1,3,2,0,0,3,2,1,1,3,3,1,0,3,2,1,2,0,0,2,0,3,2,1,2,0,1,3,0,3,2,1,2,0,2,0,0,3,2,1,2,0,3,1,0,3,2,1,3,1,0,2,0,3,2,1,3,1,1,3,0,3,2,1,3,1,2,0,0,3,2,1,3,1,3,1,0,3,3,0,0,2,0,2,0,3,3,0,0,2,1,3,0,3,3,0,0,2,2,0,0,3,3,0,0,2,3,1,0,3,3,0,1,3,0,2,0,3,3,0,1,3,1,3,0,3,3,0,1,3,2,0,0,3,3,0,1,3,3,1,0,3,3,0,2,0,0,2,0,3,3,0,2,0,1,3,0,3,3,0,2,0,2,0,0,3,3,0,2,0,3,1,0,3,3,0,3,1,0,2,0,3,3,0,3,1,1,3,0,3,3,0,3,1,2,0,0,3,3,0,3,1,3,1,1,2,0,3,0,2,0,2,1,2,0,3,0,2,1,3,1,2,0,3,0,2,2,0,1,2,0,3,0,2,3,1,1,2,0,3,1,3,0,2,1,2,0,3,1,3,1,3,1,2,0,3,1,3,2,0,1,2,0,3,1,3,3,1,1,2,0,3,2,0,0,2,1,2,0,3,2,0,1,3,1,2,0,3,2,0,2,0,1,2,0,3,2,0,3,1,1,2,0,3,3,1,0,2,1,2,0,3,3,1,1,3,1,2,0,3,3,1,2,0,1,2,0,3,3,1,3,1,1,2,1,2,0,2,0,2,1,2,1,2,0,2,1,3,1,2,1,2,0,2,2,0,1,2,1,2,0,2,3,1,1,2,1,2,1,3,0,2,1,2,1,2,1,3,1,3,1,2,1,2,1,3,2,0,1,2,1,2,1,3,3,1,1,2,1,2,2,0,0,2,1,2,1,2,2,0,1,3,1,2,1,2,2,0,2,0,1,2,1,2,2,0,3,1,1,2,1,2,3,1,0,2,1,2,1,2,3,1,1,3,1,2,1,2,3,1,2,0,1,2,1,2,3,1,3,1,1,2,2,1,0,2,0,2,1,2,2,1,0,2,1,3,1,2,2,1,0,2,2,0,1,2,2,1,0,2,3,1,1,2,2,1,1,3,0,2,1,2,2,1,1,3,1,3,1,2,2,1,1,3,2,0,1,2,2,1,1,3,3,1,1,2,2,1,2,0,0,2,1,2,2,1,2,0,1,3,1,2,2,1,2,0,2,0,1,2,2,1,2,0,3,1,1,2,2,1,3,1,0,2,1,2,2,1,3,1,1,3,1,2,2,1,3,1,2,0,1,2,2,1,3,1,3,1,1,2,3,0,0,2,0,2,1,2,3,0,0,2,1,3,1,2,3,0,0,2,2,0,1,2,3,0,0,2,3,1,1,2,3,0,1,3,0,2,1,2,3,0,1,3,1,3,1,2,3,0,1,3,2,0,1,2,3,0,1,3,3,1,1,2,3,0,2,0,0,2,1,2,3,0,2,0,1,3,1,2,3,0,2,0,2,0,1,2,3,0,2,0,3,1,1,2,3,0,3,1,0,2,1,2,3,0,3,1,1,3,1,2,3,0,3,1,2,0,1,2,3,0,3,1,3,1,2,1,0,3,0,2,0,2,2,1,0,3,0,2,1,3,2,1,0,3,0,2,2,0,2,1,0,3,0,2,3,1,2,1,0,3,1,3,0,2,2,1,0,3,1,3,1,3,2,1,0,3,1,3,2,0,2,1,0,3,1,3,3,1,2,1,0,3,2,0,0,2,2,1,0,3,2,0,1,3,2,1,0,3,2,0,2,0,2,1,0,3,2,0,3,1,2,1,0,3,3,1,0,2,2,1,0,3,3,1,1,3,2,1,0,3,3,1,2,0,2,1,0,3,3,1,3,1,2,1,1,2,0,2,0,2,2,1,1,2,0,2,1,3,2,1,1,2,0,2,2,0,2,1,1,2,0,2,3,1,2,1,1,2,1,3,0,2,2,1,1,2,1,3,1,3,2,1,1,2,1,3,2,0,2,1,1,2,1,3,3,1,2,1,1,2,2,0,0,2,2,1,1,2,2,0,1,3,2,1,1,2,2,0,2,0,2,1,1,2,2,0,3,1,2,1,1,2,3,1,0,2,2,1,1,2,3,1,1,3,2,1,1,2,3,1,2,0,2,1,1,2,3,1,3,1,2,1,2,1,0,2,0,2,2,1,2,1,0,2,1,3,2,1,2,1,0,2,2,0,2,1,2,1,0,2,3,1,2,1,2,1,1,3,0,2,2,1,2,1,1,3,1,3,2,1,2,1,1,3,2,0,2,1,2,1,1,3,3,1,2,1,2,1,2,0,0,2,2,1,2,1,2,0,1,3,2,1,2,1,2,0,2,0,2,1,2,1,2,0,3,1,2,1,2,1,3,1,0,2,2,1,2,1,3,1,1,3,2,1,2,1,3,1,2,0,2,1,2,1,3,1,3,1,2,1,3,0,0,2,0,2,2,1,3,0,0,2,1,3,2,1,3,0,0,2,2,0,2,1,3,0,0,2,3,1,2,1,3,0,1,3,0,2,2,1,3,0,1,3,1,3,2,1,3,0,1,3,2,0,2,1,3,0,1,3,3,1,2,1,3,0,2,0,0,2,2,1,3,0,2,0,1,3,2,1,3,0,2,0,2,0,2,1,3,0,2,0,3,1,2,1,3,0,3,1,0,2,2,1,3,0,3,1,1,3,2,1,3,0,3,1,2,0,2,1,3,0,3,1,3,1,3,0,0,3,0,2,0,2,3,0,0,3,0,2,1,3,3,0,0,3,0,2,2,0,3,0,0,3,0,2,3,1,3,0,0,3,1,3,0,2,3,0,0,3,1,3,1,3,3,0,0,3,1,3,2,0,3,0,0,3,1,3,3,1,3,0,0,3,2,0,0,2,3,0,0,3,2,0,1,3,3,0,0,3,2,0,2,0,3,0,0,3,2,0,3,1,3,0,0,3,3,1,0,2,3,0,0,3,3,1,1,3,3,0,0,3,3,1,2,0,3,0,0,3,3,1,3,1,3,0,1,2,0,2,0,2,3,0,1,2,0,2,1,3,3,0,1,2,0,2,2,0,3,0,1,2,0,2,3,1,3,0,1,2,1,3,0,2,3,0,1,2,1,3,1,3,3,0,1,2,1,3,2,0,3,0,1,2,1,3,3,1,3,0,1,2,2,0,0,2,3,0,1,2,2,0,1,3,3,0,1,2,2,0,2,0,3,0,1,2,2,0,3,1,3,0,1,2,3,1,0,2,3,0,1,2,3,1,1,3,3,0,1,2,3,1,2,0,3,0,1,2,3,1,3,1,3,0,2,1,0,2,0,2,3,0,2,1,0,2,1,3,3,0,2,1,0,2,2,0,3,0,2,1,0,2,3,1,3,0,2,1,1,3,0,2,3,0,2,1,1,3,1,3,3,0,2,1,1,3,2,0,3,0,2,1,1,3,3,1,3,0,2,1,2,0,0,2,3,0,2,1,2,0,1,3,3,0,2,1,2,0,2,0,3,0,2,1,2,0,3,1,3,0,2,1,3,1,0,2,3,0,2,1,3,1,1,3,3,0,2,1,3,1,2,0,3,0,2,1,3,1,3,1,3,0,3,0,0,2,0,2,3,0,3,0,0,2,1,3,3,0,3,0,0,2,2,0,3,0,3,0,0,2,3,1,3,0,3,0,1,3,0,2,3,0,3,0,1,3,1,3,3,0,3,0,1,3,2,0,3,0,3,0,1,3,3,1,3,0,3,0,2,0,0,2,3,0,3,0,2,0,1,3,3,0,3,0,2,0,2,0,3,0,3,0,2,0,3,1,3,0,3,0,3,1,0,2,3,0,3,0,3,1,1,3,3,0,3,0,3,1,2,0,3,0,3,0,3,1,3,1};
float GK_RTR_values[256] = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
short int GK_Delta_indices[3][16][4] = {0,0,0,0,0,0,1,1,0,0,2,2,0,0,3,3,1,1,0,0,1,1,1,1,1,1,2,2,1,1,3,3,2,2,0,0,2,2,1,1,2,2,2,2,2,2,3,3,3,3,0,0,3,3,1,1,3,3,2,2,3,3,3,3,0,0,0,0,0,0,1,1,0,0,2,2,0,0,3,3,1,1,0,0,1,1,1,1,1,1,2,2,1,1,3,3,2,2,0,0,2,2,1,1,2,2,2,2,2,2,3,3,3,3,0,0,3,3,1,1,3,3,2,2,3,3,3,3,0,1,0,1,0,1,1,0,0,1,2,3,0,1,3,2,1,0,0,1,1,0,1,0,1,0,2,3,1,0,3,2,2,3,0,1,2,3,1,0,2,3,2,3,2,3,3,2,3,2,0,1,3,2,1,0,3,2,2,3,3,2,3,2};
float GK_Delta_values[3][16] = {1,-1,-1,1,-1,1,1,-1,-1,1,1,-1,1,-1,-1,1,1,1,-1,-1,1,1,-1,-1,-1,-1,1,1,-1,-1,1,1,1,1,-1,-1,1,1,-1,-1,-1,-1,1,1,-1,-1,1,1};
// for mpi use global  variables
MPI_Group GK_fullGroup , GK_spaceGroup , GK_timeGroup;
MPI_Comm GK_spaceComm , GK_timeComm;
int GK_localRank;
int GK_localSize;
int GK_timeRank;
int GK_timeSize;

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

void plegma::initialize(PLEGMA_params *params){
  
  if(GK_init_PLEGMA_flag == false){
    GK_alphaAPE = params->alphaAPE;
    GK_alphaGauss = params->alphaGauss;
    GK_nsmearAPE = params->nsmearAPE;
    GK_nsmearGauss = params->nsmearGauss;
    createMomenta(params->Q_sq);
    // from now on depends on lattice and break format we choose
    
    for(int i = 0 ; i < N_DIMS ; i++)
      GK_nProc[i] = comm_dim(i);
    
    for(int i = 0 ; i < N_DIMS ; i++){   // take local and total lattice
      GK_localL[i] = params->lL[i];
      GK_totalL[i] = GK_nProc[i] * GK_localL[i];
    }
    
    for(int i = 0 ; i < N_DIMS ; i++){
      if( GK_localL[i] < GK_totalL[i])
	GK_dimBreak[i] = true;
      else
	GK_dimBreak[i] = false;
    }
    
    GK_localVolume = 1;
    GK_totalVolume = 1;
    for(int i = 0 ; i < N_DIMS ; i++){
      GK_localVolume *= GK_localL[i];
      GK_totalVolume *= GK_totalL[i];
    }

    GK_strideFull = GK_localVolume;

    for (int i=0; i<N_DIMS; i++) {
      GK_surface3D[i] = 1;
      for (int j=0; j<N_DIMS; j++) {
	if (i==j) continue;
	GK_surface3D[i] *= GK_localL[j];
      }
    }
        
    for(int i = 0 ; i < N_DIMS ; i++)
      if( GK_localL[i] == GK_totalL[i] )
	GK_surface3D[i] = 0;

    for(int i=0; i<N_DIMS; i++){
      for(int j=0; j<N_DIMS; j++){
	if(i!=j && GK_dimBreak[i] && GK_dimBreak[j]){
	  GK_surface2D[i][j] = 1;
	  for(int k=0; k<N_DIMS; k++)
	    GK_surface2D[i][j] *= (k!=i && k!=j) ? GK_localL[k] : 1;
	}
	else GK_surface2D[i][j] = 0;
      }
    }

    
    for(int i = 0 ; i < N_DIMS ; i++){
      GK_plusGhost[i] = 0;
      GK_minusGhost[i] = 0;
    }

    for(int i=0; i<2*N_DIMS; i++){
      for(int j=0; j<2*N_DIMS; j++){
	GK_cornerGhost[i][j] = 0;
      }
    }
    
#ifdef MULTI_GPU

    size_t lastIndex = GK_localVolume;
    for(int i = 0 ; i < N_DIMS ; i++)
      if( GK_localL[i] < GK_totalL[i] ){
	GK_plusGhost[i] = lastIndex ;
	GK_minusGhost[i] = lastIndex + GK_surface3D[i];
	lastIndex += 2*GK_surface3D[i];
      }

    for(int i=0; i<2*N_DIMS; i++){
      for(int j=0; j<2*N_DIMS; j++){
	if( (i%N_DIMS != j%N_DIMS ) && GK_dimBreak[i%N_DIMS] && GK_dimBreak[j%N_DIMS] ){
	  GK_cornerGhost[i][j] = lastIndex;
	  lastIndex += GK_surface2D[i%N_DIMS][j%N_DIMS];
	}
      }
    }
    
#endif

    const int eps[6][3]=
      {
	{0,1,2},
	{2,0,1},
	{1,2,0},
	{2,1,0},
	{0,2,1},
	{1,0,2}
      };
    
    const int sgn_eps[6]=
      {
	+1,+1,+1,-1,-1,-1
      };

    int procPosition[4];
    
    for(int i= 0 ; i < 4 ; i++)
      procPosition[i] = comm_coords(default_topo)[i];

    // put it zero but change it later
    GK_Nsources = params->Nsources;
    if(GK_Nsources > MAX_NSOURCES) errorQuda("Error you exceeded maximum number of source position\n");

    for(int is = 0 ; is < GK_Nsources ; is++)
      for(int i = 0 ; i < 4 ; i++)
	GK_sourcePosition[is][i] = params->sourcePosition[is][i];

    // initialization consist also from define device constants
    cudaMemcpyToSymbol(c_stride, &GK_strideFull, sizeof(int) );
    int tmp = GK_strideFull/GK_localL[3];
    cudaMemcpyToSymbol(c_stride_spatial, &tmp, sizeof(int) );
    cudaMemcpyToSymbol(c_alphaAPE, &GK_alphaAPE , sizeof(double) );
    cudaMemcpyToSymbol(c_alphaGauss, &GK_alphaGauss , sizeof(double) );
    cudaMemcpyToSymbol(c_threads , &GK_localVolume , sizeof(int) ); // may change

    cudaMemcpyToSymbol(c_dimBreak , GK_dimBreak , N_DIMS*sizeof(bool) );
    cudaMemcpyToSymbol(c_localL , GK_localL , N_DIMS*sizeof(int) );
    cudaMemcpyToSymbol(c_totalL , GK_totalL , N_DIMS*sizeof(int) );
    cudaMemcpyToSymbol(c_plusGhost , GK_plusGhost , N_DIMS*sizeof(int) );
    cudaMemcpyToSymbol(c_minusGhost , GK_minusGhost , N_DIMS*sizeof(int) );
    cudaMemcpyToSymbol(c_cornerGhost, GK_cornerGhost, 4*N_DIMS*N_DIMS*sizeof(int));
    cudaMemcpyToSymbol(c_surface , GK_surface3D , N_DIMS*sizeof(int) );
    
    cudaMemcpyToSymbol(c_eps, &(eps[0][0]) , 6*3*sizeof(int) );
    cudaMemcpyToSymbol(c_sgn_eps, sgn_eps , 6*sizeof(int) );

    cudaMemcpyToSymbol(c_procPosition, procPosition, N_DIMS*sizeof(int));

    cudaMemcpyToSymbol(c_Nmoms, &GK_Nmoms, sizeof(int));
    cudaMemcpyToSymbol(c_moms, GK_moms, MAX_NMOMENTA*3*sizeof(short int));
    cudaMemcpyToSymbol(c_mesons_indices,GK_mesons_indices,10*16*4*sizeof(short int));
    cudaMemcpyToSymbol(c_NTN_indices,GK_NTN_indices,16*4*sizeof(short int));
    cudaMemcpyToSymbol(c_NTR_indices,GK_NTR_indices,64*6*sizeof(short int));
    cudaMemcpyToSymbol(c_RTN_indices,GK_RTN_indices,64*6*sizeof(short int));
    cudaMemcpyToSymbol(c_RTR_indices,GK_RTR_indices,256*8*sizeof(short int));
    cudaMemcpyToSymbol(c_Delta_indices,GK_Delta_indices,3*16*4*sizeof(short int));
    
    cudaMemcpyToSymbol(c_mesons_values,GK_mesons_values,10*16*sizeof(float));
    cudaMemcpyToSymbol(c_NTN_values,GK_NTN_values,16*sizeof(float));
    cudaMemcpyToSymbol(c_NTR_values,GK_NTR_values,64*sizeof(float));
    cudaMemcpyToSymbol(c_RTN_values,GK_RTN_values,64*sizeof(float));
    cudaMemcpyToSymbol(c_RTR_values,GK_RTR_values,256*sizeof(float));
    cudaMemcpyToSymbol(c_Delta_values,GK_Delta_values,3*16*sizeof(float));

    checkCudaError();

    // create groups of process to use mpi reduce only on spatial points
    MPI_Comm_group(MPI_COMM_WORLD, &GK_fullGroup);
    int space3D_proc;
    space3D_proc = GK_nProc[0] * GK_nProc[1] * GK_nProc[2];
    int *ranks = (int*) malloc(space3D_proc*sizeof(int));

    for(int i= 0 ; i < space3D_proc ; i++)
      ranks[i] = comm_coords(default_topo)[3] + GK_nProc[3]*i;

    //    for(int i= 0 ; i < space3D_proc ; i++)
    //      printf("%d (%d,%d,%d,%d)\n",comm_rank(),comm_coords(default_topo)[0],comm_coords(default_topo)[1],comm_coords(default_topo)[2],comm_coords(default_topo)[3]);

    //  for(int i= 0 ; i < space3D_proc ; i++)
    //printf("%d %d\n",comm_rank(),ranks[i]);

    MPI_Group_incl(GK_fullGroup,space3D_proc,ranks,&GK_spaceGroup);
    MPI_Group_rank(GK_spaceGroup,&GK_localRank);
    MPI_Group_size(GK_spaceGroup,&GK_localSize);
    MPI_Comm_create(MPI_COMM_WORLD, GK_spaceGroup , &GK_spaceComm);

    //if(GK_spaceComm == MPI_COMM_NULL) printf("NULL %d\n",comm_rank());
    //exit(-1);
    // create group of process to use mpi gather
    int *ranksTime = (int*) malloc(GK_nProc[3]*sizeof(int));

    for(int i=0 ; i < GK_nProc[3] ; i++)
      ranksTime[i] = i;
    
    MPI_Group_incl(GK_fullGroup,GK_nProc[3], ranksTime, &GK_timeGroup);
    MPI_Group_rank(GK_timeGroup, &GK_timeRank);
    MPI_Group_size(GK_timeGroup, &GK_timeSize);
    MPI_Comm_create(MPI_COMM_WORLD, GK_timeGroup, &GK_timeComm);

    //////////////////////////////////////////////////////////////////////////////
    free(ranks);
    free(ranksTime);

    GK_init_PLEGMA_flag = true;
    printfQuda("PLEGMA has been initialized\n");
  }  

  else{
    printfQuda("???\n");
    return;
  }
  
}

void plegma::print_status(){

  if(GK_init_PLEGMA_flag == false) errorQuda("You must initialize init_PLEGMA first");
  printfQuda("Number of colors is %d\n",N_COLS);
  printfQuda("Number of spins is %d\n",N_SPINS);
  printfQuda("Number of dimensions is %d\n",N_DIMS);
  printfQuda("Number of process in each direction is (x,y,z,t) %d x %d x %d x %d\n",GK_nProc[0],GK_nProc[1],GK_nProc[2],GK_nProc[3]);
  printfQuda("Total lattice is (x,y,z,t) %d x %d x %d x %d\n",GK_totalL[0],GK_totalL[1],GK_totalL[2],GK_totalL[3]);
  printfQuda("Local lattice is (x,y,z,t) %d x %d x %d x %d\n",GK_localL[0],GK_localL[1],GK_localL[2],GK_localL[3]);
  printfQuda("Total volume is %d\n",GK_totalVolume);
  printfQuda("Local volume is %d\n",GK_localVolume);
  printfQuda("Surface is (x,y,z,t) ( %d , %d , %d , %d)\n",GK_surface3D[0],GK_surface3D[1],GK_surface3D[2],GK_surface3D[3]);
  printfQuda("The plus Ghost points in directions (x,y,z,t) ( %d , %d , %d , %d )\n",GK_plusGhost[0],GK_plusGhost[1],GK_plusGhost[2],GK_plusGhost[3]);
  printfQuda("The Minus Ghost points in directixons (x,y,z,t) ( %d , %d , %d , %d )\n",GK_minusGhost[0],GK_minusGhost[1],GK_minusGhost[2],GK_minusGhost[3]);
  printfQuda("For APE smearing we use nsmear = %d , alpha = %lf\n",GK_nsmearAPE,GK_alphaAPE);
  printfQuda("For Gauss smearing we use nsmear = %d , alpha = %lf\n",GK_nsmearGauss,GK_alphaGauss);
  printfQuda("I got %d source positions to work on\n",GK_Nsources);
  printfQuda("I got %d number of momenta to work on\n",GK_Nmoms);
}
