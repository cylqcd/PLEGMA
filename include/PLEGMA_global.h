#include <mpi.h>
#include <cuda.h>
#include <typeinfo>
#include <cstdlib>
#include <cstdio>
#include <quda_internal.h>
#include <quda.h>
#include <cublas_v2.h>
#ifndef _PLEGMA_GLOBAL_H
#define _PLEGMA_GLOBAL_H

#define PI 3.141592653589793

#define N_MESONS  10
#define N_DIMS     4
#define N_COLS     3
#define N_SPINS    4

#define gaugeSiteSize  2*N_COLS*N_COLS // real numbers per link
#define spinorSiteSize 2*N_COLS*N_SPINS // real numbers per spinor
#define cloverSiteSize 2*N_COLS*N_COLS*N_DIMS // real numbers per block-diagonal clover matrix
#define momSiteSize    10 // real numbers per momentum
#define hwSiteSize     N_COLS*N_SPINS // real numbers per half wilson

#define MAX_NSOURCES 1000
#define MAX_NMOMENTA 5000
#define MAX_TSINK 10
#define MAX_DEFLSTEPS 10
#define MAX_LP_CRIT 10
#define MAX_PROJS 5

#define LEXIC(it,iz,iy,ix,L) ( (it)*L[0]*L[1]*L[2] + (iz)*L[0]*L[1] + (iy)*L[0] + (ix) )
#define LEXIC_TZY(it,iz,iy,L) ( (it)*L[1]*L[2] + (iz)*L[1] + (iy) )
#define LEXIC_TZX(it,iz,ix,L) ( (it)*L[0]*L[2] + (iz)*L[0] + (ix) )
#define LEXIC_TYX(it,iy,ix,L) ( (it)*L[0]*L[1] + (iy)*L[0] + (ix) )
#define LEXIC_ZYX(iz,iy,ix,L) ( (iz)*L[0]*L[1] + (iy)*L[0] + (ix) )
#define LEXIC_TZ(it,iz,L) ( (it)*L[2] + (iz) )
#define	LEXIC_TY(it,iy,L) ( (it)*L[1] + (iy) )
#define	LEXIC_TX(it,ix,L) ( (it)*L[0] + (ix) )
#define	LEXIC_ZY(iz,iy,L) ( (iz)*L[1] + (iy) )
#define	LEXIC_ZX(iz,ix,L) ( (iz)*L[0] + (ix) )
#define	LEXIC_YX(iy,ix,L) ( (iy)*L[0] + (ix) )

template<typename Float> inline MPI_Datatype MPI_Type(Float a);
template<> inline MPI_Datatype MPI_Type<float>(float a) { return MPI_FLOAT; }
template<> inline MPI_Datatype MPI_Type<float*>(float* a) { return MPI_FLOAT; }
template<> inline MPI_Datatype MPI_Type<double>(double a) { return MPI_DOUBLE; }
template<> inline MPI_Datatype MPI_Type<double*>(double* a) { return MPI_DOUBLE; }

extern Topology *default_topo;

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
extern int GK_sideGhost[2*N_DIMS];
extern int GK_cornerGhost[2*N_DIMS][2*N_DIMS];
extern int GK_surface3D[N_DIMS];
extern int GK_surface2D[N_DIMS][N_DIMS];
extern bool GK_init_PLEGMA_flag;
extern int GK_Nsources;
extern int GK_sourcePosition[MAX_NSOURCES][N_DIMS];
extern int GK_Nmoms;
extern short int GK_moms[MAX_NMOMENTA][3];
// for mpi use global variables
extern MPI_Group GK_fullGroup , GK_spaceGroup , GK_timeGroup;
extern MPI_Comm GK_spaceComm , GK_timeComm;
extern int GK_localRank;
extern int GK_localSize;
extern int GK_timeRank;
extern int GK_timeSize;
// for cublas use
extern cublasHandle_t cublas_handle;
namespace plegma {
  template<typename Float> struct texture;

  // enum
  enum COMPLEX{REAL,IMAG};
  enum SOURCE_T{UNITY,RANDOM};
  enum CORR_SPACE{POSITION_SPACE,MOMENTUM_SPACE};
  enum FILE_WRITE_FORMAT{ASCII_FORM,HDF5_FORM};
  enum WHICHSPECTRUM{SR,LR,SM,LM,SI,LI};
  
  enum ALLOCATION_FLAG{NONE,HOST,DEVICE,BOTH};

  enum CLASS_ENUM{FIELD,SU3FIELD,GAUGE,VECTOR,PROPAGATOR,PROPAGATOR3D,VECTOR3D,QLOOPS};
  enum GHOST_FLAG{NO_GHOSTS,FIRST_SIDE,FIRST_CORNER};
  enum WHICHPARTICLE{PROTON,NEUTRON};
  enum WHICHPROJECTOR{P4_P,P4G5G1_P,P4G5G2_P,P4G5G3_P,P4_M,P4G5G1_M,P4G5G2_M,P4G5G3_M}; // Do not change this order

  enum THRP_TYPE{THRP_LOCAL2,THRP_NOETHER2,THRP_ONED2};

  enum LATDIMS{DIM_X,DIM_Y,DIM_Z,DIM_T};

  enum APEDIM{D3,D4};
  enum GAMMAS {ONE,G1,G2,G3,G4,G5,G5G1,G5G2,G5G3,G5G4,S12,S13,S23,S41,S42,S43}; // Do not change this order
  enum ACCUM_TYPE{ACC_ZERO, ACC_PLUS, ACC_MINUS};
  typedef struct {
    int nsmearAPE;
    int nsmearGauss;
    double alphaAPE;
    double alphaGauss;
    int lL[N_DIMS];
    int procs[N_DIMS];
    int Nsources;
    int sourcePosition[MAX_NSOURCES][N_DIMS];
    QudaPrecision Precision;
    int Q_sq;
    int Ntsink;
    int Nproj[MAX_TSINK];
    int traj;
    bool check_files;
    char *corr_dir;
    char *thrp_type[3];
    char *thrp_proj_type[5];
    char *baryon_type[10];
    char *meson_type[10];
    int tsinkSource[MAX_TSINK];
    int proj_list[MAX_TSINK][MAX_PROJS];
    int run3pt_src[MAX_NSOURCES];
    FILE_WRITE_FORMAT CorrFileFormat;
    SOURCE_T source_type;
    CORR_SPACE CorrSpace;
    bool HighMomForm;
    bool isEven;
    double kappa;
    double mu;
    double csw;
    double inv_tol;
  } PLEGMA_params;
}

#endif
