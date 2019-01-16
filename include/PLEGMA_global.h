#pragma once
#include <mpi.h>  
#include <stdlib.h>
#include <stdio.h>
#include <typeinfo>

// Constants values
#define PI 3.141592653589793

#define N_DIMS     4
#define N_COLS     3
#define N_SPINS    4

#define gaugeSiteSize  2*N_COLS*N_COLS // real numbers per link
#define spinorSiteSize 2*N_COLS*N_SPINS // real numbers per spinor
#define cloverSiteSize 2*N_COLS*N_COLS*N_DIMS // real numbers per block-diagonal clover matrix
#define hwSiteSize     N_COLS*N_SPINS // real numbers per half wilson

// Macros
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

namespace plegma {

  // Custom types and functions
  template<typename Float> struct texture;

  template<typename Float> inline MPI_Datatype MPI_Type(Float a);
  template<> inline MPI_Datatype MPI_Type<float>(float a) { return MPI_FLOAT; }
  template<> inline MPI_Datatype MPI_Type<float*>(float* a) { return MPI_FLOAT; }
  template<> inline MPI_Datatype MPI_Type<double>(double a) { return MPI_DOUBLE; }
  template<> inline MPI_Datatype MPI_Type<double*>(double* a) { return MPI_DOUBLE; }

  // Enumerations
  enum COMPLEX{REAL,IMAG};
  enum SOURCE_T{UNITY,RANDOM};
  enum CORR_SPACE{POSITION_SPACE,MOMENTUM_SPACE};
  enum FILE_WRITE_FORMAT{ASCII_FORM,HDF5_FORM};
  
  enum ALLOCATION_FLAG{HOST,DEVICE,BOTH};

  enum CLASS_ENUM{FIELD,SU3FIELD,GAUGE,VECTOR,PROPAGATOR,PROPAGATOR3D,VECTOR3D,QLOOPS};
  enum GHOST_FLAG{NO_GHOSTS,FIRST_SIDE,FIRST_CORNER};
  enum WHICHPARTICLE{PROTON,NEUTRON};
  enum WHICHPROJECTOR{G4,G5G123,G5G1,G5G2,G5G3};

  enum THRP_TYPE{THRP_LOCAL2,THRP_NOETHER2,THRP_ONED2};

  enum LATDIMS{DIM_X,DIM_Y,DIM_Z,DIM_T};

  enum ACCUM_TYPE{ACC_ZERO, ACC_PLUS, ACC_MINUS};

  // Preparation for global variables
  // here we collect the global variables for then running some default functions on them (print and copy to device)
  struct global_vars {
    // globals on host only
    std::vector<void*> host_only_pointer;
    std::vector<size_t> host_only_size;
    std::vector<std::type_info> host_only_type;
    std::vector<char*> host_only_name;
    
    // globals on both, host and device
    std::vector<std::array<void*, 2>> both_pointer;
    std::vector<size_t> both_size;
    std::vector<std::type_info> both_type;
    std::vector<char*> both_name;

    template<typename hostT, typename deviceT>
    void add(char* name, hostT &host, deviceT &device=NULL) {
      if(device == NULL) {
	host_only_pointer.push_back((void*) &host);
	host_only_size.push_back(sizeof(host));
	host_only_type.push_back(typeof(host));	
	host_only_name.push_back(name);	
      } else {
	both_pointer.push_back({(void*) &host, (void*) &device});
	both_size.push_back(sizeof(host));
	both_type.push_back(typeof(host));	
	both_name.push_back(name);
      }
    };
    void copyToDevice() {
      for(int i = 0; i != both_pointer.size(); i++) {
	cudaMemcpyToSymbol( both_pointer[i][1], both_pointer[i][0], both_size[i]);
      }
      checkCudaError();
    };
    void print() {

    };
  };
  
//These generates global constants either on host (global_host) only or also on device (global_both). It will append respectively HGC_ and DGC_.
#ifndef ALLOCATE
  extern global_vars globals;
#define global_host(dtype, name, ...) extern dtype "HGC_"##name __VA_ARGS__;
#define global_both(dtype, name, ...) extern dtype "HGC_"##name __VA_ARGS__;\
                         extern __constant__ dtype "DGC_"##name __VA_ARGS__; 
#else
  global_vars globals;
#define global_host(dtype, name, ...) dtype "HGC_"##name __VA_ARGS__;\
                         globals.add(#name, "HGC_"##name);
#define global_both(dtype, name, ...) dtype "HGC_"##name __VA_ARGS__;\
                         __constant__ dtype "DGC_"##name __VA_ARGS__;\
			 globals.add(#name, "HGC_"##name, "DGC_"##name);
#endif
  
  // Global variables
  global_host(bool, init_PLEGMA_flag);
  global_host(float, deviceMemory);

  // variables visible on both host and device
  global_both(size_t, stride_full);
  global_both(size_t, stride_spatial);
  global_both(size_t, localVolume);
  global_both(size_t, totalVolume);
  global_both(int, localL, [N_DIMS]);
  global_both(int, totalL, [N_DIMS]);
  global_both(int, procPosition, [N_DIMS]);
  global_both(size_t, sideGhost, [2*N_DIMS]);
  global_both(size_t, cornerGhost, [2*N_DIMS][2*N_DIMS]);
  global_both(size_t, surface3D, [N_DIMS]);
  global_both(size_t, surface2D, [N_DIMS][N_DIMS]);

  // for mpi use global variables (host only)
  global_host(Topology *, default_topo);
  global_host(bool, dimBreak, [N_DIMS]);
  global_host(int, nProc, [N_DIMS]);
  global_host(MPI_Group, fullGroup);
  global_host(MPI_Group, spaceGroup);
  global_host(MPI_Group, timeGroup);
  global_host(MPI_Comm, spaceComm);
  global_host(MPI_Comm, timeComm);
  global_host(int, localRank);
  global_host(int, localSize);
  global_host(int, timeRank);
  global_host(int, timeSize);

  // for cublas use
  global_host(cublasHandle_t, cublas_handle);
}

#undef global_both
#undef global_host
