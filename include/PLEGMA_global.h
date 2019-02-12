#pragma once
#include <mpi.h>
#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include <typeinfo>
#include <typeindex>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <array>
#include <algorithm>
#include <quda.h>
#include <quda_internal.h>
#include <cublas_v2.h>
#ifdef __GNUG__ // gnu C++ compiler
#include <cxxabi.h>
#include <stdlib.h>
#endif
#include <malloc.h>
#include <new> // for availability of std::bad_alloc exception

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

  // Enumerations
  enum COMPLEX{REAL,IMAG};
  enum SOURCE_T{UNITY,RANDOM};
  enum CORR_SPACE{POSITION_SPACE,MOMENTUM_SPACE};
  enum FILE_WRITE_FORMAT{ASCII_FORM,HDF5_FORM};
  
  enum ALLOCATION_FLAG{HOST,DEVICE,BOTH};

  enum CLASS_ENUM{CUSTOM,SCALAR,SU3FIELD,GAUGE,VECTOR,PROPAGATOR,PROPAGATOR3D,VECTOR3D,QLOOPS};
  enum GHOST_FLAG{NO_GHOSTS,FIRST_SIDE,FIRST_CORNER};
  enum WHICHPARTICLE{PROTON,NEUTRON};
  enum WHICHPROJECTOR{P4_P,P4G5G1_P,P4G5G2_P,P4G5G3_P,P4_M,P4G5G1_M,P4G5G2_M,P4G5G3_M}; // Do not change this order

  enum THRP_TYPE{THRP_LOCAL2,THRP_NOETHER2,THRP_ONED2};

  enum LATDIMS{DIM_X,DIM_Y,DIM_Z,DIM_T};

  enum GAMMAS {ONE,G1,G2,G3,G4,G5,G5G1,G5G2,G5G3,G5G4,S12,S13,S23,S41,S42,S43}; // Do not change this order
  const std::string GAMMAS_STR[16] {"1","g1","g2","g3","g4","g5","g5g1","g5g2","g5g3","g5g4",
      "s12","s13","s23","s41","s42","s43"};
  static inline std::string getGammasString(std::vector<GAMMAS> gammas) {
    std::string s = "";
    std::for_each(gammas.begin(), gammas.end(), [&] (GAMMAS n) {s += GAMMAS_STR[(int) n]+",";});
    return s;
  }
  
  enum ACCUM_TYPE{ACC_ZERO, ACC_PLUS, ACC_MINUS};
  enum LEFTRIGHT {LEFT, RIGHT};

  // Custom types and functions
  template<typename Float> struct texture;

  template<typename Float> inline MPI_Datatype MPI_Type();
  template<typename Float> inline MPI_Datatype MPI_Type(Float a){ return MPI_Type<Float>(); }
  template<> inline MPI_Datatype MPI_Type<float>() { return MPI_FLOAT; }
  template<> inline MPI_Datatype MPI_Type<float*>() { return MPI_FLOAT; }
  template<> inline MPI_Datatype MPI_Type<double>() { return MPI_DOUBLE; }
  template<> inline MPI_Datatype MPI_Type<double*>() { return MPI_DOUBLE; }

  template<typename T> inline char type_char(){ return 'p';};
  template<typename T> inline char type_char(T a){ return type_char<T>();};
  template<> inline char type_char<char>() { return 'c'; }
  template<> inline char type_char<int>() { return 'd'; }
  template<> inline char type_char<float>() { return 'e'; }
  template<> inline char type_char<double>() { return 'e'; }
  template<> inline char type_char<char*>() { return 's'; }
  template<> inline char type_char<const char*>() { return 's'; }
  template<> inline char type_char<void*>() { return 'p'; }

  static inline std::string demangle( const char* mangled_name ) {
#ifdef __GNUG__ // gnu C++ compiler
    std::string result ;
    std::size_t len = 0 ;
    int status = 0 ;
    char* ptr = __cxxabiv1::__cxa_demangle( mangled_name, nullptr, &len, &status ) ;

    if( status == 0 ) result = ptr ; // hope that this won't throw
    else result = "demangle error" ;
    ::free(ptr) ;
    if(result.find("basic_string") != std::string::npos) return "std::string"; 
    return result ;
#else
    return std:string(mangled_name);
#endif
  }

  // Global variable for mom list
  struct tex_mom_list {
    size_t Nmoms;
    cudaTextureObject_t tex;
    inline __device__ int4 get(size_t i){
#ifdef __NVCC__
      return tex1Dfetch<int4>(tex,i);
#else
      return make_int4(0,0,0,0);
#endif
    };
    void free(){
      cudaResourceDesc desc;
      cudaGetTextureObjectResourceDesc(&desc, tex);
      cudaFree(desc.res.linear.devPtr);
      cudaDestroyTextureObject(tex);
      checkCudaError();
      Nmoms=0;
    };
  };

  // here we collect the global variables for then running some default functions on them (print and copy to device)
  struct global_vars {
    // globals on host only
    std::vector<void*> host_only_pointer;
    std::vector<int> host_only_size;
    std::vector<size_t> host_only_bytes;
    std::vector<char> host_only_type;
    std::vector<std::string> host_only_type_name;
    std::vector<std::string> host_only_name;
    
    // globals on both, host and device
    std::vector<std::array<void*, 2>> both_pointer;
    std::vector<int> both_size;
    std::vector<size_t> both_bytes;
    std::vector<char> both_type;
    std::vector<std::string> both_type_name;
    std::vector<std::string> both_name;

    template<typename hostT>
    void add(const char* name, hostT &host, int size=1) {
      host_only_pointer.push_back((void*) &host);
      host_only_size.push_back(size);
      host_only_bytes.push_back(sizeof(hostT)*size);
      host_only_type.push_back(type_char<hostT>());	
      host_only_type_name.push_back(demangle(typeid(host).name()));	
      host_only_name.push_back(name);	
    }
    template<typename hostT, typename deviceT>
    void add(const char* name, hostT &host, deviceT &device, int size=1) {
      both_pointer.push_back({(void*) &host, (void*) &device});
      both_size.push_back(size);
      both_bytes.push_back(sizeof(hostT)*size);
      both_type.push_back(type_char<hostT>());	
      both_type_name.push_back(demangle(typeid(host).name()));	
      both_name.push_back(name);
    }
    void copyToDevice() {
      for(int i = 0; i != both_pointer.size(); i++) {
	cudaMemcpyToSymbol( *((char**) both_pointer[i][1]), both_pointer[i][0], both_bytes[i]);
      }
      checkCudaError();
    }
    void print() {

    }
  };

  #ifdef ALLOCATE
  global_vars HGC_globals_vars;
  #else
  extern global_vars HGC_globals_vars;
  #define EXTERNAL
  #endif
  #include <PLEGMA_global_vars.h>
  #undef EXTERNAL

  template<typename T> inline void buffer_malloc(T &ptr, size_t size) {
#ifdef PLEGMA_HAVE_MEMALIGN
    ptr = static_cast<T>(memalign(PLEGMA_ALIGNMENT, size));
#else
    ptr = static_cast<T>(malloc(size));
#endif
    if(ptr == static_cast<T>(NULL) ){
      warningQuda("Bad alloc. Total memory in use: %lu\n", HGC_used_memory);
      throw( std::bad_alloc() );
    }
    HGC_used_memory += sizeof(T)*size;
  }
  template<typename T> inline void buffer_free(T &ptr, size_t size) {
    free(ptr);
    ptr=NULL;
    HGC_used_memory -= sizeof(T)*size;
  }
}
using namespace plegma; // TODO: This one shouldn't be here.. But helps avoiding missing namespace.
