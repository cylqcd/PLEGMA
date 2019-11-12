/* 
 * In this header file we add dynamically the list of global variables.
 * 
 * This header can be included in three ways, with respective flags:
 *   - EXTERNAL: variables are declared as externals
 *   - ALLOCATE: variables are allocated (used once in PLEGMA.cu)
 *   - ADD_TO_GLOBAL: variables are added to the global_vars structure (used once in PLEGMA.cu)
 *
 * Variables can be added after (find ADD_FROM_HERE) using:
 *   - global_host(...): a global variable defined only on the host and with prefix HGC_
 *   - global_both(...): a global variable defined on the host with prefix HGC_ and on the device with prefix DGC_
 *
 * The arguments (...) are (dtype, name, s1, s2, ..) where s1, s2, .. is an optional variadic list of sizes.
 * The defined global variable then will be:
 *   - on host:                "dtype" HGC_"name" "[s1][s2][..]"
 *   - on device: __constant__ "dtype" DGC_"name" "[s1][s2][..]"
 */

#ifdef ADD_TO_GLOBAL

#define global_host(dtype, name, ...)					\
  HGC_global_vars.add<dtype,dtype>(#name,				\
				   PRODUCT(__VA_ARGS__),		\
				   &HGC_##name PARENTHESES(0,__VA_ARGS__))

#define global_both(dtype, name, ...)					\
  HGC_global_vars.add<dtype,dtype>(#name,				\
				   PRODUCT(__VA_ARGS__),		\
				   &HGC_##name PARENTHESES(0,__VA_ARGS__), \
				   &DGC_##name PARENTHESES(0,__VA_ARGS__))

#else
#ifdef ALLOCATE

#define global_host(dtype, name, ...)					\
  dtype HGC_##name PARENTHESES(1,__VA_ARGS__);
#ifdef __NVCC__
#define global_both(dtype, name, ...)					\
  dtype HGC_##name PARENTHESES(1,__VA_ARGS__);				\
  __constant__ dtype DGC_##name PARENTHESES(1,__VA_ARGS__);		
#else
#define global_both(dtype, name, ...)					\
  dtype HGC_##name PARENTHESES(1,__VA_ARGS__);				
#endif

#else

#define global_host(dtype, name, ...)					\
  extern dtype HGC_##name PARENTHESES(1,__VA_ARGS__);
#ifdef __NVCC__
#define global_both(dtype, name, ...)					\
  extern dtype HGC_##name PARENTHESES(1,__VA_ARGS__);			\
  extern __constant__ dtype DGC_##name PARENTHESES(1,__VA_ARGS__); 
#else
#define global_both(dtype, name, ...)			\
  extern dtype HGC_##name PARENTHESES(1,__VA_ARGS__);
#endif
#endif
#endif

// ---------------- ADD_FROM_HERE ------------- //

// Global variables
global_host(bool, init_PLEGMA_flag);
global_host(float, deviceMemory);
global_host(int, verbosity);
global_host(long int, used_memory);
global_host(global_vars, global_vars);
global_host(Options *, options);
global_host(bool, hold_exit);

// variables visible on both host and device
global_both(size_t, localVolume);
global_both(size_t, localVolume3D);
global_both(size_t, totalVolume);
global_both(int, localL, N_DIMS);
global_both(int, totalL, N_DIMS);
global_both(int, procPosition, N_DIMS);
global_both(size_t, sideGhost, N_DIMS, DIR_BOTH);
global_both(size_t, cornerGhost, N_DIMS, N_DIMS, DIR_BOTH, DIR_BOTH);
global_both(size_t, surface3D, N_DIMS);
global_both(size_t, surface2D, N_DIMS, N_DIMS);

// for mpi use global variables (host only)
global_both(bool, dimBreak, N_DIMS);
global_host(Topology *, default_topo);
global_host(int, nProc, N_DIMS);
global_host(MPI_Group, fullGroup);
global_host(MPI_Group, spaceGroup);
global_host(MPI_Group, timeGroup);
global_host(MPI_Comm, fullComm);
global_host(MPI_Comm, spaceComm);
global_host(MPI_Comm, timeComm);
global_host(int, fullRank);
global_host(int, fullSize);
global_host(int, spaceRank);
global_host(int, spaceSize);
global_host(int, timeRank);
global_host(int, timeSize);

// for cublas use
global_host(cublasHandle_t, cublas_handle);

#undef global_both
#undef global_host
#undef PAR_VALUE
