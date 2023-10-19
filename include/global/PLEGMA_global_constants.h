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
#ifdef __HIP__
#include<hipblas.h>
#else
#include<cublas_v2.h>
#endif

/*
#define global_host(dtype, name, ...)					\
  HGC_global_vars.add<dtype>(#name,					\
				   PRODUCT(__VA_ARGS__),		\
				   &HGC_##name PARENTHESES(0,__VA_ARGS__))

#if defined (__HIP__)
#define global_both(dtype, name, ...)					\
  HGC_global_vars.add<dtype>(#name,					\
				   PRODUCT(__VA_ARGS__),		\
				   &HGC_##name PARENTHESES(0,__VA_ARGS__), \
			           (void**) &DGC_##name)
#else
#define global_both(dtype, name, ...)                                   \
  HGC_global_vars.add<dtype>(#name,                                     \
                                   PRODUCT(__VA_ARGS__),                \
                                   &HGC_##name PARENTHESES(0,__VA_ARGS__), \
                                   (void**) &DGC_##name)

#endif
#else
*/
#ifdef ALLOCATE

#define global_host(dtype, name, ...)					\
  dtype HGC_##name PARENTHESES(1,__VA_ARGS__);
#ifdef __NVCC__
#define global_both(dtype, name, ...)					\
  dtype HGC_##name PARENTHESES(1,__VA_ARGS__);				\
  __constant__ dtype DGC_##name PARENTHESES(1,__VA_ARGS__);		
#elif defined (__HIP__)                                                 \
#define global_both(dtype, name, ...)                                   \
  dtype HGC_##name PARENTHESES(1,__VA_ARGS__);                          \
  __device__ __constant__ dtype DGC_##name PARENTHESES(1,__VA_ARGS__);  \
#else                                                                     
#define global_both(dtype, name, ...)					\
  dtype HGC_##name PARENTHESES(1,__VA_ARGS__);				
#endif

#else

#define global_host(dtype, name, ...)					\
  extern dtype HGC_##name PARENTHESES(1,__VA_ARGS__);
#ifdef __NVCC__
#define global_both(dtype, name, ...)					\
  extern dtype HGC PARENTHESES(1,__VA_ARGS__);			\
  extern __constant__ dtype DGC_##name PARENTHESES(1,__VA_ARGS__); 
#elif defined ( __HIP__ )
#define global_both(dtype, name, ...)                                   \
  extern dtype HGC_##name PARENTHESES(1,__VA_ARGS__);                   \
  extern __device__ __constant__ dtype DGC_##name PARENTHESES(1,__VA_ARGS__);
#else
#define global_both(dtype, name, ...)			\
  extern dtype HGC_##name PARENTHESES(1,__VA_ARGS__);
#endif
#endif

// ---------------- ADD_FROM_HERE ------------- //


#pragma message " HGC "
#ifdef ALLOCATE
struct global_vars_host HGC;
struct global_vars_both *DGC_ptr;
#if defined (__NVCC__) | (__HIP__)
__constant__ struct global_vars_both DGC_const;
#endif
#else
//static __device__ __constant__ volatile struct global_vars_both DGCS;

//static __device__ volatile struct global_vars_both *DGC;
#if defined (__NVCC__) | (__HIP__)
static __device__ struct global_vars_both *DGC;
#endif
extern struct global_vars_host HGC;
extern struct global_vars_both *DGC_ptr;
#endif

// variables visible on both host and device

// for mpi use global variables (host only)

#undef global_both
#undef global_host
#undef PAR_VALUE
