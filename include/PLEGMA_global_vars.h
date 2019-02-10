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

//--------------- Here some macros -----------------//

#define BOOST_PP_VARIADICS 1
#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/control/if.hpp>
#include <boost/preprocessor/tuple/to_seq.hpp>
#include <boost/preprocessor/arithmetic/mul.hpp>
#include <boost/preprocessor/facilities/overload.hpp>
#include <boost/preprocessor/facilities/empty.hpp>
#include <utils/MACRO_IS_EMPTY.hpp>

#define ADD_PARENTHESES(r, data, elem) BOOST_PP_IF( data,  [elem], [0])
#define FOR_EACH(...) BOOST_PP_SEQ_FOR_EACH(ADD_PARENTHESES, PAR_VALUE, BOOST_PP_TUPLE_TO_SEQ((__VA_ARGS__)))
#define NOTHING(...)
#define PARENTHESES(...) BOOST_PP_IF(IS_EMPTY(__VA_ARGS__),	\
				     NOTHING,			\
				     FOR_EACH) (__VA_ARGS__)

#define MUL_1(a) (a)
#define MUL_2(a,b) (a)*(b)
#define MUL_3(a,b,c) MUL_2(MUL_2(a,b),c)
#define MUL_4(a,b,c,d) MUL_3(MUL_2(a,b),c,d)
// define more if needed
#define ONE(...) 1
#define MUL(...) BOOST_PP_OVERLOAD(MUL_,__VA_ARGS__)(__VA_ARGS__)
#define PRODUCT(...) BOOST_PP_IF(IS_EMPTY(__VA_ARGS__),			\
				 ONE,					\
				 MUL) (__VA_ARGS__)


#ifdef ADD_TO_GLOBAL

#define PAR_VALUE 0
#define global_host(dtype, name, ...)					\
  HGC_globals_vars.add<dtype>(#name,					\
			      HGC_##name PARENTHESES(__VA_ARGS__),	\
			      PRODUCT(__VA_ARGS__))
    
#define global_both(dtype, name, ...)					\
  HGC_globals_vars.add<dtype,dtype>(#name,				\
				    HGC_##name PARENTHESES(__VA_ARGS__), \
				    DGC_##name PARENTHESES(__VA_ARGS__), \
				    PRODUCT(__VA_ARGS__))

#else
#ifdef EXTERNAL

#define PAR_VALUE 1
#define global_host(dtype, name, ...)					\
  extern dtype HGC_##name PARENTHESES(__VA_ARGS__);			\
  extern dtype GK_##name PARENTHESES(__VA_ARGS__) __attribute__((deprecated)); // This line should be removed
#define global_both(dtype, name, ...)					\
  extern dtype HGC_##name PARENTHESES(__VA_ARGS__);			\
  extern dtype GK_##name PARENTHESES(__VA_ARGS__) __attribute__((deprecated)); \
  extern __constant__ dtype DGC_##name PARENTHESES(__VA_ARGS__);	\
  extern __constant__ dtype c_##name PARENTHESES(__VA_ARGS__) __attribute__((deprecated)); // This line should be removed

#else
#ifdef ALLOCATE

#define PAR_VALUE 1
#define global_host(dtype, name, ...)					\
  dtype HGC_##name PARENTHESES(__VA_ARGS__);				
#define global_both(dtype, name, ...)					\
  dtype HGC_##name PARENTHESES(__VA_ARGS__);				\
  __constant__ dtype DGC_##name PARENTHESES(__VA_ARGS__);		

#else
#error "PLEGMA_global_vars.h should be included with either ALLOCATE, EXTERNAL, or ADD_TO_GLOBAL"
#endif
#endif
#endif

// ---------------- ADD_FROM_HERE ------------- //

// Global variables
global_host(bool, init_PLEGMA_flag);
global_host(float, deviceMemory);

// variables visible on both host and device
global_both(tex_mom_list, moms);
global_both(size_t, stride);
global_both(size_t, stride_spatial);
global_both(size_t, localVolume);
global_both(size_t, totalVolume);
global_both(int, localL, N_DIMS);
global_both(int, totalL, N_DIMS);
global_both(int, procPosition, N_DIMS);
global_both(size_t, sideGhost, 2*N_DIMS);
global_both(size_t, cornerGhost, 2*N_DIMS, 2*N_DIMS);
global_both(size_t, surface3D, N_DIMS);
global_both(size_t, surface2D, N_DIMS, N_DIMS);

// for mpi use global variables (host only)
global_both(bool, dimBreak, N_DIMS);
global_host(Topology *, default_topo);
global_host(int, nProc, N_DIMS);
global_host(MPI_Group, fullGroup);
global_host(MPI_Group, spaceGroup);
global_host(MPI_Group, timeGroup);
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
