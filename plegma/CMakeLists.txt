set(PLEGMA_LIBS PLEGMA PLEGMAutils ${CMAKE_THREAD_LIBS_INIT} ${QUDA_LIB} ${CUDA_LIBS} cuda)

LIST(APPEND PLEGMA_LIBS ${MPI_CXX_LIBRARIES})

if(PLEGMA_ARPACK)
  LIST(APPEND PLEGMA_LIBS ${PLEGMA_MPIFH_LIB})
  LIST(APPEND PLEGMA_LIBS ${PLEGMA_Z_LIB})
  LIST(APPEND PLEGMA_LIBS ${PLEGMA_GFORTRAN_LIB})
  LIST(APPEND PLEGMA_LIBS ${PLEGMA_PARPACK_LIB})
  LIST(APPEND PLEGMA_LIBS ${PLEGMA_ARPACK_LIB})
endif()

LIST(APPEND PLEGMA_LIBS ${PLEGMA_GSL_LIB})
LIST(APPEND PLEGMA_LIBS ${PLEGMA_HDF5_LIB})
LIST(APPEND PLEGMA_LIBS ${PLEGMA_LIME_LIB})

if(PLEGMA_OPENBLAS)
  LIST(APPEND PLEGMA_LIBS ${PLEGMA_OPENBLAS_GOMP})
  LIST(APPEND PLEGMA_LIBS ${PLEGMA_OPENBLAS_LIB})
endif(PLEGMA_OPENBLAS)

if(PLEGMA_MKL)
  LIST(APPEND PLEGMA_LIBS ${PLEGMA_MKL_LIB_GFORTRAN})
  LIST(APPEND PLEGMA_LIBS ${PLEGMA_MKL_LIB_IOMP5})
  LIST(APPEND PLEGMA_LIBS ${PLEGMA_MKL_LIB_LP64})
  LIST(APPEND PLEGMA_LIBS ${PLEGMA_MKL_LIB_THREAD})
  LIST(APPEND PLEGMA_LIBS ${PLEGMA_MKL_LIB_CORE})
endif(PLEGMA_MKL)

#define PLEGMA executables

cuda_add_executable(Plaquette Plaquette.cpp)
target_link_libraries(Plaquette ${PLEGMA_LIBS})

if(PLEGMA_2FLAVMG)
  #CalcMG_2pt3pt_EvenOdd
  # This function will calculate hadronic correlation functions using a 
  # handful of randomly separated, gaussian smeared point sources, 
  #constructed with a smeared gauge field. Even/Odd preconditioning is 
  # employed, as well as QUDA's MG solver routines.
  #cuda_add_executable(CalcMG_2pt3pt_EvenOdd CalcMG_2pt3pt_EvenOdd.cpp)
  #target_link_libraries(CalcMG_2pt3pt_EvenOdd ${PLEGMA_LIBS})
endif()  
