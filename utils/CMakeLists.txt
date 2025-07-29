#build a common library for all test utilities
file(GLOB PLEGMA_UTILS
     "*.cpp"
)

#cuda_add_library(PLEGMAutils STATIC ${PLEGMA_UTILS})
add_library(PLEGMAutils STATIC ${PLEGMA_UTILS})
if(${PLEGMA_TARGET_TYPE} STREQUAL "HIP")
target_link_libraries(PLEGMAutils PRIVATE ${QUDA_HOME}/lib/libquda.so -fPIC)
target_include_directories(PLEGMAutils SYSTEM PRIVATE ${QUDA_SRC}/lib ${QUDA_HOME}/include/targets/hip /users/pittlerf/code/quda/build2/_deps/eigen-src/ ${PLEGMA_HDF5HOME}/include -fPIC)
endif()
if(${PLEGMA_TARGET_TYPE} STREQUAL "CUDA")
target_include_directories(PLEGMAutils SYSTEM PRIVATE ${QUDA_SRC}/lib ${QUDA_HOME}/include/targets/cuda /users/pittlerf/code/quda/build2/_deps/eigen-src/)
endif()

add_dependencies(PLEGMAutils BoostPP)
