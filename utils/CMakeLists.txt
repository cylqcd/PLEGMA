#build a common library for all test utilities
file(GLOB PLEGMA_UTILS
     "*.cpp"
)

#cuda_add_library(PLEGMAutils STATIC ${PLEGMA_UTILS})
add_library(PLEGMAutils STATIC ${PLEGMA_UTILS})
target_link_libraries(PLEGMAutils PRIVATE ${QUDA_HOME}/lib/libquda.so)
if(${PLEGMA_TARGET_TYPE} STREQUAL "HIP")
target_include_directories(PLEGMAutils SYSTEM PRIVATE ${QUDA_SRC}/lib ${QUDA_HOME}/include/targets/hip)
endif()
if(${PLEGMA_TARGET_TYPE} STREQUAL "CUDA")
target_include_directories(PLEGMAutils SYSTEM PRIVATE ${QUDA_SRC}/lib ${QUDA_HOME}/include/targets/cuda)
endif()

add_dependencies(PLEGMAutils BoostPP)
