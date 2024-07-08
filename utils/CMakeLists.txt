#build a common library for all test utilities
file(GLOB PLEGMA_UTILS
     "*.cpp"
)

cuda_add_library(PLEGMAutils STATIC ${PLEGMA_UTILS})
target_include_directories(PLEGMAutils SYSTEM PRIVATE ${QUDA_SRC}/lib ${QUDA_HOME}/_deps/eigen-src/ )
add_dependencies(PLEGMAutils BoostPP)
