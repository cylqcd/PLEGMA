#build a common library for all test utilities
file(GLOB PLEGMA_UTILS
     "*.cpp"
)
cuda_add_library(PLEGMAutils STATIC ${PLEGMA_UTILS})
