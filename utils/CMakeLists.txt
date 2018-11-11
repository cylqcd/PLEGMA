#build a common library for all test utilities
set(PLEGMA_UTILS
    quda_types.cpp
    quda_params.cpp
    quda_interface.cpp
    read_command_line.cpp
    read_conf.cpp
    mapping_parity.cpp
    utils.cpp
    )
cuda_add_library(PLEGMAutils STATIC ${PLEGMA_UTILS})
