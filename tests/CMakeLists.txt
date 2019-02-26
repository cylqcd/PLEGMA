find_package(GTest)
if (NOT GTest_FOUND AND PLEGMA_RUN_TESTS)
  message(WARNING
    "GTest was not found; we will download it.")
  set(EXTERNAL_INSTALL_LOCATION ${CMAKE_BINARY_DIR}/external)
  ExternalProject_Add(googletest
    GIT_REPOSITORY https://github.com/google/googletest
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_LOCATION}
  )
  include_directories(${EXTERNAL_INSTALL_LOCATION}/include)
  link_directories(${EXTERNAL_INSTALL_LOCATION}/lib)
endif ()

