find_package(GTest)
if (NOT GTest_FOUND)

  include_directories(.)
  message(WARNING
    "GTest was not found; tests will run old version of already included GTest")
  set(PLEGMA_TEST_COMMON gtest-all.cc)
  cuda_add_library(plegma_test STATIC ${PLEGMA_TEST_COMMON})

  set(TEST_LIBS plegma plegma_test )

  MACRO (PLEGMA_CHECKBUILDTEST mytarget plegmabuildtests)
      IF(NOT ${plegmabuildtests})
            set_property(TARGET ${mytarget} PROPERTY EXCLUDE_FROM_ALL 1)
              ENDIF()
            ENDMACRO()
            #define tests
  return()
endif ()

