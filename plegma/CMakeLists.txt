set(PLEGMA_LIBS PLEGMA PLEGMAutils ${CMAKE_THREAD_LIBS_INIT} ${QUDA_LIB} ${CUDA_LIBS} cuda)

LIST(APPEND PLEGMA_LIBS ${MPI_CXX_LIBRARIES})

if(PLEGMA_ARPACK)
  LIST(APPEND PLEGMA_LIBS ${PLEGMA_MPIFH_LIB})
  LIST(APPEND PLEGMA_LIBS ${PLEGMA_Z_LIB})
  LIST(APPEND PLEGMA_LIBS ${PLEGMA_GFORTRAN_LIB})
  LIST(APPEND PLEGMA_LIBS ${PLEGMA_PARPACK_LIB})
  LIST(APPEND PLEGMA_LIBS ${PLEGMA_ARPACK_LIB})
endif()

if(PLEGMA_PRIMME)
  LIST(APPEND PLEGMA_LIBS ${PLEGMA_GFORTRAN_LIB})
  LIST(APPEND PLEGMA_LIBS ${PLEGMA_PRIMME_LIB})
endif()

LIST(APPEND PLEGMA_LIBS ${PLEGMA_GSL_LIB})
LIST(APPEND PLEGMA_LIBS ${HDF5_C_LIBRARY_hdf5})
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

cuda_add_executable(omega omega.cpp)
target_link_libraries(omega ${PLEGMA_LIBS})

cuda_add_executable(omega_smear omega_smear.cpp)
target_link_libraries(omega_smear ${PLEGMA_LIBS})

cuda_add_executable(pion pion.cpp)
target_link_libraries(pion ${PLEGMA_LIBS})

cuda_add_executable(pion_mix pion_mix.cpp)
target_link_libraries(pion_mix ${PLEGMA_LIBS})

cuda_add_executable(pion_mix_SIB pion_mix_SIB.cpp)
target_link_libraries(pion_mix_SIB ${PLEGMA_LIBS})

cuda_add_executable(pion_defl pion_defl.cpp)
target_link_libraries(pion_defl ${PLEGMA_LIBS})

cuda_add_executable(pion_mix_LIBE pion_mix_LIBE.cpp)
target_link_libraries(pion_mix_LIBE ${PLEGMA_LIBS})

cuda_add_executable(pion_mix_qed pion_mix_qed.cpp)
target_link_libraries(pion_mix_qed ${PLEGMA_LIBS})

cuda_add_executable(Topo Topo.cpp)
target_link_libraries(Topo ${PLEGMA_LIBS})

cuda_add_executable(Calc_2pt Calc_2pt.cpp)
target_link_libraries(Calc_2pt ${PLEGMA_LIBS})

cuda_add_executable(PDFs_nucleon PDFs_nucleon.cpp)
target_link_libraries(PDFs_nucleon ${PLEGMA_LIBS})

cuda_add_executable(GPDs_nucleon GPDs_nucleon.cpp)
target_link_libraries(GPDs_nucleon ${PLEGMA_LIBS})

cuda_add_executable(nucleon_2pt_3pt nucleon_2pt_3pt.cpp)
target_link_libraries(nucleon_2pt_3pt ${PLEGMA_LIBS})

#cuda_add_executable(meson_2pt_3pt meson_2pt_3pt.cpp)
#target_link_libraries(meson_2pt_3pt ${PLEGMA_LIBS})

cuda_add_executable(Source EXCLUDE_FROM_ALL Source.cpp)
target_link_libraries(Source ${PLEGMA_LIBS})

cuda_add_executable(Random EXCLUDE_FROM_ALL Random.cpp)
target_link_libraries(Random ${PLEGMA_LIBS})

cuda_add_executable(Dilution EXCLUDE_FROM_ALL Dilution.cpp)
target_link_libraries(Dilution ${PLEGMA_LIBS})

cuda_add_executable(testLinksSmearings EXCLUDE_FROM_ALL testLinksSmearings.cpp)
target_link_libraries(testLinksSmearings ${PLEGMA_LIBS})

cuda_add_executable(testHDF5 EXCLUDE_FROM_ALL testHDF5.cpp)
target_link_libraries(testHDF5 ${PLEGMA_LIBS})

cuda_add_executable(qLoops qLoops.cpp)
target_link_libraries(qLoops ${PLEGMA_LIBS})

cuda_add_executable(qLoops_WilsonLine qLoops_WilsonLine.cpp)
target_link_libraries(qLoops_WilsonLine ${PLEGMA_LIBS})

cuda_add_executable(gLoops_WilsonLine gLoops_WilsonLine.cpp)
target_link_libraries(gLoops_WilsonLine ${PLEGMA_LIBS})

cuda_add_executable(FT EXCLUDE_FROM_ALL FT.cpp)
target_link_libraries(FT ${PLEGMA_LIBS})

cuda_add_executable(LowModesCalc LowModesCalc.cpp)
target_link_libraries(LowModesCalc ${PLEGMA_LIBS})

cuda_add_executable(MG_tuner MG_tuner.cpp)
target_link_libraries(MG_tuner ${PLEGMA_LIBS})

cuda_add_executable(benchmark benchmark.cpp)
target_link_libraries(benchmark ${PLEGMA_LIBS})

cuda_add_executable(quark_rms EXCLUDE_FROM_ALL quark_rms.cpp)
target_link_libraries(quark_rms ${PLEGMA_LIBS})

cuda_add_executable(Zfac_Vgg EXCLUDE_FROM_ALL Zfac_Vgg.cpp)
target_link_libraries(Zfac_Vgg ${PLEGMA_LIBS})

cuda_add_executable(testShifts EXCLUDE_FROM_ALL testShifts.cpp)
target_link_libraries(testShifts ${PLEGMA_LIBS})

cuda_add_executable(Zfac Zfac.cpp)
target_link_libraries(Zfac ${PLEGMA_LIBS})

if (PLEGMA_SCATTERING_CONTRACTIONS)
  cuda_add_executable(testScattReductions EXCLUDE_FROM_ALL testScattReductions.cpp)
  target_link_libraries(testScattReductions ${PLEGMA_LIBS})
  cuda_add_executable(piNdiagrams piNdiagrams.cpp)
  target_link_libraries(piNdiagrams ${PLEGMA_LIBS})
endif(PLEGMA_SCATTERING_CONTRACTIONS)


cuda_add_executable(gaugeFixing EXCLUDE_FROM_ALL gaugeFixing.cpp)
target_link_libraries(gaugeFixing ${PLEGMA_LIBS})

cuda_add_executable(invert EXCLUDE_FROM_ALL invert.cpp)
target_link_libraries(invert ${PLEGMA_LIBS})
