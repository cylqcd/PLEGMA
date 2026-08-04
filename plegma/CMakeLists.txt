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

cuda_add_executable(Topo Topo.cpp)
target_link_libraries(Topo ${PLEGMA_LIBS})

cuda_add_executable(Calc_2pt Calc_2pt.cpp)
target_link_libraries(Calc_2pt ${PLEGMA_LIBS})

cuda_add_executable(Calc_2pt_wilson Calc_2pt_wilson.cpp)
target_link_libraries(Calc_2pt_wilson ${PLEGMA_LIBS})

#cuda_add_executable(Tetraquark Tetraquark.cpp)
#target_link_libraries(Tetraquark ${PLEGMA_LIBS})

#cuda_add_executable(TetraquarkBCUD TetraquarkBCUD.cpp)
#target_link_libraries(TetraquarkBCUD ${PLEGMA_LIBS})

cuda_add_executable(TestFrame EXCLUDE_FROM_ALL TestFrame.cpp)
target_link_libraries(TestFrame ${PLEGMA_LIBS})

#cuda_add_executable(StochasticTetraquark StochasticTetraquark.cpp)
#target_link_libraries(StochasticTetraquark ${PLEGMA_LIBS})

#cuda_add_executable(StochasticTetraquarkBCUD StochasticTetraquarkBCUD.cpp)
#target_link_libraries(StochasticTetraquarkBCUD ${PLEGMA_LIBS})

cuda_add_executable(ComputeLightStochProp ComputeLightStochProp.cpp)
target_link_libraries(ComputeLightStochProp ${PLEGMA_LIBS})

cuda_add_executable(ComputeLightPointProp ComputeLightPointProp.cpp)
target_link_libraries(ComputeLightPointProp ${PLEGMA_LIBS})

cuda_add_executable(Calc_2pt_charmonium Calc_2pt_charmonium.cpp)
target_link_libraries(Calc_2pt_charmonium ${PLEGMA_LIBS})

#cuda_add_executable(PDFs_nucleon PDFs_nucleon.cpp)
#target_link_libraries(PDFs_nucleon ${PLEGMA_LIBS})

#cuda_add_executable(PDFs_qgq_nucleon PDFs_qgq_nucleon.cpp)
#target_link_libraries(PDFs_qgq_nucleon ${PLEGMA_LIBS})

#cuda_add_executable(nucleon_2pt_3pt nucleon_2pt_3pt.cpp)
#target_link_libraries(nucleon_2pt_3pt ${PLEGMA_LIBS})

#cuda_add_executable(nucleon_2pt_3pt_npi nucleon_2pt_3pt_npi.cpp)
#target_link_libraries(nucleon_2pt_3pt_npi ${PLEGMA_LIBS})

#cuda_add_executable(mesons_3pt mesons_3pt.cpp)
#target_link_libraries(mesons_3pt ${PLEGMA_LIBS})

cuda_add_executable(TMDs_nucleon TMDs_nucleon.cpp)
target_link_libraries(TMDs_nucleon ${PLEGMA_LIBS})

cuda_add_executable(TMDs_nucleon_p1 TMDs_nucleon_p1.cpp)
target_link_libraries(TMDs_nucleon_p1 ${PLEGMA_LIBS})

cuda_add_executable(TMDs_nucleon_p2 TMDs_nucleon_p2.cpp)
target_link_libraries(TMDs_nucleon_p2 ${PLEGMA_LIBS})

cuda_add_executable(TMDs_test TMDs_test.cpp)
target_link_libraries(TMDs_test ${PLEGMA_LIBS})

cuda_add_executable(PDF_staple PDF_staple.cpp)
target_link_libraries(PDF_staple ${PLEGMA_LIBS})

cuda_add_executable(staple staple.cpp)
target_link_libraries(staple ${PLEGMA_LIBS})

cuda_add_executable(Wloop_PDF Wloop_PDF.cpp)
target_link_libraries(Wloop_PDF ${PLEGMA_LIBS})

cuda_add_executable(PDFs_staple PDFs_staple.cpp)
target_link_libraries(PDFs_staple ${PLEGMA_LIBS})

cuda_add_executable(SoftFunctionWall SoftFunctionWall.cpp)
target_link_libraries(SoftFunctionWall ${PLEGMA_LIBS})

cuda_add_executable(SoftFunction SoftFunction.cpp)
target_link_libraries(SoftFunction ${PLEGMA_LIBS})

cuda_add_executable(SoftFunction2 SoftFunction2.cpp)
target_link_libraries(SoftFunction2 ${PLEGMA_LIBS})

cuda_add_executable(SoftFunction_source_loop SoftFunction_source_loop.cpp)
target_link_libraries(SoftFunction_source_loop ${PLEGMA_LIBS})

cuda_add_executable(SoftFunction_new SoftFunction_new.cpp)
target_link_libraries(SoftFunction_new ${PLEGMA_LIBS})

cuda_add_executable(formfactor formfactor.cpp)
target_link_libraries(formfactor ${PLEGMA_LIBS})

cuda_add_executable(TMDPDFs_nucleon TMDPDFs_nucleon.cpp)
target_link_libraries(TMDPDFs_nucleon ${PLEGMA_LIBS})

cuda_add_executable(TMDPDFs_nucleon_optim TMDPDFs_nucleon_optim.cpp)
target_link_libraries(TMDPDFs_nucleon_optim ${PLEGMA_LIBS})

cuda_add_executable(TMDPDFs_nucleon_source_loop TMDPDFs_nucleon_source_loop.cpp)
target_link_libraries(TMDPDFs_nucleon_source_loop ${PLEGMA_LIBS})

cuda_add_executable(correcting_ZM EXCLUDE_FROM_ALL correcting_ZM.cpp)
target_link_libraries(correcting_ZM ${PLEGMA_LIBS})

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

#cuda_add_executable(benchmark benchmark.cpp)
#target_link_libraries(benchmark ${PLEGMA_LIBS})

#cuda_add_executable(TMDWFs_mesons  TMDWFs_mesons.cpp)
#target_link_libraries(TMDWFs_mesons ${PLEGMA_LIBS})

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
  cuda_add_executable(nucleon_delta_test EXCLUDE_FROM_ALL nucleon_delta_test.cpp)
  target_link_libraries(nucleon_delta_test ${PLEGMA_LIBS})
  cuda_add_executable(piNdiagrams_oet piNdiagrams_oet.cpp)
  target_link_libraries(piNdiagrams_oet ${PLEGMA_LIBS})
  cuda_add_executable(checking_Doet checking_Doet.cpp)
  target_link_libraries(checking_Doet ${PLEGMA_LIBS})
  cuda_add_executable(omegamass EXCLUDE_FROM_ALL omegamass.cpp)
  target_link_libraries(omegamass ${PLEGMA_LIBS})
endif(PLEGMA_SCATTERING_CONTRACTIONS)


cuda_add_executable(gaugeFixing gaugeFixing.cpp)
target_link_libraries(gaugeFixing ${PLEGMA_LIBS})

cuda_add_executable(invert invert.cpp)
target_link_libraries(invert ${PLEGMA_LIBS})

cuda_add_executable(polarN EXCLUDE_FROM_ALL polarN.cpp)
target_link_libraries(polarN ${PLEGMA_LIBS})

cuda_add_executable(Zfac_GI Zfac_GI.cpp)
target_link_libraries(Zfac_GI ${PLEGMA_LIBS})

cuda_add_executable(Zfac_GI_meson3pt Zfac_GI_meson3pt.cpp)
target_link_libraries(Zfac_GI_meson3pt ${PLEGMA_LIBS})

cuda_add_executable(mesons_2pt  mesons_2pt.cpp)
target_link_libraries(mesons_2pt ${PLEGMA_LIBS})

cuda_add_executable(mesons_simple mesons_simple.cpp)
target_link_libraries(mesons_simple ${PLEGMA_LIBS})

cuda_add_executable(Zfac_diagGI Zfac_diagGI.cpp)
target_link_libraries(Zfac_diagGI ${PLEGMA_LIBS})
