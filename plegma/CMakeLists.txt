set(PLEGMA_LIBS PLEGMA PLEGMAutils ${CMAKE_THREAD_LIBS_INIT} ${QUDA_LIB} ${CUDA_LIBS} cuda)
set(CMAKE_CUDA_SEPARABLE_COMPILATION OFF CACHE BOOL "" FORCE)
add_compile_definitions(PLEGMA_CONST_TO_DEVICE)
include_directories(${QUDA_HOME}/_deps/eigen-src/)


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
LIST(APPEND PLEGMA_LIBS HDF5::HDF5)
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
# Each group is OFF by default. PLEGMA_HIGHER_DERIV (Wilson flow targets) is
# controlled separately in the top-level CMakeLists.txt.

# --- Gauge observables: plaquette, topology, gauge fixing ---
option(PLEGMA_BUILD_GAUGE_OBS "Build gauge observables (Plaquette, Topo, gaugeFixing)" OFF)
if(PLEGMA_BUILD_GAUGE_OBS)
  cuda_add_executable(Plaquette Plaquette.cpp)
  target_link_libraries(Plaquette ${PLEGMA_LIBS})

  cuda_add_executable(Topo Topo.cpp)
  target_link_libraries(Topo ${PLEGMA_LIBS})

  cuda_add_executable(gaugeFixing gaugeFixing.cpp)
  target_link_libraries(gaugeFixing ${PLEGMA_LIBS})
endif()

# --- Solver tools: inversion, low modes, MG tuning, propagator generation ---
option(PLEGMA_BUILD_SOLVER_TOOLS "Build solver utilities (invert, LowModesCalc, MG_tuner, propagators, pion_LowModeAvg)" OFF)
if(PLEGMA_BUILD_SOLVER_TOOLS)
  cuda_add_executable(invert invert.cpp)
  target_link_libraries(invert ${PLEGMA_LIBS})

  cuda_add_executable(LowModesCalc LowModesCalc.cpp)
  target_link_libraries(LowModesCalc ${PLEGMA_LIBS})

  cuda_add_executable(MG_tuner MG_tuner.cpp)
  target_link_libraries(MG_tuner ${PLEGMA_LIBS})

  cuda_add_executable(ComputeLightStochProp ComputeLightStochProp.cpp)
  target_link_libraries(ComputeLightStochProp ${PLEGMA_LIBS})

  cuda_add_executable(ComputeLightPointProp ComputeLightPointProp.cpp)
  target_link_libraries(ComputeLightPointProp ${PLEGMA_LIBS})

  cuda_add_executable(pion_LowModeAvg pion_LowModeAvg.cpp)
  target_link_libraries(pion_LowModeAvg ${PLEGMA_LIBS})
endif()

# --- Meson 2pt/3pt correlators ---
option(PLEGMA_BUILD_MESON_CORR "Build meson 2pt/3pt correlators" OFF)
if(PLEGMA_BUILD_MESON_CORR)
  cuda_add_executable(Calc_2pt Calc_2pt.cpp)
  target_link_libraries(Calc_2pt ${PLEGMA_LIBS})

  cuda_add_executable(Calc_2pt_wilson Calc_2pt_wilson.cpp)
  target_link_libraries(Calc_2pt_wilson ${PLEGMA_LIBS})

  cuda_add_executable(Calc_2pt_marconi Calc_2pt_marconi.cpp)
  target_link_libraries(Calc_2pt_marconi ${PLEGMA_LIBS})

  cuda_add_executable(Calc_2pt_charmonium Calc_2pt_charmonium.cpp)
  target_link_libraries(Calc_2pt_charmonium ${PLEGMA_LIBS})

  cuda_add_executable(mesons_2pt mesons_2pt.cpp)
  target_link_libraries(mesons_2pt ${PLEGMA_LIBS})

  cuda_add_executable(mesons_simple mesons_simple.cpp)
  target_link_libraries(mesons_simple ${PLEGMA_LIBS})

  # cuda_add_executable(mesons_3pt mesons_3pt.cpp)
  # target_link_libraries(mesons_3pt ${PLEGMA_LIBS})
endif()

# --- Nucleon 2pt/3pt correlators and structure functions ---
option(PLEGMA_BUILD_NUCLEON_CORR "Build nucleon 2pt/3pt correlators and PDFs" OFF)
if(PLEGMA_BUILD_NUCLEON_CORR)
  cuda_add_executable(nucleon_2pt_3pt nucleon_2pt_3pt.cpp)
  target_link_libraries(nucleon_2pt_3pt ${PLEGMA_LIBS})

  cuda_add_executable(PDFs_nucleon PDFs_nucleon.cpp)
  target_link_libraries(PDFs_nucleon ${PLEGMA_LIBS})

  cuda_add_executable(polarN EXCLUDE_FROM_ALL polarN.cpp)
  target_link_libraries(polarN ${PLEGMA_LIBS})
endif()

# --- Disconnected quark/gluon loops with Wilson line insertions ---
option(PLEGMA_BUILD_DISC_LOOPS "Build disconnected quark/gluon loops (qLoops, Wilson line)" OFF)
if(PLEGMA_BUILD_DISC_LOOPS)
  cuda_add_executable(qLoops qLoops.cpp)
  target_link_libraries(qLoops ${PLEGMA_LIBS})

  cuda_add_executable(qLoops_WilsonLine qLoops_WilsonLine.cpp)
  target_link_libraries(qLoops_WilsonLine ${PLEGMA_LIBS})

  cuda_add_executable(gLoops_WilsonLine gLoops_WilsonLine.cpp)
  target_link_libraries(gLoops_WilsonLine ${PLEGMA_LIBS})
endif()

# --- Renormalization Z-factors ---
option(PLEGMA_BUILD_RENORM "Build renormalization Z-factor executables" OFF)
if(PLEGMA_BUILD_RENORM)
  cuda_add_executable(Zfac Zfac.cpp)
  target_link_libraries(Zfac ${PLEGMA_LIBS})

  cuda_add_executable(ZfacStaple ZfacStaple.cpp)
  target_link_libraries(ZfacStaple ${PLEGMA_LIBS})

  cuda_add_executable(Zfac_GI Zfac_GI.cpp)
  target_link_libraries(Zfac_GI ${PLEGMA_LIBS})

  cuda_add_executable(Zfac_GI_meson3pt Zfac_GI_meson3pt.cpp)
  target_link_libraries(Zfac_GI_meson3pt ${PLEGMA_LIBS})

  cuda_add_executable(Zfac_diagGI Zfac_diagGI.cpp)
  target_link_libraries(Zfac_diagGI ${PLEGMA_LIBS})

  cuda_add_executable(Zfac_Vgg EXCLUDE_FROM_ALL Zfac_Vgg.cpp)
  target_link_libraries(Zfac_Vgg ${PLEGMA_LIBS})
endif()

# --- TMD structure functions (mesons) ---
option(PLEGMA_BUILD_TMDPDF "Build TMD PDF executables for mesons" OFF)
if(PLEGMA_BUILD_TMDPDF)
  cuda_add_executable(TMDPDFs_mesons_Juwels TMDPDFs_mesons_Juwels.cpp)
  target_link_libraries(TMDPDFs_mesons_Juwels ${PLEGMA_LIBS})

  cuda_add_executable(new_TMDPDFs_mesons_Juwels new_TMDPDFs_mesons_Juwels.cpp)
  target_link_libraries(new_TMDPDFs_mesons_Juwels ${PLEGMA_LIBS})
endif()

# --- piN scattering contractions (PLEGMA_SCATTERING_CONTRACTIONS set in top-level CMakeLists.txt) ---
if(PLEGMA_SCATTERING_CONTRACTIONS)
  cuda_add_executable(nucleon_2pt_3pt_npi nucleon_2pt_3pt_npi.cpp)
  target_link_libraries(nucleon_2pt_3pt_npi ${PLEGMA_LIBS})

  cuda_add_executable(piNdiagrams piNdiagrams.cpp)
  target_link_libraries(piNdiagrams ${PLEGMA_LIBS})

  cuda_add_executable(piNdiagrams_oet piNdiagrams_oet.cpp)
  target_link_libraries(piNdiagrams_oet ${PLEGMA_LIBS})

  cuda_add_executable(checking_Doet checking_Doet.cpp)
  target_link_libraries(checking_Doet ${PLEGMA_LIBS})

  cuda_add_executable(testScattReductions EXCLUDE_FROM_ALL testScattReductions.cpp)
  target_link_libraries(testScattReductions ${PLEGMA_LIBS})

  cuda_add_executable(nucleon_delta_test EXCLUDE_FROM_ALL nucleon_delta_test.cpp)
  target_link_libraries(nucleon_delta_test ${PLEGMA_LIBS})

  cuda_add_executable(nucleon_3pt_n_npi EXCLUDE_FROM_ALL nucleon_3pt_n_npi.cpp)
  target_link_libraries(nucleon_3pt_n_npi ${PLEGMA_LIBS})

  cuda_add_executable(nucleon_3pt_n_npi_wd1ii EXCLUDE_FROM_ALL nucleon_3pt_n_npi_wd1ii.cpp)
  target_link_libraries(nucleon_3pt_n_npi_wd1ii ${PLEGMA_LIBS})

  cuda_add_executable(omegamass EXCLUDE_FROM_ALL omegamass.cpp)
  target_link_libraries(omegamass ${PLEGMA_LIBS})

  cuda_add_executable(pi0Insertion EXCLUDE_FROM_ALL pi0Insertion.cpp)
  target_link_libraries(pi0Insertion ${PLEGMA_LIBS})

  cuda_add_executable(QuarkLoops_pi0_insertion EXCLUDE_FROM_ALL QuarkLoops_pi0_insertion.cpp)
  target_link_libraries(QuarkLoops_pi0_insertion ${PLEGMA_LIBS})

  cuda_add_executable(testYan EXCLUDE_FROM_ALL testYan.cpp)
  target_link_libraries(testYan ${PLEGMA_LIBS})

  cuda_add_executable(piN_scattering_length_12 EXCLUDE_FROM_ALL piN_scattering_length_12.cpp)
  target_link_libraries(piN_scattering_length_12 ${PLEGMA_LIBS})
endif(PLEGMA_SCATTERING_CONTRACTIONS)

# --- Higher-order covariant derivative applications (requires PLEGMA_HIGHER_DERIV=ON) ---
if(PLEGMA_HIGHER_DERIV)
  cuda_add_executable(meson_2pt_3pt_flowed meson_2pt_3pt_flowed.cpp)
  target_link_libraries(meson_2pt_3pt_flowed ${PLEGMA_LIBS})

  cuda_add_executable(meson_2pt_3pt_flowed_stoch meson_2pt_3pt_flowed_stoch.cpp)
  target_link_libraries(meson_2pt_3pt_flowed_stoch ${PLEGMA_LIBS})

  cuda_add_executable(qLoops_patterns qLoops_patterns.cpp)
  target_link_libraries(qLoops_patterns ${PLEGMA_LIBS})
endif()
