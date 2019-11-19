#pragma once
#include <PLEGMA.h>
#include <utils/PLEGMA_params.h>
#include <utils/PLEGMA_types.h>
#include <utils/PLEGMA_readList.h>
#include <utils/PLEGMA_auxiliary.h>
#include <utils/PLEGMA_eigSolver.h>
#include <utils/QUDA_params.h>
#include <utils/QUDA_types.h>
#include <utils/QUDA_interface.h>

using namespace plegma;

//============ QUDA_interface.cpp ===============================//
void initComms(int argc, char **argv, const int *commDims);
void finalizeComms();
void initGaugeQuda(PLEGMA_Gauge<double> &gauge, bool antiperiodic = true, QudaLinkType type = QUDA_WILSON_LINKS);
void updateGaugeQuda(PLEGMA_Gauge<double> &gauge, bool antiperiodic = true, QudaLinkType type = QUDA_WILSON_LINKS);
void finalizeGaugeQuda();
void plaqQuda();
void gFixingLandauOVR_QUDA(PLEGMA_Gauge<double> &gaugeOut, PLEGMA_Gauge<double> &gaugeIn, double overelaxPar=1.5, double tolerance=1e-12,
			   int maxiter=10000,int verbosePerSteps=1, int reunit_interval=1, int stop_theta=0);

//============= QUDA_params.cpp ===================================//
void infoQuda();
void setGaugeParam(QudaGaugeParam &gauge_param);
void setMultigridParam(QudaMultigridParam &mg_param);
void setInvertParam(QudaInvertParam &inv_param);
#ifdef QUDA_INCLUDES_COMMIT_775a033
void setEigMultigridParam(QudaMultigridParam &mg_param, QudaEigParam *mg_eig_param);
#endif
//============== PLEGMA_utils.cpp =======================================//
void createMom(int *Nmom, int momElem[][3], int Q_qs);
extern const std::vector<std::string> listAvailOptPLEGMA;
void initializeOptions(int argc, char **argv, bool withQuda=true, std::vector<std::string> listOptPLEGMA = listAvailOptPLEGMA);
void updateOptions(std::string filename, std::vector<std::string>& listOpt, std::function<void(Options&)> add_options = nullptr);
void updateOptions(WHICHFLAVOR fl);
void initializePLEGMA();
void finalize();
template<typename FloatOut, typename FloatIn> void unpackGaugeToEvenOdd(FloatOut *buf[4], PLEGMA_Gauge<FloatIn> &gauge);
template<typename FloatOut, typename FloatIn> void packGaugeToNormal(PLEGMA_Gauge<FloatOut> &gauge, FloatIn *buf[4]);
template<typename Float> void applyAntiperiodicBoundary(Float **buf);
template<typename Float> void applyBoundaryConditions(PLEGMA_Gauge<Float> &gauge, bool antiperiodic);
std::vector<int> createR2(std::vector<int> &vec);

//=================== PLEGMA_Options.cpp ==========================//
void plegmaOptions(Options &opt, std::vector<std::string> list, bool update_params = false);
void qudaOptions(Options &opt);

