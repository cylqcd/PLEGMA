#pragma once
#include <PLEGMA.h>
#include <utils/PLEGMA_params.h>
#include <utils/QUDA_params.h>
#include <utils/QUDA_types.h>
#include <utils/QUDA_interface.h>
#include <utils/PLEGMA_Options.h>
#include <utils/PLEGMA_lime.h>

using namespace plegma;

//============ QUDA_interface.cpp ===============================//
void initComms(int argc, char **argv, const int *commDims);
void finalizeComms();
void initGaugeQuda(PLEGMA_Gauge<double> &gauge, bool antiperiodic = true, QudaLinkType type = QUDA_WILSON_LINKS);
void updateGaugeQuda(PLEGMA_Gauge<double> &gauge, bool antiperiodic = true, QudaLinkType type = QUDA_WILSON_LINKS);
void finalizeGaugeQuda();
void plaqQuda();

//============= QUDA_params.cpp ===================================//
void infoQuda();
void setGaugeParam(QudaGaugeParam &gauge_param);
void setMultigridParam(QudaMultigridParam &mg_param);
void setInvertParam(QudaInvertParam &inv_param);

//============== PLEGMA_utils.cpp =======================================//
void createMom(int *Nmom, int momElem[][3], int Q_qs);
void initialize(int argc, char** argv, bool withQuda=true);
void finalize();


//=================== PLEGMA_Options.cpp ==========================//
void basicOptions(Options &opt);
void qudaSolverOptions(Options &opt);
