#pragma once
#include <PLEGMA.h>
#include <utils/PLEGMA_params.h>
#include <utils/PLEGMA_Buffers.h>
#include <utils/QUDA_params.h>
#include <utils/QUDA_types.h>
#include <utils/QUDA_interface.h>
#include <utils/PLEGMA_Options.h>

using namespace plegma;

//============ QUDA_interface.cpp ===============================//
void initComms(int argc, char **argv, const int *commDims);
void finalizeComms();
void initGaugeQuda(void* gauge, QudaGaugeParam gauge_param);
void updateGaugeQuda(void* gauge, QudaGaugeParam gauge_param);
void finalizeGaugeQuda();

//============= QUDA_params.cpp ===================================//
void print_info();
void setGaugeParam(QudaGaugeParam &gauge_param);
void setMultigridParam(QudaMultigridParam &mg_param);
void setInvertParam(QudaInvertParam &inv_param);

//============== PLEGMA_utils.cpp =======================================//
void createMom(int *Nmom, int momElem[][3], int Q_qs);
void initialize(int argc, char** argv);
void initPlegma(int argc, char **argv);
void finalize();
void applyBoundaryCondition(double **gauge, int Vh ,QudaGaugeParam *gauge_param);
void applyBoundaryCondition(double **gauge, int lL[4] ,QudaGaugeParam *gauge_param);
void mapNormalToEvenOddGauge(double **gauge, QudaGaugeParam &param, int nx , int ny , int nz, int nt);
void mapNormalToEvenOddGauge(double **gauge, QudaGaugeParam &param, int lL[4]);
void mapEvenOddToNormalGauge(double **gauge, QudaGaugeParam &param, int nx , int ny , int nz, int nt);
void mapEvenOddToNormalGauge(double **gauge, QudaGaugeParam &param, int lL[4]);
void mapNormalToEvenOdd(void *spinor, QudaInvertParam param, int nx , int ny , int nz, int nt);
void mapEvenOddToNormal(void *spinor, QudaInvertParam param, int nx , int ny , int nz, int nt);

//=================== PLEGMA_Options.cpp ==========================//
void read_command_line(int argc, char** argv);
void basicOptions(Options &opt, bool showThem  = false);
void basicOptionsWsolver(Options &opt, bool showThem = false);

//================== PLEGMA_lime.cpp ===================================//
void readLimeGauge(double **gauge, char *fname, QudaGaugeParam *param, int gridSize[4]);
