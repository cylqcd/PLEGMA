#include <PLEGMA.h>

#ifndef _PLEGMA_UTILS_H
#define _PLEGMA_UTILS_H
using namespace plegma;

// quda_interface.cpp
void initComms(int argc, char **argv, const int *commDims);
void finalizeComms();

// quda_params.cpp
void print_info();
void setGaugeParam(QudaGaugeParam &gauge_param);
void setMultigridParam(QudaMultigridParam &mg_param);
void setInvertParam(QudaInvertParam &inv_param);

// utils.cpp
void createMom(int *Nmom, int momElem[][3], int Q_qs);

// read_command_line.cpp
void read_command_line(int argc, char** argv, PLEGMA_params *params);

// read_conf.cpp
void readLimeGauge(double **gauge, char *fname, QudaGaugeParam *param, int gridSize[4]);
void applyBoundaryCondition(double **gauge, int Vh ,QudaGaugeParam *gauge_param);

// mapping_parity.cpp
void mapNormalToEvenOddGauge(double **gauge, QudaGaugeParam param, int nx , int ny , int nz, int nt);
void mapEvenOddToNormalGauge(double **gauge, QudaGaugeParam param, int nx , int ny , int nz, int nt);
void mapNormalToEvenOdd(void *spinor, QudaInvertParam param, int nx , int ny , int nz, int nt);
void mapEvenOddToNormal(void *spinor, QudaInvertParam param, int nx , int ny , int nz, int nt);

#endif
