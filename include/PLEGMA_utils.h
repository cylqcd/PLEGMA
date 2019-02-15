#include <PLEGMA.h>
#include <PLEGMA_buffers.h>
#include <quda_params.h>
#include <quda_interface.h>

#ifndef _PLEGMA_UTILS_H
#define _PLEGMA_UTILS_H
using namespace plegma;

//------------------//
// PLEGMA Parameters //
//------------------//
extern char latfile_smeared[];
extern char verbosity_level[];
extern int traj;
extern bool isEven;

extern int src[];
extern int Ntsink;
extern char pathList_tsink[];
extern int Q_sq;
extern int nsmearAPE;
extern int nsmearGauss;
extern double alphaAPE;
extern double alphaGauss;
extern char twop_filename[];
extern char threep_filename[];

extern char prop_path[];
extern double csw;

extern int numSourcePositions;
extern char pathListSourcePositions[];
extern char pathListRun3pt[];
extern char run3pt[];
extern char *corr_file_format;
extern char check_file_exist[];

extern int Nproj;
extern char proj_list_file[];

extern char *corr_write_space;
extern int dim_partitioned[];

// quda_interface.cpp
void initComms(int argc, char **argv, const int *commDims);
void finalizeComms();
void initGaugeQuda(void* gauge, QudaGaugeParam gauge_param);
void updateGaugeQuda(void* gauge, QudaGaugeParam gauge_param);
void finalizeGaugeQuda();

//============= quda_params.cpp ===================================//
void print_info();
void setGaugeParam(QudaGaugeParam &gauge_param);
void setMultigridParam(QudaMultigridParam &mg_param);
void setInvertParam(QudaInvertParam &inv_param);

//============== utils.cpp =======================================//
void createMom(int *Nmom, int momElem[][3], int Q_qs);
void initialize(int argc, char** argv, PLEGMA_params *params);
void finalize();

//=================== read_command_line.cpp ==========================//
void read_command_line(int argc, char** argv, PLEGMA_params *params);

//================== read_conf.cpp ===================================//
void readLimeGauge(double **gauge, char *fname, QudaGaugeParam *param, int gridSize[4]);
void applyBoundaryCondition(double **gauge, int Vh ,QudaGaugeParam *gauge_param);
void applyBoundaryCondition(double **gauge, int lL[4] ,QudaGaugeParam *gauge_param);

//================ mapping_parity.cpp ================================//

template<typename Float> void mapNormalToEvenOddGauge(Float **gauge, int lL[4], QudaGaugeFieldOrder order = QUDA_QDP_GAUGE_ORDER);
template<typename Float> void mapEvenOddToNormalGauge(Float **gauge, int lL[4], QudaGaugeFieldOrder order = QUDA_QDP_GAUGE_ORDER);
template<typename Float> void mapNormalToEvenOdd(Float *spinor, int lL[4], QudaDiracFieldOrder order = QUDA_DIRAC_ORDER);
template<typename Float> void mapEvenOddToNormal(Float *spinor, int lL[4], QudaDiracFieldOrder order = QUDA_DIRAC_ORDER);
template<typename Float> void mapNormalToEvenOddGPUformat(Float *spinor, int lL[4], QudaDiracFieldOrder order = QUDA_DIRAC_ORDER);
template<typename Float> void mapEvenOddToNormalGPUformat(Float *spinor, int lL[4], QudaDiracFieldOrder order = QUDA_DIRAC_ORDER);
#endif
