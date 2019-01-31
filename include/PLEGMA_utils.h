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
int getVecToInd(std::vector<int> x, std::vector<int> L );
std::vector<int> getIndToVec(int ind, std::vector<int> L);

class Hprobing{
private:
  int k;  // index for the coloring distance
  int Nc; // Number of colors = Number of Hadamard vectors
  short d; // Number of dimension of Hprob (For now d=4)
  short D; // Distance of coloring D=2^k
  short Lu; // extent of the elementaty coloring block (assume symmetric block)
  int* h_arrVc; // array to hold the coloring of the lattice on HOST
  int* d_arrVc; // array to hold the coloring of the lattice on Device
  int* arrlc; // array to hold the elementary coloring block
  void createElemColBlock(){for(int i = 0; i < Nc; i++) arrlc[i]=i;}
  void createColLattice();
  void checkColoring();
public:
  Hprobing(int k_probing, int d=4, bool check = false);
  ~Hprobing();
  int* H_arrVc() const{return h_arrVc;}
  int* D_arrVc() const{return d_arrVc;}
};


//=================== read_command_line.cpp ==========================//
void read_command_line(int argc, char** argv, PLEGMA_params *params);

//================== read_conf.cpp ===================================//
void readLimeGauge(double **gauge, char *fname, QudaGaugeParam *param, int gridSize[4]);
void applyBoundaryCondition(double **gauge, int Vh ,QudaGaugeParam *gauge_param);
void applyBoundaryCondition(double **gauge, int lL[4] ,QudaGaugeParam *gauge_param);

//================ mapping_parity.cpp ================================//
void mapNormalToEvenOddGauge(double **gauge, QudaGaugeParam &param, int nx , int ny , int nz, int nt);
void mapNormalToEvenOddGauge(double **gauge, QudaGaugeParam &param, int lL[4]);
void mapEvenOddToNormalGauge(double **gauge, QudaGaugeParam &param, int nx , int ny , int nz, int nt);
void mapEvenOddToNormalGauge(double **gauge, QudaGaugeParam &param, int lL[4]);
void mapNormalToEvenOdd(void *spinor, QudaInvertParam param, int nx , int ny , int nz, int nt);
void mapEvenOddToNormal(void *spinor, QudaInvertParam param, int nx , int ny , int nz, int nt);
#endif
