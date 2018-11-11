#include <PLEGMA_global.h>
#include <PLEGMA_Field.h>
#include <PLEGMA_Gauge.h>
#include <PLEGMA_Vector.h>
#include <PLEGMA_Propagator.h>
#include <PLEGMA_Correlator.h>

#ifndef _PLEGMA__H
#define _PLEGMA__H

namespace plegma {

  typedef struct {
    int nsmearAPE;
    int nsmearGauss;
    double alphaAPE;
    double alphaGauss;
    int lL[N_DIMS];
    int procs[N_DIMS];
    int Nsources;
    int sourcePosition[MAX_NSOURCES][N_DIMS];
    QudaPrecision Precision;
    int Q_sq;
    int Ntsink;
    int Nproj[MAX_TSINK];
    int traj;
    bool check_files;
    char *thrp_type[3];
    char *thrp_proj_type[5];
    char *baryon_type[10];
    char *meson_type[10];
    int tsinkSource[MAX_TSINK];
    int proj_list[MAX_TSINK][MAX_PROJS];
    int run3pt_src[MAX_NSOURCES];
    FILE_WRITE_FORMAT CorrFileFormat;
    SOURCE_T source_type;
    CORR_SPACE CorrSpace;
    bool HighMomForm;
    bool isEven;
    double kappa;
    double mu;
    double csw;
    double inv_tol;
  } PLEGMA_params;

  void initialize(PLEGMA_params *params);

  void print_status();
}
#endif
