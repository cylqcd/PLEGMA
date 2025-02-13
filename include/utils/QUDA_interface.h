#pragma once
#include <PLEGMA.h>
#include <invert_quda.h>
#include <dirac_quda.h>
using namespace plegma;

namespace quda {  
  class QUDA_solver {

  protected:
    TimeProfile *profiler;
    Solver *solver;
    SolverParam *solverParam;
    void *mg_preconditioner;
    QudaInvertParam inv_param;
    QudaInvertParam mg_inv_param;
    QudaMultigridParam mg_param;
    Dirac *D, *DSloppy, *DPre;
    DiracMatrix *M, *MSloppy, *MPre;
    std::vector<ColorSpinorField> b, x;
#ifdef QUDA_INCLUDES_COMMIT_775a033
    QudaEigParam *mg_eig_param;
#endif

  public:
    QudaInvertParam getInvParams() const{return inv_param;}
    SolverParam* getSolverParam() const{return solverParam;}
    void UpdateSolver();
    QUDA_solver(double mu, int nsrc);
    virtual ~QUDA_solver();
    std::vector<ColorSpinorField> solve(std::vector<ColorSpinorField>& rhs);

    template<typename Float> std::vector<ColorSpinorField> solve(PLEGMA_Vector<Float> &vectorIn);
    template<typename Float> std::vector<ColorSpinorField> solve(PLEGMA_Propagator<Float> &vectorIn);

    template<typename Float> void gSmear_QUDA(PLEGMA_Gauge<Float> &gaugeOut,PLEGMA_Gauge<Float> &gaugeIn, bool antiperiodic);


/*
    std::vector<ColorSpinorField> solve(PLEGMA_Vector<float> &vectorIn);
    std::vector<ColorSpinorField> solve(PLEGMA_Vector<double> &vectorIn);

    std::vector<ColorSpinorField> solve(PLEGMA_Propagator<float> &vectorIn);
    std::vector<ColorSpinorField> solve(PLEGMA_Propagator<double> &vectorIn);*/


    template<typename Float> void solve(PLEGMA_Propagator<Float> &out, PLEGMA_Propagator<Float> &in);
    template<typename Float> void solve(PLEGMA_Vector<Float> &out, PLEGMA_Vector<Float> &in);
/*
    void solve( PLEGMA_Vector<float> &out, PLEGMA_Vector<float> &in); 
    void solve( PLEGMA_Vector<double> &out, PLEGMA_Vector<double> &in);

    void solve( PLEGMA_Propagator<float> &out, PLEGMA_Propagator<float> &in);
    void solve( PLEGMA_Propagator<double> &out, PLEGMA_Propagator<double> &in);*/

    template<typename Float> void runOneIter( PLEGMA_Vector<Float> &out, PLEGMA_Vector<Float> &in);
    template<typename Float> void runOneIter( PLEGMA_Propagator<Float> &out, PLEGMA_Propagator<Float> &in);

/*
    void runOneIter( PLEGMA_Vector<float> &out, PLEGMA_Vector<float> &in);
    void runOneIter( PLEGMA_Vector<double> &out, PLEGMA_Vector<double> &in);

    void runOneIter( PLEGMA_Propagator<float> &out, PLEGMA_Propagator<float> &in);
    void runOneIter( PLEGMA_Propagator<double> &out, PLEGMA_Propagator<double> &in);*/

    template<bool bl, typename Float>
    std::vector<ColorSpinorField> solve(typename std::conditional<bl==true, PLEGMA_Vector<Float>,  PLEGMA_Propagator<Float>>::type &vectorIn);
    template<bool bl, typename Float>
    void solve(typename std::conditional<bl==true, PLEGMA_Vector<Float>,  PLEGMA_Propagator<Float>>::type &out, typename std::conditional<bl==true, PLEGMA_Vector<Float>,  PLEGMA_Propagator<Float>>::type &in);
    template<bool bl, typename Float>
    void runOneIter(typename std::conditional<bl==true, PLEGMA_Vector<Float>,  PLEGMA_Propagator<Float>>::type &out, typename std::conditional<bl==true, PLEGMA_Vector<Float>,  PLEGMA_Propagator<Float>>::type &in);
  };

  enum APP_TYPE {M,Mdag,MdagM,MMdag};
  class QUDA_dirac {
  private:
    DiracParam dParam;
    QudaInvertParam inv_param;
    Dirac *D;
    std::vector<ColorSpinorField> in, out;
    template<APP_TYPE type> void apply();
  public:
     //only QUDA_WILSON_DSLASH, QUDA_CLOVER_WILSON_DSLASH, QUDA_TWISTED_MASS_DSLASH, QUDA_TWISTED_CLOVER_DSLASH
    QUDA_dirac(QudaDslashType dslashType);
    virtual ~QUDA_dirac();
    void print(){dParam.print();}
    void switchMu(double mu);
    void switchKappa(double kappa);
    template<APP_TYPE type, bool bl, typename Float> void apply(typename std::conditional<bl==true, PLEGMA_Vector<Float>,  PLEGMA_Propagator<Float>>::type &out, typename std::conditional<bl==true, PLEGMA_Vector<Float>,  PLEGMA_Propagator<Float>>::type &in, QudaMassNormalization normType = QUDA_KAPPA_NORMALIZATION); // Default it is without any normalization
    template<APP_TYPE type, typename Float> void apply(PLEGMA_Vector<Float> &out, PLEGMA_Vector<Float> &in, QudaMassNormalization normType = QUDA_KAPPA_NORMALIZATION); // Default it is without any normalization
    template<APP_TYPE type, typename Float> void apply(PLEGMA_Propagator<Float> &out, PLEGMA_Propagator<Float> &in, QudaMassNormalization normType = QUDA_KAPPA_NORMALIZATION); // Default it is without any normalization

    template<APP_TYPE type, typename Float> void apply(Float *dout, Float *din, QudaMassNormalization normType = QUDA_KAPPA_NORMALIZATION); // Default is without any normalization // Note that dout and din are device pointers
  };
}
