#include <PLEGMA.h>
#include <invert_quda.h>

#ifndef _QUDA_SOLVER_H
#define _QUDA_SOLVER_H

namespace quda {
  ////////////////////////
  // CLASS: QUDA_solver //
  ////////////////////////
  
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
    DiracM *M, *MSloppy, *MPre;
  
  public:
    QUDA_solver(double mu);
    virtual ~QUDA_solver();
    template<typename Float>
    void solve(PLEGMA_Vector<Float> &out, PLEGMA_Vector<Float> &in);
  };
}
#endif
