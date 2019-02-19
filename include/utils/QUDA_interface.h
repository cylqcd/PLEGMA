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
    DiracM *M, *MSloppy, *MPre;
    cudaColorSpinorField *b, *x;
    
  public:
    QudaInvertParam getInvParams() const{return inv_param;}
    QUDA_solver(double mu);
    virtual ~QUDA_solver();
    cudaColorSpinorField* solve(cudaColorSpinorField * rhs);
    template<typename Float>
    cudaColorSpinorField* solve(PLEGMA_Vector<Float> &vectorIn);
    template<typename Float>
    void solve(PLEGMA_Vector<Float> &out, PLEGMA_Vector<Float> &in);
  };

  enum APP_TYPE {M,Mdag,MdagM,MMdag};
  class QUDA_dirac {
  private:
    DiracParam dParam;
    QudaInvertParam inv_param;
    Dirac *D;
    cudaColorSpinorField *in, *out;
    template<APP_TYPE type> void apply();
  public:
     //only QUDA_WILSON_DSLASH, QUDA_CLOVER_WILSON_DSLASH, QUDA_TWISTED_MASS_DSLASH, QUDA_TWISTED_CLOVER_DSLASH
    QUDA_dirac(QudaDslashType dslashType);
    virtual ~QUDA_dirac();
    void print(){dParam.print();}
    void switchMu(double mu);
    void switchKappa(double kappa);
    template<APP_TYPE type, typename Float> void apply(PLEGMA_Vector<Float> &out, PLEGMA_Vector<Float> &in, QudaMassNormalization normType = QUDA_KAPPA_NORMALIZATION); // Default it is without any normalization
    template<APP_TYPE type, typename Float> void apply(Float *dout, Float *din, QudaMassNormalization normType = QUDA_KAPPA_NORMALIZATION); // Default is without any normalization // Note that dout and din are device pointers
  };
}

template<typename FloatOut, typename FloatIn>
static inline void unpackGaugeToEvenOdd(FloatOut *buf[4], PLEGMA_Gauge<FloatIn> &gauge)
{
  int VOLUME=gauge.Total_length();
  int VOLUMEh = VOLUME / 2;
  int gSize = N_COLS*N_COLS;
 
  for(int even = 0; even < VOLUMEh; even++) {
    int odd = even+VOLUMEh;
    int norm_coord = 2 * even;
    
    int evenSiteBit = 0;
    int tmp = norm_coord/dims[0];
    for(int i=1; i<N_DIMS; i++) {
      evenSiteBit += tmp%dims[i];
      tmp /= dims[i];
    }
    evenSiteBit = evenSiteBit % 2;
    int oddSiteBit  = evenSiteBit ^ 1;
    
    for(int dir = 0 ; dir < N_DIMS; dir++)
      for(int c = 0; c < gSize; c++) {
	buf[dir][(even*gSize + c)*2 + 0] = gauge.H_elem()[((dir*gSize+c)*VOLUME + norm_coord + evenSiteBit)*2  +0];
	buf[dir][(even*gSize + c)*2 + 1] = gauge.H_elem()[((dir*gSize+c)*VOLUME + norm_coord + evenSiteBit)*2  +1];
	buf[dir][(odd*gSize + c)*2 + 0] = gauge.H_elem()[((dir*gSize+c)*VOLUME + norm_coord + oddSiteBit)*2  +0];
	buf[dir][(odd*gSize + c)*2 + 1] = gauge.H_elem()[((dir*gSize+c)*VOLUME + norm_coord + oddSiteBit)*2  +1];
      }
  }
}

template<typename Float>
static inline void applyAntiperiodicBoundary(Float **buf)
{
  // only apply T-boundary at edge nodes
#ifdef MULTI_GPU
  bool last_node_in_t = (commCoords(3) == commDim(3)-1) ? true : false;
#else
  bool last_node_in_t = true;
#endif

  // Apply boundary conditions to temporal links
  if (last_node_in_t) {
    int gSize = N_COLS*N_COLS*2;
    size_t Vh = dims[0]*dims[1]*dims[2]*dims[3]/2;
    for (int j = Vh-dims[0]*dims[1]*dims[2]/2; j < Vh; j++) {
      for (int i = 0; i < gSize; i++) {
	buf[3][j*gSize+i] *= -1.0;
	buf[3][(Vh+j)*gSize+i] *= -1.0;
      }
    }
  }
}
