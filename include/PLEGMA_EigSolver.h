#include <PLEGMA_utils.h>
#ifndef PLEGMA_EIGSOLVER_H
#define PLEGMA_EIGSOLVER_H

#if defined(ARPACK_EIGSOLVER) && defined(PRIMME_EIGSOLVER)
#error Cannot have both ARPACK and PRIMME
#endif

#if defined(PRIMME_EIGSOLVER)
#include <primme.h>
#elif defined(ARPACK_EIGSOLVER)
extern "C"{
  extern int initlog_(int*, char*, int);
  extern int finilog_(int*);
  extern int pznaupd_(int *comm, int *ido, char *bmat, int *n, char *which, int *nev, double *tol,
			    std::complex<double> *resid, int *ncv, std::complex<double> *v, int *ldv, 
			    int *iparam, int *ipntr, std::complex<double> *workd, std::complex<double> *workl, 
			    int *lworkl, double *rwork, int *info, int bmat_size, int which_size );
  extern int pzneupd_(int *comm, int *comp_evecs, char *howmany, int *select, std::complex<double> *evals, 
			    std::complex<double> *v, int *ldv, std::complex<double> *sigma, std::complex<double> *workev, 
			    char *bmat, int *n, char *which, int *nev, double *tol, std::complex<double> *resid, 
			    int *ncv, std::complex<double> *v1, int *ldv1, int *iparam, int *ipntr, 
			    std::complex<double> *workd, std::complex<double> *workl, int *lworkl, double *rwork, int *info,
			    int howmany_size, int bmat_size, int which_size);
  extern int pmcinitdebug_(int*,int*,int*,int*,int*,int*,int*,int*);
}
#else
#error Neither PRIMME nor ARPACK have been defined
#endif

class  EigSolverParams{
  int NeV; // total number of eigenvalues & eigenvectors
  int NkV; // Krylov space size > NeV

  bool isACC; // In case we want to use Polymonial accelarator
  int PolyDeg; // Order of the Polynomial
  double amin; // Low boundary for polymonial accelator
  double amax; // High boundary for polynomial accelator
#if defined(ARPACK_EIGSOLVER)
  std::string spectrumPart; // available options for arpack are (SR,LR,SM,LM,SI,LI)
  double tol;
  int maxIters;
  int mode; 
#elif defined(PRIMME_EIGSOLVER)
  
#else
#endif
};

class PLEGMA_EigSolver{
 private:
  EigSolverParams p;
  bool verbose;
  int field_length;
  size_t size_per_Vec;
  size_t size_NeV;
  size_t size_NkV;
  size_t size_total;

  double *h_eigVecs;
  double *h_eigVals;

  QUDA_dirac *dOp;

  PLEGMA_Vector<double> *din;
  PLEGMA_Vector<double> *dout;
  PLEGMA_Vector<double> *tmp1;
  PLEGMA_Vector<double> *tmp2;

  void applyPolyOperator(double *out, double *in);
  void initEigSolver();
  void computeEigVecs();
  void computeEigVals();
  
 public:
  PLEGMA_EigSolver(EigSolverParams params, QudaDslashType dslashType, bool verbose=false);
  virtual ~PLEGMA_EigSolver();
  void projectVector(QKXTM_Vector<double> &vecOut, QKXTM_Vector<double> &vecIn);
};
#endif /* PLEGMA_EIGSOLVER_H */
