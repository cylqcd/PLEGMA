#include <PLEGMA.h>
#ifndef PLEGMA_EIGSOLVER_H
#define PLEGMA_EIGSOLVER_H

#if defined(ARPACK_EIGSOLVER) && defined(PRIMME_EIGSOLVER)
#error Cannot have both ARPACK and PRIMME
#endif

#if defined(PRIMME_EIGSOLVER)
#include <primme.h>
#elif defined(ARPACK_EIGSOLVER)

#else
#error Neither PRIMME nor ARPACK have been defined
#endif

class  PLEGMA_EigSolverParams{
 protected:
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

class PLEGMA_EigSolver: private PLEGMA_EigSolverParams{
 private:
  void applyOperator();
  void initEigSolver();
  void computeEigVectors();
  void computeEigValues();
  
 public:
  PLEGMA_EigSolver(PLEGMA_EigSolverParams params, bool verbose);
  virtual ~PLEGMA_EigSolver();
  void projectVector(QKXTM_Vector<double> &vecOut, QKXTM_Vector<double> &vecIn);
};
#endif /* PLEGMA_EIGSOLVER_H */
