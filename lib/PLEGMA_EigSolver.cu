#include <PLEGMA_EigSolver.h>
#include <PLEGMA_BLAS.h>
PLEGMA_EigSolver::PLEGMA_EigSolver(EigSolverParams params, QudaDslashType dslashType, bool verbose):verbose(verbose),p(params),
												    h_eigVecs(nullptr),h_eigVals(nullptr),dOp(nullptr),
												    d_in(nullptr),d_out(nullptr),
												    tmp1(nullptr),tmp2(nullptr)
												    {
  if(p.NeV <=0 ){
    printfQuda("Warning: Eigensolver instructed to use NeV=%d, skipping eigenvectors calculation\n",p.NeV);
    return;
  }
  if(p.NkV < p.NeV) errorQuda("The NkV should be larger than NeV");
  field_length = N_SPINS * N_COLS;
  size_per_Vec = GK_totalVolume * field_length;
  size_NeV = p.NeV * size_per_Vec;
  size_NkV = p.NkV * size_per_Vec;
  size_total = size_NeV + size_NkV;
  bytes_per_Vec = size_per_Vec * 2 * sizeof(double);
  bytes_NeV = size_NeV * 2 * sizeof(double);
  bytes_NkV = size_NkV * 2 * sizeof(double);
  bytes_total = size_total * 2 * sizeof(double);
  try{
    h_eigVecs = new double[size_total*2];
    h_eigVals = new double[p.NeV*2];
  }
  catch (std::bad_alloc& err){
    errorQuda(err.what());
  }
  
  //start the diracOp
  dOp = new quda::QUDA_dirac(dslashType);

  d_in = new PLEGMA_Vector<double>(DEVICE);
  d_out = new PLEGMA_Vector<double>(DEVICE);
  tmp1 = new PLEGMA_Vector<double>(DEVICE);
  tmp2 = new PLEGMA_Vector<double>(DEVICE);

  // check the spectrumPart
  if(p.spectrumPart != "SR" && p.spectrumPart != "LR") errorQuda("Allowed values for spectrum part are SR or LR");
  if(verbose) print();
  
  initEigSolver();
  computeEigVecs();
  delete d_in;
  delete d_out;
  delete tmp1;
  delete tmp2;
  computeEigVals();
}

PLEGMA_EigSolver::~PLEGMA_EigSolver(){
  delete[] h_eigVecs;
  delete[] h_eigVals;
  delete dOp;
}

void PLEGMA_EigSolver::applyOperator(double *out, double *in){
  cudaMemcpy(d_in->D_elem(),in,bytes_per_Vec,cudaMemcpyHostToDevice);
  if(!p.isACC) dOp->apply<quda::MdagM>(*d_out,*d_in);
  else{
    double delta,theta;
    double sigma,sigma1,sigma_old;
    std::complex<double> d1(0.,0.),d2(0.,0.),d3(0.,0.);
    double a = p.amin;
    double b = p.amax;
    delta = (b-a)/2.0;
    theta = (b+a)/2.0;
    sigma1 = -delta/theta;
    d1.real(sigma1/delta);
    d2.real(1.0);
    dOp->apply<quda::MdagM>(*d_out,*d_in);
    cuBLAS::scal(size_per_Vec, d1.real(), d_out->D_elem() );
    cuBLAS::axpy(size_per_Vec, reinterpret_cast<double(&)[2]>(d2), d_in->D_elem(), d_out->D_elem());
    if(p.PolyDeg > 1){
      tmp1->copy(*d_in);
      tmp2->copy(*d_out);
      for(int i=2; i <= p.PolyDeg; i++){
	sigma = 1.0/(2.0/sigma1-sigma_old);
	d1.real(2.0*sigma/delta);
	d2.real(-d1.real()*theta);
	d3.real(-sigma*sigma_old);
	dOp->apply<quda::MdagM>(*d_out, *tmp2);
	plegma::axpbypcz(size_per_Vec,reinterpret_cast<double(&)[2]>(d3),tmp1->D_elem(),
			 reinterpret_cast<double(&)[2]>(d2),tmp2->D_elem(),
			 reinterpret_cast<double(&)[2]>(d1),d_out->D_elem());
	tmp1->copy(*tmp2);
	tmp2->copy(*d_out);
	sigma_old=sigma;
      }
    }
  }
  cudaMemcpy(out,d_out->D_elem(),bytes_per_Vec,cudaMemcpyDeviceToHost);
}

void PLEGMA_EigSolver::print(){
  printfQuda("Number of eigenvalues requested: %d\n",p.NeV);
  printfQuda("Size of Krylov space requested: %d\n",p.NkV);
  if(p.isACC) printfQuda("Using Polynomial acceleration with parameters: Degree=%d, amin=%+e, amax=%+e\n",p.PolyDeg,p.amin,p.amax);
  
}
