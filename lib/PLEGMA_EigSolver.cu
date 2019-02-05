#include <PLEGMA_EigSolver.h>

PLEGMA_EigSolver::PLEGMA_EigSolver(EigSolverParams params, QudaDslashType dslashType, bool verbose):verbose(verbose),p(params),
												    h_eigVecs(nullptr),h_eigVals(nullptr),dOp(nullptr)
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
  try{
    h_eigVecs = new double[size_total*2];
    h_eigVals = new double[p.NeV*2];
  }
  catch (std::bad_alloc& err){
    errorQuda(err.what());
  }
  
  //start the diracOp
  dOp = new QUDA_dirac(dslashType);

  din = new PLEGMA_Vector<double>(DEVICE);
  dout = new PLEGMA_Vector<double>(DEVICE);
  tmp1 = new PLEGMA_Vector<double>(DEVICE);
  tmp2 = new PLEGMA_Vector<double>(DEVICE);

  initEigSolver();
  computeEigVecs();
  delete din;
  delete dout;
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
  cudaMemcpy(d_in.D_elem(),in,size_per_Vec*sizeof(double),cudaMemcpyHostToDevice);
  if(!p.isACC) dOp->appy<MdagM>(*d_out,*d_in);
  else{
    double delta,theta;
    double sigma,sigma1,sigma_old;
    std::complex<double> d1(0.,0.),d2(0.,0.),d3(0.,0.);
    double a = p.amin;
    double b = p.amax;
    delta = (b-a)/2.0;
    theta = (b+a)/2.0;
    sigma1 = -delta/theta;
    d1.real() =  sigma1/delta;
    d2.real() =  1.0;
    dOp->appy<MdagM>(*d_out,*d_in);
    cuBLAS::scal(size_per_Vec, d1.real, d_out->D_elem() );
    cuBLAS::axpy(size_per_Vec, (double*) d2, d_in->D_elem(), d_out->D_elem());
    if(p.PolyDeg > 1){
      tmp1->copy(*d_in);
      tmp2->copy(*d_out);
      for(int i=2; i <= p.PolyDeg; i++){
	sigma = 1.0/(2.0/sigma1-sigma_old);
	d1.real = 2.0*sigma/delta;
	d2.real = -d1.real*theta;
	d3.real = -sigma*sigma_old;
	dOp->apply<MdagM>(*d_out, *tmp2);
	plegma::axpbypcz(size_per_Vec,(double*) d3,tmp1->D_elem(), (double*) d2,tmp2->D_elem(),(double*) d1,d_out->D_elem());
	tmp1->copy(*tmp2);
	tmp2->copy(*d_out);
	sigma_old=sigma;
      }
    }
  }
  cudaMemcpy(out,d_out.D_elem(),size_per_Vec*sizeof(double),cudaMemcpyDeviceToHost);
}
