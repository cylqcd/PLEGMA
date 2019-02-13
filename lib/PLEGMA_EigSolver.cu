#include <PLEGMA_EigSolver.h>
#include <PLEGMA_BLAS.h>
#include <algorithm>
using namespace plegma;
using namespace quda;

PLEGMA_EigSolver::PLEGMA_EigSolver(EigSolverParams params, QudaDslashType dslashType, bool verbose):verbose(verbose),p(params),
												    h_eigVecs(nullptr),h_eigVals(nullptr),dOp(nullptr),
												    d_in(nullptr),d_out(nullptr),
												    tmp1(nullptr),tmp2(nullptr)
{
  if(!GK_init_PLEGMA_flag) errorQuda("Initialize PLEGMA first");
    
  if(p.NeV <=0 ){
    printfQuda("Warning: Eigensolver instructed to use NeV=%d, skipping eigenvectors calculation\n",p.NeV);
    return;
  }
  if(p.NkV < p.NeV) errorQuda("The NkV should be larger than NeV");
  field_length = N_SPINS * N_COLS;
  size_per_Vec = GK_localVolume * field_length;
  size_NeV = p.NeV * size_per_Vec;
  size_NkV = p.NkV * size_per_Vec;
  bytes_per_Vec = size_per_Vec * 2 * sizeof(double);
  bytes_NeV = size_NeV * 2 * sizeof(double);
  bytes_NkV = size_NkV * 2 * sizeof(double);
  try{
    h_eigVecs = new double[size_NkV*2];
    h_eigVals = new double[p.NkV*2];
  }
  catch (std::bad_alloc& err){
    errorQuda(err.what());
  }
  
  //start the diracOp
  dOp = new QUDA_dirac(dslashType);

  d_in = new PLEGMA_Vector<double>(DEVICE);
  d_out = new PLEGMA_Vector<double>(DEVICE);
  tmp1 = new PLEGMA_Vector<double>(DEVICE);
  tmp2 = new PLEGMA_Vector<double>(DEVICE);

  // check the spectrumPart
  if(p.spectrumPart != "SR" && p.spectrumPart != "LR") errorQuda("Allowed values for spectrum part are SR or LR");
  if(p.isACC){
    if(p.spectrumPart == "SR") p.spectrumPart = "LR";
    else if(p.spectrumPart == "LR") p.spectrumPart = "SR";
    else errorQuda("Not implemented");
  }
  if(verbose) print();
  
  initEigSolver();
  computeEigVecs();
  computeEigVals();
  for (int j = 0; j < p.NeV; ++j)  mapEvenOddToNormalGPUformat(h_eigVecs+j*size_per_Vec*2,GK_localL);
  delete d_in;
  delete d_out;
  delete tmp1;
  delete tmp2;
  delete dOp;
  delete[] h_eigVals;
}

PLEGMA_EigSolver::~PLEGMA_EigSolver(){
  delete[] h_eigVecs;
}

void PLEGMA_EigSolver::applyOperator(double *out, double *in){
  cudaMemcpy(d_in->D_elem(),in,bytes_per_Vec,cudaMemcpyHostToDevice);
  checkCudaError();
  if(!p.isACC) dOp->apply<MdagM>(*d_out,*d_in);
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
    dOp->apply<MdagM>(*d_out,*d_in);
    cuBLAS::scal(size_per_Vec, d1.real(), d_out->D_elem() );
    cuBLAS::axpy(size_per_Vec, reinterpret_cast<double(&)[2]>(d2), d_in->D_elem(), d_out->D_elem());
    if(p.PolyDeg > 1){
      tmp1->copy(*d_in);
      tmp2->copy(*d_out);
      sigma_old = sigma1;
      for(int i=2; i <= p.PolyDeg; i++){
	sigma = 1.0/(2.0/sigma1-sigma_old);
	d1.real(2.0*sigma/delta);
	d2.real(-d1.real()*theta);
	d3.real(-sigma*sigma_old);
	dOp->apply<MdagM>(*d_out, *tmp2);
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
  checkCudaError();
}

void PLEGMA_EigSolver::print(){
  printfQuda("Number of eigenvalues requested: %d\n",p.NeV);
  printfQuda("Size of Krylov space requested: %d\n",p.NkV);
  if(p.isACC) printfQuda("Using Polynomial acceleration with parameters: Degree=%d, amin=%+e, amax=%+e\n",p.PolyDeg,p.amin,p.amax);
  printfQuda("Part of the spectrum to be computed: %s",p.spectrumPart.c_str());
  if(p.isACC) printfQuda("\n Flipped due to polynomial acceleration\n");
  else printfQuda("\n");
  printfQuda("Tolerance for eigenSolver is %+e\n",p.tol);
  printfQuda("Max number of iterations for eigenSolver is %d\n",p.maxIters);
  printfQuda("The eigensolver logfile is: %s\n",p.logFile.c_str());
}

void PLEGMA_EigSolver::initEigSolver(){
#ifdef HAVE_ARPACK
  int arpack_log_u = 9999;
  if(!p.logFile.empty() && comm_rank() == 0){
    char *tmps = strdup(p.logFile.c_str());
    initlog_(&arpack_log_u, tmps, p.logFile.length());
    free(tmps);
    int msglvl0 = 0;//, msglvl1 = 1, msglvl2 = 2, msglvl3 = 3;
    int msglvl3 = 3;
    pmcinitdebug_(&arpack_log_u, &msglvl3, &msglvl3, &msglvl0, &msglvl3, &msglvl0, &msglvl0, &msglvl3);
  }
#elif HAVE_PRIMME
  errorQuda("Not implemented")
#else
  errorQuda("Not implemented")
#endif
}

void PLEGMA_EigSolver::computeEigVecs(){
  MPI_Fint mpi_comm_f = MPI_Comm_c2f(MPI_COMM_WORLD);
  char *bmat = strdup("I");
  int rvec = 1;
  int lworkl = (3*p.NkV*p.NkV+5*p.NkV)*2;
  std::complex<double> sigma;
  char *howmany = strdup("P");
  char *which_evals = strdup(p.spectrumPart.c_str());
  std::complex<double> * pin=nullptr;
  std::complex<double> * pout=nullptr;
  int *ipntr, *select, *iparam; 
  double *rwork; 
  std::complex<double> *resid, *workd, *workl, *workev;
  try{
    ipntr              = new int[14];
    select             = new int[p.NkV];
    iparam             = new int[11];
    rwork        = new double[p.NkV];
  
    resid  = new std::complex<double>[size_per_Vec];
    workd  = new std::complex<double>[3*size_per_Vec];
    workl  = new std::complex<double>[lworkl];
    workev = new std::complex<double>[2*p.NkV];
  }
  catch (std::bad_alloc& err){
    errorQuda(err.what());
  }
  
  iparam[0] = 1;  //use exact shifts
  iparam[2] = p.maxIters;
  iparam[3] = 1;
  iparam[6] = 1;
  iparam[7] = 1; // arpack mode

  int info = 0; // random initial guess
  int nconv;
  bool checkIdo = true;
  int arpack_iter = 0;
  int ido = 0;

  // start solver
  do{
    pznaupd_(&mpi_comm_f, &ido,bmat, &size_per_Vec, which_evals, 
	     &p.NeV, &p.tol, resid, &p.NkV,
	     (std::complex<double>*)h_eigVecs, &size_per_Vec, iparam, ipntr, workd, 
	     workl, &lworkl,rwork,&info,1,2);
    if(checkIdo){
      pin=workd+ipntr[0]-1;
      pout=workd+ipntr[1]-1;
      checkIdo = false;
    }
    if (ido == 99 || info == 1)break;
    if( (ido==-1) || (ido==1) ) applyOperator((double*)pout, (double*)pin);
    if(verbose)printfQuda("arpack iter = %d\n", arpack_iter);
    arpack_iter++;
  }while(ido != 99);
    
  if(info < 0) errorQuda("EigSolver error: info = %d\n",info);
  nconv = iparam[4];
  if(nconv != p.NeV) errorQuda("Converged number of eigvalues %d != %d\n", nconv, p.NeV);
  if(verbose) printfQuda("EigSolver: Number of converged eigenvalues: %d\n", nconv);

  // compute eigenvectors
  pzneupd_(&mpi_comm_f,&rvec,howmany, select, (std::complex<double>*) h_eigVals,(std::complex<double>*) h_eigVecs, &size_per_Vec,&sigma, 
	   workev,bmat,&size_per_Vec,which_evals,&p.NeV,&p.tol, resid,&p.NkV, 
	   (std::complex<double>*) h_eigVecs,&size_per_Vec,iparam,ipntr,workd,workl,&lworkl,rwork,&info,1,1,2);

  if(info == 1) printfQuda("Warning: Maximum number of iterations reached.\n");
  if(info == 3) errorQuda("No shifts could be applied during implicit, Arnoldi update, try increasing NkV\n");
  int arpack_log_u = 9999;
  if(!p.logFile.empty() && comm_rank() == 0)finilog_(&arpack_log_u);

  free(bmat);
  free(howmany);
  free(which_evals);
  delete[] ipntr;
  delete[] select;
  delete[] iparam;
  delete[] rwork;
  delete[] resid;
  delete[] workd;
  delete[] workl;
  delete[] workev;
}

void PLEGMA_EigSolver::computeEigVals(){
  for(int j = 0 ; j < p.NeV; j++){
    double one[2] = {1.,0.};
    cudaMemcpy(tmp1->D_elem(),h_eigVecs+j*size_per_Vec*2,bytes_per_Vec,cudaMemcpyHostToDevice);
    checkCudaError();
    dOp->apply<MdagM>(*tmp2,*tmp1);
    std::complex<double> eval;
    std::complex<double> res;
    cuBLAS::dot(reinterpret_cast<double(&)[2]>(eval), size_per_Vec, tmp1->D_elem(), tmp2->D_elem(), MPI_COMM_WORLD);
    cuBLAS::scal(size_per_Vec,-eval.real(),tmp1->D_elem());
    cuBLAS::axpy(size_per_Vec,one,tmp2->D_elem(),tmp1->D_elem());
    cuBLAS::dot(reinterpret_cast<double(&)[2]>(res), size_per_Vec, tmp1->D_elem(), tmp1->D_elem(), MPI_COMM_WORLD);
    evalsOrdered.push_back(std::make_tuple(eval.real(), eval.imag(), std::sqrt(res.real()), j));
  }
  std::sort(evalsOrdered.begin(), evalsOrdered.end());
  if(verbose)
    for (int j = 0; j < p.NeV; ++j)
      printfQuda("Eval[%04d] = (%+e,%+e), Residual: %+e, Order Index: %d\n", j, std::get<0>(evalsOrdered[j]), std::get<1>(evalsOrdered[j]),
		 std::get<2>(evalsOrdered[j]), std::get<3>(evalsOrdered[j]));
}

// vecOut = (1 - U * U^\dag) vecIn
void PLEGMA_EigSolver::projectVector(PLEGMA_Vector<double> &vecOut, PLEGMA_Vector<double> &vecIn){
  if(p.NeV <= 0){
    if(verbose) printfQuda("Skipping deflation of source vector since NeV=%d\n",p.NeV);
    vecOut.copy(vecIn);
    return;
  }
  if(!vecOut.IsAllocHost() || !vecIn.IsAllocHost()) errorQuda("This functions needs both vecs to have also Host allocation");
  vecIn.unload();
  double aP[2]={1.,0.}, b[2]={0.,0.}, aM[2]={-1.,0.};
  double *tmpArr = nullptr;
  try { tmpArr = new double[p.NeV*2]; } catch (std::bad_alloc &err) { errorQuda(err.what());}
  memset(tmpArr,0,p.NeV*2*sizeof(double));
  cBLAS::gemv(DAGGER, size_per_Vec, p.NeV, aP, h_eigVecs, vecIn.H_elem(), b, tmpArr, MPI_COMM_WORLD);
  cBLAS::gemv(NOTRANS, size_per_Vec, p.NeV, aM, h_eigVecs, tmpArr, b, vecOut.H_elem());
  cBLAS::axpy(size_per_Vec, aP, vecIn.H_elem(), vecOut.H_elem());
  vecOut.load();
  delete[] tmpArr;
}

void PLEGMA_EigSolver::dumpEvalsVdagG5V(std::string filename){
  PLEGMA_Vector<double> g5V(DEVICE);
  PLEGMA_Vector<double> V(DEVICE);
  std::vector<double> VdagG5V;
  for (int j = 0; j < p.NeV; ++j) {
    cudaMemcpy(g5V.D_elem(),h_eigVecs+j*size_per_Vec*2,bytes_per_Vec,cudaMemcpyHostToDevice);
    checkCudaError();
    V.copy(g5V);
    g5V.apply_gamma(G5);
    std::complex<double> res;
    cuBLAS::dot(reinterpret_cast<double(&)[2]>(res), size_per_Vec, V.D_elem(), g5V.D_elem(),MPI_COMM_WORLD);
    VdagG5V.push_back(res.real());
  }
  if(comm_rank() == 0){
    FILE *ptr = fopen(filename.c_str(), "w");
    if(ptr == NULL) errorQuda("Cannot open file:%s for writting\n",filename.c_str());
    for (int i = 0; i < p.NeV; ++i) fprintf(ptr, "%+e %+e\n", std::get<0>(evalsOrdered[i]), VdagG5V[std::get<3>(evalsOrdered[i])]);
    fclose(ptr);
  }
}

