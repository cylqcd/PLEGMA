#include <PLEGMA.h>
#include <PLEGMA_utils.h>
#include <complex>

using namespace plegma;

static std::vector<std::string> listOpt = {"verbosity", "load-gauge-list-filename","nsmear-stout","alpha-stout", 
					   "nsrc","src-filename"}; // this will be used for the momenta

inline std::complex<double> middlePointPhase(int mu, std::vector<double> mom){
  double arg = (PI/HGC_totalL[mu]) * mom[mu];
  return (std::complex<double>) {cos(arg),-sin(arg)};
}

static void computeGprop(std::vector<std::vector<std::complex<double>>> &AxA, PLEGMA_FT<double> &ftAl, PLEGMA_FT<double> &ftAr,std::vector<double> mom){
  std::vector<std::complex<double>> tmp(N_DIMS*N_DIMS*N_COLS*N_COLS,(std::complex<double>){0.,0.});
  std::complex<double> *Al = (std::complex<double> *) ftAl.H_elem();
  std::complex<double> *Ar = (std::complex<double> *) ftAr.H_elem();
  
  for(int mu = 0; mu < N_DIMS; mu++){
    std::complex<double> phl = middlePointPhase(mu,mom);
    for(int nu = 0; nu < N_DIMS; nu++){
      std::complex<double> phr = std::conj(middlePointPhase(mu,mom));
      for(int c1 = 0; c1 < N_COLS; c1++)
	for(int c2 = 0; c2 < N_COLS; c2++){
	  for(int cc = 0; cc < N_COLS; cc++)
	    tmp[((mu*N_DIMS+nu)*N_COLS+c1)*N_COLS+c2] += phl * Al[(mu*N_COLS+c1)*N_COLS+cc] * Ar[(nu*N_COLS+cc)*N_COLS+c2] * phr;
	  tmp[((mu*N_DIMS+nu)*N_COLS+c1)*N_COLS+c2]  /= HGC_totalVolume;
	}
    }
  }
  AxA.push_back(tmp);
}

int main(int argc, char **argv){
  initializeOptions(argc, argv, false, listOpt);
  //==========================//
  std::string filesPrefix="./";
  HGC_options->set("output-path", "Path to the directory to dump results", verbosity, filesPrefix);
  std::string filesSuffix="r0";
  HGC_options->set("suffix", "Suffix of data filename to distiguish replicas", verbosity, filesSuffix);
  double overelaxPar = 0.2;
  HGC_options->set("overelax-param", "The value of the parameter will be used for the overelaxation",verbosity,overelaxPar);
  bool isGFixed = false;
  HGC_options->set("isGFixed", "If this is true it means that the configuration provided is alread gauge fixed", verbosity,isGFixed);
  bool doGLoops = true;
  HGC_options->set("doGLoops", "If you want to compute also gluon loops", verbosity,doGLoops);
  //==========================//
  initializePLEGMA();

  
  if(numSourcePositions <=0) PLEGMA_error("No momenta have provided");
  PLEGMA_Gauge<double> gauge1,gauge2;
  PLEGMA_Field<double> trace1(BOTH,SCALAR),trace2(BOTH,SCALAR);
  PLEGMA_FT<double> ftUL1(0,4);
  PLEGMA_FT<double> ftUL2(0,4);
  std::vector<double> gLoop;
  std::vector<int> nScount;
  for(int n=0; n<=nsmearStout;n++) nScount.push_back(n);
  std::vector<std::vector<std::complex<double>>> AxA;
  PLEGMA_FT<double> *ftAl=nullptr, *ftAr=nullptr;
  std::vector<double> twistF = {0.,0.,0.,0.}; // do not put any twist in the momentum
  std::string filenameGprop = filesPrefix + "/gProps_" + filesSuffix + ".txt";
  std::string filenameGLoops = filesPrefix + "/gLoops_" + filesSuffix +".txt";
  if(doGLoops) if(comm_rank() == 0) cleanFile(filenameGprop);
  if(comm_rank() == 0) cleanFile(filenameGLoops);
  std::vector<int> muVec,nuVec, c1Vec, c2Vec;
  for(int mu = 0; mu < N_DIMS; mu++)
    for(int nu = 0; nu < N_DIMS; nu++)
      for(int c1 = 0; c1 < N_COLS; c1++)
  	for(int c2 = 0; c2 < N_COLS; c2++){
  	  muVec.push_back(mu); nuVec.push_back(nu); c1Vec.push_back(c1); c2Vec.push_back(c2);
  	}

  if(HGC_verbosity > 1) PLEGMA_printf("Will work on %d confs",listGaugeConfs.size());
  for(int iconf=0; iconf < listGaugeConfs.size(); iconf++){
    double t1=MPI_Wtime();
    std::string confStr=basename(listGaugeConfs[iconf],'.');
    gauge1.readFile(listGaugeConfs[iconf], LIME_FORMAT);
    PLEGMA_printf("Unsmeared Plaquette is: ");
    gauge1.calculatePlaq();
    if(!isGFixed){
      double t3=MPI_Wtime();
      gauge2.gFixingLandau(gauge1,overelaxPar);
      double t4=MPI_Wtime();
      PLEGMA_printf("Gauge fixing completed in %f secs\n",t4-t3);
      gauge1.copy(gauge2);
      gauge2.gluonField(gauge1);
    }
    else{
      gauge2.gluonField(gauge1);
    }
    
    // Here we need to compute the gluon propagator which is located at gauge2 
    //std::vector<double> twistF = {0,0,0,0}; // twist in the temporal direction
    AxA.clear();    
    for(int im=0; im < numSourcePositions; im++){
      std::vector<double> mom=(std::vector<double>) {sourcePositions[im][0]+twistF[0],
  						     sourcePositions[im][1]+twistF[1],
  						     sourcePositions[im][2]+twistF[2],
  						     sourcePositions[im][3]+twistF[3]};
      std::vector<int> momVecX(N_DIMS*N_DIMS*N_COLS*N_COLS,mom[0]);
      std::vector<int> momVecY(N_DIMS*N_DIMS*N_COLS*N_COLS,mom[1]);
      std::vector<int> momVecZ(N_DIMS*N_DIMS*N_COLS*N_COLS,mom[2]);
      std::vector<int> momVecT(N_DIMS*N_DIMS*N_COLS*N_COLS,mom[3]);
      std::vector<std::string> confVec(N_DIMS*N_DIMS*N_COLS*N_COLS,confStr);
      ftAl = new PLEGMA_FT<double>(mom,4,false);
      ftAr = new PLEGMA_FT<double>(mom,4,false);
      ftAl->apply(gauge2,FT_GEMV,-1); // remember to put the twist in the temporal direction
      ftAr->apply(gauge2,FT_GEMV,+1); // remember to put the twist in the temporal direction
      computeGprop(AxA,*ftAl,*ftAr,mom);

      if(comm_rank() == 0) write_std_vecs(filenameGprop,true,confVec,momVecX,momVecY,momVecZ,momVecT,muVec,nuVec,c1Vec,c2Vec,AxA[im]);
      delete ftAl,ftAr;
    }
    // Here we compute the gluon loops
    /*
     * Plaquete definition of the gluon loops
     * Definition \mathcal{O} = \frac{-4*\beta}{9} Re{ \Tr[ \sum_i P_{3i} - sum_{i<j} P_{ij} ]}
     * We will not include the factor \frac{-4*\beta}{9} now
     * Term1 = \Tr[ \sum_i P_{i3} ]
     * Term2 = \Tr[sum_{i<j} P_{ij}]
     */
    if(doGLoops){
      // gauge1 holds the link variables in landau gauge (gluon field is not needed anymore and will be used for tmp)
      gLoop.clear();
      for(int n=0; n<=nsmearStout;n++){
	if(n%2 == 0){
	  if(n==0) gauge2.copy(gauge1);
	  else gauge2.stoutSmearing(gauge1,1,alphaStout,4);
	}
	else{
	  gauge1.stoutSmearing(gauge2,1,alphaStout,4);
	}

	trace2.zero_device();
	for(int i = 0 ; i < N_DIMS-1; i++){
	  trace1.trPmunu((n%2==0)?gauge1:gauge2, std::make_pair(3,i));
	  trace2.axpy(trace1,(std::complex<double>) {1.,0.});
	}
	ftUL1.apply(trace2,FT_GEMV);

	trace2.zero_device();
	for(int i = 0 ; i < N_DIMS-1; i++)
	  for(int j = i+1 ; j < N_DIMS-1; j++){
	    trace1.trPmunu((n%2==0)?gauge1:gauge2, std::make_pair(i,j));
	    trace2.axpy(trace1,(std::complex<double>) {1.,0.});
	  }
	ftUL2.apply(trace2,FT_GEMV);
	gLoop.push_back(ftUL1.H_elem()[0] - ftUL2.H_elem()[0]);
      }
      std::vector<std::string> confVec(nsmearStout+1,confStr);
      if(comm_rank() == 0) write_std_vecs(filenameGLoops,true,confVec,nScount,gLoop);
      double t2=MPI_Wtime();
      PLEGMA_printf("conf.%s completed in %f secs\n",confStr.c_str(),t2-t1);
    }
  }  
  // Vertex function will be create later from a multiplication of the gluon Loop with the gluon propagator
  
  finalize();
}
