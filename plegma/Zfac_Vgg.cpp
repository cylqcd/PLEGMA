#include <PLEGMA.h>
#include <PLEGMA_utils.h>
#include <complex>

using namespace plegma;

static std::vector<std::string> listOpt = {"verbosity", "load-gauge-list-filename","nsrc","src-filename"}; // this will be used for the momenta

inline std::vector<std::vector<double>> readMomList(std::string filename){
  std::vector<std::vector<double>> momList;
  std::ifstream file(filename,std::ifstream::in);
  if(file.fail()) PLEGMA_error("Cannot open file to read momentum list: %s\n",filename.c_str());
  if(file.peek() == std::ifstream::traits_type::eof()) return momList;
  std::string str;
  int counter=0;
  std::vector<double> a(4);
  while(file >> str){
    double val=atof(str.c_str());
    int ii=counter%4;
    a[ii]=val;
    if(ii==3)momList.push_back(a);
    counter++;
  }
  file.close();
  return momList;
}


static void computeGprop(std::vector<std::vector<double>> &AxA, PLEGMA_FT<double> &ftAl, PLEGMA_FT<double> &ftAr,std::vector<double> mom){
  std::vector<double> tmp(N_DIMS*(N_DIMS+1)/2,0.);
  std::complex<double> *Al = (std::complex<double> *) ftAl.H_elem();
  std::complex<double> *Ar = (std::complex<double> *) ftAr.H_elem();  
  for(int mu = 0; mu < N_DIMS; mu++){
    for(int nu = mu; nu < N_DIMS; nu++){
     for(int c1 = 0; c1 < N_COLS; c1++)
      for(int c2 = 0; c2 < N_COLS; c2++)
	tmp[(2*N_DIMS-mu-1)*mu/2+nu] += (Al[(mu*N_COLS+c1)*N_COLS+c2] * Ar[(nu*N_COLS+c2)*N_COLS+c1]).real();
     tmp[(2*N_DIMS-mu-1)*mu/2+nu]  /= HGC_totalVolume;   
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
  std::string filenameMomList="./momList.txt";
  HGC_options->set("filenameMomList", "Path where to find momenta list", verbosity, filenameMomList);
  double tolerance = 1e-08;
  HGC_options->set("tolerance", "Tolerance to use for the gauge fixing procedure", verbosity, tolerance);
  double stochoverelaxPar = 0.2;
  HGC_options->set("stochoverelax-param", "The value of this parameter will be used for the stochastic overelaxation (PLEGMA)",verbosity,stochoverelaxPar);
  double overelaxPar = 1.5;
  HGC_options->set("overelax-param", "The value of the parameter will be used for the exact overelaxation (QUDA)",verbosity,overelaxPar);
  std::string overelaxType = "exact";
  HGC_options->set("overelaxType", "Choose between exact overrelaxation and stochastic, options (stoch,exact)",verbosity,overelaxType);    
  bool isGFixed = false;
  HGC_options->set("isGFixed", "If this is true it means that the configuration provided is already gauge fixed", verbosity,isGFixed);
  bool doGLoops = true;
  HGC_options->set("doGLoops", "If you want to compute also gluon loops", verbosity,doGLoops);
  bool doGprop = true;
  HGC_options->set("doGprop", "If we want to compute the gluon propagator, default true", verbosity,doGprop);
  //  bool doSmearGprop = true;
  //  HGC_options->set("doSmearGprop", "If we want to smear the gluon propagator (only stout for now)", verbosity,doSmearGprop);
  //  int nsmearStoutGprop = 10;
  // double alphaStoutGprop = 0.129;
  // HGC_options->set("nsmear-stout-Gprop", "Number of stout smearing step for Gprop",verbosity,nsmearStoutGprop);
  // HGC_options->set("alpha-stout-Gprop", "Coefficient for the stout smearing for Gprop",verbosity,alphaStoutGprop);
  int nsmearStout = 10;
  double alphaStout = 0.129;
  HGC_options->set("nsmear-stout", "Number of stout smearing step for GLoops and Gprop",verbosity,nsmearStout);
  HGC_options->set("alpha-stout", "Coefficient for the stout smearing for GLoops and Gprop",verbosity,alphaStout);
  std::string stoutOrWF="stout";
  HGC_options->set("stoutOrWF", "Choose what smearing to do in the operator and gluon propagator, either stout or Wilson Flow. Options (stout,WF)",verbosity,stoutOrWF);
  
  //==========================//
  
  initializePLEGMA();  
  PLEGMA_Gauge<double> gauge1,gauge2,gauge3;
  PLEGMA_FT<double> *ftAl=nullptr, *ftAr=nullptr;
  std::vector<std::vector<double>> AxA; // gluon propagator
  PLEGMA_Field<double> trace1(BOTH,SCALAR),trace2(BOTH,SCALAR),trace3(BOTH,SCALAR);
  PLEGMA_FT<double> ftUL1(0,4);  
  std::vector<double> gLoopPlt_diag; // gluon loops diagonals with Plaquette definition
  std::vector<double> gLoopFST_diag; // gluon loops diagonals with clover definition
  std::vector<double> gLoopFST_off; // gluon loops off-diagonals with clover definition
  PLEGMA_Fmunu<double> fmunu;
  PLEGMA_Su3field<double> one3x3;
  one3x3.setUnit((std::vector<int>) {0,4,8});

  // Filenames for Gprop:
  std::vector<std::string> filenameGprop;
  std::vector<std::vector<double>> momList = readMomList(filenameMomList);
  for(int im=0; im < momList.size(); im++){
    std::string momStr = "_px" + convNumToStr(momList[im][0],1) +
      "_py" + convNumToStr(momList[im][1],1) +
      "_pz" + convNumToStr(momList[im][2],1) +
      "_pt" + convNumToStr(momList[im][3],1)+"_"; 
    filenameGprop.push_back(filesPrefix + "/gProps" + momStr + filesSuffix + ".txt");
    if(comm_rank()== 0) cleanFile(filenameGprop[im]);
  }

  // Filenames for GLoops:
  std::string filenameGLoopPlt_diag = filesPrefix + "/gLoopPlt_diag_" + filesSuffix +".txt";
  std::string filenameGLoopFST_diag = filesPrefix + "/gLoopFST_diag_" + filesSuffix +".txt";
  std::string filenameGLoopFST_off = filesPrefix + "/gLoopFST_off_" + filesSuffix +".txt";  
  if(comm_rank() == 0) {
    cleanFile(filenameGLoopPlt_diag);
    cleanFile(filenameGLoopFST_diag);
    cleanFile(filenameGLoopFST_off);
  }

  // Vectors with the Lorentz indices for the Gluon propagator: 
  std::vector<int> muVec;
  std::vector<int> nuVec;
  if(doGprop){ 
    for(int mu = 0; mu < N_DIMS; mu++){
      for(int nu = mu; nu < N_DIMS; nu++){
	muVec.push_back(mu);
	nuVec.push_back(nu);
      }
    }
  }

  // Vectors with the Lorentz indices for the Gluon loops:   
  std::vector<int> muVec_diag;
  std::vector<int> muVec_off;
  std::vector<int> nuVec_off;
  if(doGLoops){
    for(int mu = 0; mu < N_DIMS; mu++) muVec_diag.push_back(mu);
    for(int mu = 0; mu < N_DIMS-1; mu++){
      for(int nu = mu+1; nu < N_DIMS; nu++){
	muVec_off.push_back(mu);
	nuVec_off.push_back(nu);
      }
    }
  }
  
  if(stoutOrWF != "stout" && stoutOrWF != "WF") PLEGMA_error("Either Stout or Wilson flow are needed");
  int nSteps = (stoutOrWF == "stout")?nsmearStout:0; // number of smearing steps

  if(HGC_verbosity > 1) PLEGMA_printf("Will work on %d confs",listGaugeConfs.size());  
  for(int iconf=0; iconf < listGaugeConfs.size(); iconf++){
     double t1=MPI_Wtime();
     std::string confStr=splitStrFwd(listGaugeConfs[iconf],'.');
     gauge1.readFile(listGaugeConfs[iconf], LIME_FORMAT);
     PLEGMA_printf("Unsmeared Plaquette is: ");
     gauge1.calculatePlaq();
     
     if(!isGFixed){
       double t3=MPI_Wtime();
       if(overelaxType == "exact") gFixingLandauOVR_QUDA(gauge2,gauge1,4,overelaxPar,tolerance,10000,10000);
       else if (overelaxType == "stoch") gauge2.gFixingLandau(gauge1,stochoverelaxPar,tolerance);
       else PLEGMA_error("Overrelaxation type %s not implemented",overelaxType.c_str());
       double t4=MPI_Wtime();
       PLEGMA_printf("Gauge fixing completed in %f secs\n",t4-t3);
       gauge1.copy(gauge2);
     }
     
     for(int n=0; n<=nSteps;n++){
       if(stoutOrWF == "stout"){
	 if(n%2 == 0){
	   if(n==0) gauge2.copy(gauge1);
	   else gauge2.stoutSmearing(gauge1,1,alphaStout,4);
	 }
	 else{
	   gauge1.stoutSmearing(gauge2,1,alphaStout,4);
	 }
       }
       else{
	 gauge2.copy(gauge1);
	 gauge2.applyGradientFlow(gauge1,100,0.01); // fixed for now Nsteps*epsilon=1
       }

       if(doGprop){
	 AxA.clear();
	 gauge3.gluonField((n%2==0)?gauge2:gauge1);
	 
	 // Vectors for conf. numbers and smearing steps for the gluon propagator
	 std::vector<std::string> confVecGprop(N_DIMS*(N_DIMS+1)/2,confStr);
	 std::vector<int> nScountGprop(N_DIMS*(N_DIMS+1)/2,n);
	
	 for(int im=0; im < momList.size(); im++){
	   std::vector<double> mom=(std::vector<double>) {momList[im][0],
							  momList[im][1],
							  momList[im][2],
							  momList[im][3]};
	   ftAl = new PLEGMA_FT<double>(mom,4,false);
	   ftAr = new PLEGMA_FT<double>(mom,4,false);
	   ftAl->apply(gauge3,FT_GEMV,-1); // remember to put the twist in the temporal direction inside the momentum list
	   ftAr->apply(gauge3,FT_GEMV,+1); // remember to put the twist in the temporal direction inside the momentum list
	   computeGprop(AxA,*ftAl,*ftAr,mom); // there is a missing factor of beta/6
	   if(comm_rank() == 0) write_std_vecs(filenameGprop[im],true,confVecGprop,nScountGprop,muVec,nuVec,AxA[im]);
	   delete ftAl,ftAr;
	 }
       }

       if(doGLoops){
	 
	 gLoopPlt_diag.clear();
	 gLoopFST_diag.clear();
	 gLoopFST_off.clear();
	 
	 // Vectors for conf. numbers and smearing steps for the gluon loops:
	 std::vector<std::string> confVecGLoops_diag(N_DIMS,confStr);
	 std::vector<int> nScountGLoops_diag(N_DIMS,n);
	 std::vector<std::string> confVecGLoops_off(N_DIMS*(N_DIMS-1)/2,confStr);
	 std::vector<int> nScountGLoops_off(N_DIMS*(N_DIMS-1)/2,n);

	 
	 // We use two different lattice definitions of the gluon EMT operator:
	 
	 /* (a) Plaquette definition (ONLY for the diagonal case):
	  *     ===================================================
	  * T^g_{mu mu} = -beta/3*[sum_{rho != mu}     tr(Re(P_{mu rho})) -
	  *                        sum_{sigma<rho,     tr(Re(P_{sigma rho}))]
          *                             sigma != mu, 
          *                             rho != mu   } 
	  *
	  * In what follows, the factor (-beta/3) is missing!             
	  */

	 /* (b) Clover definition (for both diagonal and nondiagonal cases):
	  *     ============================================================
	  * Diagonal:
	  * ---------
          * T^g_{mu mu} = -beta/384*[sum_{rho != mu}   tr(F_{mu rho}^2) -
	  *                        sum_{sigma<rho,     tr(F_{sigma rho}^2)]
          *                             sigma != mu, 
          *                             rho != mu   } 
	  *
	  * In what follows, a factor of (beta/6) is missing!            
	  *
	  * Nondiagonal: (mu != nu)
	  * ------------------------
	  * T^g_{mu nu} = -beta/192*[sum_{rho != mu,  tr(F_{mu rho} F_{nu rho})]
          *                               rho != nu}
	  *
	  * In what follows, a factor of (beta/3) is missing!
	  */

	 fmunu.compute_leaves((n%2==0)?gauge2:gauge1);
	 for(int mu = 0; mu < N_DIMS; mu++){
	   double sign;
	   trace2.zero_device();
	   trace3.zero_device();
	   for(int rho = 0; rho < N_DIMS; rho++){
	     if(rho != mu){
	       trace1.trPmunu((n%2==0)?gauge2:gauge1, std::make_pair(mu,rho));
	       trace2.add(trace1,(std::complex<double>) {1.,0.});
	       if(mu<rho){
		 trace1.TrFmunuSu3FmunuSu3(fmunu, std::make_pair(mu,rho),one3x3,fmunu,std::make_pair(mu,rho),one3x3);}
	       else{
		 trace1.TrFmunuSu3FmunuSu3(fmunu, std::make_pair(rho,mu),one3x3,fmunu,std::make_pair(rho,mu),one3x3);}
	       trace3.add(trace1,(std::complex<double>) {1.,0.});
	       for(int sigma = 0; sigma < rho; sigma++){
		 if(sigma != mu){
		   trace1.trPmunu((n%2==0)?gauge2:gauge1, std::make_pair(sigma,rho));
		   trace2.add(trace1,(std::complex<double>) {-1.,0.});
		   trace1.TrFmunuSu3FmunuSu3(fmunu, std::make_pair(sigma,rho),one3x3,fmunu,std::make_pair(sigma,rho),one3x3);
		   trace3.add(trace1,(std::complex<double>) {-1.,0.});
		 }
	       }
	     }
	   }	   
	   ftUL1.apply(trace2,FT_GEMV);
	   gLoopPlt_diag.push_back(ftUL1.H_elem()[0]);	   
	   ftUL1.apply(trace3,FT_GEMV);
	   gLoopFST_diag.push_back(ftUL1.H_elem()[0]);
	   for (int nu = mu+1; nu < N_DIMS; nu++){
	     trace2.zero_device();
	     for(int rho = 0; rho < N_DIMS; rho++){
	       if((rho != mu) && (rho != nu)){
		 if(nu<rho){
		   trace1.TrFmunuSu3FmunuSu3(fmunu, std::make_pair(mu,rho),one3x3,fmunu,std::make_pair(nu,rho),one3x3); sign = +1;}
		 else{
		   if(mu<rho){
		     trace1.TrFmunuSu3FmunuSu3(fmunu, std::make_pair(mu,rho),one3x3,fmunu,std::make_pair(rho,nu),one3x3); sign = -1;}
		   else{
		     trace1.TrFmunuSu3FmunuSu3(fmunu, std::make_pair(rho,mu),one3x3,fmunu,std::make_pair(rho,nu),one3x3); sign = +1;}
		 }
	       trace2.add(trace1,(std::complex<double>) {sign,0.});
	       }
	     }
	     ftUL1.apply(trace2,FT_GEMV);
	     gLoopFST_off.push_back(ftUL1.H_elem()[0]);
	   }
	 }

	 if(comm_rank() == 0){
	   write_std_vecs(filenameGLoopPlt_diag,true,confVecGLoops_diag,nScountGLoops_diag,muVec_diag,gLoopPlt_diag);
	   write_std_vecs(filenameGLoopFST_diag,true,confVecGLoops_diag,nScountGLoops_diag,muVec_diag,gLoopFST_diag);
	   write_std_vecs(filenameGLoopFST_off,true,confVecGLoops_off,nScountGLoops_off,muVec_off,nuVec_off,gLoopFST_off);
	 }
	 
       }
     }

     double t2=MPI_Wtime();
     PLEGMA_printf("conf.%s completed in %f secs\n",confStr.c_str(),t2-t1);
  }

  // The vertex function will be created later from a multiplication of the gluon loop with the gluon propagator    
  finalize();
  
}
