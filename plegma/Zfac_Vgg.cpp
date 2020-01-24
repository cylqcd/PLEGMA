#include <PLEGMA.h>
#include <PLEGMA_utils.h>
#include <complex>

using namespace plegma;

static std::vector<std::string> listOpt = {"verbosity", "load-gauge-list-filename","nsmear-stout","alpha-stout", 
					   "nsrc","src-filename"}; // this will be used for the momenta

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


static void computeGprop(std::vector<std::vector<std::complex<double>>> &AxA, PLEGMA_FT<double> &ftAl, PLEGMA_FT<double> &ftAr,std::vector<double> mom){
  std::vector<std::complex<double>> tmp(N_DIMS,(std::complex<double>){0.,0.});
  std::complex<double> *Al = (std::complex<double> *) ftAl.H_elem();
  std::complex<double> *Ar = (std::complex<double> *) ftAr.H_elem();  
  for(int mu = 0; mu < N_DIMS; mu++){
    for(int c1 = 0; c1 < N_COLS; c1++)
      for(int c2 = 0; c2 < N_COLS; c2++)
	tmp[mu] += Al[(mu*N_COLS+c1)*N_COLS+c2] * Ar[(mu*N_COLS+c2)*N_COLS+c1];
    tmp[mu]  /= HGC_totalVolume;
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
  HGC_options->set("isGFixed", "If this is true it means that the configuration provided is alread gauge fixed", verbosity,isGFixed);
  bool doGLoops = true;
  HGC_options->set("doGLoops", "If you want to compute also gluon loops", verbosity,doGLoops);
  bool doSmearGprop = false;
  HGC_options->set("doSmearGprop", "If we want to smear the gluon propagator (only stout for now)", verbosity,doSmearGprop);
  int nsmearStoutGprop = 10;
  double alphaStoutGprop = 0.129;
  HGC_options->set("nsmear-stout-Gprop", "Number of stout smearing step for Gprop",verbosity,nsmearStoutGprop);
  HGC_options->set("alpha-stout-Gprop", "Coefficient for the stout smearing for Gprop",verbosity,alphaStoutGprop);
  std::string stoutOrWF="stout";
  HGC_options->set("stoutOrWF", "Choose what smearing to do in the operator, either stout or Wilson Flow. Options (stout,WF)",verbosity,stoutOrWF);
  
  //==========================//
  initializePLEGMA();

  
  PLEGMA_Gauge<double> gauge1,gauge2,gauge3;
  PLEGMA_Field<double> trace1(BOTH,SCALAR),trace2(BOTH,SCALAR);
  PLEGMA_FT<double> ftUL1(0,4);
  PLEGMA_FT<double> ftUL2(0,4);
  std::vector<double> gLoopPlt; // gluon loops with Plaquette definition
  std::vector<double> gLoopFST; // gluon loops diagonals with Field strength tensor
  std::vector<double> gLoopFST_off[3]; // gluon loops off-diagonals with Field strength tensor
  std::vector<std::vector<std::complex<double>>> AxA;
  PLEGMA_FT<double> *ftAl=nullptr, *ftAr=nullptr;
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
  std::string filenameGLoopPlt = filesPrefix + "/gLoopPlt_" + filesSuffix +".txt";
  std::string filenameGLoopFST = filesPrefix + "/gLoopFST_" + filesSuffix +".txt";
  std::string filenameGLoopFST_off[3] = {filesPrefix + "/gLoopFST_off0_" + filesSuffix +".txt",
					 filesPrefix + "/gLoopFST_off1_" + filesSuffix +".txt",
					 filesPrefix + "/gLoopFST_off2_" + filesSuffix +".txt"};  

  if(doGLoops) if(comm_rank() == 0) cleanFile(filenameGLoopPlt);
  if(doGLoops) if(comm_rank() == 0) cleanFile(filenameGLoopFST);
  if(doGLoops) if(comm_rank() == 0) for(int i=0;i < 3;i++) cleanFile(filenameGLoopFST_off[i]);
  

  std::vector<int> muVec;
  for(int mu = 0; mu < N_DIMS; mu++) muVec.push_back(mu);


  
  if(HGC_verbosity > 1) PLEGMA_printf("Will work on %d confs",listGaugeConfs.size());
  for(int iconf=0; iconf < listGaugeConfs.size(); iconf++){
    double t1=MPI_Wtime();
    std::string confStr=splitStrFwd(listGaugeConfs[iconf],'.');
    gauge1.readFile(listGaugeConfs[iconf], LIME_FORMAT,true);
    PLEGMA_printf("Unsmeared Plaquette is: ");
    gauge1.calculatePlaq();
    if(!isGFixed){
      double t3=MPI_Wtime();
      if(overelaxType == "exact") gFixingLandauOVR_QUDA(gauge2,gauge1,overelaxPar,tolerance,10000,10000);
      else if (overelaxType == "stoch") gauge2.gFixingLandau(gauge1,stochoverelaxPar,tolerance);
      else PLEGMA_error("Overrelaxation type %s not implemented",overelaxType.c_str());
      double t4=MPI_Wtime();
      PLEGMA_printf("Gauge fixing completed in %f secs\n",t4-t3);
      gauge1.copy(gauge2);
      if(doSmearGprop) {
	gauge3.stoutSmearing(gauge1,nsmearStoutGprop,alphaStoutGprop,4);
	gauge2.gluonField(gauge3);
      }
      else{
	gauge2.gluonField(gauge1);
      }
    }
    else{
      if(doSmearGprop){
	gauge3.stoutSmearing(gauge1,nsmearStoutGprop,alphaStoutGprop,4);
	gauge2.gluonField(gauge3);
      }
      else{
	gauge2.gluonField(gauge1);
      }
    }

    // Here we need to compute the gluon propagator which is located at gauge2 
    AxA.clear();    
    for(int im=0; im < momList.size(); im++){
      std::vector<double> mom=(std::vector<double>) {momList[im][0],
  						     momList[im][1],
  						     momList[im][2],
  						     momList[im][3]};
      std::vector<std::string> confVec(N_DIMS,confStr);
      ftAl = new PLEGMA_FT<double>(mom,4,false);
      ftAr = new PLEGMA_FT<double>(mom,4,false);
      ftAl->apply(gauge2,FT_GEMV,-1); // remember to put the twist in the temporal direction
      ftAr->apply(gauge2,FT_GEMV,+1); // remember to put the twist in the temporal direction
      computeGprop(AxA,*ftAl,*ftAr,mom);

      if(comm_rank() == 0) write_std_vecs(filenameGprop[im],true,confVec,muVec,AxA[im]);
      delete ftAl,ftAr;
    }


    PLEGMA_Fmunu<double> fmunu;
    PLEGMA_Su3field<double> one3x3;
    one3x3.setUnit((std::vector<int>) {0,4,8});
    
    // Here we compute the gluon loops
    if(doGLoops){
      // gauge1 holds the link variables in landau gauge (gluon field is not needed anymore and will be used for tmp)
      gLoopPlt.clear();
      gLoopFST.clear();
      gLoopFST_off[0].clear();      gLoopFST_off[1].clear();      gLoopFST_off[2].clear();
      if(stoutOrWF != "stout" && stoutOrWF != "WF") PLEGMA_error("Either Stout or Wilson flow are needed");
      int nSteps = (stoutOrWF == "stout")?nsmearStout:0;
      std::vector<int> nScount;
      for(int n=0; n<=nSteps;n++) nScount.push_back(n);
      
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
	// gLoops diagonal Plq definition
	/*
	 * Plaquete definition of the gluon loops
	 * Definition \mathcal{O} = \frac{-4*\beta}{9} Re{ \Tr[ \sum_i P_{3i} - sum_{i<j} P_{ij} ]}
	 * We will not include the factor \frac{-4*\beta}{9} now
	 * Term1 = \Tr[ \sum_i P_{i3} ]
	 * Term2 = \Tr[sum_{i<j} P_{ij}]
	 */
  	trace2.zero_device();
  	for(int i = 0 ; i < N_DIMS-1; i++){
  	  trace1.trPmunu((n%2==0)?gauge2:gauge1, std::make_pair(3,i));
  	  trace2.add(trace1,(std::complex<double>) {1.,0.});
  	}
  	ftUL1.apply(trace2,FT_GEMV);

  	trace2.zero_device();
  	for(int i = 0 ; i < N_DIMS-1; i++)
  	  for(int j = i+1 ; j < N_DIMS-1; j++){
  	    trace1.trPmunu((n%2==0)?gauge2:gauge1, std::make_pair(i,j));
  	    trace2.add(trace1,(std::complex<double>) {1.,0.});
  	  }
  	ftUL2.apply(trace2,FT_GEMV);
  	gLoopPlt.push_back(ftUL1.H_elem()[0] - ftUL2.H_elem()[0]);
   

      
	// gLoops diagonal FST definition
	/* Clover definition of gluon loops
	 * Definition \mathcal{O} = \frac{-4*\beta}{18} \Tr[ \sum_{i<j} F^2_{ij} - \sum_i F^2_{i3} ]
	 * Term1 = \Tr[ \sum_{i<j} F^2_{ij}]
	 * Term2 = \Tr[ \sum_i F^2_{i3} ]
	 */
	fmunu.compute_leaves((n%2==0)?gauge2:gauge1);
	trace2.zero_device();
	for(int i = 0 ; i < N_DIMS-1; i++){
	  trace1.TrFmunuSu3FmunuSu3(fmunu, std::make_pair(i,3),one3x3,fmunu,std::make_pair(i,3),one3x3);
	  trace2.add(trace1,(std::complex<double>) {1.,0.});
	}
	ftUL1.apply(trace2,FT_GEMV);
	trace2.zero_device();
	for(int i = 0 ; i < N_DIMS-1; i++)
	  for(int j = i+1 ; j < N_DIMS-1; j++){
	    trace1.TrFmunuSu3FmunuSu3(fmunu, std::make_pair(i,j), one3x3,fmunu,
				      std::make_pair(i,j), one3x3);
	    trace2.add(trace1,(std::complex<double>) {1.,0.});
	  }
	ftUL2.apply(trace2,FT_GEMV);
	gLoopFST.push_back(ftUL2.H_elem()[0] - ftUL1.H_elem()[0]);

	// gLoops off diagonal elements with FST definition
	/* Clover definition of gluon loops off diagonals
	 * Definition \mathcal{O}_i = unknown * \Tr[\sum_\mu F_{i,\mu} * F_{3,\mu}]
	 * FST indices cannot be same
	 * unknow is a factor which will be figured out later
	 */
	for(int i =0 ; i< N_DIMS-1;i++){
	  double sign;
	  trace2.zero_device();
	  for(int mu =0; mu< N_DIMS;mu++){
	    if((i!=mu) && (mu!=3)){
	      if(i<mu){ trace1.TrFmunuSu3FmunuSu3(fmunu, std::make_pair(i,mu), one3x3,fmunu,std::make_pair(mu,3), one3x3); sign=-1.;}
	      else{trace1.TrFmunuSu3FmunuSu3(fmunu, std::make_pair(mu,i), one3x3,fmunu,std::make_pair(mu,3), one3x3); sign=+1.;}
	      trace2.add(trace1,(std::complex<double>) {sign,0.});	      
	    }
	  }
	  ftUL1.apply(trace2,FT_GEMV);
	  gLoopFST_off[i].push_back(ftUL1.H_elem()[0]);
	}
	
	double t2=MPI_Wtime();
	PLEGMA_printf("conf.%s completed in %f secs\n",confStr.c_str(),t2-t1);
      }
      
      std::vector<std::string> confVec(nSteps+1,confStr);
      if(comm_rank() == 0){
	write_std_vecs(filenameGLoopPlt,true,confVec,nScount,gLoopPlt);
	write_std_vecs(filenameGLoopFST,true,confVec,nScount,gLoopFST);
	for(int i =0 ; i< N_DIMS-1;i++)
	  write_std_vecs(filenameGLoopFST_off[i],true,confVec,nScount,gLoopFST_off[i]);
      }
    }
  }
  // Vertex function will be create later from a multiplication of the gluon Loop with the gluon propagator    
  finalize();
}




  //   std::vector<double> gLoop00,gLoop11,gLoop22;

  //   if(doGLoops){
  //     // gauge1 holds the link variables in landau gauge (gluon field is not needed anymore and will be used for tmp)
  //     gLoop00.clear();      gLoop11.clear();      gLoop22.clear(); gLoop.clear();
  //     for(int n=0; n<=nsmearStout;n++){
  // 	if(n%2 == 0){
  // 	  if(n==0) gauge2.copy(gauge1);
  // 	  else gauge2.stoutSmearing(gauge1,1,alphaStout,4);
  // 	}
  // 	else{
  // 	  gauge1.stoutSmearing(gauge2,1,alphaStout,4);
  // 	}

  // 	for(int nu =0; nu < N_DIMS; nu++){
  // 	  trace2.zero_device();
  // 	  for(int i = 0 ; i < N_DIMS; i++){
  // 	    if(i != nu){
  // 	      trace1.trPmunu((n%2==0)?gauge2:gauge1, std::make_pair(nu,i));
  // 	      trace2.add(trace1,(std::complex<double>) {1.,0.});
  // 	    }
  // 	  }
  // 	  ftUL1.apply(trace2,FT_GEMV);
  // 	  if(nu == 0) gLoop00.push_back(ftUL1.H_elem()[0]);
  // 	  else if(nu==1) gLoop11.push_back(ftUL1.H_elem()[0]);
  // 	  else if(nu==2)gLoop22.push_back(ftUL1.H_elem()[0]);
  // 	  else gLoop.push_back(ftUL1.H_elem()[0]);
  // 	}	
  //     }
  //     std::vector<std::string> confVec(nsmearStout+1,confStr);
  //     if(comm_rank() == 0) write_std_vecs(filenameGLoops,true,confVec,nScount,gLoop00,gLoop11,gLoop22,gLoop);
  //     double t2=MPI_Wtime();
  //     PLEGMA_printf("conf.%s completed in %f secs\n",confStr.c_str(),t2-t1);
  //   }
  // }
