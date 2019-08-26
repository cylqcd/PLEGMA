#include <PLEGMA.h>
#include <PLEGMA_utils.h>

using namespace plegma;

static std::vector<std::string> listOpt = {"verbosity", "load-gauge","nsmear-stout", "alpha-stout","corr-file-format","maxQsq"};

static void dumpLoops(PLEGMA_FT<double> **ft,
 		      std::string filenamePrefix, std::string confID, FILE_FORMAT format){
  for(int itype = 0 ; itype < 4; itype++)
    for(int idir=0; idir < 3; idir++)
      for(int i =0; i < HGC_totalL[0]; i++)
	ft[itype*3*HGC_totalL[0]+idir*HGC_totalL[0]+i]->writeToFile(filenamePrefix + "/gLoops_WilsonLine_OperType" + std::to_string(itype)
								    + "_dir" + std::to_string(idir) + "_z" +
								    std::to_string(i) + "_" + confID + ".dat" ,format);
}


struct pairedSinged {
  std::pair<int,int> munu;
  int sign;
};
    
static pairedSinged makePairCheck(int mu, int nu){
  if( mu == nu) PLEGMA_error("Cannot choose mu == nu");
  pairedSinged prs;
  if(nu > mu){
    prs.sign = +1;
    prs.munu = std::make_pair(mu,nu);
  }
  else{
    prs.sign = -1;
    prs.munu = std::make_pair(nu,mu);
  }
  return prs;
}
    
static void computeWithType(PLEGMA_Field<double> &out,PLEGMA_Field<double> &tmp, int wilsDir, PLEGMA_Fmunu<double> &fmunu_l,
			    PLEGMA_Su3field<double> &Wl,
			    PLEGMA_Fmunu<double> &fmunu_r, PLEGMA_Su3field<double> &Wr, int type){
  if(type <0 || type > 3) PLEGMA_error("Type not implemented");
  std::vector<pairedSinged> munu_l;
  std::vector<pairedSinged> munu_r;
  std::complex<double> signC = {-1.,0.}; 
  switch(type){
  case(1):
    munu_l.push_back( makePairCheck( (wilsDir+1)%3,3) ); munu_r.push_back( makePairCheck((wilsDir+1)%3,3) );
    munu_l.push_back( makePairCheck( (wilsDir+2)%3,3) ); munu_r.push_back( makePairCheck((wilsDir+2)%3,3) );
    break;
  case(2):
    munu_l.push_back( makePairCheck((wilsDir+1)%3,wilsDir) ); munu_r.push_back( makePairCheck((wilsDir+1)%3,wilsDir) );
    munu_l.push_back( makePairCheck((wilsDir+2)%3,wilsDir) ); munu_r.push_back( makePairCheck((wilsDir+2)%3,wilsDir) );	
    break;
  case(3):
    munu_l.push_back( makePairCheck(3,(wilsDir+1)%3) ); munu_r.push_back( makePairCheck(wilsDir,(wilsDir+1)%3) );
    munu_l.push_back( makePairCheck(3,(wilsDir+2)%3) ); munu_r.push_back( makePairCheck(wilsDir,(wilsDir+2)%3) );	
    break;
  }

  if(type != 0){
    out.TrFmunuSu3FmunuSu3(fmunu_l,munu_l[0].munu,Wl,fmunu_r,munu_r[0].munu,Wr);
    if(munu_l[0].sign * munu_r[0].sign == -1) out.cscale(signC);
    tmp.TrFmunuSu3FmunuSu3(fmunu_l,munu_l[1].munu,Wl,fmunu_r,munu_r[1].munu,Wr);
    if(munu_l[1].sign * munu_r[1].sign == -1) tmp.cscale(signC);
    out.axpy(tmp,(std::complex<double>) {1.,0.});
    out.cscale((std::complex<double>) {0.5,0.}); //average the two contributions
  }
  else{
    out.zero_device();
    for(int i = 0 ; i < N_DIMS-1; i++)
      for(int j = i+1 ; j < N_DIMS-1; j++){
	tmp.TrFmunuSu3FmunuSu3(fmunu_l, std::make_pair(i,j), Wl,fmunu_r, std::make_pair(i,j), Wr);
	out.axpy(tmp,(std::complex<double>) {1.,0.});
      }
    for(int i = 0 ; i < N_DIMS-1; i++){
      tmp.TrFmunuSu3FmunuSu3(fmunu_l, std::make_pair(i,3), Wl,fmunu_r, std::make_pair(i,3), Wr);
      out.axpy(tmp,(std::complex<double>) {-1.,0.});
    }
  }
}

int main(int argc, char **argv){
  initializeOptions(argc, argv, false, listOpt);
  //===================================//
  std::string filesPrefix="./";
  HGC_options->set("output-path", "Path to the directory to dump results", verbosity, filesPrefix);
  bool IsGloopsWline=false;
  HGC_options->set("isGloopsWline", "In case we want to add gluon loops with Wilson line", verbosity, IsGloopsWline);
  int nsmearStoutWL = 30;
  double alphaStoutWL = 0.129;
  HGC_options->set("nsmear-stoutWL", "Number of stout smearing step for the Wilson line if enabled",verbosity,nsmearStoutWL);
  HGC_options->set("alpha-stoutWL", "Coefficient for the stout smearing for the Wilson line if enabled",verbosity,alphaStoutWL);
  //===================================//
  initializePLEGMA();
  PLEGMA_Gauge<double> *gauge = new PLEGMA_Gauge<double>();
  gauge->readFromLime(latfile.c_str());
  PLEGMA_printf("Unsmeared Plaquette is:");
  gauge->calculatePlaq();


  PLEGMA_Gauge<double> *smearedGaugeOp = new PLEGMA_Gauge<double>();
  smearedGaugeOp->stoutSmearing(*gauge, nsmearStout, alphaStout, 4);
  PLEGMA_printf("Smeared Plaquette with stout 4D for the field strength tensor:");
  smearedGaugeOp->calculatePlaq();

  PLEGMA_Gauge<double> *smearedGaugeWL = nullptr;
  if(IsGloopsWline){
    smearedGaugeWL = new PLEGMA_Gauge<double>();
    smearedGaugeWL->stoutSmearing(*gauge, nsmearStoutWL, alphaStoutWL, 3);
    PLEGMA_printf("Smeared Plaquette with stout 3D for Wilson Line:");
    smearedGaugeWL->calculatePlaq();
  }
  
  delete gauge;
  
  PLEGMA_Field<double> traceO1(BOTH,SCALAR);
  PLEGMA_Field<double> traceO2(BOTH,SCALAR);
  PLEGMA_FT<double> ftUL(maxQsq,3);

  std::size_t foundPos = latfile.find("conf.");
  if(foundPos == std::string::npos) PLEGMA_error("Cannot find (conf.) in configuration path to get confID");
  std::string confID = latfile.substr(foundPos+5,latfile.length());


  /*
   * Plaquete definition of the gluon loops
   * Definition \mathcal{O} = \frac{-4*\beta}{9} Re{ \Tr[ \sum_i P_{3i} - sum_{i<j} P_{ij} ]}
   * Term1 = \Tr[ \sum_i P_{i3} ]
   * Term2 = \Tr[sum_{i<j} P_{ij}]
   */
  for(int i = 0 ; i < N_DIMS-1; i++){
    traceO1.trPmunu(*smearedGaugeOp, std::make_pair(3,i));
    traceO2.axpy(traceO1,(std::complex<double>) {1.,0.});
  }
  ftUL.apply(traceO2);
  ftUL.writeToFile(filesPrefix + "/gLoops_ultralocal_Plq_def_Term1" +  "_" + confID + ".dat" ,corr_file_format);

  traceO2.zero_device();
  for(int i = 0 ; i < N_DIMS-1; i++)
    for(int j = i+1 ; j < N_DIMS-1; j++){
      traceO1.trPmunu(*smearedGaugeOp, std::make_pair(i,j));
      traceO2.axpy(traceO1,(std::complex<double>) {1.,0.});
    }
  ftUL.apply(traceO2);
  ftUL.writeToFile(filesPrefix + "/gLoops_ultralocal_Plq_def_Term2" +  "_" + confID + ".dat" ,corr_file_format);

  traceO1.zero_device();
  traceO2.zero_device();
  //================================================

  /* Clover definition of gluon loops
   * Definition \mathcal{O} = \frac{-4*\beta}{18} \Tr[ \sum_{i<j} F^2_{ij} - \sum_i F^2_{i3} ]
   * Term1 = \Tr[ \sum_{i<j} F^2_{ij}]
   * Term2 = \Tr[ \sum_i F^2_{i3} ]
   */
  PLEGMA_Fmunu<double> fmunu_l;
  PLEGMA_Su3field<double> Wl;
  PLEGMA_Su3field<double> Wr;
  
  Wl.setUnit((std::vector<int>) {0,4,8});
  Wr.setUnit((std::vector<int>) {0,4,8});
  
  fmunu_l.compute_leaves(*smearedGaugeOp);
  delete smearedGaugeOp;
  
  for(int i = 0 ; i < N_DIMS-1; i++){
    traceO1.TrFmunuSu3FmunuSu3(fmunu_l, std::make_pair(i,3),Wl,fmunu_l,std::make_pair(i,3),Wr);
    traceO2.axpy(traceO1,(std::complex<double>) {1.,0.});
  }
  ftUL.apply(traceO2);
  ftUL.writeToFile(filesPrefix + "/gLoops_ultralocal_Clv_def_Term2" +  "_" + confID + ".dat" ,corr_file_format);

  traceO2.zero_device();
  for(int i = 0 ; i < N_DIMS-1; i++)
    for(int j = i+1 ; j < N_DIMS-1; j++){
      traceO1.TrFmunuSu3FmunuSu3(fmunu_l, std::make_pair(i,j), Wl,fmunu_l, std::make_pair(i,j), Wr);
      traceO2.axpy(traceO1,(std::complex<double>) {1.,0.});
    }
  ftUL.apply(traceO2);
  ftUL.writeToFile(filesPrefix + "/gLoops_ultralocal_Clv_def_Term1" +  "_" + confID + ".dat" ,corr_file_format);

  traceO1.zero_device();
  traceO2.zero_device();

  //=====================================================
  /* Gluon loops with Wilson Line
   * We have three types of Operator in order to avoid mixing
   * T0 &=& \sum_{i<j} F_{ij}(x+n\hat{k})W(x+n\hat{k},x) F_{ij}(x) - \sum_{i} F_{i3}(x+n\hat{k})W(x+n\hat{k},x) F_{i3}(x) \\
   * T1 &=& \;\; \frac{1}{2} \sum_i F_{i3}(x+n\hat{k}) W(x+n\hat{k},x) F_{i3}(x) \;\;\;\; if \;\; k \neq i \\
   * T2 &=& \;\; \frac{1}{2} \sum_i F_{ki}(x+n\hat{k}) W(x+n\hat{k},x) F_{ki}(x) \;\;\;\; if \;\; k \neq i \\
   * T3 &=& \;\; \frac{1}{2} \sum_i F_{3i}(x+n\hat{k}) W(x+n\hat{k},x) F_{ki}(x) \;\;\;\; if \;\; k \neq i 
   */
  if(IsGloopsWline){

    PLEGMA_Su3field<double> su3l;
    PLEGMA_Su3field<double> su3r;
    PLEGMA_Su3field<double> tmp;

    PLEGMA_Fmunu<double> fmunu_r;
    PLEGMA_Fmunu<double> *fmunuExchange = nullptr;
    PLEGMA_Fmunu<double> *fmunu_ptr = nullptr;
    PLEGMA_Fmunu<double> *fmunu_In = new PLEGMA_Fmunu<double>(DEVICE);
  


    fmunu_r.copy(fmunu_l);

    if(!(HGC_totalL[0] == HGC_totalL[1] && HGC_totalL[1] == HGC_totalL[2])) PLEGMA_error("Spatial total volume should be symmetric for this to work");
    int L=HGC_totalL[0];
    if(L%2 != 0) PLEGMA_error("If spatial extent is not multiple of 2 then it will not work");
    int Lo2 = L/2;

    
    int sizeFT=4*3*HGC_totalL[0]; // Four type of Operator, 3 directions of Wilson Line, L length of Wilson Line fwd/bwd
    PLEGMA_FT<double> **FTs = new PLEGMA_FT<double>*[sizeFT];
    for(int i = 0 ; i< sizeFT; i++) FTs[i] = new PLEGMA_FT<double>(maxQsq,3);
  
    for(int wilsDir = 0 ; wilsDir < 3; wilsDir++){
      su3l.absorbDir_device(*smearedGaugeWL,wilsDir);
      su3r.absorbDir_device(*smearedGaugeWL,wilsDir);
      Wl.setUnit((std::vector<int>) {0,4,8});
      Wr.setUnit((std::vector<int>) {0,4,8});
      fmunu_ptr = &fmunu_r;
      for(int i = 0 ; i < Lo2;i++){
    	for(int itype = 0 ; itype < 4 ; itype++){
    	  computeWithType(traceO1,traceO2,wilsDir,fmunu_l,Wl,*fmunu_ptr,Wr,itype);
    	  int index = itype*3*2*Lo2 + wilsDir*2*Lo2+i;
    	  if(FTs[index]->IsAccum()) PLEGMA_error("We need accumulation off here");
    	  FTs[index]->apply(traceO1);
    	}
    	fmunuExchange = fmunu_In; fmunu_In = fmunu_ptr; fmunu_ptr = fmunuExchange;
    	Wl.wilsonLineUpdate(su3l,tmp,4+wilsDir);
	Wr.wilsonLineUpdate(su3r,tmp,4+wilsDir,true);
    	fmunu_ptr->shift(*fmunu_In,4+wilsDir);
      }

      su3l.absorbDir_device(*smearedGaugeWL,wilsDir);
      su3r.absorbDir_device(*smearedGaugeWL,wilsDir);
      Wl.setUnit((std::vector<int>) {0,4,8});
      Wr.setUnit((std::vector<int>) {0,4,8});
      fmunu_ptr->copy(fmunu_l);
      for(int i = 0 ; i < Lo2;i++){
    	for(int itype = 0 ; itype < 4 ; itype++){
    	  computeWithType(traceO1,traceO2,wilsDir,fmunu_l,Wl,*fmunu_ptr,Wr,itype);
    	  int index = itype*3*2*Lo2 + wilsDir*2*Lo2+i+Lo2;
    	  if(FTs[index]->IsAccum()) PLEGMA_error("We need accumulation off here");
    	  FTs[index]->apply(traceO1);
    	}
    	fmunuExchange = fmunu_In; fmunu_In = fmunu_ptr; fmunu_ptr = fmunuExchange;
    	Wl.wilsonLineUpdate(su3l,tmp,wilsDir);
	Wr.wilsonLineUpdate(su3r,tmp,wilsDir,true);
    	fmunu_ptr->shift(*fmunu_In,wilsDir);
      }
      fmunu_ptr->copy(fmunu_l);
    }

    dumpLoops(FTs,filesPrefix,confID,corr_file_format);
  
    delete fmunu_In;
    for(int i = 0 ; i< sizeFT; i++) delete FTs[i];
    delete[] FTs;
    delete smearedGaugeWL;
  }

  
  finalize();
}
