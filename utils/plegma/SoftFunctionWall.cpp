#include <PLEGMA.h>
#include <PLEGMA_utils.h>

using namespace plegma;
using namespace quda;

int main(int argc, char **argv)
{

  static std::vector<std::string> listOpt = {"verbosity", "load-gauge", "nsmear-APE", "alpha-APE", "nsmear-gauss", "alpha-gauss",
					     "nsrc", "src-filename", "maxQsq", "twop-filename","threep-filename",  "corr-file-format",
					     "corr-space", "tSinks","Projs","xiMomSm","sinkMom","which_particle","gammas"};

  initializeOptions(argc, argv, true, listOpt);

  double mu_ud;

  double rhoStout;
  HGC_options->set("rho_stout", "Rho parameter stout smearing", verbosity, rhoStout);

  int nStout;
  HGC_options->set("n_stout", "n parameter stout smearing", verbosity, nStout);

  int L_LEN;
  HGC_options->set("l-len", "length of l", verbosity, L_LEN);

  int L_DIR;
  HGC_options->set("l-dir", "direction of l", verbosity, L_DIR);

  int B_LEN;
  HGC_options->set("b-len", "length of b", verbosity, B_LEN);

  int B_DIR;
  HGC_options->set("b-dir", "direction of b", verbosity, B_DIR);
  
  int z = 0;
  HGC_options->set("z-len", "length of z", verbosity, z);

  std::vector<int> PMom = {0,0,0};
  HGC_options->set("P-momentum", "If added to the momentum transfer delta gives the sink momentum", verbosity, PMom);

  std::vector<int> pMom = {0,0,0};
  HGC_options->set("p-momentum", "If added to the momentum transfer delta gives the sink momentum", verbosity, pMom);

  int tsrc;
  HGC_options->set("t-source", "Time of source vector", verbosity, tsrc);

  int tsnk;
  HGC_options->set("t-sink", "Time of sink vector", verbosity, tsnk);

  std::string fourp_filename;
  HGC_options->set("fourp-filename", "Filename of the four-points correlator", verbosity, fourp_filename);

  std::string qwf_filename;
  HGC_options->set("qwf-filename", "Filename of the quasi-wave function", verbosity, qwf_filename);

  std::string ft1_filename;
  std::string ft2_filename;
  HGC_options->set("ft1-filename", "If added to the momentum transfer delta gives the sink momentum", verbosity, ft1_filename);
  HGC_options->set("ft2-filename", "If added to the momentum transfer delta gives the sink momentum", verbosity, ft2_filename);
  
  initializePLEGMA();

  std::vector<int> PpMom = {0,0,0};

  std::transform(PMom.begin(), PMom.end(), pMom.begin(), PpMom.begin(), std::minus<int>());

  std::vector<int> P2Mom = {0,0,0};

  std::transform(PMom.begin(), PMom.end(), PMom.begin(), P2Mom.begin(), std::plus<int>());

  std::vector<int> zero_mom = {0,0,0};
  
  PLEGMA_Gauge<double> gauge;
  gauge.readFile(latfile, LIME_FORMAT);
  gauge.calculatePlaq();

  initGaugeQuda(gauge, true, QUDA_WILSON_LINKS);
  plaqQuda();

  PLEGMA_Gauge<double> smearedGauge;
  smearedGauge.APEsmearing(gauge, nsmearAPE, alphaAPE, 3);
  PLEGMA_printf("Plaquette after smearing:\n");
  smearedGauge.calculatePlaq();

  PLEGMA_Gauge<float> gaugeWL;
  gaugeWL.copy(gauge);
  gaugeWL.stoutSmearing(gaugeWL,nStout,rhoStout,3);
  PLEGMA_Su3field<float> *WL = new PLEGMA_Su3field<float>(BOTH);
  PLEGMA_Su3field<float> tmp(BOTH);
  PLEGMA_Su3field<float> *u_s[4];
  for(int idir = 0; idir < 4 ; idir++){
    u_s[idir] = new PLEGMA_Su3field<float>(BOTH);
    u_s[idir]->absorbDir_device(gaugeWL,idir);
  }
  PLEGMA_Su3field<float> *WLIn = new PLEGMA_Su3field<float>(BOTH);
  PLEGMA_Su3field<float> *WLExchange = nullptr;
  
  if(mu<0)  mu*=-1.;
  mu_ud = mu;
  QUDA_solver *solver = new QUDA_solver(mu);

  PLEGMA_Vector<double> vectorIn;
  PLEGMA_Vector<double> vectorOut;
  PLEGMA_Vector<double> vectorAuxD;
  PLEGMA_Vector<float> vectorAuxF;

  PLEGMA_Propagator<float> *propUPp = new PLEGMA_Propagator<float>(BOTH);
  PLEGMA_Propagator<float> *propDNp = new PLEGMA_Propagator<float>(BOTH);
  PLEGMA_Propagator<float> *propUPPp = new PLEGMA_Propagator<float>(BOTH);
  PLEGMA_Propagator<float> *propDNPp = new PLEGMA_Propagator<float>(BOTH);
  PLEGMA_Propagator<float> *propIn = new PLEGMA_Propagator<float>(BOTH);
  PLEGMA_Propagator<float> *propExchange = nullptr;

  site& source = sourcePositions[0];
  
  if(mu != mu_ud) {
    mu = mu_ud;
    solver->UpdateSolver();
  }
    
  for(int isc=0;isc<12;isc++){
    vectorAuxD.setUnit((std::vector<int>) {isc});
    vectorIn.absorbTimeslice(vectorAuxD,tsrc,true);
    vectorIn.mulMomentumPhases((std::vector<int>) {pMom[0],pMom[1],pMom[2],0},-1);
    solver->solve(vectorOut,vectorIn);
    vectorAuxF.copy(vectorOut);
    propUPp->absorb(vectorAuxF, isc/3, isc%3);
  }

  for(int isc=0;isc<12;isc++){
    vectorAuxD.setUnit((std::vector<int>) {isc});
    vectorIn.absorbTimeslice(vectorAuxD,tsrc,true);
    vectorIn.mulMomentumPhases((std::vector<int>) {PpMom[0],PpMom[1],PpMom[2],0},+1);
    solver->solve(vectorOut,vectorIn);
    vectorAuxF.copy(vectorOut);
    propUPPp->absorb(vectorAuxF, isc/3, isc%3);
  }

  propUPPp->rotateToPhysicalBase_device(+1);
  propUPp->rotateToPhysicalBase_device(+1);
  propUPPp->applyBoundaries_device(0);
  propUPp->applyBoundaries_device(0);

  PLEGMA_FT<float> ftp(pMom,3);
  PLEGMA_FT<float> ftPp(PpMom,3);
  ftp.apply(*propUPp,FT_GEMV,+1);
  ftPp.apply(*propUPPp);
  ftp.writeFile(ft1_filename, corr_file_format);
  ftPp.writeFile(ft2_filename, corr_file_format);

  WL->setUnit((std::vector<int>) {0,4,8});

  PLEGMA_Correlator<float> corr(corr_space, source);
  corr.setFixMomVec(sinkMom);
  corr.contractTMDWFMesons_Zfac(*propUPPp,*propUPp,*WL,0,0);
  corr.writeFile(twop_filename, corr_file_format);

  if(mu != -mu_ud) {
    mu = -mu_ud;
    solver->UpdateSolver();
  }

  for(int isc=0;isc<12;isc++){
    vectorAuxD.setUnit((std::vector<int>) {isc});
    vectorIn.absorbTimeslice(vectorAuxD,tsnk,true);
    vectorIn.mulMomentumPhases((std::vector<int>) {pMom[0],pMom[1],pMom[2],0},+1);
    solver->solve(vectorOut,vectorIn);
    vectorAuxF.copy(vectorOut);
    propDNp->absorb(vectorAuxF, isc/3, isc%3);
  }

  for(int isc=0;isc<12;isc++){
    vectorAuxD.setUnit((std::vector<int>) {isc});
    vectorIn.absorbTimeslice(vectorAuxD,tsnk,true);
    vectorIn.mulMomentumPhases((std::vector<int>) {PpMom[0],PpMom[1],PpMom[2],0},-1);
    solver->solve(vectorOut,vectorIn);
    vectorAuxF.copy(vectorOut);
    propDNPp->absorb(vectorAuxF, isc/3, isc%3);
  }

  propDNp->rotateToPhysicalBase_device(-1);
  propDNp->applyBoundaries_device(source[0]);
  propDNPp->rotateToPhysicalBase_device(-1);
  propDNPp->applyBoundaries_device(source[0]);

  PLEGMA_Correlator<float> corr4p(corr_space, source);
  corr4p.setFixMomVec(P2Mom);

  PLEGMA_Correlator<float> corrqwf(corr_space, source);
  corrqwf.setFixMomVec(sinkMom);

  for(int b=0;b<B_LEN+1;b++){

    if(b>0){
      propExchange = propIn; propIn = propUPp; propUPp = propExchange;
      propUPp->shift(*propIn, 4+B_DIR);
      
      propExchange = propIn; propIn = propDNp; propDNp = propExchange;
      propDNp->shift(*propIn, 4+B_DIR);
    }
    
    corr4p.contractMesonsFourp_ultralocal(*propUPp, *propDNp, *propDNPp, *propUPPp,b);
    corr4p.writeFile(fourp_filename.c_str(), corr_file_format);
    
    for(int l=0;l<L_LEN+1;l++){

      WL->setUnit((std::vector<int>) {0,4,8});

      int spath[(2*l)+b+z];

      if((l!=0)||(b!=0)){
	for(int i=0;i<l;i++){
	  spath[i] = 4+L_DIR;
	}
	for(int j=l;j<(l+b);j++){
	  spath[j] = B_DIR;
	}
	for(int k=(l+b);k<((2*l)+b+z);k++){
	  spath[k] = L_DIR;
	}

	std::vector<int> vspath(spath,spath+(2*l)+b+z);

	WL->path(vspath, u_s, tmp);

	for(int k=(l+b);k<((2*l)+b+z);k++){
	  WLExchange = WLIn; WLIn = WL; WL = WLExchange;
	  WL->shift(*WLIn,4+L_DIR);
	}
	for(int j=l;j<(l+b);j++){
	  WLExchange = WLIn; WLIn = WL; WL = WLExchange;
	  WL->shift(*WLIn,4+B_DIR);
	}
	for(int i=0;i<l;i++){
	  WLExchange = WLIn; WLIn = WL; WL = WLExchange;
	  WL->shift(*WLIn,L_DIR);
	}

      }

      WL->conjugate();
      corrqwf.contractTMDWFMesons_Zfac(*propUPPp,*propUPp,*WL,b,l);
      corrqwf.writeFile(qwf_filename.c_str(), corr_file_format);
    }

    for(int l=1;l<L_LEN+1;l++){

      WL->setUnit((std::vector<int>) {0,4,8});

      int spath[(2*l)+b-z];

      for(int i=0;i<l;i++){
	spath[i] = L_DIR;
      }
      for(int j=l;j<(l+b);j++){
	spath[j] = B_DIR;
      }
      for(int k=(l+b);k<((2*l)+b-z);k++){
	spath[k] = 4+L_DIR;
      }

      std::vector<int> vspath(spath,spath+(2*l)+b-z);

      WL->path(vspath, u_s, tmp);

      for(int k=(l+b);k<((2*l)+b-z);k++){
	WLExchange = WLIn; WLIn = WL; WL = WLExchange;
	WL->shift(*WLIn,L_DIR);
      }
      for(int j=l;j<(l+b);j++){
	WLExchange = WLIn; WLIn = WL; WL = WLExchange;
	WL->shift(*WLIn,4+B_DIR);
      }
      for(int i=0;i<l;i++){
	WLExchange = WLIn; WLIn = WL; WL = WLExchange;
	WL->shift(*WLIn,4+L_DIR);
      }

      WL->conjugate();
      corrqwf.contractTMDWFMesons_Zfac(*propUPPp,*propUPp,*WL,b,-l);
      corrqwf.writeFile(qwf_filename.c_str(), corr_file_format);
      
    }
  }
  
  for(int b=0;b<B_LEN;b++){
    propExchange = propIn; propIn = propUPp; propUPp = propExchange;
    propUPp->shift(*propIn, B_DIR);

    propExchange = propIn; propIn = propDNp; propDNp = propExchange;
    propDNp->shift(*propIn, B_DIR);
  }
  
  for(int b=1;b<B_LEN+1;b++){

    if(b>0){
      propExchange = propIn; propIn = propUPp; propUPp = propExchange;
      propUPp->shift(*propIn, B_DIR);

      propExchange = propIn; propIn = propDNp; propDNp = propExchange;
      propDNp->shift(*propIn, B_DIR);
    }

    corr4p.contractMesonsFourp_ultralocal(*propUPp, *propDNp, *propDNPp, *propUPPp,-b);
    corr4p.writeFile(fourp_filename.c_str(), corr_file_format);

    for(int l=0;l<L_LEN+1;l++){

      WL->setUnit((std::vector<int>) {0,4,8});

      int spath[(2*l)+b+z];

      if((l!=0)||(b!=0)){
	for(int i=0;i<l;i++){
	  spath[i] = 4+L_DIR;
	}
	for(int j=l;j<(l+b);j++){
	  spath[j] = 4+B_DIR;
	}
	for(int k=(l+b);k<((2*l)+b+z);k++){
	  spath[k] = L_DIR;
	}

	std::vector<int> vspath(spath,spath+(2*l)+b+z);

	WL->path(vspath, u_s, tmp);

	for(int k=(l+b);k<((2*l)+b+z);k++){
	  WLExchange = WLIn; WLIn = WL; WL = WLExchange;
	  WL->shift(*WLIn,4+L_DIR);
	}
	for(int j=l;j<(l+b);j++){
	  WLExchange = WLIn; WLIn = WL; WL = WLExchange;
	  WL->shift(*WLIn,B_DIR);
	}
	for(int i=0;i<l;i++){
	  WLExchange = WLIn; WLIn = WL; WL = WLExchange;
	  WL->shift(*WLIn,L_DIR);
	}
      }

      WL->conjugate();
      corrqwf.contractTMDWFMesons_Zfac(*propUPPp,*propUPp,*WL,-b,l);
      corrqwf.writeFile(qwf_filename.c_str(), corr_file_format);

    }

    for(int l=1;l<L_LEN+1;l++){

      WL->setUnit((std::vector<int>) {0,4,8});

      int spath[(2*l)+b-z];

      for(int i=0;i<l;i++){
	spath[i] = L_DIR;
      }
      for(int j=l;j<(l+b);j++){
	spath[j] = 4+B_DIR;
      }
      for(int k=(l+b);k<((2*l)+b-z);k++){
	spath[k] = 4+L_DIR;
      }

      std::vector<int> vspath(spath,spath+(2*l)+b-z);

      WL->path(vspath, u_s, tmp);

      for(int k=(l+b);k<((2*l)+b-z);k++){
	WLExchange = WLIn; WLIn = WL; WL = WLExchange;
	WL->shift(*WLIn,L_DIR);
      }
      for(int j=l;j<(l+b);j++){
	WLExchange = WLIn; WLIn = WL; WL = WLExchange;
	WL->shift(*WLIn,B_DIR);
      }
      for(int i=0;i<l;i++){
	WLExchange = WLIn; WLIn = WL; WL = WLExchange;
	WL->shift(*WLIn,4+L_DIR);
      }

      WL->conjugate();
      corrqwf.contractTMDWFMesons_Zfac(*propUPPp,*propUPp,*WL,-b,-l);
      corrqwf.writeFile(qwf_filename.c_str(), corr_file_format);

    }
  }
  
  for(int idir = 0; idir < 4 ; idir++){
    delete u_s[idir];
  }
  delete WL;
  delete WLIn;
  
  delete propUPp;
  delete propUPPp;
  delete propDNp;
  delete propDNPp;
  delete propIn;

  delete solver;
  
  finalize();
  
  return 0;
}
