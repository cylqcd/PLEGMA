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

  int seed=0;
  HGC_options->set("seed", "seed", verbosity, seed);

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
  PLEGMA_Gauge<double> smearedGaugeMn;
  smearedGauge.APEsmearing(gauge, nsmearAPE, alphaAPE, 3);
  PLEGMA_printf("Plaquette after smearing:\n");
  smearedGauge.calculatePlaq();
  smearedGaugeMn.copy(smearedGauge);

  std::complex<double> momSmPlScale[N_DIMS];
  std::complex<double> momSmMnScale[N_DIMS];
  std::complex<double> I(0,1);
  for(int i = 0 ; i < N_DIMS; i++){
    momSmPlScale[i] = std::exp(-(xiMomSm*2.*PI*sinkMom[i]/HGC_totalL[i])*I);
    momSmMnScale[i] = std::exp(+(xiMomSm*2.*PI*sinkMom[i]/HGC_totalL[i])*I);
  }
  smearedGauge.scaleDirWise(momSmPlScale);
  smearedGaugeMn.scaleDirWise(momSmMnScale);

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

  PLEGMA_Vector<double> vectorIn1;
  PLEGMA_Vector<double> vectorIn2;
  PLEGMA_Vector<double> vectorDil1[12];
  PLEGMA_Vector<double> vectorDil2[12];
  PLEGMA_Vector<float> vectorC21[12];
  PLEGMA_Vector<float> vectorC22[12];
  PLEGMA_Vector<float> vectorOutUP1[12];
  PLEGMA_Vector<float> vectorOutUP2[12];
  PLEGMA_Vector<float> vectorOutDN1[12];
  PLEGMA_Vector<float> vectorOutDN2[12];
  PLEGMA_Vector<double> vectorAuxD1;
  PLEGMA_Vector<double> vectorAuxD2;
  PLEGMA_Vector<float> vectorAuxF;
  PLEGMA_Propagator<float> tmpprop(BOTH);
  PLEGMA_Propagator<float> prop1(BOTH);
  PLEGMA_Propagator<float> prop2(BOTH);
  
  vectorIn1.randInit(1234+seed);
  vectorIn2.randInit(5678+seed);
  vectorIn1.stochastic_Z(4);
  vectorIn2.stochastic_Z(4);
  for(int i=0;i<12;i++){
    vectorDil1[i].dilutespincolor(vectorIn1,i%4,i/4);
    vectorDil2[i].dilutespincolor(vectorIn2,i%4,i/4);
  }
    
  site& source = sourcePositions[0];
  
  if(mu != mu_ud) {
    mu = mu_ud;
    solver->UpdateSolver();
  }
    
  for(int isc=0;isc<12;isc++){
    vectorAuxD1.absorbTimeslice(vectorDil1[isc],tsrc,true);
    vectorAuxD1.mulMomentumPhases((std::vector<int>) {pMom[0],pMom[1],pMom[2],0},-1);
    vectorAuxD2.gaussianSmearing(vectorAuxD1, smearedGaugeMn, nsmearGauss, alphaGauss);
    vectorAuxD1.rotateToPhysicalBasis(vectorAuxD2,+1);
    solver->solve(vectorAuxD2,vectorAuxD1);
    vectorAuxD1.rotateToPhysicalBasis(vectorAuxD2,+1);
    vectorOutUP1[isc].copy(vectorAuxD1);
  }

  for(int isc=0;isc<12;isc++){
    vectorAuxD1.absorbTimeslice(vectorDil1[isc],tsrc,true);
    vectorAuxD1.mulMomentumPhases((std::vector<int>) {PpMom[0],PpMom[1],PpMom[2],0},+1);
    vectorAuxD2.gaussianSmearing(vectorAuxD1, smearedGauge, nsmearGauss, alphaGauss);
    vectorAuxD1.rotateToPhysicalBasis(vectorAuxD2,+1);
    solver->solve(vectorAuxD2,vectorAuxD1);
    vectorAuxD1.rotateToPhysicalBasis(vectorAuxD2,+1);
    vectorOutUP2[isc].copy(vectorAuxD1);
  }

  if(mu != -mu_ud) {
    mu = -mu_ud;
    solver->UpdateSolver();
  }

  for(int isc=0;isc<12;isc++){
    vectorAuxD1.absorbTimeslice(vectorDil2[isc],tsnk,true);
    vectorAuxD1.mulMomentumPhases((std::vector<int>) {pMom[0],pMom[1],pMom[2],0},+1);
    vectorAuxD2.gaussianSmearing(vectorAuxD1, smearedGauge, nsmearGauss, alphaGauss);
    vectorAuxD1.rotateToPhysicalBasis(vectorAuxD2,-1);
    solver->solve(vectorAuxD2,vectorAuxD1);
    vectorAuxD1.rotateToPhysicalBasis(vectorAuxD2,-1);
    vectorOutDN1[isc].copy(vectorAuxD1);
  }

  for(int isc=0;isc<12;isc++){
    vectorAuxD1.absorbTimeslice(vectorDil2[isc],tsnk,true);
    vectorAuxD1.mulMomentumPhases((std::vector<int>) {PpMom[0],PpMom[1],PpMom[2],0},-1);
    vectorAuxD2.gaussianSmearing(vectorAuxD1, smearedGaugeMn, nsmearGauss, alphaGauss);
    vectorAuxD1.rotateToPhysicalBasis(vectorAuxD2,-1);
    solver->solve(vectorAuxD2,vectorAuxD1);
    vectorAuxD1.rotateToPhysicalBasis(vectorAuxD2,-1);
    vectorOutDN2[isc].copy(vectorAuxD1);
  }

  for(int i=0;i<12;i++){
    vectorAuxD1.copy(vectorOutUP2[i]);
    vectorAuxD2.gaussianSmearing(vectorAuxD1, smearedGauge, nsmearGauss, alphaGauss);
    vectorC21[i].copy(vectorAuxD2);
    vectorAuxD1.copy(vectorOutUP1[i]);
    vectorAuxD2.gaussianSmearing(vectorAuxD1, smearedGaugeMn, nsmearGauss, alphaGauss);
    vectorC22[i].copy(vectorAuxD2);
  }

  prop1.zero_where(BOTH);
  for(int j=0 ; j<12 ; j++){
    tmpprop.PropmulVVdag(vectorC21[j],vectorC22[j]);
    prop1.add(tmpprop,1.0);
  }
  
  WL->setUnit((std::vector<int>) {0,4,8});
  
  PLEGMA_Correlator<float> corrqwf(corr_space, source);
  corrqwf.setFixMomVec(sinkMom);
  corrqwf.contractTMDWFMesonsTrick_Zfac(prop1,*WL,0,0);
  corrqwf.writeFile(twop_filename.c_str(), corr_file_format);

  PLEGMA_Correlator<float> corr4p(corr_space, source);
  corr4p.setFixMomVec(P2Mom);

  for(int b=0;b<B_LEN+1;b++){

    if(b>0){
      for(int isc=0;isc<12;isc++){
	vectorAuxF.shift(vectorOutUP1[isc], 4+B_DIR);
	vectorOutUP1[isc].copy(vectorAuxF);
      }
      
      for(int isc=0;isc<12;isc++){
	vectorAuxF.shift(vectorOutDN1[isc], 4+B_DIR);
	vectorOutDN1[isc].copy(vectorAuxF);
      }
    }
     
    prop1.zero_where(BOTH);
    prop2.zero_where(BOTH);
    
    for(int i=0 ; i<12 ; i++){
      tmpprop.PropmulVVdag(vectorOutDN1[i],vectorOutDN2[i]);
      prop1.add(tmpprop,1.0);
    }
    
    for(int j=0 ; j<12 ; j++){
      tmpprop.PropmulVVdag(vectorOutUP2[j],vectorOutUP1[j]);
      prop2.add(tmpprop,1.0);
    }
    
    corr4p.contractMesonsFourp_ultralocal_oneendtrick(prop1, prop2,b);
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
      corrqwf.contractTMDWFMesonsTrick_Zfac(prop2,*WL,b,l);
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
      corrqwf.contractTMDWFMesonsTrick_Zfac(prop2,*WL,b,-l);
      corrqwf.writeFile(qwf_filename.c_str(), corr_file_format);
      
    }
  }
  
  for(int b=0;b<B_LEN;b++){
    for(int isc=0;isc<12;isc++){  
      vectorAuxF.shift(vectorOutUP1[isc], B_DIR);
      vectorOutUP1[isc].copy(vectorAuxF);
    }
    for(int isc=0;isc<12;isc++){
      vectorAuxF.shift(vectorOutDN1[isc], B_DIR);
      vectorOutDN1[isc].copy(vectorAuxF);
    }
  }
  
  for(int b=1;b<B_LEN+1;b++){

    if(b>0){
      for(int isc=0;isc<12;isc++){
	vectorAuxF.shift(vectorOutUP1[isc], B_DIR);
	vectorOutUP1[isc].copy(vectorAuxF);
      }

      for(int isc=0;isc<12;isc++){
        vectorAuxF.shift(vectorOutDN1[isc], B_DIR);
        vectorOutDN1[isc].copy(vectorAuxF);
      }
    }

    prop1.zero_where(BOTH);
    prop2.zero_where(BOTH);

    for(int i=0 ; i<12 ; i++){
      tmpprop.PropmulVVdag(vectorOutDN1[i],vectorOutDN2[i]);
      prop1.add(tmpprop,1.0);
    }

    for(int j=0 ; j<12 ; j++){
      tmpprop.PropmulVVdag(vectorOutUP2[j],vectorOutUP1[j]);
      prop2.add(tmpprop,1.0);
    }

    corr4p.contractMesonsFourp_ultralocal_oneendtrick(prop1, prop2,-b);
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
      corrqwf.contractTMDWFMesonsTrick_Zfac(prop2,*WL,-b,l);
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
      corrqwf.contractTMDWFMesonsTrick_Zfac(prop2,*WL,-b,-l);
      corrqwf.writeFile(qwf_filename.c_str(), corr_file_format);

    }
  }
  
  for(int idir = 0; idir < 4 ; idir++){
    delete u_s[idir];
  }
  delete WL;
  delete WLIn;
  
  delete solver;
  
  finalize();
  
  return 0;
}
