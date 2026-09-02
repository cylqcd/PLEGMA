#include <PLEGMA.h>
#include <PLEGMA_utils.h>

using namespace plegma;
using namespace quda;

int main(int argc, char **argv)
{

  static std::vector<std::string> listOpt = {"verbosity", "load-gauge", "nsmear-APE", "alpha-APE", "nsmear-gauss", "alpha-gauss",
<<<<<<< HEAD
					     "nsrc", "src-filename", "maxQsq", "twop-filename","threep-filename",  "corr-file-format",
					     "corr-space", "tSinks","Projs","xiMomSm","sinkMom","which_particle","gammas"};

  initializeOptions(argc, argv, true, listOpt);

  double mu_ud;

=======
					     "nsrc", "src-filename", "maxQsq",  "corr-file-format",
					     "corr-space","xiMomSm","gammas", "twop-filename"};

  initializeOptions(argc, argv, true, listOpt);

>>>>>>> origin/TMD
  double rhoStout;
  HGC_options->set("rho_stout", "Rho parameter stout smearing", verbosity, rhoStout);

  int nStout;
<<<<<<< HEAD
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
=======
  HGC_options->set("nstout", "n parameter stout smearing", verbosity, nStout);

  size_t WilsDir;
  HGC_options->set("wilson_direction", "Direction of the wilson line", verbosity, WilsDir);
  if(WilsDir>N_DIMS) PLEGMA_error("The direction of the WIlson line has to be smaller than 3");

  std::vector<int> sinkMom(4);
  HGC_options->set("sinkMom", "momentum at the sink", verbosity, sinkMom);

  std::string proj;
  HGC_options->set("which_projector", "Which projector to use for 3pt function", verbosity, proj);
  WHICHPROJECTOR which_proj=get_projector(proj.c_str());
>>>>>>> origin/TMD

  int tsrc;
  HGC_options->set("t-source", "Time of source vector", verbosity, tsrc);

  int tsnk;
  HGC_options->set("t-sink", "Time of sink vector", verbosity, tsnk);

  std::string fourp_filename;
  HGC_options->set("fourp-filename", "Filename of the four-points correlator", verbosity, fourp_filename);

  std::string qwf_filename;
  HGC_options->set("qwf-filename", "Filename of the quasi-wave function", verbosity, qwf_filename);
<<<<<<< HEAD
  
  initializePLEGMA();

  std::vector<int> PpMom = {0,0,0};

  std::transform(PMom.begin(), PMom.end(), pMom.begin(), PpMom.begin(), std::minus<int>());

  std::vector<int> P2Mom = {0,0,0};

  std::transform(PMom.begin(), PMom.end(), PMom.begin(), P2Mom.begin(), std::plus<int>());
=======

  /* std::string ft1_filename;
  std::string ft2_filename;
  HGC_options->set("ft1-filename", "If added to the momentum transfer delta gives the sink momentum", verbosity, ft1_filename);
  HGC_options->set("ft2-filename", "If added to the momentum transfer delta gives the sink momentum", verbosity, ft2_filename); */
  
  initializePLEGMA();

  //set the 2 transverse directions given WilsDir
  size_t Bdir1, Bdir2;
  switch(WilsDir) {
    case 0:
      Bdir1 = 1;
      Bdir2 = 2;
      break;
    case 1:
      Bdir1 = 0;
      Bdir2 = 2;
      break;
    case 2:
      Bdir1 = 0;
      Bdir2 = 1;
      break;
    default:
      PLEGMA_error("The direction of the WIlson line has to be smaller than 3");
  }

  size_t B_max = 6; //6;
  size_t L_max = 8; //HGC_totalL[WilsDir]/2;
  size_t Z_max = 10; //HGC_totalL[WilsDir]/2;

  std::vector<int> PMom = {0,0,0};
  for(int pi = 0; pi < 3; pi++) {
    PMom[pi] = sinkMom[pi]/2;
  }
  PLEGMA_printf("PMom = %d %d %d\n",PMom[0],PMom[1],PMom[2]);

  std::vector<int> P2Mom = {0,0,0};
  for(int pi = 0; pi < 3; pi++) {
    P2Mom[pi] = 2*sinkMom[pi];
  }
  PLEGMA_printf("P2Mom = %d %d %d\n",P2Mom[0],P2Mom[1],P2Mom[2]);
>>>>>>> origin/TMD

  std::vector<int> zero_mom = {0,0,0};
  
  PLEGMA_Gauge<double> gauge;
  gauge.readFile(latfile, LIME_FORMAT);
  gauge.calculatePlaq();

  initGaugeQuda(gauge, true, QUDA_WILSON_LINKS);
  plaqQuda();

  PLEGMA_Gauge<double> smearedGauge;
<<<<<<< HEAD
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
=======
  smearedGauge.APEsmearing(gauge, nsmearAPE, alphaAPE, 3);
  PLEGMA_printf("Plaquette after smearing:\n");
  smearedGauge.calculatePlaq();
>>>>>>> origin/TMD

  PLEGMA_Gauge<float> gaugeWL;
  gaugeWL.copy(gauge);
  gaugeWL.stoutSmearing(gaugeWL,nStout,rhoStout,3);
<<<<<<< HEAD
  PLEGMA_Su3field<float> *WL = new PLEGMA_Su3field<float>(BOTH);
=======
  /* PLEGMA_Su3field<float> *WL = new PLEGMA_Su3field<float>(BOTH);
>>>>>>> origin/TMD
  PLEGMA_Su3field<float> tmp(BOTH);
  PLEGMA_Su3field<float> *u_s[4];
  for(int idir = 0; idir < 4 ; idir++){
    u_s[idir] = new PLEGMA_Su3field<float>(BOTH);
    u_s[idir]->absorbDir_device(gaugeWL,idir);
  }
  PLEGMA_Su3field<float> *WLIn = new PLEGMA_Su3field<float>(BOTH);
<<<<<<< HEAD
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
=======
  PLEGMA_Su3field<float> *WLExchange = nullptr; */
  
  if(mu<0)  mu*=-1.;
  QUDA_solver *solver = new QUDA_solver(mu);

  PLEGMA_Vector<double> vectorIn;
  PLEGMA_Vector<double> vectorOut;
  PLEGMA_Vector<double> vectorAuxD;
  PLEGMA_Vector<float> vectorAuxF;

  PLEGMA_Propagator<float> *propUPp[numSourcePositions];
  PLEGMA_Propagator<float> *propDNp[numSourcePositions];
  PLEGMA_Propagator<float> *propUPPp[numSourcePositions];
  PLEGMA_Propagator<float> *propDNPp[numSourcePositions];
  for(int is = 0; is < numSourcePositions; is++) {
    propUPp[is] = new PLEGMA_Propagator<float>(BOTH);
    propDNp[is] = new PLEGMA_Propagator<float>(BOTH);
    propUPPp[is] = new PLEGMA_Propagator<float>(BOTH);
    propDNPp[is] = new PLEGMA_Propagator<float>(BOTH);
  }
  PLEGMA_Propagator<float> *propIn = new PLEGMA_Propagator<float>(BOTH);
  PLEGMA_Propagator<float> *propExchange = nullptr;


  PLEGMA_Su3field<float> *su3_1 = new PLEGMA_Su3field<float>(BOTH);
    //PLEGMA_Su3field<float> *su3_21 = new PLEGMA_Su3field<float>(BOTH);
    //PLEGMA_Su3field<float> *su3_22 = new PLEGMA_Su3field<float>(BOTH);
  PLEGMA_Su3field<float> *su3_2_1 = new PLEGMA_Su3field<float>(BOTH);
  PLEGMA_Su3field<float> *su3_2_2 = new PLEGMA_Su3field<float>(BOTH);
  PLEGMA_Su3field<float> *su3_3_1 = new PLEGMA_Su3field<float>(BOTH);
  PLEGMA_Su3field<float> *su3_3_2 = new PLEGMA_Su3field<float>(BOTH);
  PLEGMA_Su3field<float> *su3_4_1 = new PLEGMA_Su3field<float>(BOTH);
  PLEGMA_Su3field<float> *su3_4_2 = new PLEGMA_Su3field<float>(BOTH);
  PLEGMA_Su3field<float> *WL_1 = new PLEGMA_Su3field<float>(BOTH);
  PLEGMA_Su3field<float> *WL_2_1 = new PLEGMA_Su3field<float>(BOTH);
  //PLEGMA_Su3field<float> *WL_3_1 = new PLEGMA_Su3field<float>(BOTH);
  PLEGMA_Su3field<float> *WL_2_2 = new PLEGMA_Su3field<float>(BOTH);
  //PLEGMA_Su3field<float> *WL_3_2 = new PLEGMA_Su3field<float>(BOTH);
  PLEGMA_Su3field<float> *WL_4_1 = new PLEGMA_Su3field<float>(BOTH);
  PLEGMA_Su3field<float> *WL_4_2 = new PLEGMA_Su3field<float>(BOTH);
  // PLEGMA_Su3field<float> *WL_temp = new PLEGMA_Su3field<float>(BOTH);
  // PLEGMA_Su3field<float> *WL_temp1 = new PLEGMA_Su3field<float>(BOTH);
  // PLEGMA_Su3field<float> *WL_temp2 = new PLEGMA_Su3field<float>(BOTH);
  PLEGMA_Su3field<float> *su3_in = new PLEGMA_Su3field<float>(BOTH);
  // PLEGMA_Su3field<float> *WL1 = new PLEGMA_Su3field<float>(BOTH);
  // PLEGMA_Su3field<float> *WL2 = new PLEGMA_Su3field<float>(BOTH);
  PLEGMA_Su3field<float> *su3_exchange = new PLEGMA_Su3field<float>(BOTH);
  PLEGMA_Su3field<float> tmp(BOTH);

  site& source = sourcePositions[0];
  for(int isource = 0; isource < numSourcePositions; isource++) { //loop over number of source-sink separations
    int tsrci = tsrc + 2*isource;
    int tsnki = tsnk + 2*isource;

    if(mu < 0) {
      mu *= -1;
      solver->UpdateSolver();
    }
  
    for(int isc=0;isc<12;isc++){
      vectorAuxD.setUnit((std::vector<int>) {isc});
      vectorIn.absorbTimeslice(vectorAuxD,tsrci,true);
      vectorIn.mulMomentumPhases((std::vector<int>) {PMom[0],PMom[1],PMom[2],0},-1);
      solver->solve(vectorOut,vectorIn);
      vectorAuxF.copy(vectorOut);
      propUPp[isource]->absorb(vectorAuxF, isc/3, isc%3);
    }
  
    for(int isc=0;isc<12;isc++){
      vectorAuxD.setUnit((std::vector<int>) {isc});
      vectorIn.absorbTimeslice(vectorAuxD,tsrci,true);
      vectorIn.mulMomentumPhases((std::vector<int>) {PMom[0],PMom[1],PMom[2],0},+1);
      solver->solve(vectorOut,vectorIn);
      vectorAuxF.copy(vectorOut);
      propUPPp[isource]->absorb(vectorAuxF, isc/3, isc%3);
    }
  
    propUPPp[isource]->rotateToPhysicalBase_device(+1);
    propUPp[isource]->rotateToPhysicalBase_device(+1);
    propUPPp[isource]->applyBoundaries_device(0);
    propUPp[isource]->applyBoundaries_device(0);

  /* PLEGMA_FT<float> ftp(pMom,3);
  PLEGMA_FT<float> ftPp(PpMom,3);
  ftp.apply(*propUPp,FT_GEMV,+1);
  ftPp.apply(*propUPPp);
  ftp.writeFile(ft1_filename, corr_file_format);
  ftPp.writeFile(ft2_filename, corr_file_format);

  WL->setUnit((std::vector<int>) {0,4,8});

  PLEGMA_Correlator<float> corr(corr_space, source);
  corr.setFixMomVec(sinkMom);
  corr.contractTMDWFMesons_Zfac(*propUPPp,*propUPp,*WL,0,0);
  corr.writeFile(twop_filename, corr_file_format); */

    if(mu > 0) {
      mu *= -1;
      solver->UpdateSolver();
    }

    for(int isc=0;isc<12;isc++){
      vectorAuxD.setUnit((std::vector<int>) {isc});
      vectorIn.absorbTimeslice(vectorAuxD,tsnki,true);
      vectorIn.mulMomentumPhases((std::vector<int>) {PMom[0],PMom[1],PMom[2],0},+1);
      solver->solve(vectorOut,vectorIn);
      vectorAuxF.copy(vectorOut);
      propDNp[isource]->absorb(vectorAuxF, isc/3, isc%3);
    }

    for(int isc=0;isc<12;isc++){
      vectorAuxD.setUnit((std::vector<int>) {isc});
      vectorIn.absorbTimeslice(vectorAuxD,tsnki,true);
      vectorIn.mulMomentumPhases((std::vector<int>) {PMom[0],PMom[1],PMom[2],0},-1);
      solver->solve(vectorOut,vectorIn);
      vectorAuxF.copy(vectorOut);
      propDNPp[isource]->absorb(vectorAuxF, isc/3, isc%3);
    }

    propDNp[isource]->rotateToPhysicalBase_device(-1);
    propDNp[isource]->applyBoundaries_device(source[0]);
    propDNPp[isource]->rotateToPhysicalBase_device(-1);
    propDNPp[isource]->applyBoundaries_device(source[0]);

  }//loop over number of source-sink separations
>>>>>>> origin/TMD

  PLEGMA_Correlator<float> corr4p(corr_space, source);
  corr4p.setFixMomVec(P2Mom);

<<<<<<< HEAD
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
  
=======
  PLEGMA_Correlator<float> corrqwf(corr_space, source);
  corrqwf.setFixMomVec(sinkMom);

  //positive L, positive B

  //set the branch 1 as unit
  WL_1->setUnit( (std::vector<int>) {0,4,8});

  for(int is = 0; is < numSourcePositions; is++) {//loop over number of source-sink separations
    int tsrci = tsrc + 2*is;
    corrqwf.contractTMDWFMesons_Zfac(*propUPPp[is],*propUPp[is],*WL_1,0,0,0);
    corrqwf.writeFile((twop_filename+"_tsrc_"+std::to_string(tsrci)+"_stout_"+std::to_string(nStout)+".h5").c_str(), corr_file_format);
  }//loop over number of source-sink separations

  //absorb the gauge directions for building the 4 branches
  su3_1->absorbDir_device(gaugeWL, WilsDir);
  su3_2_1->absorbDir_device(gaugeWL, Bdir1);
  su3_2_2->absorbDir_device(gaugeWL, Bdir2);
  su3_3_1->absorbDir_device(gaugeWL, WilsDir);
  su3_3_2->absorbDir_device(gaugeWL, WilsDir);

    
  for(int l = 0; l <= L_max; l++) {//loop over values of L, building "1" and "3"

    if(l != 0) {
      WL_1->wilsonLineUpdate(*su3_1, tmp, 4 + WilsDir);

      su3_exchange = su3_in; su3_in = su3_2_1; su3_2_1 = su3_exchange;
      su3_2_1->shift(*su3_in, 4 + WilsDir);
      su3_exchange = su3_in; su3_in = su3_2_2; su3_2_2 = su3_exchange;
      su3_2_2->shift(*su3_in, 4 + WilsDir);

    }

    if(l==8 || l==10) {//calculate only for certain l's
      WL_2_1->copy(*WL_1);
      WL_2_2->copy(*WL_1);
      su3_2_1->unload();
      su3_2_2->unload();
      for(int b = 0; b <= B_max; b++) {//loop over values of B, building "2"

        if(b != 0) {

          WL_2_1->wilsonLineUpdate(*su3_2_1, tmp, 4 + Bdir1);

          WL_2_2->wilsonLineUpdate(*su3_2_2, tmp, 4 + Bdir2);

        }

        for(int bi = 0; bi < b; bi++) {
          su3_exchange = su3_in; su3_in = WL_1; WL_1 = su3_exchange;
          WL_1->shift(*su3_in, 4 + Bdir1);
        }
        WL_4_1->UxUdag(*WL_2_1, *WL_1);
        for(int bi = 0; bi < b; bi++) {
          su3_exchange = su3_in; su3_in = WL_1; WL_1 = su3_exchange;
          WL_1->shift(*su3_in, Bdir1);
        }
        for(int bi = 0; bi < b; bi++) {
          su3_exchange = su3_in; su3_in = WL_1; WL_1 = su3_exchange;
          WL_1->shift(*su3_in, 4 + Bdir2);
        }
        WL_4_2->UxUdag(*WL_2_2, *WL_1);
        for(int bi = 0; bi < b; bi++) {
          su3_exchange = su3_in; su3_in = WL_1; WL_1 = su3_exchange;
          WL_1->shift(*su3_in, Bdir2);
        }

        su3_4_1->absorbDir_device(gaugeWL, WilsDir);
        su3_4_2->absorbDir_device(gaugeWL, WilsDir);
        for(int bi = 0; bi < b; bi++) {
          su3_exchange = su3_in; su3_in = su3_4_1; su3_4_1 = su3_exchange;
          su3_4_1->shift(*su3_in, 4 + Bdir1);
          su3_exchange = su3_in; su3_in = su3_4_2; su3_4_2 = su3_exchange;
          su3_4_2->shift(*su3_in, 4 + Bdir2);
        }


        for(int z = 0; z <= Z_max; z++) {//loop over values of Z, building "4"

          if(z != 0) {

            WL_4_1->wilsonLineUpdate(*su3_4_1, tmp, WilsDir);

            WL_4_2->wilsonLineUpdate(*su3_4_2, tmp, WilsDir);

          }

          //uncomment to write the staples, for testing
          //WL1->writeHDF5( (twop_filename + suff11 + "_L_" + std::to_string(l) + "_B_" + std::to_string(b) + "_Z_" + std::to_string(z) + ".h5/").c_str(), true );
          //WL2->writeHDF5( (twop_filename + suff12 + "_L_" + std::to_string(l) + "_B_" + std::to_string(b) + "_Z_" + std::to_string(z) + ".h5/").c_str(), true );

          for(int is = 0; is < numSourcePositions; is++) {//loop over number of source-sink separations
            int tsrci = tsrc + 2*is;
            int tsnki = tsnk + 2*is;
            for(int bi = 0; bi < b; bi++) {
              propExchange = propIn; propIn = propUPp[is]; propUPp[is] = propExchange;
              propUPp[is]->shift(*propIn, 4 + Bdir1);
            }
            if(z == 0){
              for(int bi = 0; bi < b; bi++) {
                propExchange = propIn; propIn = propDNp[is]; propDNp[is] = propExchange;
                propDNp[is]->shift(*propIn, 4 + Bdir1);
              }
              corr4p.contractMesonsFourp_ultralocal(*propUPp[is], *propDNp[is], *propDNPp[is], *propUPPp[is],b);
              corr4p.writeFile((fourp_filename+".Bdir_"+std::to_string(Bdir1)+"_tsrc_"+std::to_string(tsrci)+"_tsnk_"+std::to_string(tsnki)+"_stout_"+std::to_string(nStout)+".h5").c_str(), corr_file_format);
              for(int bi = 0; bi < b; bi++) {
                propExchange = propIn; propIn = propDNp[is]; propDNp[is] = propExchange;
                propDNp[is]->shift(*propIn, Bdir1);
              }
            }
            for(int zi = 0; zi < z; zi++) {
              propExchange = propIn; propIn = propUPp[is]; propUPp[is] = propExchange;
              propUPp[is]->shift(*propIn, WilsDir);
            }
            WL_4_1->conjugate();
            corrqwf.contractTMDWFMesons_Zfac(*propUPPp[is],*propUPp[is],*WL_4_1,l,b,z);
            corrqwf.writeFile((qwf_filename+".Bdir_"+std::to_string(Bdir1)+"_tsrc_"+std::to_string(tsrci)+"_stout_"+std::to_string(nStout)+".h5").c_str(), corr_file_format);
            WL_4_1->conjugate();
            for(int bi = 0; bi < b; bi++) {
              propExchange = propIn; propIn = propUPp[is]; propUPp[is] = propExchange;
              propUPp[is]->shift(*propIn, Bdir1);
            }
            for(int bi = 0; bi < b; bi++) {
              propExchange = propIn; propIn = propUPp[is]; propUPp[is] = propExchange;
              propUPp[is]->shift(*propIn, 4 + Bdir2);
            }
            if(z == 0){
              for(int bi = 0; bi < b; bi++) {
                propExchange = propIn; propIn = propDNp[is]; propDNp[is] = propExchange;
                propDNp[is]->shift(*propIn, 4 + Bdir2);
              }
              corr4p.contractMesonsFourp_ultralocal(*propUPp[is], *propDNp[is], *propDNPp[is], *propUPPp[is],b);
              corr4p.writeFile((fourp_filename+".Bdir_"+std::to_string(Bdir2)+"_tsrc_"+std::to_string(tsrci)+"_tsnk_"+std::to_string(tsnki)+"_stout_"+std::to_string(nStout)+".h5").c_str(), corr_file_format);
              for(int bi = 0; bi < b; bi++) {
                propExchange = propIn; propIn = propDNp[is]; propDNp[is] = propExchange;
                propDNp[is]->shift(*propIn, Bdir2);
              }
            }
            WL_4_2->conjugate();
            corrqwf.contractTMDWFMesons_Zfac(*propUPPp[is],*propUPp[is],*WL_4_2,l,b,z);
            corrqwf.writeFile((qwf_filename+".Bdir_"+std::to_string(Bdir2)+"_tsrc_"+std::to_string(tsrci)+"_stout_"+std::to_string(nStout)+".h5").c_str(), corr_file_format);
            WL_4_2->conjugate();
            for(int zi = 0; zi < z; zi++) {
              propExchange = propIn; propIn = propUPp[is]; propUPp[is] = propExchange;
              propUPp[is]->shift(*propIn, 4 + WilsDir);
            }
            for(int bi = 0; bi < b; bi++) {
              propExchange = propIn; propIn = propUPp[is]; propUPp[is] = propExchange;
              propUPp[is]->shift(*propIn, Bdir2);
            }
          }//loop over number of source-sink separations

        }//loop over values of Z, building "4"

          

      }//loop over values of B, building "2"
      su3_2_1->load();
      su3_2_2->load();
    }//calculate only for certain l's
        
  }//loop over values of L, building "1" and "3"

  //negative L, positive B

  //set the branch 1 as unit
  WL_1->setUnit( (std::vector<int>) {0,4,8});

  //absorb the gauge directions for building the 4 branches
  su3_1->absorbDir_device(gaugeWL, WilsDir);
  su3_2_1->absorbDir_device(gaugeWL, Bdir1);
  su3_2_2->absorbDir_device(gaugeWL, Bdir2);
  su3_3_1->absorbDir_device(gaugeWL, WilsDir);
  su3_3_2->absorbDir_device(gaugeWL, WilsDir);

    
  for(int l = 0; l <= L_max; l++) {//loop over values of L, building "1" and "3"

    if(l != 0) {
      WL_1->wilsonLineUpdate(*su3_1, tmp, WilsDir);

      su3_exchange = su3_in; su3_in = su3_2_1; su3_2_1 = su3_exchange;
      su3_2_1->shift(*su3_in, WilsDir);
      su3_exchange = su3_in; su3_in = su3_2_2; su3_2_2 = su3_exchange;
      su3_2_2->shift(*su3_in, WilsDir);

    }

    if(l==8 || l==10) {//calculate only for certain l's
      WL_2_1->copy(*WL_1);
      WL_2_2->copy(*WL_1);
      su3_2_1->unload();
      su3_2_2->unload();
      for(int b = 0; b <= B_max; b++) {//loop over values of B, building "2"

        if(b != 0) {

          WL_2_1->wilsonLineUpdate(*su3_2_1, tmp, 4 + Bdir1);

          WL_2_2->wilsonLineUpdate(*su3_2_2, tmp, 4 + Bdir2);

        }

        for(int bi = 0; bi < b; bi++) {
          su3_exchange = su3_in; su3_in = WL_1; WL_1 = su3_exchange;
          WL_1->shift(*su3_in, 4 + Bdir1);
        }
        WL_4_1->UxUdag(*WL_2_1, *WL_1);
        for(int bi = 0; bi < b; bi++) {
          su3_exchange = su3_in; su3_in = WL_1; WL_1 = su3_exchange;
          WL_1->shift(*su3_in, Bdir1);
        }
        for(int bi = 0; bi < b; bi++) {
          su3_exchange = su3_in; su3_in = WL_1; WL_1 = su3_exchange;
          WL_1->shift(*su3_in, 4 + Bdir2);
        }
        WL_4_2->UxUdag(*WL_2_2, *WL_1);
        for(int bi = 0; bi < b; bi++) {
          su3_exchange = su3_in; su3_in = WL_1; WL_1 = su3_exchange;
          WL_1->shift(*su3_in, Bdir2);
        }

        su3_4_1->absorbDir_device(gaugeWL, WilsDir);
        su3_4_2->absorbDir_device(gaugeWL, WilsDir);
        for(int bi = 0; bi < b; bi++) {
          su3_exchange = su3_in; su3_in = su3_4_1; su3_4_1 = su3_exchange;
          su3_4_1->shift(*su3_in, 4 + Bdir1);
          su3_exchange = su3_in; su3_in = su3_4_2; su3_4_2 = su3_exchange;
          su3_4_2->shift(*su3_in, 4 + Bdir2);
        }


        for(int z = 0; z <= Z_max; z++) {//loop over values of Z, building "4"

          if(z != 0) {

            WL_4_1->wilsonLineUpdate(*su3_4_1, tmp, 4 + WilsDir);

            WL_4_2->wilsonLineUpdate(*su3_4_2, tmp, 4 + WilsDir);

          }

          //uncomment to write the staples, for testing
          //WL1->writeHDF5( (twop_filename + suff11 + "_L_" + std::to_string(l) + "_B_" + std::to_string(b) + "_Z_" + std::to_string(z) + ".h5/").c_str(), true );
          //WL2->writeHDF5( (twop_filename + suff12 + "_L_" + std::to_string(l) + "_B_" + std::to_string(b) + "_Z_" + std::to_string(z) + ".h5/").c_str(), true );

          for(int is = 0; is < numSourcePositions; is++) {//loop over number of source-sink separations
            int tsrci = tsrc + 2*is;
            int tsnki = tsnk + 2*is;
            for(int bi = 0; bi < b; bi++) {
              propExchange = propIn; propIn = propUPp[is]; propUPp[is] = propExchange;
              propUPp[is]->shift(*propIn, 4 + Bdir1);
            }
            for(int zi = 0; zi < z; zi++) {
              propExchange = propIn; propIn = propUPp[is]; propUPp[is] = propExchange;
              propUPp[is]->shift(*propIn, 4 + WilsDir);
            }
            WL_4_1->conjugate();
            corrqwf.contractTMDWFMesons_Zfac(*propUPPp[is],*propUPp[is],*WL_4_1,-l,b,z);
            corrqwf.writeFile((qwf_filename+".Bdir_"+std::to_string(Bdir1)+"_tsrc_"+std::to_string(tsrci)+"_stout_"+std::to_string(nStout)+".h5").c_str(), corr_file_format);
            WL_4_1->conjugate();
            for(int bi = 0; bi < b; bi++) {
              propExchange = propIn; propIn = propUPp[is]; propUPp[is] = propExchange;
              propUPp[is]->shift(*propIn, Bdir1);
            }
            for(int bi = 0; bi < b; bi++) {
              propExchange = propIn; propIn = propUPp[is]; propUPp[is] = propExchange;
              propUPp[is]->shift(*propIn, 4 + Bdir2);
            }
            WL_4_2->conjugate();
            corrqwf.contractTMDWFMesons_Zfac(*propUPPp[is],*propUPp[is],*WL_4_2,-l,b,z);
            corrqwf.writeFile((qwf_filename+".Bdir_"+std::to_string(Bdir2)+"_tsrc_"+std::to_string(tsrci)+"_stout_"+std::to_string(nStout)+".h5").c_str(), corr_file_format);
            WL_4_2->conjugate();
            for(int zi = 0; zi < z; zi++) {
              propExchange = propIn; propIn = propUPp[is]; propUPp[is] = propExchange;
              propUPp[is]->shift(*propIn, WilsDir);
            }
            for(int bi = 0; bi < b; bi++) {
              propExchange = propIn; propIn = propUPp[is]; propUPp[is] = propExchange;
              propUPp[is]->shift(*propIn, Bdir2);
            }
          }//loop over number of source-sink separations

        }//loop over values of Z, building "4"

          

      }//loop over values of B, building "2"
      su3_2_1->load();
      su3_2_2->load();
    }//calculate only for certain l's
        
  }//loop over values of L, building "1" and "3"

  //positive L, negative B

  //set the branch 1 as unit
  WL_1->setUnit( (std::vector<int>) {0,4,8});

  //absorb the gauge directions for building the 4 branches
  su3_1->absorbDir_device(gaugeWL, WilsDir);
  su3_2_1->absorbDir_device(gaugeWL, Bdir1);
  su3_2_2->absorbDir_device(gaugeWL, Bdir2);
  su3_3_1->absorbDir_device(gaugeWL, WilsDir);
  su3_3_2->absorbDir_device(gaugeWL, WilsDir);

    
  for(int l = 0; l <= L_max; l++) {//loop over values of L, building "1" and "3"

    if(l != 0) {
      WL_1->wilsonLineUpdate(*su3_1, tmp, 4 + WilsDir);

      su3_exchange = su3_in; su3_in = su3_2_1; su3_2_1 = su3_exchange;
      su3_2_1->shift(*su3_in, 4 + WilsDir);
      su3_exchange = su3_in; su3_in = su3_2_2; su3_2_2 = su3_exchange;
      su3_2_2->shift(*su3_in, 4 + WilsDir);

    }

    if(l==8 || l==10) {//calculate only for certain l's
      WL_2_1->copy(*WL_1);
      WL_2_2->copy(*WL_1);
      su3_2_1->unload();
      su3_2_2->unload();
      for(int b = 0; b <= B_max; b++) {//loop over values of B, building "2"

        if(b != 0) {

          WL_2_1->wilsonLineUpdate(*su3_2_1, tmp, Bdir1);

          WL_2_2->wilsonLineUpdate(*su3_2_2, tmp, Bdir2);

        }

        for(int bi = 0; bi < b; bi++) {
          su3_exchange = su3_in; su3_in = WL_1; WL_1 = su3_exchange;
          WL_1->shift(*su3_in, Bdir1);
        }
        WL_4_1->UxUdag(*WL_2_1, *WL_1);
        for(int bi = 0; bi < b; bi++) {
          su3_exchange = su3_in; su3_in = WL_1; WL_1 = su3_exchange;
          WL_1->shift(*su3_in, 4 + Bdir1);
        }
        for(int bi = 0; bi < b; bi++) {
          su3_exchange = su3_in; su3_in = WL_1; WL_1 = su3_exchange;
          WL_1->shift(*su3_in, Bdir2);
        }
        WL_4_2->UxUdag(*WL_2_2, *WL_1);
        for(int bi = 0; bi < b; bi++) {
          su3_exchange = su3_in; su3_in = WL_1; WL_1 = su3_exchange;
          WL_1->shift(*su3_in, 4 + Bdir2);
        }

        su3_4_1->absorbDir_device(gaugeWL, WilsDir);
        su3_4_2->absorbDir_device(gaugeWL, WilsDir);
        for(int bi = 0; bi < b; bi++) {
          su3_exchange = su3_in; su3_in = su3_4_1; su3_4_1 = su3_exchange;
          su3_4_1->shift(*su3_in, Bdir1);
          su3_exchange = su3_in; su3_in = su3_4_2; su3_4_2 = su3_exchange;
          su3_4_2->shift(*su3_in, Bdir2);
        }


        for(int z = 0; z <= Z_max; z++) {//loop over values of Z, building "4"

          if(z != 0) {

            WL_4_1->wilsonLineUpdate(*su3_4_1, tmp, WilsDir);

            WL_4_2->wilsonLineUpdate(*su3_4_2, tmp, WilsDir);

          }

          //uncomment to write the staples, for testing
          //WL1->writeHDF5( (twop_filename + suff11 + "_L_" + std::to_string(l) + "_B_" + std::to_string(b) + "_Z_" + std::to_string(z) + ".h5/").c_str(), true );
          //WL2->writeHDF5( (twop_filename + suff12 + "_L_" + std::to_string(l) + "_B_" + std::to_string(b) + "_Z_" + std::to_string(z) + ".h5/").c_str(), true );

          for(int is = 0; is < numSourcePositions; is++) {//loop over number of source-sink separations
            int tsrci = tsrc + 2*is;
            int tsnki = tsnk + 2*is;
            for(int bi = 0; bi < b; bi++) {
              propExchange = propIn; propIn = propUPp[is]; propUPp[is] = propExchange;
              propUPp[is]->shift(*propIn, Bdir1);
            }
            if(z == 0){
              for(int bi = 0; bi < b; bi++) {
                propExchange = propIn; propIn = propDNp[is]; propDNp[is] = propExchange;
                propDNp[is]->shift(*propIn, Bdir1);
              }
              corr4p.contractMesonsFourp_ultralocal(*propUPp[is], *propDNp[is], *propDNPp[is], *propUPPp[is],-b);
              corr4p.writeFile((fourp_filename+".Bdir_"+std::to_string(Bdir1)+"_tsrc_"+std::to_string(tsrci)+"_tsnk_"+std::to_string(tsnki)+"_stout_"+std::to_string(nStout)+".h5").c_str(), corr_file_format);
              for(int bi = 0; bi < b; bi++) {
                propExchange = propIn; propIn = propDNp[is]; propDNp[is] = propExchange;
                propDNp[is]->shift(*propIn, 4 + Bdir1);
              }
            }
            for(int zi = 0; zi < z; zi++) {
              propExchange = propIn; propIn = propUPp[is]; propUPp[is] = propExchange;
              propUPp[is]->shift(*propIn, WilsDir);
            }
            WL_4_1->conjugate();
            corrqwf.contractTMDWFMesons_Zfac(*propUPPp[is],*propUPp[is],*WL_4_1,l,-b,z);
            corrqwf.writeFile((qwf_filename+".Bdir_"+std::to_string(Bdir1)+"_tsrc_"+std::to_string(tsrci)+"_stout_"+std::to_string(nStout)+".h5").c_str(), corr_file_format);
            WL_4_1->conjugate();
            for(int bi = 0; bi < b; bi++) {
              propExchange = propIn; propIn = propUPp[is]; propUPp[is] = propExchange;
              propUPp[is]->shift(*propIn, 4 + Bdir1);
            }
            for(int bi = 0; bi < b; bi++) {
              propExchange = propIn; propIn = propUPp[is]; propUPp[is] = propExchange;
              propUPp[is]->shift(*propIn, Bdir2);
            }
            if(z == 0){
              for(int bi = 0; bi < b; bi++) {
                propExchange = propIn; propIn = propDNp[is]; propDNp[is] = propExchange;
                propDNp[is]->shift(*propIn, Bdir2);
              }
              corr4p.contractMesonsFourp_ultralocal(*propUPp[is], *propDNp[is], *propDNPp[is], *propUPPp[is],-b);
              corr4p.writeFile((fourp_filename+".Bdir_"+std::to_string(Bdir2)+"_tsrc_"+std::to_string(tsrci)+"_tsnk_"+std::to_string(tsnki)+"_stout_"+std::to_string(nStout)+".h5").c_str(), corr_file_format);
              for(int bi = 0; bi < b; bi++) {
                propExchange = propIn; propIn = propDNp[is]; propDNp[is] = propExchange;
                propDNp[is]->shift(*propIn, 4 + Bdir2);
              }
            }
            WL_4_2->conjugate();
            corrqwf.contractTMDWFMesons_Zfac(*propUPPp[is],*propUPp[is],*WL_4_2,l,-b,z);
            corrqwf.writeFile((qwf_filename+".Bdir_"+std::to_string(Bdir2)+"_tsrc_"+std::to_string(tsrci)+"_stout_"+std::to_string(nStout)+".h5").c_str(), corr_file_format);
            WL_4_2->conjugate();
            for(int zi = 0; zi < z; zi++) {
              propExchange = propIn; propIn = propUPp[is]; propUPp[is] = propExchange;
              propUPp[is]->shift(*propIn, 4 + WilsDir);
            }
            for(int bi = 0; bi < b; bi++) {
              propExchange = propIn; propIn = propUPp[is]; propUPp[is] = propExchange;
              propUPp[is]->shift(*propIn, 4 + Bdir2);
            }
          }//loop over number of source-sink separations

        }//loop over values of Z, building "4"

          

      }//loop over values of B, building "2"
      su3_2_1->load();
      su3_2_2->load();
    }//calculate only for certain l's
        
  }//loop over values of L, building "1" and "3"

  //negative L, negative B

  //set the branch 1 as unit
  WL_1->setUnit( (std::vector<int>) {0,4,8});

  //absorb the gauge directions for building the 4 branches
  su3_1->absorbDir_device(gaugeWL, WilsDir);
  su3_2_1->absorbDir_device(gaugeWL, Bdir1);
  su3_2_2->absorbDir_device(gaugeWL, Bdir2);
  su3_3_1->absorbDir_device(gaugeWL, WilsDir);
  su3_3_2->absorbDir_device(gaugeWL, WilsDir);

    
  for(int l = 0; l <= L_max; l++) {//loop over values of L, building "1" and "3"

    if(l != 0) {
      WL_1->wilsonLineUpdate(*su3_1, tmp, WilsDir);

      su3_exchange = su3_in; su3_in = su3_2_1; su3_2_1 = su3_exchange;
      su3_2_1->shift(*su3_in, WilsDir);
      su3_exchange = su3_in; su3_in = su3_2_2; su3_2_2 = su3_exchange;
      su3_2_2->shift(*su3_in, WilsDir);

    }

    if(l==8 || l==10) {//calculate only for certain l's
      WL_2_1->copy(*WL_1);
      WL_2_2->copy(*WL_1);
      su3_2_1->unload();
      su3_2_2->unload();
      for(int b = 0; b <= B_max; b++) {//loop over values of B, building "2"

        if(b != 0) {

          WL_2_1->wilsonLineUpdate(*su3_2_1, tmp, Bdir1);

          WL_2_2->wilsonLineUpdate(*su3_2_2, tmp, Bdir2);

        }

        for(int bi = 0; bi < b; bi++) {
          su3_exchange = su3_in; su3_in = WL_1; WL_1 = su3_exchange;
          WL_1->shift(*su3_in, Bdir1);
        }
        WL_4_1->UxUdag(*WL_2_1, *WL_1);
        for(int bi = 0; bi < b; bi++) {
          su3_exchange = su3_in; su3_in = WL_1; WL_1 = su3_exchange;
          WL_1->shift(*su3_in, 4 + Bdir1);
        }
        for(int bi = 0; bi < b; bi++) {
          su3_exchange = su3_in; su3_in = WL_1; WL_1 = su3_exchange;
          WL_1->shift(*su3_in, Bdir2);
        }
        WL_4_2->UxUdag(*WL_2_2, *WL_1);
        for(int bi = 0; bi < b; bi++) {
          su3_exchange = su3_in; su3_in = WL_1; WL_1 = su3_exchange;
          WL_1->shift(*su3_in, 4 + Bdir2);
        }

        su3_4_1->absorbDir_device(gaugeWL, WilsDir);
        su3_4_2->absorbDir_device(gaugeWL, WilsDir);
        for(int bi = 0; bi < b; bi++) {
          su3_exchange = su3_in; su3_in = su3_4_1; su3_4_1 = su3_exchange;
          su3_4_1->shift(*su3_in, Bdir1);
          su3_exchange = su3_in; su3_in = su3_4_2; su3_4_2 = su3_exchange;
          su3_4_2->shift(*su3_in, Bdir2);
        }


        for(int z = 0; z <= Z_max; z++) {//loop over values of Z, building "4"

          if(z != 0) {

            WL_4_1->wilsonLineUpdate(*su3_4_1, tmp, 4 + WilsDir);

            WL_4_2->wilsonLineUpdate(*su3_4_2, tmp, 4 + WilsDir);

          }

          //uncomment to write the staples, for testing
          //WL1->writeHDF5( (twop_filename + suff11 + "_L_" + std::to_string(l) + "_B_" + std::to_string(b) + "_Z_" + std::to_string(z) + ".h5/").c_str(), true );
          //WL2->writeHDF5( (twop_filename + suff12 + "_L_" + std::to_string(l) + "_B_" + std::to_string(b) + "_Z_" + std::to_string(z) + ".h5/").c_str(), true );

          for(int is = 0; is < numSourcePositions; is++) {//loop over number of source-sink separations
            int tsrci = tsrc + 2*is;
            int tsnki = tsnk + 2*is;
            for(int bi = 0; bi < b; bi++) {
              propExchange = propIn; propIn = propUPp[is]; propUPp[is] = propExchange;
              propUPp[is]->shift(*propIn, Bdir1);
            }
            for(int zi = 0; zi < z; zi++) {
              propExchange = propIn; propIn = propUPp[is]; propUPp[is] = propExchange;
              propUPp[is]->shift(*propIn, 4 + WilsDir);
            }
            WL_4_1->conjugate();
            corrqwf.contractTMDWFMesons_Zfac(*propUPPp[is],*propUPp[is],*WL_4_1,-l,-b,z);
            corrqwf.writeFile((qwf_filename+".Bdir_"+std::to_string(Bdir1)+"_tsrc_"+std::to_string(tsrci)+"_stout_"+std::to_string(nStout)+".h5").c_str(), corr_file_format);
            WL_4_1->conjugate();
            for(int bi = 0; bi < b; bi++) {
              propExchange = propIn; propIn = propUPp[is]; propUPp[is] = propExchange;
              propUPp[is]->shift(*propIn, 4 + Bdir1);
            }
            for(int bi = 0; bi < b; bi++) {
              propExchange = propIn; propIn = propUPp[is]; propUPp[is] = propExchange;
              propUPp[is]->shift(*propIn, Bdir2);
            }
            WL_4_2->conjugate();
            corrqwf.contractTMDWFMesons_Zfac(*propUPPp[is],*propUPp[is],*WL_4_2,-l,-b,z);
            corrqwf.writeFile((qwf_filename+".Bdir_"+std::to_string(Bdir2)+"_tsrc_"+std::to_string(tsrci)+"_stout_"+std::to_string(nStout)+".h5").c_str(), corr_file_format);
            WL_4_2->conjugate();
            for(int zi = 0; zi < z; zi++) {
              propExchange = propIn; propIn = propUPp[is]; propUPp[is] = propExchange;
              propUPp[is]->shift(*propIn, WilsDir);
            }
            for(int bi = 0; bi < b; bi++) {
              propExchange = propIn; propIn = propUPp[is]; propUPp[is] = propExchange;
              propUPp[is]->shift(*propIn, 4 + Bdir2);
            }
          }//loop over number of source-sink separations

        }//loop over values of Z, building "4"

          

      }//loop over values of B, building "2"
      su3_2_1->load();
      su3_2_2->load();
    }//calculate only for certain l's
        
  }//loop over values of L, building "1" and "3"
  
  /* delete propUPp;
  delete propUPPp;
  delete propDNp;
  delete propDNPp;
  delete propIn; */

>>>>>>> origin/TMD
  delete solver;
  
  finalize();
  
  return 0;
}
