#include <PLEGMA.h>
#include <PLEGMA_utils.h>

using namespace plegma;
using namespace quda;

int main(int argc, char **argv) {

  static std::vector<std::string> listOpt = {"verbosity", "load-gauge", "nsmear-APE", "alpha-APE", "nsmear-gauss", "alpha-gauss",
					     "nsrc", "src-filename", "maxQsq",  "corr-file-format",
					     "corr-space","xiMomSm","gammas"};

  initializeOptions(argc, argv, true, listOpt);

  double rhoStout;
  HGC_options->set("rho_stout", "Rho parameter stout smearing", verbosity, rhoStout);

  int nStout;
  HGC_options->set("nstout", "n parameter stout smearing", verbosity, nStout);

  size_t WilsDir;
  HGC_options->set("wilson_direction", "Direction of the wilson line", verbosity, WilsDir);
  if(WilsDir>N_DIMS) PLEGMA_error("The direction of the WIlson line has to be smaller than 3");

  std::vector<int> sinkMom(4);
  HGC_options->set("sinkMom", "momentum at the sink", verbosity, sinkMom);

  std::string qwf_filename;
  HGC_options->set("qwf-filename", "Filename of the quasi-wave function", verbosity, qwf_filename);

  std::vector<int> Lvals;
  HGC_options->set("Lvals", "values of L for which to calculate the staple", verbosity, Lvals);


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

  size_t B_max = 2;
  size_t L_max = Lvals[Lvals.size()-1];

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

  // Reading from Lime file and loading to device
  PLEGMA_Gauge<double> gauge;
  gauge.readFile(latfile, LIME_FORMAT);
  gauge.calculatePlaq();

  // Loading to QUDA and computing plaquette also there
  initGaugeQuda(gauge, true, QUDA_WILSON_LINKS);
  plaqQuda();
 
  // Smearing
  PLEGMA_Gauge<double> smearedGauge;
  smearedGauge.APEsmearing(gauge, nsmearAPE, alphaAPE, 3);
  PLEGMA_printf("Plaquette after smearing:\n");
  smearedGauge.calculatePlaq();
  PLEGMA_Gauge<double> smearedGaugeMinus;
  smearedGaugeMinus.APEsmearing(gauge, nsmearAPE, alphaAPE, 3);

  PLEGMA_printf("Sink Momentum px %d, py %d, pz %d, pt %d\n",
		sinkMom[0],sinkMom[1],sinkMom[2],sinkMom[3]);
  if(maxQsq < sinkMom[0]*sinkMom[0]+sinkMom[1]*sinkMom[1]+sinkMom[2]*sinkMom[2])
      PLEGMA_error("maxQsq does not include the sink momentum\n");
  
  //Momentum smearing: put the momentum phase to smeared gauge field
  std::complex<double> momSmScale[N_DIMS];
  std::complex<double> I(0,1);
  for(int i = 0 ; i < N_DIMS; i++){
    momSmScale[i] = std::exp(-(xiMomSm*2.*PI*PMom[i]/HGC_totalL[i])*I);
  }
  smearedGauge.scaleDirWise(momSmScale);
  for(int i = 0 ; i < N_DIMS; i++){
    momSmScale[i] = std::exp(+(xiMomSm*2.*PI*PMom[i]/HGC_totalL[i])*I);
  }
  smearedGaugeMinus.scaleDirWise(momSmScale);

  //copy gauge to gaugeWL for building of the staples
  PLEGMA_Gauge<float> gaugeWL;
  gaugeWL.copy(gauge);

  //apply the stout smearing steps
  if(nStout > 0) gaugeWL.stoutSmearing(gaugeWL, nStout, rhoStout, 3);

  // Extracting the spacial sink momentum from the sink 4-momentum  
  std::vector<int> sinkMom_3D(3);
  std::copy(sinkMom.begin(),sinkMom.begin()+3,sinkMom_3D.begin());

  // ensuring mu positive
  if(mu<0)  mu*=-1.;

  QUDA_solver *solver = new QUDA_solver(mu);

  PLEGMA_Vector<double> vectorIn;
  PLEGMA_Vector<double> vectorOut;
  PLEGMA_Vector<double> vectorAuxD;
  PLEGMA_Vector<float> vectorAuxF;
  PLEGMA_Propagator<float> *propUP = new PLEGMA_Propagator<float>(BOTH);
  PLEGMA_Propagator<float> *propDN = new PLEGMA_Propagator<float>(BOTH);
  PLEGMA_Propagator<float> *propIn = new PLEGMA_Propagator<float>(BOTH);
  PLEGMA_Propagator<float> *propExchange = nullptr;
  PLEGMA_Propagator<float> *propF = new PLEGMA_Propagator<float>(BOTH);

  PLEGMA_Su3field<float> *su3_1 = new PLEGMA_Su3field<float>(BOTH);
  PLEGMA_Su3field<float> *su3_2_1 = new PLEGMA_Su3field<float>(BOTH);
  PLEGMA_Su3field<float> *su3_2_2 = new PLEGMA_Su3field<float>(BOTH);
  PLEGMA_Su3field<float> *su3_4_1 = new PLEGMA_Su3field<float>(BOTH);
  PLEGMA_Su3field<float> *su3_4_2 = new PLEGMA_Su3field<float>(BOTH);
  PLEGMA_Su3field<float> *WL_1 = new PLEGMA_Su3field<float>(BOTH);
  PLEGMA_Su3field<float> *WL_2_1 = new PLEGMA_Su3field<float>(BOTH);
  PLEGMA_Su3field<float> *WL_2_2 = new PLEGMA_Su3field<float>(BOTH);
  PLEGMA_Su3field<float> *WL_4_1 = new PLEGMA_Su3field<float>(BOTH);
  PLEGMA_Su3field<float> *WL_4_2 = new PLEGMA_Su3field<float>(BOTH);
  PLEGMA_Su3field<float> *su3_in = new PLEGMA_Su3field<float>(BOTH);
  PLEGMA_Su3field<float> *su3_exchange = new PLEGMA_Su3field<float>(BOTH);
  PLEGMA_Su3field<float> tmp(BOTH);

  //WL testing
#if 1
  PLEGMA_Su3field<float> *WL = new PLEGMA_Su3field<float>(BOTH);
  PLEGMA_Su3field<float> *u_s[4];
  for(int idir = 0; idir < 4 ; idir++){
    u_s[idir] = new PLEGMA_Su3field<float>(BOTH);
    u_s[idir]->absorbDir_device(gaugeWL,idir);
  }
  PLEGMA_Su3field<float> *WLIn = new PLEGMA_Su3field<float>(BOTH);
  PLEGMA_Su3field<float> *WLExchange = nullptr;
#endif

  for(int isource = 0; isource < numSourcePositions; isource++) { //loop over source positions
    
    PLEGMA_Correlator<float> corrqwf(corr_space, sourcePositions[isource], maxQsq);
    corrqwf.setFixMomVec(sinkMom);
    
    // UP prop
    if(mu<0){
      mu *= (-1);
      solver->UpdateSolver();
    }

    for(int isc = 0 ; isc < 12 ; isc++) {
      vectorAuxD.pointSource(sourcePositions[isource], isc/3, isc%3, DEVICE);
      vectorIn.gaussianSmearing(vectorAuxD, smearedGauge, nsmearGauss, alphaGauss);
      PLEGMA_printf("Going to invert UP for component %d\n", isc);
      solver->solve(vectorOut, vectorIn);
      vectorAuxD.gaussianSmearing(vectorOut, smearedGauge , nsmearGauss, alphaGauss);
      vectorAuxF.copy(vectorAuxD);
      propUP->absorb(vectorAuxF, isc/3, isc%3);
    }

    // DN prop
    if(mu>0){
      mu *= (-1);
      solver->UpdateSolver();
    }

    for(int isc = 0 ; isc < 12 ; isc++) {
      vectorAuxD.pointSource(sourcePositions[isource], isc/3, isc%3, DEVICE);
      vectorIn.gaussianSmearing(vectorAuxD, smearedGaugeMinus, nsmearGauss, alphaGauss);
      PLEGMA_printf("Going to invert DN for component %d\n", isc);
      solver->solve(vectorOut, vectorIn);
      vectorAuxF.copy(vectorOut);
      propDN->absorb(vectorAuxF, isc/3, isc%3);
      vectorAuxD.gaussianSmearing(vectorOut, smearedGaugeMinus , nsmearGauss, alphaGauss);
      vectorAuxF.copy(vectorAuxD);
      propDN->absorb(vectorAuxF, isc/3, isc%3);
    }

  propUP->unload();
  propDN->unload();

  propUP->rotateToPhysicalBase_device(+1);
  propUP->applyBoundaries_device(sourcePositions[isource][DIM_T]);

  //set the 4 branches as unit
  WL_1->setUnit( (std::vector<int>) {0,4,8});

  //absorb the gauge directions for building the 4 branches
  su3_1->absorbDir_device(gaugeWL, WilsDir);

  for(int l = 0; l <= L_max; l++) { //loop over values of L
    if(l != 0) {
      WL_1->wilsonLineUpdate(*su3_1, tmp, 4 + WilsDir);
    }
    if(std::count(Lvals.begin(),Lvals.end(),l)) { //calculate only for certain l's
      WL_2_1->copy(*WL_1);
      WL_2_2->copy(*WL_1);
      su3_2_1->absorbDir_device(gaugeWL, Bdir1);
      su3_2_2->absorbDir_device(gaugeWL, Bdir2);
      for(int li = 0; li < l; li++) {
        su3_exchange = su3_in; su3_in = su3_2_1; su3_2_1 = su3_exchange;
        su3_2_1->shift(*su3_in, 4 + WilsDir);
        su3_exchange = su3_in; su3_in = su3_2_2; su3_2_2 = su3_exchange;
        su3_2_2->shift(*su3_in, 4 + WilsDir);
      }
      for(int b = 0; b <= B_max; b++) { //loop over values of B
        if(b != 0) {
          WL_2_1->wilsonLineUpdate(*su3_2_1, tmp, 4 + Bdir1);
          WL_2_2->wilsonLineUpdate(*su3_2_2, tmp, 4 + Bdir2);
        }
        WL_4_1->copy(*WL_2_1);
        WL_4_2->copy(*WL_2_2);
        su3_4_1->absorbDir_device(gaugeWL, WilsDir);
        su3_4_2->absorbDir_device(gaugeWL, WilsDir);
        for(int li = 0; li < l; li ++) {
          su3_exchange = su3_in; su3_in = su3_4_1; su3_4_1 = su3_exchange;
          su3_4_1->shift(*su3_in, 4 + WilsDir);
          su3_exchange = su3_in; su3_in = su3_4_2; su3_4_2 = su3_exchange;
          su3_4_2->shift(*su3_in, 4 + WilsDir);
        }
        for(int bi = 0; bi < b; bi++) {
          su3_exchange = su3_in; su3_in = su3_4_1; su3_4_1 = su3_exchange;
          su3_4_1->shift(*su3_in, 4 + Bdir1);
          su3_exchange = su3_in; su3_in = su3_4_2; su3_4_2 = su3_exchange;
          su3_4_2->shift(*su3_in, 4 + Bdir2);
        }
        for(int z = 1; z < 2*l; z++) { //loop over values of Z
          WL_4_1->wilsonLineUpdate(*su3_4_1, tmp, WilsDir);
          WL_4_2->wilsonLineUpdate(*su3_4_2, tmp, WilsDir);
          int zval = z - l;
          propF->copy(*propDN);
          for(int bi = 0; bi < b; bi++) {
            propExchange = propIn; propIn = propF; propF = propExchange;
            propF->shift(*propIn, 4 + Bdir1);
          }
          if(zval < 0) {
            for(int zi = 0; zi < abs(zval); zi++) {
              propExchange = propIn; propIn = propF; propF = propExchange;
              propF->shift(*propIn, 4 + WilsDir);
            }
          }else {
            for(int zi = 0; zi < abs(zval); zi++) {
              propExchange = propIn; propIn = propF; propF = propExchange;
              propF->shift(*propIn, WilsDir);
            }
          }
          propF->rotateToPhysicalBase_device(-1);
          propF->applyBoundaries_device(sourcePositions[isource][DIM_T]);
          corrqwf.contractTMDWFMesonsNew(*propUP,*propF,*WL_4_1,l,b,zval);
          corrqwf.writeFile((qwf_filename+".Bdir_"+std::to_string(Bdir1)+"_stout_"+std::to_string(nStout)+".h5").c_str(), corr_file_format);
          propF->copy(*propDN);
          for(int bi = 0; bi < b; bi++) {
            propExchange = propIn; propIn = propF; propF = propExchange;
            propF->shift(*propIn, 4 + Bdir2);
          }
          if(zval < 0) {
            for(int zi = 0; zi < abs(zval); zi++) {
              propExchange = propIn; propIn = propF; propF = propExchange;
              propF->shift(*propIn, 4 + WilsDir);
            }
          }else {
            for(int zi = 0; zi < abs(zval); zi++) {
              propExchange = propIn; propIn = propF; propF = propExchange;
              propF->shift(*propIn, WilsDir);
            }
          }
          propF->rotateToPhysicalBase_device(-1);
          propF->applyBoundaries_device(sourcePositions[isource][DIM_T]);
          corrqwf.contractTMDWFMesonsNew(*propUP,*propF,*WL_4_2,l,b,zval);
          corrqwf.writeFile((qwf_filename+".Bdir_"+std::to_string(Bdir2)+"_stout_"+std::to_string(nStout)+".h5").c_str(), corr_file_format);
        }//loop over values of Z
      }//loop over values of B
    }//calculate only for certain l's
  }//loop over values of L

  }//loop over source positions

  //WL testing
#if 0
  WL->setUnit((std::vector<int>) {0,4,8});
  int spath[8] = {2,2,2,2,1,1,6,6};
  std::vector<int> vspath(spath,spath+8);
  WL->path(vspath, u_s, tmp);
  for(int ii = 0; ii < 2; ii++) {
    WLExchange = WLIn; WLIn = WL; WL = WLExchange;
    WL->shift(*WLIn,2);
  }
  for(int ii = 0; ii < 2; ii++) {
    WLExchange = WLIn; WLIn = WL; WL = WLExchange;
    WL->shift(*WLIn,5);
  }
  for(int ii = 0; ii < 4; ii++) {
    WLExchange = WLIn; WLIn = WL; WL = WLExchange;
    WL->shift(*WLIn,6);
  }
  WL->writeHDF5("WL_xcheck.h5");
#endif

  delete propUP;
  delete propDN;

  delete solver;

  finalize();
  return 0;
}