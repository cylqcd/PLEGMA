#include <PLEGMA.h>
#include <PLEGMA_utils.h>

using namespace plegma;
using namespace quda;
static std::vector<std::string> listOpt = {"verbosity", "load-gauge", "nsrc","src-filename"};
// we use src-filename to provide the momenta we want to do

int main(int argc, char **argv)
{
  initializeOptions(argc, argv, true, listOpt);
  //================ Add your options in this between initializeOptions and initializePLEGMA ================//
  int L_LEN;
  HGC_options->set("l-len", "length of l", verbosity, L_LEN);
  int L_DIR;
  HGC_options->set("l-dir", "direction of l", verbosity, L_DIR);
  int B_LEN;
  HGC_options->set("b-len", "length of b", verbosity, B_LEN);
  int B_DIR;
  HGC_options->set("b-dir", "direction of b", verbosity, B_DIR);
  int z;
  HGC_options->set("z-len", "length of z", verbosity, z);
  int nStout;
  HGC_options->set("n_stout", "n parameter stout smearing", verbosity, nStout);
  double rhoStout;
  HGC_options->set("rho_stout", "Rho parameter stout smearing", verbosity, rhoStout);
  std::string boundaryCond = "antiperiodic";
  HGC_options->set("boundary-condition", "If we want periodic or antiperiodic in the temporal direction", verbosity, boundaryCond);
  std::string filesPrefix="./";
  HGC_options->set("output-path", "Path to the directory to dump results", verbosity, filesPrefix);
  //==========================//
  initializePLEGMA();

  // Read the configurations //
  // Warning gauge fixing is needed //
  PLEGMA_Gauge<double> gauge(BOTH,FIRST_VERTEX);
  gauge.readFile(latfile, LIME_FORMAT);
  gauge.calculatePlaq();
  initGaugeQuda(gauge, boundaryCond == "antiperiodic");
  plaqQuda();
  std::string confStr=splitStrFwd(latfile,'.');
  if(boundaryCond == "antiperiodic") applyBoundaryConditions(gauge,true);
  /////////////////////////////

  if(mu < 0) mu = -mu;
  QUDA_solver solver(mu);
  PLEGMA_Vector<double> vecIn;
  PLEGMA_Vector<double> vecOut;
  PLEGMA_Propagator<double> *rprop_d = new PLEGMA_Propagator<double>(BOTH,FIRST_VERTEX);
  PLEGMA_Propagator<double> *lprop_d = new PLEGMA_Propagator<double>(BOTH,FIRST_VERTEX);
  PLEGMA_Propagator<double> *propIn = new PLEGMA_Propagator<double>(BOTH,FIRST_VERTEX);
  PLEGMA_Propagator<double> *propExchange = nullptr;
  
  PLEGMA_Gauge<double> gaugeWL(BOTH,FIRST_VERTEX);
  gaugeWL.copy(gauge);
  gaugeWL.stoutSmearing(gaugeWL,nStout,rhoStout,3);
  PLEGMA_Su3field<double> *WL = new PLEGMA_Su3field<double>(BOTH,FIRST_VERTEX);
  PLEGMA_Su3field<double> tmp(BOTH,FIRST_VERTEX);
  PLEGMA_Su3field<double> *u_s[4];
  for(int idir = 0; idir < 4 ; idir++){
    u_s[idir] = new PLEGMA_Su3field<double>(BOTH,FIRST_VERTEX);
    u_s[idir]->absorbDir_device(gaugeWL,idir);
  }
  PLEGMA_Su3field<double> *WLIn = new PLEGMA_Su3field<double>(BOTH,FIRST_VERTEX);
  PLEGMA_Su3field<double> *WLExchange = nullptr;

  site source({0,0,0,0});
  std::vector<GAMMAS> gammas = {ONE,G1,G2,G3,G4,G5,G5G1,G5G2,G5G3,G5G4,S12,S13,S23,S41,S42,S43};
  
  PLEGMA_Correlator<double> corr(MOMENTUM_SPACE, source, 0, HGC_totalL[3]);
  
  for(int i = 0; i < numSourcePositions; i++){ // iterate over the momenta we want to do
    site& sS = sourcePositions[i];
    std::vector<int> mom= {sS[0],sS[1],sS[2],sS[3]};
    std::string pxpypzpt = "px" + std::to_string(sS[0]) + "py" + std::to_string(sS[1]) +
      "pz" + std::to_string(sS[2]) + "pt" + std::to_string(sS[3]);
    //      outName = pathOut + sourceType + "_" + pxpypzpt + "." + confStr + ".data"; 	

    if(mu<0) {
      mu*=-1.;
      solver.UpdateSolver();
    }
    
    for(int isc =0; isc < 12; isc++){ // compute the prop with mom source
      std::vector<int> scInd = {isc};
      vecIn.setUnit(scInd);

      vecIn.mulMomentumPhases((std::vector<int>) {sS[0],sS[1],sS[2],sS[3]},+1);
      //      vecIn.rotate_uk_ch();
      if(boundaryCond == "antiperiodic") vecIn.mulThetaPhase(1.,true);
      vecIn.apply_gamma(G4,LEFT);
      vecIn.apply_gamma(G5,LEFT);
      solver.solve(vecOut,vecIn);
      vecOut.apply_gamma(G5,LEFT);
      vecOut.apply_gamma(G4,LEFT);
      rprop_d->absorb(vecOut, isc/3, isc%3); // propagator in g0 convention in UKQCD (implicit flavor flip from g5g4)
    }
    //rprop_d->rotateToPhysicalBase_device(-1); // propagator in g0 convention in chiral (flipped flavor sign)

    // compute the prop in momentum space
    PLEGMA_FT<double> ft((std::vector<double>) {(double)mom[0], (double)mom[1], (double)mom[2], mom[3]+0.5},4,false);
    ft.apply(*rprop_d, FT_GEMV,-1);
    ft.writeASCII(filesPrefix+"new_GdpropMom_mu000533_twistbase_"+pxpypzpt+"_conf_"+confStr+".dat",0);

    ///////////
    lprop_d->copy(*rprop_d);

    lprop_d->apply_gamma(G5,LEFT);
    lprop_d->apply_gamma(G5,RIGHT);

    //lprop_d->rotateToPhysicalBase_device(-1);
    //lprop_d->rotateToPhysicalBase_device(-1);
    lprop_d->conjugate(); // this is to make the dagger, trans is in the code

    for(int z_step=0 ; z_step <= z ; z_step++){

      if(z_step>0){
        propExchange = propIn; propIn = rprop_d; rprop_d = propExchange;
        rprop_d->shift(*propIn, (4+L_DIR)%8);
      }

      for(int b=0 ; b <= B_LEN ; b++){

        if(b>0){
          propExchange = propIn; propIn = rprop_d; rprop_d = propExchange;
          rprop_d->shift(*propIn, (4+B_DIR)%8);
        }

        for(int l=0 ; l <= L_LEN ; l++){

          WL->setUnit((std::vector<int>) {0,4,8});

          int spath[(2*l)+b+z_step];

          if((l!=0)||(b!=0)||(z_step!=0)){
            for(int i=0;i<l;i++){
              spath[i] = (4+L_DIR)%8;
	    }
            for(int j=l;j<(l+b);j++){
              spath[j] = B_DIR;
            }
            for(int k=(l+b);k<((2*l)+b+z_step);k++){
              spath[k] = (L_DIR)%8;
            }

            std::vector<int> vspath(spath,spath+(2*l)+b+z_step);

	    WL->path(vspath, u_s, tmp);

            for(int k=(l+b);k<((2*l)+b+z_step);k++){
              WLExchange = WLIn; WLIn = WL; WL = WLExchange;
              WL->shift(*WLIn,(4+L_DIR)%8);
            }
            for(int j=l;j<(l+b);j++){
              WLExchange = WLIn; WLIn = WL; WL = WLExchange;
              WL->shift(*WLIn,(4+B_DIR)%8);
            }
            for(int i=0;i<l;i++){
              WLExchange = WLIn; WLIn = WL; WL = WLExchange;
              WL->shift(*WLIn,(L_DIR)%8);
            }

          }
	  
          corr.contractNucleonThrp_staple(*lprop_d, *rprop_d, *WL, 0, gammas, true, b, l, z_step);
          corr.writeFile(filesPrefix+"new_Staple_mu000533_twistbase_ud_stout5_"+pxpypzpt+"_conf_"+confStr,HDF5_FORMAT);

	}
      }
	  
      for(int b=0 ; b < B_LEN ; b++){
        propExchange = propIn; propIn = rprop_d; rprop_d = propExchange;
        rprop_d->shift(*propIn, B_DIR);
      }
    }
  }

  delete rprop_d;
  delete lprop_d;
  delete propIn;

  for(int idir = 0; idir < 4 ; idir++){
    delete u_s[idir];
  }
  delete WL;
  delete WLIn;
  
  finalize();
  return 0;
}
