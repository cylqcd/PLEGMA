#include <PLEGMA.h>
#include <PLEGMA_utils.h>

std::vector<double> runtime;
#define TIME(fnc) runtime.push_back(MPI_Wtime()); fnc;                     \
  PLEGMA_printf("TIME for "#fnc" %f sec\n", MPI_Wtime()-runtime.back());  \
  runtime.pop_back()

std::vector<std::thread> threads;
//#define THREAD(fnc) threads.push_back(std::thread([=]() { TIME(fnc); }))
#define THREAD(fnc) saveTuneCache(false); TIME(fnc)

using namespace plegma;
using namespace quda;

int main(int argc, char **argv) {

  static std::vector<std::string> listOpt = {"verbosity", "load-gauge", "nsmear-APE", "alpha-APE", "nsmear-gauss", "alpha-gauss",
               "nsrc", "src-filename", "maxQsq", "twop-filename","threep-filename",  "corr-file-format",
               "corr-space", "tSinks","xiMomSm","gammas"};

  initializeOptions(argc, argv, true, listOpt);


  if(mu<0){
    mu *= -1.0;
  }

  size_t WilsDir;
  HGC_options->set("wilson_direction", "Direction of the wilson line", verbosity, WilsDir);
  if(WilsDir>N_DIMS) PLEGMA_error("The direction of the WIlson line has to be smaller than 3");

  double rhoStout;
  HGC_options->set("rho_stout", "Rho parameter stout smearing", verbosity, rhoStout);

  size_t nStout;
  HGC_options->set("nstout", "Number of stout smearing steps", verbosity, nStout);
  
  std::string proj;
  HGC_options->set("which_projector", "Which projector to use for 3pt function", verbosity, proj);
  WHICHPROJECTOR which_proj=get_projector(proj.c_str());
  
  bool calc3pt = true;
  HGC_options->set("calc3pt", "If true then the 3pt function is computed", verbosity, calc3pt);

  std::vector<int> sinkMom(4);
  HGC_options->set("sinkMom", "momentum at the sink", verbosity, sinkMom);

  std::vector<int> Lvals;
  HGC_options->set("Lvals", "values of L for which to calculate the staple", verbosity, Lvals);

  std::string which_particle = "proton";
  HGC_options->set("which_particle", "Choice of the nucleon interpolator to insert in the three point function (neutron,proton)", verbosity, which_particle);
  WHICHPARTICLE nucleon = get_particle(which_particle.c_str());
  
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

  size_t B_max = 6;
  size_t L_max = Lvals[Lvals.size()-1];
  size_t Z_max = HGC_totalL[WilsDir]/2;

  
  
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

  PLEGMA_printf("Sink Momentum px %d, py %d, pz %d, pt %d\n",
		sinkMom[0],sinkMom[1],sinkMom[2],sinkMom[3]);
  
  //Momentum smearing: put the momentum phase to smeared gauge field
  std::complex<double> momSmScale[N_DIMS];
  std::complex<double> I(0,1);
  for(int i = 0 ; i < N_DIMS; i++){
    momSmScale[i] = std::exp(-(xiMomSm*2.*PI*sinkMom[i]/HGC_totalL[i])*I);
  }
  smearedGauge.scaleDirWise(momSmScale);

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

  TIME(QUDA_solver *solver = new QUDA_solver(mu));

  PLEGMA_Vector<double> vectorIn;
  PLEGMA_Vector<double> vectorOut;
  PLEGMA_Vector<double> vectorAuxD;
  PLEGMA_Vector<float> vectorAuxF;
  PLEGMA_Propagator<float> *propUP = new PLEGMA_Propagator<float>(BOTH);
  PLEGMA_Propagator<float> *propDN = new PLEGMA_Propagator<float>(BOTH);
  PLEGMA_Propagator<float> *propUP_SS = new PLEGMA_Propagator<float>(BOTH);
  PLEGMA_Propagator<float> *propDN_SS = new PLEGMA_Propagator<float>(BOTH);
  PLEGMA_Propagator<float> *seqPropOut1 = new PLEGMA_Propagator<float>(BOTH);
  PLEGMA_Propagator<float> *seqPropOut2 = new PLEGMA_Propagator<float>(BOTH);
  PLEGMA_Propagator<float> *propIn = new PLEGMA_Propagator<float>(BOTH);
  PLEGMA_Propagator<float> *propExchange = nullptr;

  PLEGMA_Propagator3D<float> propUP3D;
  PLEGMA_Propagator3D<float> propDN3D;

  std::string suff11, suff12, suff21, suff22;

  //PLEGMA_Su3field<float> su3;
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

//ignore this part, only used for testing purposes

#if 0

  PLEGMA_Su3field<float> *WL = new PLEGMA_Su3field<float>(BOTH);
  PLEGMA_Su3field<float> *WLIn = new PLEGMA_Su3field<float>(BOTH);
  PLEGMA_Su3field<float> *WLExchange = nullptr;
  PLEGMA_Su3field<float> *u_s[4];

  for(int idir = 0; idir < 4 ; idir++){
    u_s[idir] = new PLEGMA_Su3field<float>(BOTH);
    u_s[idir]->absorbDir_device(gaugeWL,idir);
  }


#endif

  

  for(int isource=0;isource<numSourcePositions;isource++){

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
      vectorAuxF.copy(vectorOut);
      propUP->absorb(vectorAuxF, isc/3, isc%3);
      vectorAuxD.gaussianSmearing(vectorOut, smearedGauge , nsmearGauss, alphaGauss);
      vectorAuxF.copy(vectorAuxD);
      propUP_SS->absorb(vectorAuxF, isc/3, isc%3);
    }

    // DN prop
    if(mu>0){
      mu *= (-1);
      solver->UpdateSolver();
    }

    for(int isc = 0 ; isc < 12 ; isc++) {
      vectorAuxD.pointSource(sourcePositions[isource], isc/3, isc%3, DEVICE);
      vectorIn.gaussianSmearing(vectorAuxD, smearedGauge, nsmearGauss, alphaGauss);
      PLEGMA_printf("Going to invert DN for component %d\n", isc);
      solver->solve(vectorOut, vectorIn);
      vectorAuxF.copy(vectorOut);
      propDN->absorb(vectorAuxF, isc/3, isc%3);
      vectorAuxD.gaussianSmearing(vectorOut, smearedGauge , nsmearGauss, alphaGauss);
      vectorAuxF.copy(vectorAuxD);
      propDN_SS->absorb(vectorAuxF, isc/3, isc%3);
    }

  propUP->unload();
  propDN->unload();

  propUP_SS->unload();
  propDN_SS->unload();

#if 1

  for(int ts=0;ts<tSinks.size();ts++) {

    PLEGMA_Correlator<float> corrThrpWL(corr_space, sourcePositions[isource], 0, (tSinks[ts]+1));

    int signPer = (tSinks[ts] + sourcePositions[isource][3]) >= HGC_totalL[3] ? -1 : +1;
    int global_fixSinkTime = (tSinks[ts] + sourcePositions[isource][3])%HGC_totalL[3]; 
    int my_fixSinkTime = global_fixSinkTime - HGC_procPosition[3] * HGC_localL[3];
    bool is_myST = (my_fixSinkTime >= 0) && ( my_fixSinkTime < HGC_localL[3] );

    propUP3D.absorb(*propUP_SS,global_fixSinkTime);
    propDN3D.absorb(*propDN_SS, global_fixSinkTime);


    //calculate 3 point functions

    //CP1 part
    if(nucleon == PROTON){
      if(mu<0) {
        mu*=-1.;
        solver->UpdateSolver();
      }
    }else {
      if(mu>0) {
        mu*=-1.;
        solver->UpdateSolver();
      }
    }

    for(int nu = 0 ; nu < 4 ; nu++) {//loop over spin
      for(int c2 = 0 ; c2 < 3 ; c2++) {//loop over color

        PLEGMA_Vector3D<float> vectorAux3D;

        if(nucleon == PROTON) vectorAux3D.seqSourceNucleon(propUP3D, which_proj, nucleon, nu, c2);
        else vectorAux3D.seqSourceNucleon(propDN3D, which_proj, nucleon, nu, c2);
        vectorAux3D.mulMomentumPhases(sinkMom,-1); // put momentum at the sink
        std::complex<float> Isingle(0,1);
        float phase = 2.*PI*(((float) sinkMom[0] * sourcePositions[isource][0])/HGC_totalL[0]
          + ((float)sinkMom[1] * sourcePositions[isource][1])/HGC_totalL[1]
          + ((float)sinkMom[2] * sourcePositions[isource][2])/HGC_totalL[2]);
        vectorAux3D.cscale(std::exp<float>(+phase*Isingle)); // put momentum from the point source
        vectorAux3D.conjugate();
        vectorAux3D.apply_gamma(G5);
        vectorAuxF.absorb(vectorAux3D, global_fixSinkTime);
        vectorAuxD.copy(vectorAuxF);
        vectorIn.gaussianSmearing(vectorAuxD,smearedGauge, nsmearGauss, alphaGauss);

        double norm = vectorIn.norm();
        vectorIn.cscale(1/norm);
        solver->solve(vectorOut, vectorIn);
        vectorOut.cscale(norm);
        vectorAuxF.copy(vectorOut);
        seqPropOut1->absorb(vectorAuxF, nu, c2);
        }//loop over color

      }//loop over spin

    seqPropOut1->apply_gamma(G5);
    seqPropOut1->conjugate();
    seqPropOut1->unload();

    int signProps1 = (nucleon == PROTON) ? -1: +1;

    //repeat for CP2 part
    if(nucleon == PROTON){
      if(mu>0) {
        mu*=-1.;
        solver->UpdateSolver();
      }
    }else {
      if(mu<0) {
        mu*=-1.;
        solver->UpdateSolver();
      }
    }

    for(int nu = 0 ; nu < 4 ; nu++) {//loop over spin
      for(int c2 = 0 ; c2 < 3 ; c2++) {//loop over color

        PLEGMA_Vector3D<float> vectorAux3D;

        if(nucleon == PROTON) vectorAux3D.seqSourceNucleon(propUP3D, propDN3D, which_proj, nucleon, nu, c2);
        else vectorAux3D.seqSourceNucleon(propDN3D, propUP3D, which_proj, nucleon, nu, c2);

        vectorAux3D.mulMomentumPhases(sinkMom,-1); // put momentum at the sink
        std::complex<float> Isingle(0,1);
        float phase = 2.*PI*(((float) sinkMom[0] * sourcePositions[isource][0])/HGC_totalL[0]
          + ((float)sinkMom[1] * sourcePositions[isource][1])/HGC_totalL[1]
          + ((float)sinkMom[2] * sourcePositions[isource][2])/HGC_totalL[2]);
        vectorAux3D.cscale(std::exp<float>(+phase*Isingle)); // put momentum from the point source
        vectorAux3D.conjugate();
        vectorAux3D.apply_gamma(G5);
        vectorAuxF.absorb(vectorAux3D, global_fixSinkTime);
        vectorAuxD.copy(vectorAuxF);
        vectorIn.gaussianSmearing(vectorAuxD,smearedGauge, nsmearGauss, alphaGauss);

        double norm = vectorIn.norm();
        vectorIn.cscale(1/norm);
        solver->solve(vectorOut, vectorIn);
        vectorOut.cscale(norm);
        vectorAuxF.copy(vectorOut);
        seqPropOut2->absorb(vectorAuxF, nu, c2);

      }//loop over color

    }//loop over spin

    seqPropOut2->apply_gamma(G5);
    seqPropOut2->conjugate();
    seqPropOut2->unload();

    int signProps2 = (nucleon == PROTON) ? +1: -1;


    //propF11 for staple in Bdir1 and propF12 for staple in Bdir2
    PLEGMA_Propagator<float> *propF = new PLEGMA_Propagator<float>(BOTH);
      

      /**********************************************************************
       * now we build the staples
       * idea is to build in 4 branches
       * 
       *                     _______2_______
       *                    +               +
       *                    |               |
       *                    |               |
       *                    |               |
       *                    1               3
       *                    |               |
       *                    |               |
       *                    |               |
       *                                    +
       *                                    |
       *                                    |
       *                                    4
       *                                    |
       *                                    |
       * 
       * also do the staple by taking "2" in two directions Bdir1 and Bdir2
       * this means we calculate "1" only once
       * also do it step wise in "2" which is the B direction
       * so build "1", "2" and "3", then build "4" stepwise for fixed "1", "2" and "3"
       * then shift "3" one step in B direction and update "2"
       * do this untill you reach B_max
       * then shift "2" one step in WilsDir and update both "1" and "3"
       * now follow the same procedure but in the opposite direction for B for "2", that is you start with B_max and go to zero
       * do similar thing for "4" as well by going in opposite direction, start with Z_max and go to zero
       * keep doing this, alternating the direction of B and Z, untill you reach L_max


      ***********************************************************************/

    //set the 4 branches as unit
    WL_1->setUnit( (std::vector<int>) {0,4,8});

    //absorb the gauge directions for building the 4 branches
    su3_1->absorbDir_device(gaugeWL, WilsDir);
    su3_2_1->absorbDir_device(gaugeWL, Bdir1);
    su3_2_2->absorbDir_device(gaugeWL, Bdir2);
      


    suff11 = "_CP1_stout_"+std::to_string(nStout)+"_Bdir_"+std::to_string(Bdir1)+"_Plus_";
    suff12 = "_CP1_stout_"+std::to_string(nStout)+"_Bdir_"+std::to_string(Bdir2)+"_Plus_";

    suff21 = "_CP2_stout_"+std::to_string(nStout)+"_Bdir_"+std::to_string(Bdir1)+"_Plus_";
    suff22 = "_CP2_stout_"+std::to_string(nStout)+"_Bdir_"+std::to_string(Bdir2)+"_Plus_";

    for(int l = 0; l <= L_max; l++) {//loop over values of L, building "1" and "3"

      if(l != 0) {

        WL_1->wilsonLineUpdate(*su3_1, tmp, 4 + WilsDir);

        su3_exchange = su3_in; su3_in = su3_2_1; su3_2_1 = su3_exchange;
        su3_2_1->shift(*su3_in, 4 + WilsDir);

                    su3_exchange = su3_in; su3_in = su3_2_2; su3_2_2 = su3_exchange;
        su3_2_2->shift(*su3_in, 4 + WilsDir);

      }

      if(std::count(Lvals.begin(), Lvals.end(), l)) {//calculate only for certain l's
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

            if(nucleon == PROTON) propF->copy(*propDN);
            else propF->copy(*propUP);
            for(int bi = 0; bi < b; bi++) {
              propExchange = propIn; propIn = propF; propF = propExchange;
              propF->shift(*propIn, 4 + Bdir1);
            }
            for(int zi = 0; zi < z; zi++) {
              propExchange = propIn; propIn = propF; propF = propExchange;
              propF->shift(*propIn, WilsDir);
            }

            corrThrpWL.contractNucleonThrp_staple(*seqPropOut1, *propF, *WL_4_1, signProps1, gammas, l, b, z);
            if(signPer < 0) for(int iv = 0 ; iv < corrThrpWL.getTotalSize()*2; iv++) corrThrpWL.H_elem()[iv] *= signPer;
            corrThrpWL.writeHDF5( (threep_filename + suff11 + "_ts_" + std::to_string(tSinks[ts]) + ".h5").c_str());

            if(nucleon == PROTON) propF->copy(*propDN);
            else propF->copy(*propUP);
            for(int bi = 0; bi < b; bi++) {
              propExchange = propIn; propIn = propF; propF = propExchange;
              propF->shift(*propIn, 4 + Bdir2);
            }
            for(int zi = 0; zi < z; zi++) {
              propExchange = propIn; propIn = propF; propF = propExchange;
              propF->shift(*propIn, WilsDir);
            }

            corrThrpWL.contractNucleonThrp_staple(*seqPropOut1, *propF, *WL_4_2, signProps1, gammas, l, b, z);
            if(signPer < 0) for(int iv = 0 ; iv < corrThrpWL.getTotalSize()*2; iv++) corrThrpWL.H_elem()[iv] *= signPer;
            corrThrpWL.writeHDF5( (threep_filename + suff12 + "_ts_" + std::to_string(tSinks[ts]) + ".h5").c_str());

            if(nucleon == PROTON) propF->copy(*propUP);
            else propF->copy(*propDN);
            for(int bi = 0; bi < b; bi++) {
              propExchange = propIn; propIn = propF; propF = propExchange;
              propF->shift(*propIn, 4 + Bdir1);
            }
            for(int zi = 0; zi < z; zi++) {
              propExchange = propIn; propIn = propF; propF = propExchange;
              propF->shift(*propIn, WilsDir);
            }

            corrThrpWL.contractNucleonThrp_staple(*seqPropOut2, *propF, *WL_4_1, signProps2, gammas, l, b, z);
            if(signPer < 0) for(int iv = 0 ; iv < corrThrpWL.getTotalSize()*2; iv++) corrThrpWL.H_elem()[iv] *= signPer;
            corrThrpWL.writeHDF5( (threep_filename + suff21 + "_ts_" + std::to_string(tSinks[ts]) + ".h5").c_str());

            if(nucleon == PROTON) propF->copy(*propUP);
            else propF->copy(*propDN);
            for(int bi = 0; bi < b; bi++) {
              propExchange = propIn; propIn = propF; propF = propExchange;
              propF->shift(*propIn, 4 + Bdir2);
            }
            for(int zi = 0; zi < z; zi++) {
              propExchange = propIn; propIn = propF; propF = propExchange;
              propF->shift(*propIn, WilsDir);
            }

            corrThrpWL.contractNucleonThrp_staple(*seqPropOut2, *propF, *WL_4_2, signProps2, gammas, l, b, z);
            if(signPer < 0) for(int iv = 0 ; iv < corrThrpWL.getTotalSize()*2; iv++) corrThrpWL.H_elem()[iv] *= signPer;
            corrThrpWL.writeHDF5( (threep_filename + suff22 + "_ts_" + std::to_string(tSinks[ts]) + ".h5").c_str());

            

          }//loop over values of Z, building "4"

          

        }//loop over values of B, building "2"
        su3_2_1->load();
        su3_2_2->load();
        //WL_1->load();
      }//calculate only for certain l's
        
    }//loop over values of L, building "1" and "3"


      //now repeat the same thing only going in negative boost direction for the staple
      //set the 4 branches as unit
      WL_1->setUnit( (std::vector<int>) {0,4,8});

      //absorb the gauge directions for building the 4 branches
      su3_1->absorbDir_device(gaugeWL, WilsDir);
      su3_2_1->absorbDir_device(gaugeWL, Bdir1);
      su3_2_2->absorbDir_device(gaugeWL, Bdir2);

      suff11 = "_CP1_stout_"+std::to_string(nStout)+"_Bdir_"+std::to_string(Bdir1)+"_Minus_";
      suff12 = "_CP1_stout_"+std::to_string(nStout)+"_Bdir_"+std::to_string(Bdir2)+"_Minus_";

      suff21 = "_CP2_stout_"+std::to_string(nStout)+"_Bdir_"+std::to_string(Bdir1)+"_Minus_";
      suff22 = "_CP2_stout_"+std::to_string(nStout)+"_Bdir_"+std::to_string(Bdir2)+"_Minus_";

      for(int l = 0; l <= L_max; l++) {//loop over values of L, building "1" and "3"

        if(l != 0) {

          WL_1->wilsonLineUpdate(*su3_1, tmp, WilsDir);

          su3_exchange = su3_in; su3_in = su3_2_1; su3_2_1 = su3_exchange;
          su3_2_1->shift(*su3_in, WilsDir);

          su3_exchange = su3_in; su3_in = su3_2_2; su3_2_2 = su3_exchange;
          su3_2_2->shift(*su3_in, WilsDir);

        }

        if(std::count(Lvals.begin(), Lvals.end(), l)) {//calculate only for certain l's
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

            if(nucleon == PROTON) propF->copy(*propDN);
            else propF->copy(*propUP);
            for(int bi = 0; bi < b; bi++) {
              propExchange = propIn; propIn = propF; propF = propExchange;
              propF->shift(*propIn, 4 + Bdir1);
            }
            for(int zi = 0; zi < z; zi++) {
              propExchange = propIn; propIn = propF; propF = propExchange;
              propF->shift(*propIn, 4 + WilsDir);
            }

            corrThrpWL.contractNucleonThrp_staple(*seqPropOut1, *propF, *WL_4_1, signProps1, gammas, l, b, z);
            if(signPer < 0) for(int iv = 0 ; iv < corrThrpWL.getTotalSize()*2; iv++) corrThrpWL.H_elem()[iv] *= signPer;
            corrThrpWL.writeHDF5( (threep_filename + suff11 + "_ts_" + std::to_string(tSinks[ts]) + ".h5").c_str());

            if(nucleon == PROTON) propF->copy(*propDN);
            else propF->copy(*propUP);
            for(int bi = 0; bi < b; bi++) {
              propExchange = propIn; propIn = propF; propF = propExchange;
              propF->shift(*propIn, 4 + Bdir2);
            }
            for(int zi = 0; zi < z; zi++) {
              propExchange = propIn; propIn = propF; propF = propExchange;
              propF->shift(*propIn, 4 + WilsDir);
            }

            corrThrpWL.contractNucleonThrp_staple(*seqPropOut1, *propF, *WL_4_2, signProps1, gammas, l, b, z);
            if(signPer < 0) for(int iv = 0 ; iv < corrThrpWL.getTotalSize()*2; iv++) corrThrpWL.H_elem()[iv] *= signPer;
            corrThrpWL.writeHDF5( (threep_filename + suff12 + "_ts_" + std::to_string(tSinks[ts]) + ".h5").c_str());

            if(nucleon == PROTON) propF->copy(*propUP);
            else propF->copy(*propDN);
            for(int bi = 0; bi < b; bi++) {
              propExchange = propIn; propIn = propF; propF = propExchange;
              propF->shift(*propIn, 4 + Bdir1);
            }
            for(int zi = 0; zi < z; zi++) {
              propExchange = propIn; propIn = propF; propF = propExchange;
              propF->shift(*propIn, 4 + WilsDir);
            }

            corrThrpWL.contractNucleonThrp_staple(*seqPropOut2, *propF, *WL_4_1, signProps2, gammas, l, b, z);
            if(signPer < 0) for(int iv = 0 ; iv < corrThrpWL.getTotalSize()*2; iv++) corrThrpWL.H_elem()[iv] *= signPer;
            corrThrpWL.writeHDF5( (threep_filename + suff21 + "_ts_" + std::to_string(tSinks[ts]) + ".h5").c_str());

            if(nucleon == PROTON) propF->copy(*propUP);
            else propF->copy(*propDN);
            for(int bi = 0; bi < b; bi++) {
              propExchange = propIn; propIn = propF; propF = propExchange;
              propF->shift(*propIn, 4 + Bdir2);
            }
            for(int zi = 0; zi < z; zi++) {
              propExchange = propIn; propIn = propF; propF = propExchange;
              propF->shift(*propIn, 4 + WilsDir);
            }

            corrThrpWL.contractNucleonThrp_staple(*seqPropOut2, *propF, *WL_4_2, signProps2, gammas, l, b, z);
            if(signPer < 0) for(int iv = 0 ; iv < corrThrpWL.getTotalSize()*2; iv++) corrThrpWL.H_elem()[iv] *= signPer;
            corrThrpWL.writeHDF5( (threep_filename + suff22 + "_ts_" + std::to_string(tSinks[ts]) + ".h5").c_str());

            

          }//loop over values of Z, building "4"

          

        }//loop over values of B, building "2"
        su3_2_1->load();
        su3_2_2->load();
      }//calculate only for certain l's
        
      }//loop over values of L, building "1" and "3"


  }

#endif

#if 0
  int z1 = 1;
  int L_DIR = WilsDir;
  int B_DIR = 0;
  int B_LEN = 1;
  int L_LEN = 2;
  for(int z_step=0 ; z_step <= z1 ; z_step++){

            for(int b=0 ; b <= B_LEN ; b++){

              for(int l=0 ; l <= L_LEN ; l++){

                WL->setUnit((std::vector<int>) {0,4,8});

                int spath[(2*l)+b+z_step];

                if((l!=0)||(b!=0)||(z_step!=0)){
                  for(int i=0;i<l;i++){
                    spath[i] = 4+L_DIR;
                  }
                  for(int j=l;j<(l+b);j++){
                    spath[j] = 4+B_DIR;
                  }
                  for(int k=(l+b);k<((2*l)+b+z_step);k++){
                    spath[k] = L_DIR;
                  }

                  std::vector<int> vspath(spath,spath+(2*l)+b+z_step);

                  WL->path(vspath, u_s, tmp);

                  /* for(int k=(l+b);k<((2*l)+b+z_step);k++){
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
                  } */

                }

                WL->writeHDF5( (twop_filename + suff12 + "_xcheck_L_" + std::to_string(l) + "_B_" + std::to_string(b) + "_Z_" + std::to_string(z_step) + ".h5/").c_str(), true );
              }
            }
          }


#endif

  propUP_SS->load();
  propDN_SS->load();

  /* for(int nu = 0 ; nu < 4 ; nu++)
    for(int c2 = 0 ; c2 < 3 ; c2++){
      vectorAuxF.absorb(*propUP, nu, c2);
      vectorAuxD.copy(vectorAuxF);
      vectorOut.gaussianSmearing(vectorAuxD, smearedGauge, nsmearGauss, alphaGauss);
      vectorAuxF.copy(vectorOut);
      propUP->absorb(vectorAuxF, nu, c2);

      vectorAuxF.absorb(*propDN, nu, c2);
      vectorAuxD.copy(vectorAuxF);
      vectorOut.gaussianSmearing(vectorAuxD, smearedGauge, nsmearGauss, alphaGauss);
      vectorAuxF.copy(vectorOut);
      propDN->absorb(vectorAuxF, nu, c2);
      
    } */

  
  propUP_SS->rotateToPhysicalBase_device(+1);
  propDN_SS->rotateToPhysicalBase_device(-1);
  propUP_SS->applyBoundaries_device(sourcePositions[isource][3]);
  propDN_SS->applyBoundaries_device(sourcePositions[isource][3]);
  
  PLEGMA_Correlator<float> corr(corr_space, sourcePositions[isource], 0);
  corr.setFixMomVec(sinkMom_3D);
  
  TIME(corr.contractBaryons(*propUP_SS, *propDN_SS));
  THREAD(corr.writeFile((twop_filename).c_str(), corr_file_format));
  
  }

  delete propUP;
  delete propDN;

  delete solver;

  finalize();
  return 0;

}
