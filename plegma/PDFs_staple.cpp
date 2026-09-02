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
  
  std::string proj;
  HGC_options->set("which_projector", "Which projector to use for 3pt function", verbosity, proj);
  WHICHPROJECTOR which_proj=get_projector(proj.c_str());

  std::vector<int> sinkMom(4);
  HGC_options->set("sinkMom", "momentum at the sink", verbosity, sinkMom);

  std::string which_particle = "proton";
  HGC_options->set("which_particle", "Choice of the nucleon interpolator to insert in the three point function (neutron,proton)", verbosity, which_particle);
  WHICHPARTICLE nucleon = get_particle(which_particle.c_str());

  size_t Zdir = 2;
  HGC_options->set("Zdir", "Direction of longitudinal part of staple", verbosity, Zdir);

  size_t Zmax = 0;
  HGC_options->set("Zmax", "Maximum length of the longitudinal direction of staple", verbosity, Zmax);

  size_t Tmax = 0;
  HGC_options->set("Tmax", "Maximum length of the time direction of staple", verbosity, Tmax);

  double rhoStout;
  HGC_options->set("rho_stout", "Rho parameter stout smearing", verbosity, rhoStout);

  size_t nStout;
  HGC_options->set("nstout", "Number of stout smearing steps", verbosity, nStout);
  
  initializePLEGMA();

  if(Tmax == 0) Tmax = HGC_totalL[3]/2;
  if(Zmax == 0) Zmax = HGC_totalL[Zdir]/2;

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

  // Extracting the spacial sink momentum from the sink 4-momentum  
  std::vector<int> sinkMom_3D(3);
  std::copy(sinkMom.begin(),sinkMom.begin()+3,sinkMom_3D.begin());

  PLEGMA_Gauge<float> gaugeWL;
  gaugeWL.copy(gauge);

  //apply the stout smearing steps
  if(nStout > 0) gaugeWL.stoutSmearing(gaugeWL, nStout, rhoStout, 4);
  
  // ensuring mu positive
  if(mu<0)  mu*=-1.;

  TIME(QUDA_solver *solver = new QUDA_solver(mu));

  PLEGMA_Vector<double> vectorIn;
  PLEGMA_Vector<double> vectorOut;
  PLEGMA_Vector<double> vectorAuxD;
  PLEGMA_Vector<float> vectorAuxF;
  PLEGMA_Propagator<float> *propUP = new PLEGMA_Propagator<float>(BOTH);
  PLEGMA_Propagator<float> *propDN = new PLEGMA_Propagator<float>(BOTH);
  PLEGMA_Propagator<float> *seqPropOut1 = new PLEGMA_Propagator<float>(BOTH);
  PLEGMA_Propagator<float> *seqPropOut2 = new PLEGMA_Propagator<float>(BOTH);
  PLEGMA_Propagator<float> *propIn = new PLEGMA_Propagator<float>(BOTH);
  PLEGMA_Propagator<float> *propExchange = new PLEGMA_Propagator<float>(BOTH);
  PLEGMA_Propagator<float> *propF1 = new PLEGMA_Propagator<float>(BOTH);
  PLEGMA_Propagator<float> *propF2 = new PLEGMA_Propagator<float>(BOTH);

  PLEGMA_Propagator3D<float> propUP3D;
  PLEGMA_Propagator3D<float> propDN3D;

  PLEGMA_Su3field<float> *su3_1 = new PLEGMA_Su3field<float>(BOTH);
  PLEGMA_Su3field<float> *su3_2 = new PLEGMA_Su3field<float>(BOTH);
  PLEGMA_Su3field<float> *WL_1 = new PLEGMA_Su3field<float>(BOTH);
  PLEGMA_Su3field<float> *WL_2 = new PLEGMA_Su3field<float>(BOTH);
  PLEGMA_Su3field<float> *WL_3 = new PLEGMA_Su3field<float>(BOTH);
  PLEGMA_Su3field<float> *WL = new PLEGMA_Su3field<float>(BOTH);
  PLEGMA_Su3field<float> *su3_in = new PLEGMA_Su3field<float>(BOTH);
  PLEGMA_Su3field<float> *su3_exchange = new PLEGMA_Su3field<float>(BOTH);
  PLEGMA_Su3field<float> tmp(BOTH);

  std::string suff11, suff12, suff21, suff22;

  for(int isource=0;isource<numSourcePositions;isource++){//loop over source positions
    
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
    }

    for(int ts=0;ts<tSinks.size();ts++) {//loop over tSinks

      PLEGMA_Correlator<float> corrThrpWL(corr_space, sourcePositions[isource], 0, (tSinks[ts]+1));
      
      int signPer = (tSinks[ts] + sourcePositions[isource][3]) >= HGC_totalL[3] ? -1 : +1;
      int global_fixSinkTime = (tSinks[ts] + sourcePositions[isource][3])%HGC_totalL[3]; 
      int my_fixSinkTime = global_fixSinkTime - HGC_procPosition[3] * HGC_localL[3];
      bool is_myST = (my_fixSinkTime >= 0) && ( my_fixSinkTime < HGC_localL[3] );
    
      for(int nu = 0 ; nu < 4 ; nu++)
      for(int c2 = 0 ; c2 < 3 ; c2++){
        vectorAuxF.absorb(*propUP, nu, c2);
        vectorAuxD.copy(vectorAuxF);
        vectorOut.gaussianSmearing(vectorAuxD, smearedGauge, nsmearGauss, alphaGauss);
        vectorAuxF.copy(vectorOut);
        propUP3D.absorb(vectorAuxF,global_fixSinkTime, nu, c2);

        vectorAuxF.absorb(*propDN, nu, c2);
        vectorAuxD.copy(vectorAuxF);
        vectorOut.gaussianSmearing(vectorAuxD, smearedGauge, nsmearGauss, alphaGauss);
        vectorAuxF.copy(vectorOut);
        propDN3D.absorb(vectorAuxF,global_fixSinkTime, nu, c2);
      }

      //calculate sequential propagators

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

      int signProps1 = (nucleon == PROTON) ? -1: +1;

      int signProps2 = (nucleon == PROTON) ? +1: -1;

      //set the 4 branches as unit
      WL_1->setUnit( (std::vector<int>) {0,4,8});
    
      suff11 = "_CP1_stout_"+std::to_string(nStout)+"_TPlus_ZPlus_";
      suff12 = "_CP1_stout_"+std::to_string(nStout)+"_TPlus_ZMinus_";

      suff21 = "_CP2_stout_"+std::to_string(nStout)+"_TPlus_ZPlus_";
      suff22 = "_CP2_stout_"+std::to_string(nStout)+"_TPlus_ZMinus_";

      //absorb the gauge directions for building the 4 branches
      su3_1->absorbDir_device(gaugeWL, 3);

      for(int t = 0; t <= Tmax; t++) {//loop over values of L, building "1" and "3"

        if(t != 0) {

          WL_1->wilsonLineUpdate(*su3_1, tmp, 7);

        }

        WL_2->copy(*WL_1);
        WL_3->copy(*WL_1);
        su3_2->absorbDir_device(gaugeWL, Zdir);
        for(int ti = 0; ti < t; ti++){
          su3_exchange = su3_in; su3_in = su3_2; su3_2 = su3_exchange;
          su3_2->shift(*su3_in, 7);
        }

        if(nucleon == PROTON) propF1->copy(*propDN);
        else propF1->copy(*propUP);
    
        if(nucleon == PROTON) propF2->copy(*propUP);
        else propF2->copy(*propDN);
        for(int z = 0; z <= Zmax; z++) {//loop over values of B, building "2"

          if(z != 0) {

            WL_2->wilsonLineUpdate(*su3_2, tmp, 4 + Zdir);

            su3_exchange = su3_in; su3_in = WL_3; WL_3 = su3_exchange;
            WL_3->shift(*su3_in, 4 + Zdir);

            propExchange = propIn; propIn = propF1; propF1 = propExchange;
            propF1->shift(*propIn, 4 + Zdir);

            propExchange = propIn; propIn = propF2; propF2 = propExchange;
            propF2->shift(*propIn, 4 + Zdir);

          }

          WL->UxUdag(*WL_2, *WL_3);

          //uncomment to write the staples, for testing
          //WL->writeHDF5( (twop_filename + suff11 + "_L_" + std::to_string(l) + "_B_" + std::to_string(b) + "_Z_" + std::to_string(z) + ".h5/").c_str(), true );

          corrThrpWL.contractNucleonThrp_staple(*seqPropOut1, *propF1, *WL, signProps1, gammas, t, z, 0);
          if(signPer < 0) for(int iv = 0 ; iv < corrThrpWL.getTotalSize()*2; iv++) corrThrpWL.H_elem()[iv] *= signPer;
          corrThrpWL.writeHDF5( (threep_filename + suff11 + "_ts_" + std::to_string(tSinks[ts]) + ".h5").c_str());

          corrThrpWL.contractNucleonThrp_staple(*seqPropOut2, *propF2, *WL, signProps2, gammas, t, z, 0);
          if(signPer < 0) for(int iv = 0 ; iv < corrThrpWL.getTotalSize()*2; iv++) corrThrpWL.H_elem()[iv] *= signPer;
          corrThrpWL.writeHDF5( (threep_filename + suff21 + "_ts_" + std::to_string(tSinks[ts]) + ".h5").c_str());         

        }//loop over values of B, building "2"

        WL_2->copy(*WL_1);
        WL_3->copy(*WL_1);
        su3_2->absorbDir_device(gaugeWL, Zdir);
        for(int ti = 0; ti < t; ti++){
          su3_exchange = su3_in; su3_in = su3_2; su3_2 = su3_exchange;
          su3_2->shift(*su3_in, 7);
        }

        if(nucleon == PROTON) propF1->copy(*propDN);
        else propF1->copy(*propUP);
    
        if(nucleon == PROTON) propF2->copy(*propUP);
        else propF2->copy(*propDN);
        for(int z = 0; z <= Zmax; z++) {//loop over values of B, building "2"

          if(z != 0) {

            WL_2->wilsonLineUpdate(*su3_2, tmp, Zdir);

            su3_exchange = su3_in; su3_in = WL_3; WL_3 = su3_exchange;
            WL_3->shift(*su3_in, Zdir);

            propExchange = propIn; propIn = propF1; propF1 = propExchange;
            propF1->shift(*propIn, Zdir);

            propExchange = propIn; propIn = propF2; propF2 = propExchange;
            propF2->shift(*propIn, Zdir);

          }

          WL->UxUdag(*WL_2, *WL_3);

          //uncomment to write the staples, for testing
          //WL->writeHDF5( (twop_filename + suff11 + "_L_" + std::to_string(l) + "_B_" + std::to_string(b) + "_Z_" + std::to_string(z) + ".h5/").c_str(), true );

          corrThrpWL.contractNucleonThrp_staple(*seqPropOut1, *propF1, *WL, signProps1, gammas, t, z, 0);
          if(signPer < 0) for(int iv = 0 ; iv < corrThrpWL.getTotalSize()*2; iv++) corrThrpWL.H_elem()[iv] *= signPer;
          corrThrpWL.writeHDF5( (threep_filename + suff12 + "_ts_" + std::to_string(tSinks[ts]) + ".h5").c_str());

          corrThrpWL.contractNucleonThrp_staple(*seqPropOut2, *propF2, *WL, signProps2, gammas, t, z, 0);
          if(signPer < 0) for(int iv = 0 ; iv < corrThrpWL.getTotalSize()*2; iv++) corrThrpWL.H_elem()[iv] *= signPer;
          corrThrpWL.writeHDF5( (threep_filename + suff22 + "_ts_" + std::to_string(tSinks[ts]) + ".h5").c_str());         

        }//loop over values of B, building "2"
        
      }//loop over values of L, building "1" and "3"


      //set the 4 branches as unit
      WL_1->setUnit( (std::vector<int>) {0,4,8});
    
      suff11 = "_CP1_stout_"+std::to_string(nStout)+"_TMinus_ZPlus_";
      suff12 = "_CP1_stout_"+std::to_string(nStout)+"_TMinus_ZMinus_";

      suff21 = "_CP2_stout_"+std::to_string(nStout)+"_TMinus_ZPlus_";
      suff22 = "_CP2_stout_"+std::to_string(nStout)+"_TMinus_ZMinus_";

      //absorb the gauge directions for building the 4 branches
      su3_1->absorbDir_device(gaugeWL, 3);

      for(int t = 0; t <= Tmax; t++) {//loop over values of L, building "1" and "3"

        if(t != 0) {

          WL_1->wilsonLineUpdate(*su3_1, tmp, 3);

        }

        WL_2->copy(*WL_1);
        WL_3->copy(*WL_1);
        su3_2->absorbDir_device(gaugeWL, Zdir);
        for(int ti = 0; ti < t; ti++){
          su3_exchange = su3_in; su3_in = su3_2; su3_2 = su3_exchange;
          su3_2->shift(*su3_in, 3);
        }

        if(nucleon == PROTON) propF1->copy(*propDN);
        else propF1->copy(*propUP);
    
        if(nucleon == PROTON) propF2->copy(*propUP);
        else propF2->copy(*propDN);
        for(int z = 0; z <= Zmax; z++) {//loop over values of B, building "2"

          if(z != 0) {

            WL_2->wilsonLineUpdate(*su3_2, tmp, 4 + Zdir);

            su3_exchange = su3_in; su3_in = WL_3; WL_3 = su3_exchange;
            WL_3->shift(*su3_in, 4 + Zdir);

            propExchange = propIn; propIn = propF1; propF1 = propExchange;
            propF1->shift(*propIn, 4 + Zdir);

            propExchange = propIn; propIn = propF2; propF2 = propExchange;
            propF2->shift(*propIn, 4 + Zdir);

          }

          WL->UxUdag(*WL_2, *WL_3);

          //uncomment to write the staples, for testing
          //WL->writeHDF5( (twop_filename + suff11 + "_L_" + std::to_string(l) + "_B_" + std::to_string(b) + "_Z_" + std::to_string(z) + ".h5/").c_str(), true );

          corrThrpWL.contractNucleonThrp_staple(*seqPropOut1, *propF1, *WL, signProps1, gammas, t, z, 0);
          if(signPer < 0) for(int iv = 0 ; iv < corrThrpWL.getTotalSize()*2; iv++) corrThrpWL.H_elem()[iv] *= signPer;
          corrThrpWL.writeHDF5( (threep_filename + suff11 + "_ts_" + std::to_string(tSinks[ts]) + ".h5").c_str());

          corrThrpWL.contractNucleonThrp_staple(*seqPropOut2, *propF2, *WL, signProps2, gammas, t, z, 0);
          if(signPer < 0) for(int iv = 0 ; iv < corrThrpWL.getTotalSize()*2; iv++) corrThrpWL.H_elem()[iv] *= signPer;
          corrThrpWL.writeHDF5( (threep_filename + suff21 + "_ts_" + std::to_string(tSinks[ts]) + ".h5").c_str());         

        }//loop over values of B, building "2"

        WL_2->copy(*WL_1);
        WL_3->copy(*WL_1);
        su3_2->absorbDir_device(gaugeWL, Zdir);
        for(int ti = 0; ti < t; ti++){
          su3_exchange = su3_in; su3_in = su3_2; su3_2 = su3_exchange;
          su3_2->shift(*su3_in, 3);
        }

        if(nucleon == PROTON) propF1->copy(*propDN);
        else propF1->copy(*propUP);
    
        if(nucleon == PROTON) propF2->copy(*propUP);
        else propF2->copy(*propDN);
        for(int z = 0; z <= Zmax; z++) {//loop over values of B, building "2"

          if(z != 0) {

            WL_2->wilsonLineUpdate(*su3_2, tmp, Zdir);

            su3_exchange = su3_in; su3_in = WL_3; WL_3 = su3_exchange;
            WL_3->shift(*su3_in, Zdir);

            propExchange = propIn; propIn = propF1; propF1 = propExchange;
            propF1->shift(*propIn, Zdir);

            propExchange = propIn; propIn = propF2; propF2 = propExchange;
            propF2->shift(*propIn, Zdir);

          }

          WL->UxUdag(*WL_2, *WL_3);

          //uncomment to write the staples, for testing
          //WL->writeHDF5( (twop_filename + suff11 + "_L_" + std::to_string(l) + "_B_" + std::to_string(b) + "_Z_" + std::to_string(z) + ".h5/").c_str(), true );

          corrThrpWL.contractNucleonThrp_staple(*seqPropOut1, *propF1, *WL, signProps1, gammas, t, z, 0);
          if(signPer < 0) for(int iv = 0 ; iv < corrThrpWL.getTotalSize()*2; iv++) corrThrpWL.H_elem()[iv] *= signPer;
          corrThrpWL.writeHDF5( (threep_filename + suff12 + "_ts_" + std::to_string(tSinks[ts]) + ".h5").c_str());

          corrThrpWL.contractNucleonThrp_staple(*seqPropOut2, *propF2, *WL, signProps2, gammas, t, z, 0);
          if(signPer < 0) for(int iv = 0 ; iv < corrThrpWL.getTotalSize()*2; iv++) corrThrpWL.H_elem()[iv] *= signPer;
          corrThrpWL.writeHDF5( (threep_filename + suff22 + "_ts_" + std::to_string(tSinks[ts]) + ".h5").c_str());         

        }//loop over values of B, building "2"
        
      }//loop over values of L, building "1" and "3"


    }//loop over tSinks

    //two point correlator
    for(int nu = 0 ; nu < 4 ; nu++)
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
      
    }

  
    propUP->rotateToPhysicalBase_device(+1);
    propDN->rotateToPhysicalBase_device(-1);
    propUP->applyBoundaries_device(sourcePositions[isource][3]);
    propDN->applyBoundaries_device(sourcePositions[isource][3]);
  
    PLEGMA_Correlator<float> corr(corr_space, sourcePositions[isource], 0);
    corr.setFixMomVec(sinkMom_3D);
  
    TIME(corr.contractBaryons(*propUP, *propDN));
    THREAD(corr.writeFile((twop_filename).c_str(), corr_file_format));

  }//loop over source positions

  delete propUP;
  delete propDN;

  delete solver;

  finalize();

  return 0;
}