#include <PLEGMA.h>
#include <PLEGMA_utils.h>

using namespace plegma;
using namespace quda;

int main(int argc, char **argv)
{

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
  QUDA_solver *solver = new QUDA_solver(mu);

  PLEGMA_Vector<double> vectorIn;
  PLEGMA_Vector<double> vectorOut;
  PLEGMA_Vector<double> vectorAuxD;
  PLEGMA_Vector<float> vectorAuxF;
  PLEGMA_Propagator<float> *propUP[numSourcePositions];
  PLEGMA_Propagator<float> *propDN[numSourcePositions];
  PLEGMA_Propagator<float> *propUP_SS[numSourcePositions];
  PLEGMA_Propagator<float> *propDN_SS[numSourcePositions];
  PLEGMA_Propagator<float> *seqPropOut1[numSourcePositions];
  PLEGMA_Propagator<float> *seqPropOut2[numSourcePositions];
  PLEGMA_Propagator<float> *propIn = new PLEGMA_Propagator<float>(BOTH);
  PLEGMA_Propagator<float> *propExchange = nullptr;

  PLEGMA_Propagator3D<float> propUP3D;
  PLEGMA_Propagator3D<float> propDN3D;

  std::string suff11, suff12, suff21, suff22;

  for(int is = 0; is < numSourcePositions; is++) {
    propUP[is] = new PLEGMA_Propagator<float>(BOTH);
    propDN[is] = new PLEGMA_Propagator<float>(BOTH);
    propUP_SS[is] = new PLEGMA_Propagator<float>(BOTH);
    propDN_SS[is] = new PLEGMA_Propagator<float>(BOTH);
    seqPropOut1[is] = new PLEGMA_Propagator<float>(BOTH);
    seqPropOut2[is] = new PLEGMA_Propagator<float>(BOTH);
  }
  PLEGMA_Propagator<float> *prop1 = new PLEGMA_Propagator<float>(BOTH);
  PLEGMA_Propagator<float> *prop2 = new PLEGMA_Propagator<float>(BOTH);


  PLEGMA_Su3field<float> *su3_1 = new PLEGMA_Su3field<float>(BOTH);
    //PLEGMA_Su3field<float> *su3_21 = new PLEGMA_Su3field<float>(BOTH);
    //PLEGMA_Su3field<float> *su3_22 = new PLEGMA_Su3field<float>(BOTH);
  PLEGMA_Su3field<float> *su3_2_1 = new PLEGMA_Su3field<float>(BOTH);
  PLEGMA_Su3field<float> *su3_2_2 = new PLEGMA_Su3field<float>(BOTH);
  //PLEGMA_Su3field<float> *su3_3_1 = new PLEGMA_Su3field<float>(BOTH);
  //PLEGMA_Su3field<float> *su3_3_2 = new PLEGMA_Su3field<float>(BOTH);
  PLEGMA_Su3field<float> *su3_4_1 = new PLEGMA_Su3field<float>(BOTH);
  PLEGMA_Su3field<float> *su3_4_2 = new PLEGMA_Su3field<float>(BOTH);
  PLEGMA_Su3field<float> *WL_1 = new PLEGMA_Su3field<float>(BOTH);
  PLEGMA_Su3field<float> *WL_2_1 = new PLEGMA_Su3field<float>(BOTH);
  //PLEGMA_Su3field<float> *WL_3_1 = new PLEGMA_Su3field<float>(BOTH);
  PLEGMA_Su3field<float> *WL_2_2 = new PLEGMA_Su3field<float>(BOTH);
  //PLEGMA_Su3field<float> *WL_3_2 = new PLEGMA_Su3field<float>(BOTH);
  PLEGMA_Su3field<float> *WL_4_1 = new PLEGMA_Su3field<float>(BOTH);
  PLEGMA_Su3field<float> *WL_4_2 = new PLEGMA_Su3field<float>(BOTH);
  //PLEGMA_Su3field<float> *WL_temp = new PLEGMA_Su3field<float>(BOTH);
  PLEGMA_Su3field<float> *su3_in = new PLEGMA_Su3field<float>(BOTH);
  PLEGMA_Su3field<float> *WL1 = new PLEGMA_Su3field<float>(BOTH);
  PLEGMA_Su3field<float> *WL2 = new PLEGMA_Su3field<float>(BOTH);
  //PLEGMA_Su3field<float> *WL2 = new PLEGMA_Su3field<float>(BOTH);
  PLEGMA_Su3field<float> *su3_exchange = new PLEGMA_Su3field<float>(BOTH);
  PLEGMA_Su3field<float> tmp(BOTH);

  //WL testing
#if 0
  PLEGMA_Su3field<float> *WL = new PLEGMA_Su3field<float>(BOTH);
  PLEGMA_Su3field<float> *u_s[4];
  for(int idir = 0; idir < 4 ; idir++){
    u_s[idir] = new PLEGMA_Su3field<float>(BOTH);
    u_s[idir]->absorbDir_device(gaugeWL,idir);
  }
  PLEGMA_Su3field<float> *WLIn = new PLEGMA_Su3field<float>(BOTH);
  PLEGMA_Su3field<float> *WLExchange = nullptr;
#endif

  // UP props
  if(mu<0){
    mu *= (-1);
    solver->UpdateSolver();
  }

  for(int isource = 0; isource < numSourcePositions; isource++) { //loop over source positions
  
    for(int isc = 0 ; isc < 12 ; isc++) {
      vectorAuxD.pointSource(sourcePositions[isource], isc/3, isc%3, DEVICE);
      vectorIn.gaussianSmearing(vectorAuxD, smearedGauge, nsmearGauss, alphaGauss);
      PLEGMA_printf("Going to invert UP for component %d\n", isc);
      solver->solve(vectorOut, vectorIn);
      vectorAuxF.copy(vectorOut);
      propUP[isource]->absorb(vectorAuxF, isc/3, isc%3);
      vectorAuxD.gaussianSmearing(vectorOut, smearedGauge , nsmearGauss, alphaGauss);
      vectorAuxF.copy(vectorAuxD);
      propUP_SS[isource]->absorb(vectorAuxF, isc/3, isc%3);
    }
  
  }//loop over source positions

  // DN props
    if(mu>0){
      mu *= (-1);
      solver->UpdateSolver();
    }

  for(int isource = 0; isource < numSourcePositions; isource++) { //loop over source positions
    
    for(int isc = 0 ; isc < 12 ; isc++) {
      vectorAuxD.pointSource(sourcePositions[isource], isc/3, isc%3, DEVICE);
      vectorIn.gaussianSmearing(vectorAuxD, smearedGauge, nsmearGauss, alphaGauss);
      PLEGMA_printf("Going to invert DN for component %d\n", isc);
      solver->solve(vectorOut, vectorIn);
      vectorAuxF.copy(vectorOut);
      propDN[isource]->absorb(vectorAuxF, isc/3, isc%3);
      vectorAuxD.gaussianSmearing(vectorOut, smearedGauge , nsmearGauss, alphaGauss);
      vectorAuxF.copy(vectorAuxD);
      propDN_SS[isource]->absorb(vectorAuxF, isc/3, isc%3);
    }

  }//loop over source positions
  
  for(int ts = 0; ts < tSinks.size(); ts++) { //loop over tSinks
    
    //seqProps CP1
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
    for(int isource = 0; isource < numSourcePositions; isource++) { //loop over source positions
      int signPer = (tSinks[ts] + sourcePositions[isource][3]) >= HGC_totalL[3] ? -1 : +1;
      int global_fixSinkTime = (tSinks[ts] + sourcePositions[isource][3])%HGC_totalL[3]; 
      int my_fixSinkTime = global_fixSinkTime - HGC_procPosition[3] * HGC_localL[3];
      bool is_myST = (my_fixSinkTime >= 0) && ( my_fixSinkTime < HGC_localL[3] );

      propUP3D.absorb(*propUP_SS[isource], global_fixSinkTime);
      propDN3D.absorb(*propDN_SS[isource], global_fixSinkTime);

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
          seqPropOut1[isource]->absorb(vectorAuxF, nu, c2);
        }//loop over color

      }//loop over spin

      seqPropOut1[isource]->apply_gamma(G5);
      seqPropOut1[isource]->conjugate();

    } //loop over source positions
    int signProps1 = (nucleon == PROTON) ? -1: +1;

    //seqProps CP2
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
    for(int isource = 0; isource < numSourcePositions; isource++) { //loop over source positions
      int signPer = (tSinks[ts] + sourcePositions[isource][3]) >= HGC_totalL[3] ? -1 : +1;
      int global_fixSinkTime = (tSinks[ts] + sourcePositions[isource][3])%HGC_totalL[3]; 
      int my_fixSinkTime = global_fixSinkTime - HGC_procPosition[3] * HGC_localL[3];
      bool is_myST = (my_fixSinkTime >= 0) && ( my_fixSinkTime < HGC_localL[3] );

      propUP3D.absorb(*propUP_SS[isource], global_fixSinkTime);
      propDN3D.absorb(*propDN_SS[isource], global_fixSinkTime);

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
          seqPropOut2[isource]->absorb(vectorAuxF, nu, c2);
  
        }//loop over color

      }//loop over spin

      seqPropOut2[isource]->apply_gamma(G5);
      seqPropOut2[isource]->conjugate();
      seqPropOut2[isource]->unload();
    } //loop over source positions

    int signProps2 = (nucleon == PROTON) ? +1: -1;
  

  //positive L, positive B

  for(int l:Lvals) {//loop over values of L
    //absorb the gauge directions for building the 4 branches
    su3_1->absorbDir_device(gaugeWL, WilsDir);
    su3_2_1->absorbDir_device(gaugeWL, Bdir1);
    su3_2_2->absorbDir_device(gaugeWL, Bdir2);
    //su3_3_1->absorbDir_device(gaugeWL, WilsDir);
    //su3_3_2->absorbDir_device(gaugeWL, WilsDir);
    su3_4_1->absorbDir_device(gaugeWL, WilsDir);
    su3_4_2->absorbDir_device(gaugeWL, WilsDir);

    WL_1->setUnit( (std::vector<int>) {0,4,8});
    for(int li = 0; li < l; li++) {
      WL_1->wilsonLineUpdate(*su3_1, tmp, 4 + WilsDir);
      su3_exchange = su3_in; su3_in = su3_2_1; su3_2_1 = su3_exchange;
      su3_2_1->shift(*su3_in, 4 + WilsDir);
      su3_exchange = su3_in; su3_in = su3_2_2; su3_2_2 = su3_exchange;
      su3_2_2->shift(*su3_in, 4 + WilsDir);
    }
    WL_4_1->copy(*WL_1);
    WL_4_2->copy(*WL_1);
    WL_4_2->Udag();
    su3_2_1->unload();
    su3_2_2->unload();
    for(int z = 0; z < l; z++) {//loop  over values of Z
      if(z!=0){
        WL_4_1->wilsonLineUpdate(*su3_4_1, tmp, 4 + WilsDir, true);
        WL_4_2->wilsonLineUpdate(*su3_4_2, tmp, WilsDir);
      }
      WL_2_1->copy(*WL_4_1);
      WL_2_2->copy(*WL_4_1);
      for(int b = 0; b <= B_max; b++) {//loop over values of B
        if(b != 0) {
          WL_2_1->wilsonLineUpdate(*su3_2_1, tmp, 4 + Bdir1);
          WL_2_2->wilsonLineUpdate(*su3_2_2, tmp, 4 + Bdir2);
        }
        for(int bi = 0; bi < b; bi++) {
          su3_exchange = su3_in; su3_in = WL_4_2; WL_4_2 = su3_exchange;
          WL_4_2->shift(*su3_in, 4 + Bdir1);
        }
        WL1->UxU(*WL_2_1,*WL_4_2);
        //WL1->writeHDF5(("WL_test.Bdir1.h5/l"+std::to_string(l)+"b"+std::to_string(b)+"z"+std::to_string(z)).c_str());
        for(int bi = 0; bi < b; bi++) {
          su3_exchange = su3_in; su3_in = WL_4_2; WL_4_2 = su3_exchange;
          WL_4_2->shift(*su3_in, Bdir1);
        } 
        for(int bi = 0; bi < b; bi++) {
          su3_exchange = su3_in; su3_in = WL_4_2; WL_4_2 = su3_exchange;
          WL_4_2->shift(*su3_in, 4 + Bdir2);
        }
        WL2->UxU(*WL_2_2,*WL_4_2);
        //WL2->writeHDF5(("WL_test.Bdir2.h5/l"+std::to_string(l)+"b"+std::to_string(b)+"z"+std::to_string(z)).c_str());
        for(int bi = 0; bi < b; bi++) {
          su3_exchange = su3_in; su3_in = WL_4_2; WL_4_2 = su3_exchange;
          WL_4_2->shift(*su3_in, Bdir2);
        }
        for(int is = 0; is < numSourcePositions; is++) {//loop over source positions
          PLEGMA_Correlator<float> corrThrpWL(corr_space, sourcePositions[is], 0, (tSinks[ts]+1));
          int signPer = (tSinks[ts] + sourcePositions[is][3]) >= HGC_totalL[3] ? -1 : +1;

          if(nucleon == PROTON) prop1->copy(*propDN[is]);
          else prop1->copy(*propUP[is]);
          prop2->copy(*seqPropOut1[is]);
          for(int zi = 0; zi < z; zi++) {
            propExchange = propIn; propIn = prop1; prop1 = propExchange;
            prop1->shift(*propIn, WilsDir);
            propExchange = propIn; propIn = prop2; prop2 = propExchange;
            prop2->shift(*propIn, 4 + WilsDir);
          }
          for(int bi = 0; bi < b; bi++) {
            propExchange = propIn; propIn = prop1; prop1 = propExchange;
            prop1->shift(*propIn, 4 + Bdir1);
          }
          corrThrpWL.contractNucleonThrp_staple(*prop2, *prop1, *WL1, signProps1, gammas, l, b, z);
          if(signPer < 0) for(int iv = 0 ; iv < corrThrpWL.getTotalSize()*2; iv++) corrThrpWL.H_elem()[iv] *= signPer;
          corrThrpWL.writeHDF5( (threep_filename +".CP1_Bdir_"+std::to_string(Bdir1)+"_ts_"+std::to_string(tSinks[ts])+"_stout_"+std::to_string(nStout)+".h5").c_str());
          for(int bi = 0; bi < b; bi++) {
            propExchange = propIn; propIn = prop1; prop1 = propExchange;
            prop1->shift(*propIn, Bdir1);
          }
          for(int bi = 0; bi < b; bi++) {
            propExchange = propIn; propIn = prop1; prop1 = propExchange;
            prop1->shift(*propIn, 4 + Bdir2);
          }
          corrThrpWL.contractNucleonThrp_staple(*prop2, *prop1, *WL2, signProps1, gammas, l, b, z);
          if(signPer < 0) for(int iv = 0 ; iv < corrThrpWL.getTotalSize()*2; iv++) corrThrpWL.H_elem()[iv] *= signPer;
          corrThrpWL.writeHDF5( (threep_filename +".CP1_Bdir_"+std::to_string(Bdir2)+"_ts_"+std::to_string(tSinks[ts])+"_stout_"+std::to_string(nStout)+".h5").c_str());

          if(nucleon == PROTON) prop1->copy(*propUP[is]);
          else prop1->copy(*propDN[is]);
          prop2->copy(*seqPropOut2[is]);
          for(int zi = 0; zi < z; zi++) {
            propExchange = propIn; propIn = prop1; prop1 = propExchange;
            prop1->shift(*propIn, WilsDir);
            propExchange = propIn; propIn = prop2; prop2 = propExchange;
            prop2->shift(*propIn, 4 + WilsDir);
          }
          for(int bi = 0; bi < b; bi++) {
            propExchange = propIn; propIn = prop1; prop1 = propExchange;
            prop1->shift(*propIn, 4 + Bdir1);
          }
          corrThrpWL.contractNucleonThrp_staple(*prop2, *prop1, *WL1, signProps2, gammas, l, b, z);
          if(signPer < 0) for(int iv = 0 ; iv < corrThrpWL.getTotalSize()*2; iv++) corrThrpWL.H_elem()[iv] *= signPer;
          corrThrpWL.writeHDF5( (threep_filename +".CP2_Bdir_"+std::to_string(Bdir1)+"_ts_"+std::to_string(tSinks[ts])+"_stout_"+std::to_string(nStout)+".h5").c_str());
          for(int bi = 0; bi < b; bi++) {
            propExchange = propIn; propIn = prop1; prop1 = propExchange;
            prop1->shift(*propIn, Bdir1);
          }
          for(int bi = 0; bi < b; bi++) {
            propExchange = propIn; propIn = prop1; prop1 = propExchange;
            prop1->shift(*propIn, 4 + Bdir2);
          }
          corrThrpWL.contractNucleonThrp_staple(*prop2, *prop1, *WL2, signProps2, gammas, l, b, z);
          if(signPer < 0) for(int iv = 0 ; iv < corrThrpWL.getTotalSize()*2; iv++) corrThrpWL.H_elem()[iv] *= signPer;
          corrThrpWL.writeHDF5( (threep_filename +".CP2_Bdir_"+std::to_string(Bdir2)+"_ts_"+std::to_string(tSinks[ts])+"_stout_"+std::to_string(nStout)+".h5").c_str());
          //delete corrThrpWL;
        }//loop over source positions
      }//loop over values of B
      su3_2_1->load();
      su3_2_2->load();
      WL_2_1->copy(*WL_4_2);
      WL_2_2->copy(*WL_4_2);
      WL_2_1->Udag();
      WL_2_2->Udag();
      for(int b = 0; b <= B_max; b++) {//loop over values of B
        if(b != 0) {
          WL_2_1->wilsonLineUpdate(*su3_2_1, tmp, 4 + Bdir1);
          WL_2_2->wilsonLineUpdate(*su3_2_2, tmp, 4 + Bdir2);
        }
        for(int bi = 0; bi < b; bi++) {
          su3_exchange = su3_in; su3_in = WL_4_1; WL_4_1 = su3_exchange;
          WL_4_1->shift(*su3_in, 4 + Bdir1);
        }
        WL1->UxUdag(*WL_2_1,*WL_4_1);
        //WL1->writeHDF5(("WL_test.Bdir1.h5/l"+std::to_string(l)+"b"+std::to_string(b)+"z"+std::to_string(-z)).c_str());
        for(int bi = 0; bi < b; bi++) {
          su3_exchange = su3_in; su3_in = WL_4_1; WL_4_1 = su3_exchange;
          WL_4_1->shift(*su3_in, Bdir1);
        } 
        for(int bi = 0; bi < b; bi++) {
          su3_exchange = su3_in; su3_in = WL_4_1; WL_4_1 = su3_exchange;
          WL_4_1->shift(*su3_in, 4 + Bdir2);
        }
        WL2->UxUdag(*WL_2_2,*WL_4_1);
        //WL2->writeHDF5(("WL_test.Bdir2.h5/l"+std::to_string(l)+"b"+std::to_string(b)+"z"+std::to_string(-z)).c_str());
        for(int bi = 0; bi < b; bi++) {
          su3_exchange = su3_in; su3_in = WL_4_1; WL_4_1 = su3_exchange;
          WL_4_1->shift(*su3_in, Bdir2);
        }
        for(int is = 0; is < numSourcePositions; is++) {//loop over source positions
          PLEGMA_Correlator<float> corrThrpWL(corr_space, sourcePositions[is], 0, (tSinks[ts]+1));
          int signPer = (tSinks[ts] + sourcePositions[is][3]) >= HGC_totalL[3] ? -1 : +1;

          if(nucleon == PROTON) prop1->copy(*propDN[is]);
          else prop1->copy(*propUP[is]);
          prop2->copy(*seqPropOut1[is]);
          for(int zi = 0; zi < z; zi++) {
            propExchange = propIn; propIn = prop1; prop1 = propExchange;
            prop1->shift(*propIn, 4 + WilsDir);
            propExchange = propIn; propIn = prop2; prop2 = propExchange;
            prop2->shift(*propIn, WilsDir);
          }
          for(int bi = 0; bi < b; bi++) {
            propExchange = propIn; propIn = prop1; prop1 = propExchange;
            prop1->shift(*propIn, 4 + Bdir1);
          }
          corrThrpWL.contractNucleonThrp_staple(*prop2, *prop1, *WL1, signProps1, gammas, l, b, -z);
          if(signPer < 0) for(int iv = 0 ; iv < corrThrpWL.getTotalSize()*2; iv++) corrThrpWL.H_elem()[iv] *= signPer;
          corrThrpWL.writeHDF5( (threep_filename +".CP1_Bdir_"+std::to_string(Bdir1)+"_ts_"+std::to_string(tSinks[ts])+"_stout_"+std::to_string(nStout)+".h5").c_str());
          for(int bi = 0; bi < b; bi++) {
            propExchange = propIn; propIn = prop1; prop1 = propExchange;
            prop1->shift(*propIn, Bdir1);
          }
          for(int bi = 0; bi < b; bi++) {
            propExchange = propIn; propIn = prop1; prop1 = propExchange;
            prop1->shift(*propIn, 4 + Bdir2);
          }
          corrThrpWL.contractNucleonThrp_staple(*prop2, *prop1, *WL2, signProps1, gammas, l, b, -z);
          if(signPer < 0) for(int iv = 0 ; iv < corrThrpWL.getTotalSize()*2; iv++) corrThrpWL.H_elem()[iv] *= signPer;
          corrThrpWL.writeHDF5( (threep_filename +".CP1_Bdir_"+std::to_string(Bdir2)+"_ts_"+std::to_string(tSinks[ts])+"_stout_"+std::to_string(nStout)+".h5").c_str());

          if(nucleon == PROTON) prop1->copy(*propUP[is]);
          else prop1->copy(*propDN[is]);
          prop2->copy(*seqPropOut2[is]);
          for(int zi = 0; zi < z; zi++) {
            propExchange = propIn; propIn = prop1; prop1 = propExchange;
            prop1->shift(*propIn, 4 + WilsDir);
            propExchange = propIn; propIn = prop2; prop2 = propExchange;
            prop2->shift(*propIn, WilsDir);
          }
          for(int bi = 0; bi < b; bi++) {
            propExchange = propIn; propIn = prop1; prop1 = propExchange;
            prop1->shift(*propIn, 4 + Bdir1);
          }
          corrThrpWL.contractNucleonThrp_staple(*prop2, *prop1, *WL1, signProps2, gammas, l, b, -z);
          if(signPer < 0) for(int iv = 0 ; iv < corrThrpWL.getTotalSize()*2; iv++) corrThrpWL.H_elem()[iv] *= signPer;
          corrThrpWL.writeHDF5( (threep_filename +".CP2_Bdir_"+std::to_string(Bdir1)+"_ts_"+std::to_string(tSinks[ts])+"_stout_"+std::to_string(nStout)+".h5").c_str());
          for(int bi = 0; bi < b; bi++) {
            propExchange = propIn; propIn = prop1; prop1 = propExchange;
            prop1->shift(*propIn, Bdir1);
          }
          for(int bi = 0; bi < b; bi++) {
            propExchange = propIn; propIn = prop1; prop1 = propExchange;
            prop1->shift(*propIn, 4 + Bdir2);
          }
          corrThrpWL.contractNucleonThrp_staple(*prop2, *prop1, *WL2, signProps2, gammas, l, b, -z);
          if(signPer < 0) for(int iv = 0 ; iv < corrThrpWL.getTotalSize()*2; iv++) corrThrpWL.H_elem()[iv] *= signPer;
          corrThrpWL.writeHDF5( (threep_filename +".CP2_Bdir_"+std::to_string(Bdir2)+"_ts_"+std::to_string(tSinks[ts])+"_stout_"+std::to_string(nStout)+".h5").c_str());
          //delete corrThrpWL;
        }//loop over source positions
      }//loop over values of B
      su3_2_1->load();
      su3_2_2->load();
    }//loop over values of Z
  }//loop over values of L

  //negative L, positive B

  for(int l:Lvals) {//loop over values of L
    //absorb the gauge directions for building the 4 branches
    su3_1->absorbDir_device(gaugeWL, WilsDir);
    su3_2_1->absorbDir_device(gaugeWL, Bdir1);
    su3_2_2->absorbDir_device(gaugeWL, Bdir2);
    //su3_3_1->absorbDir_device(gaugeWL, WilsDir);
    //su3_3_2->absorbDir_device(gaugeWL, WilsDir);
    su3_4_1->absorbDir_device(gaugeWL, WilsDir);
    su3_4_2->absorbDir_device(gaugeWL, WilsDir);

    WL_1->setUnit( (std::vector<int>) {0,4,8});
    for(int li = 0; li < l; li++) {
      WL_1->wilsonLineUpdate(*su3_1, tmp, WilsDir);
      su3_exchange = su3_in; su3_in = su3_2_1; su3_2_1 = su3_exchange;
      su3_2_1->shift(*su3_in, WilsDir);
      su3_exchange = su3_in; su3_in = su3_2_2; su3_2_2 = su3_exchange;
      su3_2_2->shift(*su3_in, WilsDir);
    }
    WL_4_1->copy(*WL_1);
    WL_4_2->copy(*WL_1);
    WL_4_2->Udag();
    su3_2_1->unload();
    su3_2_2->unload();
    for(int z = 0; z < l; z++) {//loop  over values of Z
      if(z!=0){
        WL_4_1->wilsonLineUpdate(*su3_4_1, tmp, WilsDir, true);
        WL_4_2->wilsonLineUpdate(*su3_4_2, tmp, 4 + WilsDir);
      }
      WL_2_1->copy(*WL_4_1);
      WL_2_2->copy(*WL_4_1);
      for(int b = 0; b <= B_max; b++) {//loop over values of B
        if(b != 0) {
          WL_2_1->wilsonLineUpdate(*su3_2_1, tmp, 4 + Bdir1);
          WL_2_2->wilsonLineUpdate(*su3_2_2, tmp, 4 + Bdir2);
        }
        for(int bi = 0; bi < b; bi++) {
          su3_exchange = su3_in; su3_in = WL_4_2; WL_4_2 = su3_exchange;
          WL_4_2->shift(*su3_in, 4 + Bdir1);
        }
        WL1->UxU(*WL_2_1,*WL_4_2);
        //WL1->writeHDF5(("WL_test.Bdir1.h5/l"+std::to_string(l)+"b"+std::to_string(b)+"z"+std::to_string(z)).c_str());
        for(int bi = 0; bi < b; bi++) {
          su3_exchange = su3_in; su3_in = WL_4_2; WL_4_2 = su3_exchange;
          WL_4_2->shift(*su3_in, Bdir1);
        } 
        for(int bi = 0; bi < b; bi++) {
          su3_exchange = su3_in; su3_in = WL_4_2; WL_4_2 = su3_exchange;
          WL_4_2->shift(*su3_in, 4 + Bdir2);
        }
        WL2->UxU(*WL_2_2,*WL_4_2);
        //WL2->writeHDF5(("WL_test.Bdir2.h5/l"+std::to_string(l)+"b"+std::to_string(b)+"z"+std::to_string(z)).c_str());
        for(int bi = 0; bi < b; bi++) {
          su3_exchange = su3_in; su3_in = WL_4_2; WL_4_2 = su3_exchange;
          WL_4_2->shift(*su3_in, Bdir2);
        }
        for(int is = 0; is < numSourcePositions; is++) {//loop over source positions
          PLEGMA_Correlator<float> corrThrpWL(corr_space, sourcePositions[is], 0, (tSinks[ts]+1));
          int signPer = (tSinks[ts] + sourcePositions[is][3]) >= HGC_totalL[3] ? -1 : +1;

          if(nucleon == PROTON) prop1->copy(*propDN[is]);
          else prop1->copy(*propUP[is]);
          prop2->copy(*seqPropOut1[is]);
          for(int zi = 0; zi < z; zi++) {
            propExchange = propIn; propIn = prop1; prop1 = propExchange;
            prop1->shift(*propIn, 4 + WilsDir);
            propExchange = propIn; propIn = prop2; prop2 = propExchange;
            prop2->shift(*propIn, WilsDir);
          }
          for(int bi = 0; bi < b; bi++) {
            propExchange = propIn; propIn = prop1; prop1 = propExchange;
            prop1->shift(*propIn, 4 + Bdir1);
          }
          corrThrpWL.contractNucleonThrp_staple(*prop2, *prop1, *WL1, signProps1, gammas, -l, b, z);
          if(signPer < 0) for(int iv = 0 ; iv < corrThrpWL.getTotalSize()*2; iv++) corrThrpWL.H_elem()[iv] *= signPer;
          corrThrpWL.writeHDF5( (threep_filename +".CP1_Bdir_"+std::to_string(Bdir1)+"_ts_"+std::to_string(tSinks[ts])+"_stout_"+std::to_string(nStout)+".h5").c_str());
          for(int bi = 0; bi < b; bi++) {
            propExchange = propIn; propIn = prop1; prop1 = propExchange;
            prop1->shift(*propIn, Bdir1);
          }
          for(int bi = 0; bi < b; bi++) {
            propExchange = propIn; propIn = prop1; prop1 = propExchange;
            prop1->shift(*propIn, 4 + Bdir2);
          }
          corrThrpWL.contractNucleonThrp_staple(*prop2, *prop1, *WL2, signProps1, gammas, -l, b, z);
          if(signPer < 0) for(int iv = 0 ; iv < corrThrpWL.getTotalSize()*2; iv++) corrThrpWL.H_elem()[iv] *= signPer;
          corrThrpWL.writeHDF5( (threep_filename +".CP1_Bdir_"+std::to_string(Bdir2)+"_ts_"+std::to_string(tSinks[ts])+"_stout_"+std::to_string(nStout)+".h5").c_str());

          if(nucleon == PROTON) prop1->copy(*propUP[is]);
          else prop1->copy(*propDN[is]);
          prop2->copy(*seqPropOut2[is]);
          for(int zi = 0; zi < z; zi++) {
            propExchange = propIn; propIn = prop1; prop1 = propExchange;
            prop1->shift(*propIn, 4 + WilsDir);
            propExchange = propIn; propIn = prop2; prop2 = propExchange;
            prop2->shift(*propIn, WilsDir);
          }
          for(int bi = 0; bi < b; bi++) {
            propExchange = propIn; propIn = prop1; prop1 = propExchange;
            prop1->shift(*propIn, 4 + Bdir1);
          }
          corrThrpWL.contractNucleonThrp_staple(*prop2, *prop1, *WL1, signProps2, gammas, -l, b, z);
          if(signPer < 0) for(int iv = 0 ; iv < corrThrpWL.getTotalSize()*2; iv++) corrThrpWL.H_elem()[iv] *= signPer;
          corrThrpWL.writeHDF5( (threep_filename +".CP2_Bdir_"+std::to_string(Bdir1)+"_ts_"+std::to_string(tSinks[ts])+"_stout_"+std::to_string(nStout)+".h5").c_str());
          for(int bi = 0; bi < b; bi++) {
            propExchange = propIn; propIn = prop1; prop1 = propExchange;
            prop1->shift(*propIn, Bdir1);
          }
          for(int bi = 0; bi < b; bi++) {
            propExchange = propIn; propIn = prop1; prop1 = propExchange;
            prop1->shift(*propIn, 4 + Bdir2);
          }
          corrThrpWL.contractNucleonThrp_staple(*prop2, *prop1, *WL2, signProps2, gammas, -l, b, z);
          if(signPer < 0) for(int iv = 0 ; iv < corrThrpWL.getTotalSize()*2; iv++) corrThrpWL.H_elem()[iv] *= signPer;
          corrThrpWL.writeHDF5( (threep_filename +".CP2_Bdir_"+std::to_string(Bdir2)+"_ts_"+std::to_string(tSinks[ts])+"_stout_"+std::to_string(nStout)+".h5").c_str());
          //delete corrThrpWL;
        }//loop over source positions
      }//loop over values of B
      su3_2_1->load();
      su3_2_2->load();
      WL_2_1->copy(*WL_4_2);
      WL_2_2->copy(*WL_4_2);
      WL_2_1->Udag();
      WL_2_2->Udag();
      for(int b = 0; b <= B_max; b++) {//loop over values of B
        if(b != 0) {
          WL_2_1->wilsonLineUpdate(*su3_2_1, tmp, 4 + Bdir1);
          WL_2_2->wilsonLineUpdate(*su3_2_2, tmp, 4 + Bdir2);
        }
        for(int bi = 0; bi < b; bi++) {
          su3_exchange = su3_in; su3_in = WL_4_1; WL_4_1 = su3_exchange;
          WL_4_1->shift(*su3_in, 4 + Bdir1);
        }
        WL1->UxUdag(*WL_2_1,*WL_4_1);
        //WL1->writeHDF5(("WL_test.Bdir1.h5/l"+std::to_string(l)+"b"+std::to_string(b)+"z"+std::to_string(-z)).c_str());
        for(int bi = 0; bi < b; bi++) {
          su3_exchange = su3_in; su3_in = WL_4_1; WL_4_1 = su3_exchange;
          WL_4_1->shift(*su3_in, Bdir1);
        } 
        for(int bi = 0; bi < b; bi++) {
          su3_exchange = su3_in; su3_in = WL_4_1; WL_4_1 = su3_exchange;
          WL_4_1->shift(*su3_in, 4 + Bdir2);
        }
        WL2->UxUdag(*WL_2_2,*WL_4_1);
        //WL2->writeHDF5(("WL_test.Bdir2.h5/l"+std::to_string(l)+"b"+std::to_string(b)+"z"+std::to_string(-z)).c_str());
        for(int bi = 0; bi < b; bi++) {
          su3_exchange = su3_in; su3_in = WL_4_1; WL_4_1 = su3_exchange;
          WL_4_1->shift(*su3_in, Bdir2);
        }
        for(int is = 0; is < numSourcePositions; is++) {//loop over source positions
          PLEGMA_Correlator<float> corrThrpWL(corr_space, sourcePositions[is], 0, (tSinks[ts]+1));
          int signPer = (tSinks[ts] + sourcePositions[is][3]) >= HGC_totalL[3] ? -1 : +1;

          if(nucleon == PROTON) prop1->copy(*propDN[is]);
          else prop1->copy(*propUP[is]);
          prop2->copy(*seqPropOut1[is]);
          for(int zi = 0; zi < z; zi++) {
            propExchange = propIn; propIn = prop1; prop1 = propExchange;
            prop1->shift(*propIn, WilsDir);
            propExchange = propIn; propIn = prop2; prop2 = propExchange;
            prop2->shift(*propIn, 4 + WilsDir);
          }
          for(int bi = 0; bi < b; bi++) {
            propExchange = propIn; propIn = prop1; prop1 = propExchange;
            prop1->shift(*propIn, 4 + Bdir1);
          }
          corrThrpWL.contractNucleonThrp_staple(*prop2, *prop1, *WL1, signProps1, gammas, -l, b, -z);
          if(signPer < 0) for(int iv = 0 ; iv < corrThrpWL.getTotalSize()*2; iv++) corrThrpWL.H_elem()[iv] *= signPer;
          corrThrpWL.writeHDF5( (threep_filename +".CP1_Bdir_"+std::to_string(Bdir1)+"_ts_"+std::to_string(tSinks[ts])+"_stout_"+std::to_string(nStout)+".h5").c_str());
          for(int bi = 0; bi < b; bi++) {
            propExchange = propIn; propIn = prop1; prop1 = propExchange;
            prop1->shift(*propIn, Bdir1);
          }
          for(int bi = 0; bi < b; bi++) {
            propExchange = propIn; propIn = prop1; prop1 = propExchange;
            prop1->shift(*propIn, 4 + Bdir2);
          }
          corrThrpWL.contractNucleonThrp_staple(*prop2, *prop1, *WL2, signProps1, gammas, -l, b, -z);
          if(signPer < 0) for(int iv = 0 ; iv < corrThrpWL.getTotalSize()*2; iv++) corrThrpWL.H_elem()[iv] *= signPer;
          corrThrpWL.writeHDF5( (threep_filename +".CP1_Bdir_"+std::to_string(Bdir2)+"_ts_"+std::to_string(tSinks[ts])+"_stout_"+std::to_string(nStout)+".h5").c_str());

          if(nucleon == PROTON) prop1->copy(*propUP[is]);
          else prop1->copy(*propDN[is]);
          prop2->copy(*seqPropOut2[is]);
          for(int zi = 0; zi < z; zi++) {
            propExchange = propIn; propIn = prop1; prop1 = propExchange;
            prop1->shift(*propIn, WilsDir);
            propExchange = propIn; propIn = prop2; prop2 = propExchange;
            prop2->shift(*propIn, 4 + WilsDir);
          }
          for(int bi = 0; bi < b; bi++) {
            propExchange = propIn; propIn = prop1; prop1 = propExchange;
            prop1->shift(*propIn, 4 + Bdir1);
          }
          corrThrpWL.contractNucleonThrp_staple(*prop2, *prop1, *WL1, signProps2, gammas, -l, b, -z);
          if(signPer < 0) for(int iv = 0 ; iv < corrThrpWL.getTotalSize()*2; iv++) corrThrpWL.H_elem()[iv] *= signPer;
          corrThrpWL.writeHDF5( (threep_filename +".CP2_Bdir_"+std::to_string(Bdir1)+"_ts_"+std::to_string(tSinks[ts])+"_stout_"+std::to_string(nStout)+".h5").c_str());
          for(int bi = 0; bi < b; bi++) {
            propExchange = propIn; propIn = prop1; prop1 = propExchange;
            prop1->shift(*propIn, Bdir1);
          }
          for(int bi = 0; bi < b; bi++) {
            propExchange = propIn; propIn = prop1; prop1 = propExchange;
            prop1->shift(*propIn, 4 + Bdir2);
          }
          corrThrpWL.contractNucleonThrp_staple(*prop2, *prop1, *WL2, signProps2, gammas, -l, b, -z);
          if(signPer < 0) for(int iv = 0 ; iv < corrThrpWL.getTotalSize()*2; iv++) corrThrpWL.H_elem()[iv] *= signPer;
          corrThrpWL.writeHDF5( (threep_filename +".CP2_Bdir_"+std::to_string(Bdir2)+"_ts_"+std::to_string(tSinks[ts])+"_stout_"+std::to_string(nStout)+".h5").c_str());
          //delete corrThrpWL;
        }//loop over source positions
      }//loop over values of B
      su3_2_1->load();
      su3_2_2->load();
    }//loop over values of Z
  }//loop over values of L

  //positive L, negative B

  for(int l:Lvals) {//loop over values of L
    //absorb the gauge directions for building the 4 branches
    su3_1->absorbDir_device(gaugeWL, WilsDir);
    su3_2_1->absorbDir_device(gaugeWL, Bdir1);
    su3_2_2->absorbDir_device(gaugeWL, Bdir2);
    //su3_3_1->absorbDir_device(gaugeWL, WilsDir);
    //su3_3_2->absorbDir_device(gaugeWL, WilsDir);
    su3_4_1->absorbDir_device(gaugeWL, WilsDir);
    su3_4_2->absorbDir_device(gaugeWL, WilsDir);

    WL_1->setUnit( (std::vector<int>) {0,4,8});
    for(int li = 0; li < l; li++) {
      WL_1->wilsonLineUpdate(*su3_1, tmp, 4 + WilsDir);
      su3_exchange = su3_in; su3_in = su3_2_1; su3_2_1 = su3_exchange;
      su3_2_1->shift(*su3_in, 4 + WilsDir);
      su3_exchange = su3_in; su3_in = su3_2_2; su3_2_2 = su3_exchange;
      su3_2_2->shift(*su3_in, 4 + WilsDir);
    }
    WL_4_1->copy(*WL_1);
    WL_4_2->copy(*WL_1);
    WL_4_2->Udag();
    su3_2_1->unload();
    su3_2_2->unload();
    for(int z = 0; z < l; z++) {//loop  over values of Z
      if(z!=0){
        WL_4_1->wilsonLineUpdate(*su3_4_1, tmp, 4 + WilsDir, true);
        WL_4_2->wilsonLineUpdate(*su3_4_2, tmp, WilsDir);
      }
      WL_2_1->copy(*WL_4_1);
      WL_2_2->copy(*WL_4_1);
      for(int b = 0; b <= B_max; b++) {//loop over values of B
        if(b != 0) {
          WL_2_1->wilsonLineUpdate(*su3_2_1, tmp, Bdir1);
          WL_2_2->wilsonLineUpdate(*su3_2_2, tmp, Bdir2);
        }
        for(int bi = 0; bi < b; bi++) {
          su3_exchange = su3_in; su3_in = WL_4_2; WL_4_2 = su3_exchange;
          WL_4_2->shift(*su3_in, Bdir1);
        }
        WL1->UxU(*WL_2_1,*WL_4_2);
        //WL1->writeHDF5(("WL_test.Bdir1.h5/l"+std::to_string(l)+"b"+std::to_string(b)+"z"+std::to_string(z)).c_str());
        for(int bi = 0; bi < b; bi++) {
          su3_exchange = su3_in; su3_in = WL_4_2; WL_4_2 = su3_exchange;
          WL_4_2->shift(*su3_in, 4 + Bdir1);
        } 
        for(int bi = 0; bi < b; bi++) {
          su3_exchange = su3_in; su3_in = WL_4_2; WL_4_2 = su3_exchange;
          WL_4_2->shift(*su3_in, Bdir2);
        }
        WL2->UxU(*WL_2_2,*WL_4_2);
        //WL2->writeHDF5(("WL_test.Bdir2.h5/l"+std::to_string(l)+"b"+std::to_string(b)+"z"+std::to_string(z)).c_str());
        for(int bi = 0; bi < b; bi++) {
          su3_exchange = su3_in; su3_in = WL_4_2; WL_4_2 = su3_exchange;
          WL_4_2->shift(*su3_in, 4 + Bdir2);
        }
        for(int is = 0; is < numSourcePositions; is++) {//loop over source positions
          PLEGMA_Correlator<float> corrThrpWL(corr_space, sourcePositions[is], 0, (tSinks[ts]+1));
          int signPer = (tSinks[ts] + sourcePositions[is][3]) >= HGC_totalL[3] ? -1 : +1;

          if(nucleon == PROTON) prop1->copy(*propDN[is]);
          else prop1->copy(*propUP[is]);
          prop2->copy(*seqPropOut1[is]);
          for(int zi = 0; zi < z; zi++) {
            propExchange = propIn; propIn = prop1; prop1 = propExchange;
            prop1->shift(*propIn, WilsDir);
            propExchange = propIn; propIn = prop2; prop2 = propExchange;
            prop2->shift(*propIn, 4 + WilsDir);
          }
          for(int bi = 0; bi < b; bi++) {
            propExchange = propIn; propIn = prop1; prop1 = propExchange;
            prop1->shift(*propIn, Bdir1);
          }
          corrThrpWL.contractNucleonThrp_staple(*prop2, *prop1, *WL1, signProps1, gammas, l, -b, z);
          if(signPer < 0) for(int iv = 0 ; iv < corrThrpWL.getTotalSize()*2; iv++) corrThrpWL.H_elem()[iv] *= signPer;
          corrThrpWL.writeHDF5( (threep_filename +".CP1_Bdir_"+std::to_string(Bdir1)+"_ts_"+std::to_string(tSinks[ts])+"_stout_"+std::to_string(nStout)+".h5").c_str());
          for(int bi = 0; bi < b; bi++) {
            propExchange = propIn; propIn = prop1; prop1 = propExchange;
            prop1->shift(*propIn, 4 + Bdir1);
          }
          for(int bi = 0; bi < b; bi++) {
            propExchange = propIn; propIn = prop1; prop1 = propExchange;
            prop1->shift(*propIn, Bdir2);
          }
          corrThrpWL.contractNucleonThrp_staple(*prop2, *prop1, *WL2, signProps1, gammas, l, -b, z);
          if(signPer < 0) for(int iv = 0 ; iv < corrThrpWL.getTotalSize()*2; iv++) corrThrpWL.H_elem()[iv] *= signPer;
          corrThrpWL.writeHDF5( (threep_filename +".CP1_Bdir_"+std::to_string(Bdir2)+"_ts_"+std::to_string(tSinks[ts])+"_stout_"+std::to_string(nStout)+".h5").c_str());

          if(nucleon == PROTON) prop1->copy(*propUP[is]);
          else prop1->copy(*propDN[is]);
          prop2->copy(*seqPropOut2[is]);
          for(int zi = 0; zi < z; zi++) {
            propExchange = propIn; propIn = prop1; prop1 = propExchange;
            prop1->shift(*propIn, WilsDir);
            propExchange = propIn; propIn = prop2; prop2 = propExchange;
            prop2->shift(*propIn, 4 + WilsDir);
          }
          for(int bi = 0; bi < b; bi++) {
            propExchange = propIn; propIn = prop1; prop1 = propExchange;
            prop1->shift(*propIn, Bdir1);
          }
          corrThrpWL.contractNucleonThrp_staple(*prop2, *prop1, *WL1, signProps2, gammas, l, -b, z);
          if(signPer < 0) for(int iv = 0 ; iv < corrThrpWL.getTotalSize()*2; iv++) corrThrpWL.H_elem()[iv] *= signPer;
          corrThrpWL.writeHDF5( (threep_filename +".CP2_Bdir_"+std::to_string(Bdir1)+"_ts_"+std::to_string(tSinks[ts])+"_stout_"+std::to_string(nStout)+".h5").c_str());
          for(int bi = 0; bi < b; bi++) {
            propExchange = propIn; propIn = prop1; prop1 = propExchange;
            prop1->shift(*propIn, 4 + Bdir1);
          }
          for(int bi = 0; bi < b; bi++) {
            propExchange = propIn; propIn = prop1; prop1 = propExchange;
            prop1->shift(*propIn, Bdir2);
          }
          corrThrpWL.contractNucleonThrp_staple(*prop2, *prop1, *WL2, signProps2, gammas, l, -b, z);
          if(signPer < 0) for(int iv = 0 ; iv < corrThrpWL.getTotalSize()*2; iv++) corrThrpWL.H_elem()[iv] *= signPer;
          corrThrpWL.writeHDF5( (threep_filename +".CP2_Bdir_"+std::to_string(Bdir2)+"_ts_"+std::to_string(tSinks[ts])+"_stout_"+std::to_string(nStout)+".h5").c_str());
          //delete corrThrpWL;
        }//loop over source positions
      }//loop over values of B
      su3_2_1->load();
      su3_2_2->load();
      WL_2_1->copy(*WL_4_2);
      WL_2_2->copy(*WL_4_2);
      WL_2_1->Udag();
      WL_2_2->Udag();
      for(int b = 0; b <= B_max; b++) {//loop over values of B
        if(b != 0) {
          WL_2_1->wilsonLineUpdate(*su3_2_1, tmp, Bdir1);
          WL_2_2->wilsonLineUpdate(*su3_2_2, tmp, Bdir2);
        }
        for(int bi = 0; bi < b; bi++) {
          su3_exchange = su3_in; su3_in = WL_4_1; WL_4_1 = su3_exchange;
          WL_4_1->shift(*su3_in, Bdir1);
        }
        WL1->UxUdag(*WL_2_1,*WL_4_1);
        //WL1->writeHDF5(("WL_test.Bdir1.h5/l"+std::to_string(l)+"b"+std::to_string(b)+"z"+std::to_string(-z)).c_str());
        for(int bi = 0; bi < b; bi++) {
          su3_exchange = su3_in; su3_in = WL_4_1; WL_4_1 = su3_exchange;
          WL_4_1->shift(*su3_in, 4 + Bdir1);
        } 
        for(int bi = 0; bi < b; bi++) {
          su3_exchange = su3_in; su3_in = WL_4_1; WL_4_1 = su3_exchange;
          WL_4_1->shift(*su3_in, Bdir2);
        }
        WL2->UxUdag(*WL_2_2,*WL_4_1);
        //WL2->writeHDF5(("WL_test.Bdir2.h5/l"+std::to_string(l)+"b"+std::to_string(b)+"z"+std::to_string(-z)).c_str());
        for(int bi = 0; bi < b; bi++) {
          su3_exchange = su3_in; su3_in = WL_4_1; WL_4_1 = su3_exchange;
          WL_4_1->shift(*su3_in, 4 + Bdir2);
        }
        for(int is = 0; is < numSourcePositions; is++) {//loop over source positions
          PLEGMA_Correlator<float> corrThrpWL(corr_space, sourcePositions[is], 0, (tSinks[ts]+1));
          int signPer = (tSinks[ts] + sourcePositions[is][3]) >= HGC_totalL[3] ? -1 : +1;

          if(nucleon == PROTON) prop1->copy(*propDN[is]);
          else prop1->copy(*propUP[is]);
          prop2->copy(*seqPropOut1[is]);
          for(int zi = 0; zi < z; zi++) {
            propExchange = propIn; propIn = prop1; prop1 = propExchange;
            prop1->shift(*propIn, 4 + WilsDir);
            propExchange = propIn; propIn = prop2; prop2 = propExchange;
            prop2->shift(*propIn, WilsDir);
          }
          for(int bi = 0; bi < b; bi++) {
            propExchange = propIn; propIn = prop1; prop1 = propExchange;
            prop1->shift(*propIn, Bdir1);
          }
          corrThrpWL.contractNucleonThrp_staple(*prop2, *prop1, *WL1, signProps1, gammas, l, -b, -z);
          if(signPer < 0) for(int iv = 0 ; iv < corrThrpWL.getTotalSize()*2; iv++) corrThrpWL.H_elem()[iv] *= signPer;
          corrThrpWL.writeHDF5( (threep_filename +".CP1_Bdir_"+std::to_string(Bdir1)+"_ts_"+std::to_string(tSinks[ts])+"_stout_"+std::to_string(nStout)+".h5").c_str());
          for(int bi = 0; bi < b; bi++) {
            propExchange = propIn; propIn = prop1; prop1 = propExchange;
            prop1->shift(*propIn, 4 + Bdir1);
          }
          for(int bi = 0; bi < b; bi++) {
            propExchange = propIn; propIn = prop1; prop1 = propExchange;
            prop1->shift(*propIn, Bdir2);
          }
          corrThrpWL.contractNucleonThrp_staple(*prop2, *prop1, *WL2, signProps1, gammas, l, -b, -z);
          if(signPer < 0) for(int iv = 0 ; iv < corrThrpWL.getTotalSize()*2; iv++) corrThrpWL.H_elem()[iv] *= signPer;
          corrThrpWL.writeHDF5( (threep_filename +".CP1_Bdir_"+std::to_string(Bdir2)+"_ts_"+std::to_string(tSinks[ts])+"_stout_"+std::to_string(nStout)+".h5").c_str());

          if(nucleon == PROTON) prop1->copy(*propUP[is]);
          else prop1->copy(*propDN[is]);
          prop2->copy(*seqPropOut2[is]);
          for(int zi = 0; zi < z; zi++) {
            propExchange = propIn; propIn = prop1; prop1 = propExchange;
            prop1->shift(*propIn, 4 + WilsDir);
            propExchange = propIn; propIn = prop2; prop2 = propExchange;
            prop2->shift(*propIn, WilsDir);
          }
          for(int bi = 0; bi < b; bi++) {
            propExchange = propIn; propIn = prop1; prop1 = propExchange;
            prop1->shift(*propIn, Bdir1);
          }
          corrThrpWL.contractNucleonThrp_staple(*prop2, *prop1, *WL1, signProps2, gammas, l, -b, -z);
          if(signPer < 0) for(int iv = 0 ; iv < corrThrpWL.getTotalSize()*2; iv++) corrThrpWL.H_elem()[iv] *= signPer;
          corrThrpWL.writeHDF5( (threep_filename +".CP2_Bdir_"+std::to_string(Bdir1)+"_ts_"+std::to_string(tSinks[ts])+"_stout_"+std::to_string(nStout)+".h5").c_str());
          for(int bi = 0; bi < b; bi++) {
            propExchange = propIn; propIn = prop1; prop1 = propExchange;
            prop1->shift(*propIn, 4 + Bdir1);
          }
          for(int bi = 0; bi < b; bi++) {
            propExchange = propIn; propIn = prop1; prop1 = propExchange;
            prop1->shift(*propIn, Bdir2);
          }
          corrThrpWL.contractNucleonThrp_staple(*prop2, *prop1, *WL2, signProps2, gammas, l, -b, -z);
          if(signPer < 0) for(int iv = 0 ; iv < corrThrpWL.getTotalSize()*2; iv++) corrThrpWL.H_elem()[iv] *= signPer;
          corrThrpWL.writeHDF5( (threep_filename +".CP2_Bdir_"+std::to_string(Bdir2)+"_ts_"+std::to_string(tSinks[ts])+"_stout_"+std::to_string(nStout)+".h5").c_str());
          //delete corrThrpWL;
        }//loop over source positions
      }//loop over values of B
      su3_2_1->load();
      su3_2_2->load();
    }//loop over values of Z
  }//loop over values of L

  //negative L, negative B

  for(int l:Lvals) {//loop over values of L
    //absorb the gauge directions for building the 4 branches
    su3_1->absorbDir_device(gaugeWL, WilsDir);
    su3_2_1->absorbDir_device(gaugeWL, Bdir1);
    su3_2_2->absorbDir_device(gaugeWL, Bdir2);
    //su3_3_1->absorbDir_device(gaugeWL, WilsDir);
    //su3_3_2->absorbDir_device(gaugeWL, WilsDir);
    su3_4_1->absorbDir_device(gaugeWL, WilsDir);
    su3_4_2->absorbDir_device(gaugeWL, WilsDir);

    WL_1->setUnit( (std::vector<int>) {0,4,8});
    for(int li = 0; li < l; li++) {
      WL_1->wilsonLineUpdate(*su3_1, tmp, WilsDir);
      su3_exchange = su3_in; su3_in = su3_2_1; su3_2_1 = su3_exchange;
      su3_2_1->shift(*su3_in, WilsDir);
      su3_exchange = su3_in; su3_in = su3_2_2; su3_2_2 = su3_exchange;
      su3_2_2->shift(*su3_in, WilsDir);
    }
    WL_4_1->copy(*WL_1);
    WL_4_2->copy(*WL_1);
    WL_4_2->Udag();
    su3_2_1->unload();
    su3_2_2->unload();
    for(int z = 0; z < l; z++) {//loop  over values of Z
      if(z!=0){
        WL_4_1->wilsonLineUpdate(*su3_4_1, tmp, WilsDir, true);
        WL_4_2->wilsonLineUpdate(*su3_4_2, tmp, 4 + WilsDir);
      }
      WL_2_1->copy(*WL_4_1);
      WL_2_2->copy(*WL_4_1);
      for(int b = 0; b <= B_max; b++) {//loop over values of B
        if(b != 0) {
          WL_2_1->wilsonLineUpdate(*su3_2_1, tmp, Bdir1);
          WL_2_2->wilsonLineUpdate(*su3_2_2, tmp, Bdir2);
        }
        for(int bi = 0; bi < b; bi++) {
          su3_exchange = su3_in; su3_in = WL_4_2; WL_4_2 = su3_exchange;
          WL_4_2->shift(*su3_in, Bdir1);
        }
        WL1->UxU(*WL_2_1,*WL_4_2);
        //WL1->writeHDF5(("WL_test.Bdir1.h5/l"+std::to_string(l)+"b"+std::to_string(b)+"z"+std::to_string(z)).c_str());
        for(int bi = 0; bi < b; bi++) {
          su3_exchange = su3_in; su3_in = WL_4_2; WL_4_2 = su3_exchange;
          WL_4_2->shift(*su3_in, 4 + Bdir1);
        } 
        for(int bi = 0; bi < b; bi++) {
          su3_exchange = su3_in; su3_in = WL_4_2; WL_4_2 = su3_exchange;
          WL_4_2->shift(*su3_in, Bdir2);
        }
        WL2->UxU(*WL_2_2,*WL_4_2);
        //WL2->writeHDF5(("WL_test.Bdir2.h5/l"+std::to_string(l)+"b"+std::to_string(b)+"z"+std::to_string(z)).c_str());
        for(int bi = 0; bi < b; bi++) {
          su3_exchange = su3_in; su3_in = WL_4_2; WL_4_2 = su3_exchange;
          WL_4_2->shift(*su3_in, 4 + Bdir2);
        }
        for(int is = 0; is < numSourcePositions; is++) {//loop over source positions
          PLEGMA_Correlator<float> corrThrpWL(corr_space, sourcePositions[is], 0, (tSinks[ts]+1));
          int signPer = (tSinks[ts] + sourcePositions[is][3]) >= HGC_totalL[3] ? -1 : +1;

          if(nucleon == PROTON) prop1->copy(*propDN[is]);
          else prop1->copy(*propUP[is]);
          prop2->copy(*seqPropOut1[is]);
          for(int zi = 0; zi < z; zi++) {
            propExchange = propIn; propIn = prop1; prop1 = propExchange;
            prop1->shift(*propIn, 4 + WilsDir);
            propExchange = propIn; propIn = prop2; prop2 = propExchange;
            prop2->shift(*propIn, WilsDir);
          }
          for(int bi = 0; bi < b; bi++) {
            propExchange = propIn; propIn = prop1; prop1 = propExchange;
            prop1->shift(*propIn, Bdir1);
          }
          corrThrpWL.contractNucleonThrp_staple(*prop2, *prop1, *WL1, signProps1, gammas, -l, -b, z);
          if(signPer < 0) for(int iv = 0 ; iv < corrThrpWL.getTotalSize()*2; iv++) corrThrpWL.H_elem()[iv] *= signPer;
          corrThrpWL.writeHDF5( (threep_filename +".CP1_Bdir_"+std::to_string(Bdir1)+"_ts_"+std::to_string(tSinks[ts])+"_stout_"+std::to_string(nStout)+".h5").c_str());
          for(int bi = 0; bi < b; bi++) {
            propExchange = propIn; propIn = prop1; prop1 = propExchange;
            prop1->shift(*propIn, 4 + Bdir1);
          }
          for(int bi = 0; bi < b; bi++) {
            propExchange = propIn; propIn = prop1; prop1 = propExchange;
            prop1->shift(*propIn, Bdir2);
          }
          corrThrpWL.contractNucleonThrp_staple(*prop2, *prop1, *WL2, signProps1, gammas, -l, -b, z);
          if(signPer < 0) for(int iv = 0 ; iv < corrThrpWL.getTotalSize()*2; iv++) corrThrpWL.H_elem()[iv] *= signPer;
          corrThrpWL.writeHDF5( (threep_filename +".CP1_Bdir_"+std::to_string(Bdir2)+"_ts_"+std::to_string(tSinks[ts])+"_stout_"+std::to_string(nStout)+".h5").c_str());

          if(nucleon == PROTON) prop1->copy(*propUP[is]);
          else prop1->copy(*propDN[is]);
          prop2->copy(*seqPropOut2[is]);
          for(int zi = 0; zi < z; zi++) {
            propExchange = propIn; propIn = prop1; prop1 = propExchange;
            prop1->shift(*propIn, 4 + WilsDir);
            propExchange = propIn; propIn = prop2; prop2 = propExchange;
            prop2->shift(*propIn, WilsDir);
          }
          for(int bi = 0; bi < b; bi++) {
            propExchange = propIn; propIn = prop1; prop1 = propExchange;
            prop1->shift(*propIn, Bdir1);
          }
          corrThrpWL.contractNucleonThrp_staple(*prop2, *prop1, *WL1, signProps2, gammas, -l, -b, z);
          if(signPer < 0) for(int iv = 0 ; iv < corrThrpWL.getTotalSize()*2; iv++) corrThrpWL.H_elem()[iv] *= signPer;
          corrThrpWL.writeHDF5( (threep_filename +".CP2_Bdir_"+std::to_string(Bdir1)+"_ts_"+std::to_string(tSinks[ts])+"_stout_"+std::to_string(nStout)+".h5").c_str());
          for(int bi = 0; bi < b; bi++) {
            propExchange = propIn; propIn = prop1; prop1 = propExchange;
            prop1->shift(*propIn, 4 + Bdir1);
          }
          for(int bi = 0; bi < b; bi++) {
            propExchange = propIn; propIn = prop1; prop1 = propExchange;
            prop1->shift(*propIn, Bdir2);
          }
          corrThrpWL.contractNucleonThrp_staple(*prop2, *prop1, *WL2, signProps2, gammas, -l, -b, z);
          if(signPer < 0) for(int iv = 0 ; iv < corrThrpWL.getTotalSize()*2; iv++) corrThrpWL.H_elem()[iv] *= signPer;
          corrThrpWL.writeHDF5( (threep_filename +".CP2_Bdir_"+std::to_string(Bdir2)+"_ts_"+std::to_string(tSinks[ts])+"_stout_"+std::to_string(nStout)+".h5").c_str());
          //delete corrThrpWL;
        }//loop over source positions
      }//loop over values of B
      su3_2_1->load();
      su3_2_2->load();
      WL_2_1->copy(*WL_4_2);
      WL_2_2->copy(*WL_4_2);
      WL_2_1->Udag();
      WL_2_2->Udag();
      for(int b = 0; b <= B_max; b++) {//loop over values of B
        if(b != 0) {
          WL_2_1->wilsonLineUpdate(*su3_2_1, tmp, Bdir1);
          WL_2_2->wilsonLineUpdate(*su3_2_2, tmp, Bdir2);
        }
        for(int bi = 0; bi < b; bi++) {
          su3_exchange = su3_in; su3_in = WL_4_1; WL_4_1 = su3_exchange;
          WL_4_1->shift(*su3_in, Bdir1);
        }
        WL1->UxUdag(*WL_2_1,*WL_4_1);
        //WL1->writeHDF5(("WL_test.Bdir1.h5/l"+std::to_string(l)+"b"+std::to_string(b)+"z"+std::to_string(-z)).c_str());
        for(int bi = 0; bi < b; bi++) {
          su3_exchange = su3_in; su3_in = WL_4_1; WL_4_1 = su3_exchange;
          WL_4_1->shift(*su3_in, 4 + Bdir1);
        } 
        for(int bi = 0; bi < b; bi++) {
          su3_exchange = su3_in; su3_in = WL_4_1; WL_4_1 = su3_exchange;
          WL_4_1->shift(*su3_in, Bdir2);
        }
        WL2->UxUdag(*WL_2_2,*WL_4_1);
        //WL2->writeHDF5(("WL_test.Bdir2.h5/l"+std::to_string(l)+"b"+std::to_string(b)+"z"+std::to_string(-z)).c_str());
        for(int bi = 0; bi < b; bi++) {
          su3_exchange = su3_in; su3_in = WL_4_1; WL_4_1 = su3_exchange;
          WL_4_1->shift(*su3_in, 4 + Bdir2);
        }
        for(int is = 0; is < numSourcePositions; is++) {//loop over source positions
          PLEGMA_Correlator<float> corrThrpWL(corr_space, sourcePositions[is], 0, (tSinks[ts]+1));
          int signPer = (tSinks[ts] + sourcePositions[is][3]) >= HGC_totalL[3] ? -1 : +1;

          if(nucleon == PROTON) prop1->copy(*propDN[is]);
          else prop1->copy(*propUP[is]);
          prop2->copy(*seqPropOut1[is]);
          for(int zi = 0; zi < z; zi++) {
            propExchange = propIn; propIn = prop1; prop1 = propExchange;
            prop1->shift(*propIn, WilsDir);
            propExchange = propIn; propIn = prop2; prop2 = propExchange;
            prop2->shift(*propIn, 4 + WilsDir);
          }
          for(int bi = 0; bi < b; bi++) {
            propExchange = propIn; propIn = prop1; prop1 = propExchange;
            prop1->shift(*propIn, Bdir1);
          }
          corrThrpWL.contractNucleonThrp_staple(*prop2, *prop1, *WL1, signProps1, gammas, -l, -b, -z);
          if(signPer < 0) for(int iv = 0 ; iv < corrThrpWL.getTotalSize()*2; iv++) corrThrpWL.H_elem()[iv] *= signPer;
          corrThrpWL.writeHDF5( (threep_filename +".CP1_Bdir_"+std::to_string(Bdir1)+"_ts_"+std::to_string(tSinks[ts])+"_stout_"+std::to_string(nStout)+".h5").c_str());
          for(int bi = 0; bi < b; bi++) {
            propExchange = propIn; propIn = prop1; prop1 = propExchange;
            prop1->shift(*propIn, 4 + Bdir1);
          }
          for(int bi = 0; bi < b; bi++) {
            propExchange = propIn; propIn = prop1; prop1 = propExchange;
            prop1->shift(*propIn, Bdir2);
          }
          corrThrpWL.contractNucleonThrp_staple(*prop2, *prop1, *WL2, signProps1, gammas, -l, -b, -z);
          if(signPer < 0) for(int iv = 0 ; iv < corrThrpWL.getTotalSize()*2; iv++) corrThrpWL.H_elem()[iv] *= signPer;
          corrThrpWL.writeHDF5( (threep_filename +".CP1_Bdir_"+std::to_string(Bdir2)+"_ts_"+std::to_string(tSinks[ts])+"_stout_"+std::to_string(nStout)+".h5").c_str());

          if(nucleon == PROTON) prop1->copy(*propUP[is]);
          else prop1->copy(*propDN[is]);
          prop2->copy(*seqPropOut2[is]);
          for(int zi = 0; zi < z; zi++) {
            propExchange = propIn; propIn = prop1; prop1 = propExchange;
            prop1->shift(*propIn, WilsDir);
            propExchange = propIn; propIn = prop2; prop2 = propExchange;
            prop2->shift(*propIn, 4 + WilsDir);
          }
          for(int bi = 0; bi < b; bi++) {
            propExchange = propIn; propIn = prop1; prop1 = propExchange;
            prop1->shift(*propIn, Bdir1);
          }
          corrThrpWL.contractNucleonThrp_staple(*prop2, *prop1, *WL1, signProps2, gammas, -l, -b, -z);
          if(signPer < 0) for(int iv = 0 ; iv < corrThrpWL.getTotalSize()*2; iv++) corrThrpWL.H_elem()[iv] *= signPer;
          corrThrpWL.writeHDF5( (threep_filename +".CP2_Bdir_"+std::to_string(Bdir1)+"_ts_"+std::to_string(tSinks[ts])+"_stout_"+std::to_string(nStout)+".h5").c_str());
          for(int bi = 0; bi < b; bi++) {
            propExchange = propIn; propIn = prop1; prop1 = propExchange;
            prop1->shift(*propIn, 4 + Bdir1);
          }
          for(int bi = 0; bi < b; bi++) {
            propExchange = propIn; propIn = prop1; prop1 = propExchange;
            prop1->shift(*propIn, Bdir2);
          }
          corrThrpWL.contractNucleonThrp_staple(*prop2, *prop1, *WL2, signProps2, gammas, -l, -b, -z);
          if(signPer < 0) for(int iv = 0 ; iv < corrThrpWL.getTotalSize()*2; iv++) corrThrpWL.H_elem()[iv] *= signPer;
          corrThrpWL.writeHDF5( (threep_filename +".CP2_Bdir_"+std::to_string(Bdir2)+"_ts_"+std::to_string(tSinks[ts])+"_stout_"+std::to_string(nStout)+".h5").c_str());
          //delete corrThrpWL;
        }//loop over source positions
      }//loop over values of B
      su3_2_1->load();
      su3_2_2->load();
    }//loop over values of Z
  }//loop over values of L

  } //loop over tSinks

  for(int is = 0; is < numSourcePositions; is++) {//loop over source positions
    propUP_SS[is]->rotateToPhysicalBase_device(+1);
    propDN_SS[is]->rotateToPhysicalBase_device(-1);
    propUP_SS[is]->applyBoundaries_device(sourcePositions[is][3]);
    propDN_SS[is]->applyBoundaries_device(sourcePositions[is][3]);
    
    PLEGMA_Correlator<float> corr(corr_space, sourcePositions[is], 0);
    corr.setFixMomVec(sinkMom_3D);
    
    corr.contractBaryons(*propUP_SS[is], *propDN_SS[is]);
    corr.writeFile((twop_filename).c_str(), corr_file_format);
  }

  //WL testing
#if 0
  WL->setUnit((std::vector<int>) {0,4,8});
  int spath[11] = {0,0,0,0,0,1,1,1,4,4,4};
  std::vector<int> vspath(spath,spath+11);
  WL->path(vspath, u_s, tmp);
  for(int ii = 0; ii < 3; ii++) {
    WLExchange = WLIn; WLIn = WL; WL = WLExchange;
    WL->shift(*WLIn,0);
  }
  for(int ii = 0; ii < 3; ii++) {
    WLExchange = WLIn; WLIn = WL; WL = WLExchange;
    WL->shift(*WLIn,5);
  }
  for(int ii = 0; ii < 4; ii++) {
    WLExchange = WLIn; WLIn = WL; WL = WLExchange;
    WL->shift(*WLIn,4);
  }
  WL->writeHDF5("WL_xcheck.h5");
#endif

  delete solver;
  
  finalize();
  
  return 0;
}
