#include <PLEGMA.h>
#include <PLEGMA_utils.h>

using namespace plegma;
using namespace quda;

int main(int argc, char **argv)
{
    
  initializeOptions(argc, argv); // Put list of Options later
  //================ Add your options in this between initializeOptions and initializePLEGMA ================//

  //=========================================================================================================//
  initializePLEGMA();

  //  int nsmearAPE = 20;
  // double alphaAPE = 0.5;
  double xiMomSm = 0.6; //remember the sign later
  std::vector<int> moms = {0,0,1,0};

  // Reading from Lime file and loading to device
  PLEGMA_Gauge<double> gauge;
  gauge.readFromLime(latfile.c_str());
  gauge.load();
  gauge.calculatePlaq();

  // Loading to QUDA and computing plaquette also there
  initGaugeQuda(gauge, true, QUDA_WILSON_LINKS);
  plaqQuda();
  
  // Smearing
  PLEGMA_Gauge<double> smearedGauge;
  smearedGauge.APEsmearing(gauge, nsmearAPE, alphaAPE, 3);
  PLEGMA_printf("Plaquette after smearing:\n");
  smearedGauge.calculatePlaq();

  //Momentum smearing: put the momentum phase to smeared gauge field
  std::complex<double> momSmScale[N_DIMS];
  std::complex<double> I(0,1);
  for(int i = 0 ; i < N_DIMS; i++) momSmScale[i] = std::exp(-(xiMomSm*2.*PI*moms[i]/HGC_totalL[i])*I);
  smearedGauge.scaleDirWise(momSmScale);

  PLEGMA_Gauge<float> gaugeWL;
  gaugeWL.copy(gauge);

  // ensuring mu positive
  if(mu<0) mu*=-1.;
  QUDA_solver *solverUP = new QUDA_solver(mu);
  mu*=-1.;
  QUDA_solver *solverDN = new QUDA_solver(mu);

  PLEGMA_Vector<double> vectorIn;
  PLEGMA_Vector<double> vectorOut;
  PLEGMA_Vector<double> vectorAuxD;
  PLEGMA_Vector<float> vectorAuxF;
  PLEGMA_Propagator<float> *propUP = new PLEGMA_Propagator<float>(BOTH);
  PLEGMA_Propagator<float> *propDN = new PLEGMA_Propagator<float>(BOTH);
  PLEGMA_Propagator<float> *seqPropOut = new PLEGMA_Propagator<float>(BOTH);
  PLEGMA_Propagator<float> *propIn = new PLEGMA_Propagator<float>(BOTH);
  PLEGMA_Propagator<float> *propExchange = nullptr;
  
  PLEGMA_Propagator3D<float> propUP3D;
  PLEGMA_Propagator3D<float> propDN3D;

  PLEGMA_Correlator<float> corrThrpWL(MOMENTUM_SPACE,0);
  
  // PLEGMA_Correlator<float> *nucleonThrpWLP_CP1 = new PLEGMA_Correlator<float>(MOMENTUM_SPACE,0)[HGC_totalL[2]]; // if is the z direction
  // PLEGMA_Correlator<float> *nucleonThrpWLP_CP2 = new PLEGMA_Correlator<float>(MOMENTUM_SPACE,0)[HGC_totalL[2]];

  // PLEGMA_Correlator<float> *nucleonThrpWLM_CP1 = new PLEGMA_Correlator<float>(MOMENTUM_SPACE,0)[HGC_totalL[2]];
  // PLEGMA_Correlator<float> *nucleonThrpWLM_CP2 = new PLEGMA_Correlator<float>(MOMENTUM_SPACE,0)[HGC_totalL[2]];

  // PLEGMA_Correlator<float> *nucleonThrpWL_CP2[2];
  // nucleonThrpWL_CP2[0] = nucleonThrpWLP_CP2;
  // nucleonThrpWL_CP2[1] = nucleonThrpWLM_CP2;

  // PLEGMA_Correlator<float> *nucleonThrpWL_CP1[2];
  // nucleonThrpWL_CP1[0] = nucleonThrpWLP_CP1;
  // nucleonThrpWL_CP1[1] = nucleonThrpWLM_CP1;


    // for the test use sinkSourceSep = 10;
  int  isource=0;

  int tsinkMtsource = 10;//test with tsink 10
  int signPer = (tsinkMtsource + sourcePositions[isource][3]) >= HGC_totalL[3] ? -1 : +1;
  int global_fixSinkTime = (tsinkMtsource + sourcePositions[isource][3])%HGC_totalL[3]; 
  int my_fixSinkTime = global_fixSinkTime - comm_coords(HGC_default_topo)[3] * HGC_localL[3];
  bool is_myST = (my_fixSinkTime >= 0) && ( my_fixSinkTime < HGC_localL[3] );

  for(int isc = 0 ; isc < 12 ; isc++){
    vectorAuxD.pointSource(sourcePositions[isource], isc/3, isc%3, DEVICE);
    vectorIn.gaussianSmearing(vectorAuxD, smearedGauge, nsmearGauss, alphaGauss);
    
    PLEGMA_printf("Going to invert UP for component %d\n", isc);
    solverUP->solve(vectorOut, vectorIn);
    vectorAuxF.copy(vectorOut);
    propUP->absorb(vectorAuxF, isc/3, isc%3);
    vectorAuxD.gaussianSmearing(vectorOut, smearedGauge , nsmearGauss, alphaGauss);
    vectorAuxF.copy(vectorAuxD);
    propUP3D.absorb(vectorAuxF,global_fixSinkTime, isc/3, isc%3);
    
    
    PLEGMA_printf("Going to invert DN for component %d\n", isc);
    solverDN->solve(vectorOut, vectorIn);
    vectorAuxF.copy(vectorOut);
    propDN->absorb(vectorAuxF, isc/3, isc%3);
    vectorAuxD.gaussianSmearing(vectorOut, smearedGauge , nsmearGauss, alphaGauss);
    vectorAuxF.copy(vectorAuxD);
    propDN3D.absorb(vectorAuxF,global_fixSinkTime, isc/3, isc%3);
  }

  WHICHPARTICLE nucleon = NEUTRON; // for the test is NEUTRON, later we can provide an option

  PLEGMA_Su3field<float> su3;
  PLEGMA_Su3field<float> WL;
  PLEGMA_Su3field<float> tmp;

  propUP->unload();
  propDN->unload();
  //seq source part 2Props and contraction block
  {
    for(int nu = 0 ; nu < 4 ; nu++)
      for(int c2 = 0 ; c2 < 3 ; c2++){
	if(nucleon == PROTON)
	  vectorAuxF.seqSourceNucleon(propUP3D, propDN3D, P4_P, nucleon, global_fixSinkTime, nu, c2); //test case unpolarized proj
	else
	  vectorAuxF.seqSourceNucleon(propDN3D, propUP3D, P4_P, nucleon, global_fixSinkTime, nu, c2);
	vectorAuxF.mulMomentumPhases(moms,-1); // put momentum at the sink
	std::complex<float> Isingle(0,1);
	float phase = 2.*PI*(((float) moms[0] * sourcePositions[isource][0])/HGC_totalL[0]
			     + ((float)moms[1] * sourcePositions[isource][1])/HGC_totalL[1]
			     + ((float)moms[2] * sourcePositions[isource][2])/HGC_totalL[2]);
	vectorAuxF.cscale(std::exp<float>(+phase*Isingle)); // put momentum from the point source
	vectorAuxF.conjugate();
	vectorAuxF.apply_gamma(G5);
	vectorAuxD.copy(vectorAuxF);
	vectorIn.gaussianSmearing(vectorAuxD,smearedGauge, nsmearGauss, alphaGauss);
	// check if we need to normalize the seqsource for mix precision solver
	if(nucleon == PROTON) solverDN->solve(vectorOut, vectorIn); else solverUP->solve(vectorOut, vectorIn);
	// if we normalize the seqsource we have to take it out here
	vectorAuxF.copy(vectorOut);
	seqPropOut->absorb(vectorAuxF, nu, c2);
      }
    seqPropOut->apply_gamma(G5);
    seqPropOut->conjugate();
    
    int signProps = (nucleon == PROTON) ? +1: -1;

    PLEGMA_Propagator<float> *propF = (nucleon == PROTON) ? propUP : propDN;
    std::vector<GAMMAS> gammas = {G3};

    //!!!!!!!!!!!!!!!!!!!!!!!!! if spatial extent is not multiple of 2 then it will not work
    
    su3.absorbDir_device(gaugeWL, 2); // only for z direction
    WL.setUnit( (std::vector<int>) {0,4,8});
    for(int i = 0 ; i < HGC_totalL[2]/2;i++){ // HGC_totalL[2] only for z direction
      corrThrpWL.contractNucleonThrp_wilsonLine(*seqPropOut, *propF, WL, signProps, gammas, sourcePositions[isource]);
      if(signPer < 0) for(int iv = 0 ; iv < corrThrpWL.getTotalSize()*2; iv++) (corrThrpWL.getCorr())[iv] *= signPer;
      corrThrpWL.writeASCII( ("/onyx/noether/h/khadjiyiannakou/runs/threep_PDFs_CP2_Plus_new" + std::to_string(i) + ".dat").c_str() );
      propExchange = propIn; propIn = propF; propF = propExchange;
      WL.wilsonLineUpdate(su3, tmp, 4+2); // build Wilson line in the +z direction
      propF->shift(*propIn, 4+2);
    }
    
    propF->load();
    su3.absorbDir_device(gaugeWL, 2); // only for z direction
    WL.setUnit( (std::vector<int>) {0,4,8});
    for(int i = 0 ; i < HGC_totalL[2]/2;i++){ // HGC_totalL[2] only for z direction
      corrThrpWL.contractNucleonThrp_wilsonLine(*seqPropOut, *propF, WL, signProps, gammas, sourcePositions[isource]);
      if(signPer < 0) for(int iv = 0 ; iv < corrThrpWL.getTotalSize()*2; iv++) (corrThrpWL.getCorr())[iv] *= signPer;
      corrThrpWL.writeASCII( ("/onyx/noether/h/khadjiyiannakou/runs/threep_PDFs_CP2_Minus_new" + std::to_string(i) + ".dat").c_str() );
      propExchange = propIn; propIn = propF; propF = propExchange;
      WL.wilsonLineUpdate(su3, tmp, 2); // build Wilson line in the +z direction
      propF->shift(*propIn, 2);
    }
    propF->load();
  }


  //seq source part 1Props and contraction block
  {
    for(int nu = 0 ; nu < 4 ; nu++)
      for(int c2 = 0 ; c2 < 3 ; c2++){
	if(nucleon == PROTON)
	  vectorAuxF.seqSourceNucleon(propUP3D, P4_P, nucleon, global_fixSinkTime, nu, c2); //test case unpolarized proj
	else
	  vectorAuxF.seqSourceNucleon(propDN3D, P4_P, nucleon, global_fixSinkTime, nu, c2);
	vectorAuxF.mulMomentumPhases(moms,-1); // put momentum at the sink
	std::complex<float> Isingle(0,1);
	float phase = 2.*PI*(((float) moms[0] * sourcePositions[isource][0])/HGC_totalL[0]
			     + ((float)moms[1] * sourcePositions[isource][1])/HGC_totalL[1]
			     + ((float)moms[2] * sourcePositions[isource][2])/HGC_totalL[2]);
	vectorAuxF.cscale(std::exp<float>(+phase*Isingle)); // put momentum from the point source
	vectorAuxF.conjugate();
	vectorAuxF.apply_gamma(G5);
	vectorAuxD.copy(vectorAuxF);
	vectorIn.gaussianSmearing(vectorAuxD,smearedGauge, nsmearGauss, alphaGauss);
	// check if we need to normalize the seqsource for mix precision solver
	if(nucleon == PROTON) solverUP->solve(vectorOut, vectorIn); else solverDN->solve(vectorOut, vectorIn);
	// if we normalize the seqsource we have to take it out here
	vectorAuxF.copy(vectorOut);
	seqPropOut->absorb(vectorAuxF, nu, c2);
      }
    seqPropOut->apply_gamma(G5);
    seqPropOut->conjugate();
    
    int signProps = (nucleon == PROTON) ? -1: +1;

    PLEGMA_Propagator<float> *propF = (nucleon == PROTON) ? propDN : propUP;
    std::vector<GAMMAS> gammas = {G3};

    //!!!!!!!!!!!!!!!!!!!!!!!!! if spatial extent is not multiple of 2 then it will not work
    
    su3.absorbDir_device(gaugeWL, 2); // only for z direction
    WL.setUnit( (std::vector<int>) {0,4,8});
    for(int i = 0 ; i < HGC_totalL[2]/2;i++){ // HGC_totalL[2] only for z direction
      corrThrpWL.contractNucleonThrp_wilsonLine(*seqPropOut, *propF, WL, signProps, gammas, sourcePositions[isource]);
      if(signPer < 0) for(int iv = 0 ; iv < corrThrpWL.getTotalSize()*2; iv++) (corrThrpWL.getCorr())[iv] *= signPer;
      corrThrpWL.writeASCII( ("/onyx/noether/h/khadjiyiannakou/runs/threep_PDFs_CP1_Plus_new" + std::to_string(i) + ".dat").c_str() );
      propExchange = propIn; propIn = propF; propF = propExchange;
      WL.wilsonLineUpdate(su3, tmp, 4+2); // build Wilson line in the +z direction
      propF->shift(*propIn, 4+2);
    }
    
    propF->load();
    su3.absorbDir_device(gaugeWL, 2); // only for z direction
    WL.setUnit( (std::vector<int>) {0,4,8});
    for(int i = 0 ; i < HGC_totalL[2]/2;i++){ // HGC_totalL[2] only for z direction
      corrThrpWL.contractNucleonThrp_wilsonLine(*seqPropOut, *propF, WL, signProps, gammas, sourcePositions[isource]);
      if(signPer < 0) for(int iv = 0 ; iv < corrThrpWL.getTotalSize()*2; iv++) (corrThrpWL.getCorr())[iv] *= signPer;
      corrThrpWL.writeASCII( ("/onyx/noether/h/khadjiyiannakou/runs/threep_PDFs_CP1_Minus_new" + std::to_string(i) + ".dat").c_str() );
      propExchange = propIn; propIn = propF; propF = propExchange;
      WL.wilsonLineUpdate(su3, tmp, 2); // build Wilson line in the +z direction
      propF->shift(*propIn, 2);
    }
    propF->load();
  }

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
  
  PLEGMA_Correlator<float> corr(corr_space, maxQsq);
  corr.contractMesons(*propUP, *propDN, sourcePositions[isource]);
  corr.writeFile(twop_filename.c_str(), corr_file_format);

  //!!!!!!!!!! maybe later we choose the specific momentum when this allows it
  corr.contractBaryons(*propUP, *propDN, sourcePositions[isource]);
  corr.writeFile(twop_filename.c_str(), corr_file_format);

  delete propUP;
  delete propDN;
  delete propIn;
  delete seqPropOut;
  
  // delete[] nucleonThrpWLP_CP1;
  // delete[] nucleonThrpWLP_CP2;

  // delete[] nucleonThrpWLM_CP1;
  // delete[] nucleonThrpWLM_CP2;

  delete solverUP;
  delete solverDN;
  
  finalize();
  return 0;
}

