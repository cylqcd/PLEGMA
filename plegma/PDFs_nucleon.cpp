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

  size_t WilsDir;
  HGC_options->set("wilson_direction", "Direction of the wilson line", verbosity, WilsDir);
  if(WilsDir>N_DIMS) PLEGMA_error("The direction of the WIlson line has to be smaller than 3");

  double rhoStout;
  HGC_options->set("rho_stout", "Rho parameter stout smearing", verbosity, rhoStout);

  size_t maxStout;
  HGC_options->set("max_stout", "Maximum number of stout smearing steps", verbosity, maxStout);

  size_t stepStout;
  HGC_options->set("step_stout", "Save the PDFs every step_stout stout smearing step", verbosity, stepStout);

  bool calc3pt = true ;
  HGC_options->set("calc3pt", "If true then the 3pt function is computed", verbosity, calc3pt);

  
  std::string proj ;
  HGC_options->set("which_projector", "Which projector to use for 3pt function", verbosity, proj);
  WHICHPROJECTOR which_proj=get_projector(proj.c_str());

  initializePLEGMA();
  
  // Reading from Lime file and loading to device
  PLEGMA_Gauge<double> gauge;
  gauge.readFile(latfile, LIME_FORMAT);
  gauge.load();
  gauge.calculatePlaq();

  // Loading to QUDA and computing plaquette also there
  initGaugeQuda(gauge, true, QUDA_WILSON_LINKS);
  plaqQuda();
  
  WHICHPARTICLE nucleon = which_particle;
  if(nucleon!=NEUTRON && nucleon!=PROTON) PLEGMA_error("Only nucleon PDFs have been implemented so far\n");
  
  // Smearing
  PLEGMA_Gauge<double> smearedGauge;
  smearedGauge.APEsmearing(gauge, nsmearAPE, alphaAPE, 3);
  PLEGMA_printf("Plaquette after smearing:\n");
  smearedGauge.calculatePlaq();

  //Momentum smearing: put the momentum phase to smeared gauge field
  std::complex<double> momSmScale[N_DIMS];
  std::complex<double> I(0,1);
  for(int i = 0 ; i < N_DIMS; i++) momSmScale[i] = std::exp(-(xiMomSm*2.*PI*sinkMom[i]/HGC_totalL[i])*I);
  smearedGauge.scaleDirWise(momSmScale);

  PLEGMA_Gauge<float> gaugeWL;
  gaugeWL.copy(gauge);

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
  PLEGMA_Propagator<float> *seqPropOut = new PLEGMA_Propagator<float>(BOTH);
  PLEGMA_Propagator<float> *propIn = new PLEGMA_Propagator<float>(BOTH);
  PLEGMA_Propagator<float> *propExchange = nullptr;
  
  PLEGMA_Propagator3D<float> propUP3D;
  PLEGMA_Propagator3D<float> propDN3D;

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

  PLEGMA_Correlator<float> corrThrpWL(corr_space, sourcePositions[isource], 0); 
  
  for(int ts=0;ts<tSinks.size();ts++)
    {
      int signPer = (tSinks[ts] + sourcePositions[isource][3]) >= HGC_totalL[3] ? -1 : +1;
      int global_fixSinkTime = (tSinks[ts] + sourcePositions[isource][3])%HGC_totalL[3]; 
      int my_fixSinkTime = global_fixSinkTime - comm_coords(HGC_default_topo)[3] * HGC_localL[3];
      bool is_myST = (my_fixSinkTime >= 0) && ( my_fixSinkTime < HGC_localL[3] );

      for(int isc = 0 ; isc < 12 ; isc++){
	vectorAuxD.pointSource(sourcePositions[isource], isc/3, isc%3, DEVICE);
	vectorIn.gaussianSmearing(vectorAuxD, smearedGauge, nsmearGauss, alphaGauss);
    
	PLEGMA_printf("Going to invert UP for component %d\n", isc);
	solver->solve(vectorOut, vectorIn);
	vectorAuxF.copy(vectorOut);
	propUP->absorb(vectorAuxF, isc/3, isc%3);
	vectorAuxD.gaussianSmearing(vectorOut, smearedGauge , nsmearGauss, alphaGauss);
	vectorAuxF.copy(vectorAuxD);
	propUP3D.absorb(vectorAuxF,global_fixSinkTime, isc/3, isc%3);
      }
	

      if(mu>0) {
	mu*=-1.;
	solver->UpdateSolver();
      }
      for(int isc = 0 ; isc < 12 ; isc++){
	vectorAuxD.pointSource(sourcePositions[isource], isc/3, isc%3, DEVICE);
	vectorIn.gaussianSmearing(vectorAuxD, smearedGauge, nsmearGauss, alphaGauss);
    
	PLEGMA_printf("Going to invert DN for component %d\n", isc);
	solver->solve(vectorOut, vectorIn);
	vectorAuxF.copy(vectorOut);
	propDN->absorb(vectorAuxF, isc/3, isc%3);
	vectorAuxD.gaussianSmearing(vectorOut, smearedGauge , nsmearGauss, alphaGauss);
	vectorAuxF.copy(vectorAuxD);
	propDN3D.absorb(vectorAuxF,global_fixSinkTime, isc/3, isc%3);
      }

      
      if(calc3pt){

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
		vectorAuxF.seqSourceNucleon(propUP3D, propDN3D, which_proj, nucleon, global_fixSinkTime, nu, c2);
	      else
		vectorAuxF.seqSourceNucleon(propDN3D, propUP3D, which_proj, nucleon, global_fixSinkTime, nu, c2);
	      vectorAuxF.mulMomentumPhases(sinkMom,-1); // put momentum at the sink
	      std::complex<float> Isingle(0,1);
	      float phase = 2.*PI*(((float) sinkMom[0] * sourcePositions[isource][0])/HGC_totalL[0]
				   + ((float)sinkMom[1] * sourcePositions[isource][1])/HGC_totalL[1]
				   + ((float)sinkMom[2] * sourcePositions[isource][2])/HGC_totalL[2]);
	      vectorAuxF.cscale(std::exp<float>(+phase*Isingle)); // put momentum from the point source
	      vectorAuxF.conjugate();
	      vectorAuxF.apply_gamma(G5);
	      vectorAuxD.copy(vectorAuxF);
	      vectorIn.gaussianSmearing(vectorAuxD,smearedGauge, nsmearGauss, alphaGauss);
	      // check if we need to normalize the seqsource for mix precision solver
	      if(nucleon == PROTON){
		if(mu<0) {
		  mu*=-1.;
		  solver->UpdateSolver();
		}
	      }
	      else{
		if(mu>0) {
		  mu*=-1.;
		  solver->UpdateSolver();
		}
	      }
	      double norm = vectorIn.norm();
	      vectorIn.cscale(1/norm);
	      solver->solve(vectorOut, vectorIn);
	      vectorOut.cscale(norm);    
	      vectorAuxF.copy(vectorOut);
	      seqPropOut->absorb(vectorAuxF, nu, c2);
	    }
	  seqPropOut->apply_gamma(G5);
	  seqPropOut->conjugate();
    
	  int signProps = (nucleon == PROTON) ? +1: -1;

	  PLEGMA_Propagator<float> *propF = (nucleon == PROTON) ? propUP : propDN;


	  //!!!!!!!!!!!!!!!!!!!!!!!!! if spatial extent is not multiple of 2 then it will not work
	  for(int stIt=0;stIt<=(int)(maxStout/stepStout);stIt++){
	    std::string suff = "_CP2_stout_"+std::to_string(stIt*stepStout)+"_Plus_";
	    if(stIt>0) gaugeWL.stoutSmearing(gaugeWL,stepStout,rhoStout,3);
	    su3.absorbDir_device(gaugeWL, WilsDir);
	    WL.setUnit( (std::vector<int>) {0,4,8});
	    for(int i = 0 ; i < HGC_totalL[WilsDir]/2;i++){ 
	      corrThrpWL.contractNucleonThrp_wilsonLine(*seqPropOut, *propF, WL, signProps, gammas);
	      if(signPer < 0) for(int iv = 0 ; iv < corrThrpWL.getTotalSize()*2; iv++) corrThrpWL.H_elem()[iv] *= signPer;
	      corrThrpWL.writeASCII( (threep_filename +  suff + std::to_string(i) + "_ts_" + std::to_string(tSinks[ts])  + ".dat").c_str() ); 
	      propExchange = propIn; propIn = propF; propF = propExchange;
	      WL.wilsonLineUpdate(su3, tmp, 4+WilsDir); 
	      propF->shift(*propIn, 4+WilsDir);
	    }

	    suff="_CP2_stout_"+std::to_string(stIt*stepStout)+"_Minus_";
	    propF->load();
	    su3.absorbDir_device(gaugeWL, WilsDir); // only for z direction
	    WL.setUnit( (std::vector<int>) {0,4,8});
	    for(int i = 0 ; i < HGC_totalL[WilsDir]/2;i++){ // HGC_totalL[2] only for z direction
	      corrThrpWL.contractNucleonThrp_wilsonLine(*seqPropOut, *propF, WL, signProps, gammas);
	      if(signPer < 0) for(int iv = 0 ; iv < corrThrpWL.getTotalSize()*2; iv++) corrThrpWL.H_elem()[iv] *= signPer;
	      corrThrpWL.writeASCII( (threep_filename + suff + std::to_string(i) +  "_ts_" + std::to_string(tSinks[ts]) + ".dat").c_str() );
	      propExchange = propIn; propIn = propF; propF = propExchange;
	      WL.wilsonLineUpdate(su3, tmp, WilsDir); // build Wilson line in the +z direction
	      propF->shift(*propIn, WilsDir);
	    }
	    propF->load();
	  }
	}

	//seq source part 1Props and contraction block
	{
	  for(int nu = 0 ; nu < 4 ; nu++)
	    for(int c2 = 0 ; c2 < 3 ; c2++){
	      if(nucleon == PROTON)
		vectorAuxF.seqSourceNucleon(propUP3D, which_proj, nucleon, global_fixSinkTime, nu, c2);
	      else
		vectorAuxF.seqSourceNucleon(propDN3D, which_proj, nucleon, global_fixSinkTime, nu, c2);
	      vectorAuxF.mulMomentumPhases(sinkMom,-1); // put momentum at the sink
	      std::complex<float> Isingle(0,1);
	      float phase = 2.*PI*(((float) sinkMom[0] * sourcePositions[isource][0])/HGC_totalL[0]
				   + ((float)sinkMom[1] * sourcePositions[isource][1])/HGC_totalL[1]
				   + ((float)sinkMom[2] * sourcePositions[isource][2])/HGC_totalL[2]);
	      vectorAuxF.cscale(std::exp<float>(+phase*Isingle)); // put momentum from the point source
	      vectorAuxF.conjugate();
	      vectorAuxF.apply_gamma(G5);
	      vectorAuxD.copy(vectorAuxF);
	      vectorIn.gaussianSmearing(vectorAuxD,smearedGauge, nsmearGauss, alphaGauss);
	    
	      if(nucleon == PROTON){
		if(mu<0) {
		  mu*=-1.;
		  solver->UpdateSolver();
		}
	      }
	      else{
		if(mu>0) {
		  mu*=-1.;
		  solver->UpdateSolver();
		}
	      }
	      double norm = vectorIn.norm();
	      vectorIn.cscale(1/norm);
	      solver->solve(vectorOut, vectorIn);
	      vectorOut.cscale(norm);
	      vectorAuxF.copy(vectorOut);
	      seqPropOut->absorb(vectorAuxF, nu, c2);
	    }
	  seqPropOut->apply_gamma(G5);
	  seqPropOut->conjugate();
    
	  int signProps = (nucleon == PROTON) ? -1: +1;

	  PLEGMA_Propagator<float> *propF = (nucleon == PROTON) ? propDN : propUP;
	  gaugeWL.copy(gauge);

	  //!!!!!!!!!!!!!!!!!!!!!!!!! if spatial extent is not multiple of 2 then it will not work
	  for(int stIt=0;stIt<=(int)(maxStout/stepStout);stIt++){
	    if(stIt>0) gaugeWL.stoutSmearing(gaugeWL,stepStout,rhoStout,3);
	    std::string suff="_CP1_stout_"+std::to_string(stIt*stepStout)+"_Plus_";
	    su3.absorbDir_device(gaugeWL, WilsDir); 
	    WL.setUnit( (std::vector<int>) {0,4,8});
	    for(int i = 0 ; i < HGC_totalL[WilsDir]/2;i++){ // HGC_totalL[2] only for z direction
	      corrThrpWL.contractNucleonThrp_wilsonLine(*seqPropOut, *propF, WL, signProps, gammas);
	      if(signPer < 0) for(int iv = 0 ; iv < corrThrpWL.getTotalSize()*2; iv++) corrThrpWL.H_elem()[iv] *= signPer;
	      corrThrpWL.writeASCII( (threep_filename + suff + std::to_string(i) + "_ts_" + std::to_string(tSinks[ts]) + ".dat").c_str() );
	      propExchange = propIn; propIn = propF; propF = propExchange;
	      WL.wilsonLineUpdate(su3, tmp, 4+WilsDir); // build Wilson line in the +z direction
	      propF->shift(*propIn, 4+WilsDir);
	    }
    
	    propF->load();
	    suff="_CP1_stout_"+std::to_string(stIt*stepStout)+"_Minus_";
	    su3.absorbDir_device(gaugeWL, WilsDir); 
	    WL.setUnit( (std::vector<int>) {0,4,8});
	    for(int i = 0 ; i < HGC_totalL[WilsDir]/2;i++){ // HGC_totalL[2] only for z direction
	      corrThrpWL.contractNucleonThrp_wilsonLine(*seqPropOut, *propF, WL, signProps, gammas);
	      if(signPer < 0) for(int iv = 0 ; iv < corrThrpWL.getTotalSize()*2; iv++) corrThrpWL.H_elem()[iv] *= signPer;
	      corrThrpWL.writeASCII( (threep_filename + suff + std::to_string(i) + "_ts_" + std::to_string(tSinks[ts]) + ".dat").c_str() );
	      propExchange = propIn; propIn = propF; propF = propExchange;
	      WL.wilsonLineUpdate(su3, tmp, WilsDir); // build Wilson line in the +z direction
	      propF->shift(*propIn, WilsDir);
	    }
	    propF->load();
	  }
	}
      }
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

  PLEGMA_Correlator<float> corr(corr_space, sourcePositions[isource]);
  corr.setFixMomVec(sinkMom_3D);
  corr.contractMesons(*propUP, *propDN);
  corr.writeFile(twop_filename.c_str(), corr_file_format);

  //!!!!!!!!!! maybe later we choose the specific momentum when this allows it
  corr.contractBaryons(*propUP, *propDN);
  corr.writeFile(twop_filename.c_str(), corr_file_format);
    
  delete propUP;
  delete propDN;
  delete propIn;
  delete seqPropOut;
  
  // delete[] nucleonThrpWLP_CP1;
  // delete[] nucleonThrpWLP_CP2;

  // delete[] nucleonThrpWLM_CP1;
  // delete[] nucleonThrpWLM_CP2;

  delete solver;
  
  finalize();
  return 0;
}

