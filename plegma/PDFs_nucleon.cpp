#include <PLEGMA.h>
#include <PLEGMA_utils.h>

using namespace plegma;
using namespace quda;

int main(int argc, char **argv)
{

  static std::vector<std::string> listOpt = {"verbosity", "load-gauge", "nsmear-APE", "alpha-APE", "nsmear-gauss", "alpha-gauss",
					     "nsrc", "src-filename", "maxQsq", "twop-filename","threep-filename",  "corr-file-format",
					     "corr-space", "tSinks","Projs","xiMomSm","gammas"};

  initializeOptions(argc, argv, true, listOpt);

  //================ Add your options in this between initializeOptions and initializePLEGMA ================//

  size_t WilsDir;
  HGC_options->set("wilson-direction", "Direction of the wilson line", verbosity, WilsDir);
  

  double rhoStout = 0.;
  HGC_options->set("rho-stout", "Rho parameter stout smearing", verbosity, rhoStout);

  size_t maxStout = 1;
  HGC_options->set("max-stout", "Maximum number of stout smearing steps", verbosity, maxStout);

  size_t stepStout = 5;
  HGC_options->set("stout-steps", "Save the PDFs every step_stout stout smearing step", verbosity, stepStout);

  bool calc3pt = true ;
  HGC_options->set("calc3pt", "If true then the 3pt function is computed", verbosity, calc3pt);
  
  /*
    We consider the momenta in the symmetric frame. Both P-momentum and Delta-momentum are vectors
    having three components. 
    The source momentum is given by delta/2-p, while the sink momentum by delta/2+p.
  */

  std::vector<int> PMom = {0,0,0};
  HGC_options->set("P-momentum", "If added to the momentum transfer delta gives the sink momentum", verbosity, PMom);

  std::vector<int> DeltaMom = {0,0,0};
  HGC_options->set("Delta-momentum", "Square root of the momentum transfer", verbosity, DeltaMom);

  std::string aux_str = "proton";
  HGC_options->set("which-particle", "Choice of the nucleon interpolator to insert in the three point function (neutron,proton)", verbosity, aux_str);
  WHICHPARTICLE nucleon = get_particle(aux_str.c_str());

  
  //=========================================================================================================//
  initializePLEGMA();

  if(DeltaMom.size() > 3 || PMom.size() > 3) PLEGMA_error("DeltaMom and PMom have to be vectors with lenght three\n");
  if(DeltaMom.size()!=PMom.size()) PLEGMA_error("PMom has to have the same size of DeltaMom\n");
  if(WilsDir>N_DIMS) PLEGMA_error("The direction of the Wilson line has to be smaller than 3\n");
  if(nucleon!=NEUTRON && nucleon!=PROTON) PLEGMA_error("Only nucleon PDFs have been implemented so far\n");

  bool isGPD = !std::all_of(DeltaMom.begin(), DeltaMom.end(), [](int i) { return i==0; });
  if(isGPD && gammas.size()!=4) PLEGMA_warning("Not all the insertion relevant for the computation of GPDs have been set in the input file\n");

  // Reading from Lime file and loading to device
  PLEGMA_Gauge<double> gauge;
  gauge.readFile(latfile, LIME_FORMAT);
  gauge.load();
  gauge.calculatePlaq();

  // Loading to QUDA and computing plaquette also there
  initGaugeQuda(gauge, true, QUDA_WILSON_LINKS);
  plaqQuda();
 
  
  //Smearing
  PLEGMA_Gauge<double> smearedGauge;
  smearedGauge.APEsmearing(gauge, nsmearAPE, alphaAPE, 3);
  PLEGMA_printf("Plaquette after smearing:\n");
  smearedGauge.calculatePlaq();



  DeltaMom.push_back(0);
  
  std::vector<int> HalfDelta = {0,0,0,0}; 
  std::vector<int> sinkMom = {0,0,0,0};
  std::vector<int> sourceMom = {0,0,0,0};
  float SourcePhase;
  
  // Compute the source and sink momenta
  
  std::transform(DeltaMom.begin(), DeltaMom.end(), HalfDelta.begin(), [](int &c) { return (int)(c/2);});
  std::transform(PMom.begin(), PMom.end(), HalfDelta.begin(), sinkMom.begin(), std::plus<int>());
  std::transform(PMom.begin(), PMom.end(), HalfDelta.begin(), sourceMom.begin(), std::minus<int>());
  
  PLEGMA_printf("Source Momentum px %d, py %d, pz %d, pt %d\nSink Momentum px %d, py %d, pz %d, pt %d\n",
		sourceMom[0],sourceMom[1],sourceMom[2],sourceMom[3],sinkMom[0],sinkMom[1],sinkMom[2],sinkMom[3]);
 

  PLEGMA_Gauge<double> *smearedGauge_sink;
  if(isGPD){
    smearedGauge_sink = new PLEGMA_Gauge<double>(BOTH);
    smearedGauge_sink->copy(smearedGauge);}
  
  
  //Momentum smearing: put the momentum phase to smeared gauge field
  std::complex<double> momSmScale[N_DIMS];
  std::complex<double> I(0,1);
  for(int i = 0 ; i < N_DIMS; i++) momSmScale[i] = std::exp(-(xiMomSm*2.*PI*sourceMom[i]/HGC_totalL[i])*I);
  smearedGauge.scaleDirWise(momSmScale);
  
  for(int i = 0 ; i < N_DIMS; i++) momSmScale[i] = std::exp(-(xiMomSm*2.*PI*sinkMom[i]/HGC_totalL[i])*I);
  if(isGPD) smearedGauge_sink->scaleDirWise(momSmScale);

  
  PLEGMA_Gauge<float> gaugeWL;
  gaugeWL.copy(gauge);

  // Extracting the spacial sink momentum from the sink 4-momentum  
  std::vector<int> sinkMom_3D(3);
  std::copy(sourceMom.begin(),sourceMom.begin()+3,sinkMom_3D.begin());
  
  
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


  std::vector<int> DeltaMom_3D(3);
  std::copy(DeltaMom.begin(),DeltaMom.begin()+3,DeltaMom_3D.begin());
  
  
  
  PLEGMA_Gauge<double> *AuxSinkGauge;
  if(isGPD) AuxSinkGauge = smearedGauge_sink;
  else AuxSinkGauge = &smearedGauge;

  for(int isource=0;isource<numSourcePositions;isource++){
    for(int ts=0;ts<tSinks.size();ts++)
      {
	PLEGMA_Correlator<float> corrThrpWL(corr_space,sourcePositions[isource],0,tSinks[ts]+1);
	corrThrpWL.setFixMomVec(DeltsMom_3D);
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
	  vectorAuxD.gaussianSmearing(vectorOut, *AuxSinkGauge , nsmearGauss, alphaGauss);
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
	  vectorAuxD.gaussianSmearing(vectorOut, *AuxSinkGauge , nsmearGauss, alphaGauss);
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
	    for(std::string proj_str : Projs){ 
	      WHICHPROJECTOR which_proj = get_projector(proj_str);
	      for(int nu = 0 ; nu < 4 ; nu++)
		for(int c2 = 0 ; c2 < 3 ; c2++){
		  if(nucleon == PROTON)
		    vectorAuxF.seqSourceNucleon(propUP3D, propDN3D, which_proj, nucleon, global_fixSinkTime, nu, c2);
		  else
		    vectorAuxF.seqSourceNucleon(propDN3D, propUP3D, which_proj, nucleon, global_fixSinkTime, nu, c2);
		  vectorAuxF.mulMomentumPhases(sinkMom,-1); // put momentum at the sink
		  std::complex<float> Isingle(0,1);
		  float phase = 2.*PI*(((float) sourceMom[0] * sourcePositions[isource][0])/HGC_totalL[0]
				       + ((float) sourceMom[1] * sourcePositions[isource][1])/HGC_totalL[1]
				       + ((float) sourceMom[2] * sourcePositions[isource][2])/HGC_totalL[2]);
		  vectorAuxF.cscale(std::exp<float>(+phase*Isingle)); // put momentum from the point source
		  vectorAuxF.conjugate();
		  vectorAuxF.apply_gamma(G5);
		  vectorAuxD.copy(vectorAuxF);
		  vectorIn.gaussianSmearing(vectorAuxD,*AuxSinkGauge, nsmearGauss, alphaGauss);
		  // check if we need to normalize the seqsource for mix precision solver
		  if(nucleon == PROTON){
		    if(mu>0) {
		      mu*=-1.;
		      solver->UpdateSolver();
		    }
		  }
		  else{
		    if(mu<0) {
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
	      std::string aux_str = (nucleon == PROTON) ? "UP" : "DOWN";

	      //!!!!!!!!!!!!!!!!!!!!!!!!! if spatial extent is not multiple of 2 then it will not work
	      for(int stIt=0;stIt<=(int)(maxStout/stepStout);stIt++){
		std::string suff = "_stout_"+std::to_string(stIt*stepStout);
		if(stIt>0) gaugeWL.stoutSmearing(gaugeWL,stepStout,rhoStout,3);
		su3.absorbDir_device(gaugeWL, WilsDir);
		WL.setUnit( (std::vector<int>) {0,4,8});
		for(int i = 0 ; i < HGC_totalL[WilsDir]/2;i++){ 
		  corrThrpWL.contractNucleonThrp_wilsonLine(*seqPropOut, *propF, WL, signProps, gammas, i,aux_str);
		  if(signPer < 0) for(int iv = 0 ; iv < corrThrpWL.getTotalSize()*2; iv++) (corrThrpWL.getCorr())[iv] *= signPer;
		  corrThrpWL.writeHDF5( (threep_filename + suff + "_ts_" + std::to_string(tSinks[ts])  + "_Proj_" + proj_str.c_str()).c_str() ); 
		  propExchange = propIn; propIn = propF; propF = propExchange;
		  WL.wilsonLineUpdate(su3, tmp, 4+WilsDir); 
		  propF->shift(*propIn, 4+WilsDir);
		}

		suff="_stout_"+std::to_string(stIt*stepStout);
		propF->load();
		su3.absorbDir_device(gaugeWL, WilsDir); // only for z direction
		WL.setUnit( (std::vector<int>) {0,4,8});
		for(int i = 0 ; i < HGC_totalL[WilsDir]/2;i++){ // HGC_totalL[2] only for z direction
		  corrThrpWL.contractNucleonThrp_wilsonLine(*seqPropOut, *propF, WL, signProps, gammas ,-i,aux_str);
		  if(signPer < 0) for(int iv = 0 ; iv < corrThrpWL.getTotalSize()*2; iv++) (corrThrpWL.getCorr())[iv] *= signPer;
		  corrThrpWL.writeHDF5( (threep_filename + suff  +  "_ts_" + std::to_string(tSinks[ts]) + "_Proj_" + proj_str.c_str()).c_str() );
		  propExchange = propIn; propIn = propF; propF = propExchange;
		  WL.wilsonLineUpdate(su3, tmp, WilsDir); // build Wilson line in the +z direction
		  propF->shift(*propIn, WilsDir);
		}
	    
		propF->load();
	      }
	    }
	  }

	  //seq source part 1Props and contraction block
	  {
	    for(std::string proj_str : Projs){
	      WHICHPROJECTOR which_proj = get_projector(proj_str);
	      for(int nu = 0 ; nu < 4 ; nu++)
		for(int c2 = 0 ; c2 < 3 ; c2++){
		  if(nucleon == PROTON)
		    vectorAuxF.seqSourceNucleon(propUP3D, which_proj, nucleon, global_fixSinkTime, nu, c2);
		  else
		    vectorAuxF.seqSourceNucleon(propDN3D, which_proj, nucleon, global_fixSinkTime, nu, c2);
		  vectorAuxF.mulMomentumPhases(sinkMom,-1); // put momentum at the sink
		  std::complex<float> Isingle(0,1);
		  float phase = 2.*PI*(((float) sourceMom[0] * sourcePositions[isource][0])/HGC_totalL[0]
				       + ((float) sourceMom[1] * sourcePositions[isource][1])/HGC_totalL[1]
				       + ((float) sourceMom[2] * sourcePositions[isource][2])/HGC_totalL[2]);
		  vectorAuxF.cscale(std::exp<float>(+phase*Isingle)); // put momentum from the point source
		  vectorAuxF.conjugate();
		  vectorAuxF.apply_gamma(G5);
		  vectorAuxD.copy(vectorAuxF);
		  vectorIn.gaussianSmearing(vectorAuxD,*AuxSinkGauge, nsmearGauss, alphaGauss);
	    
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
	      aux_str = (nucleon == PROTON)? "DOWN" : "UP";
	      //!!!!!!!!!!!!!!!!!!!!!!!!! if spatial extent is not multiple of 2 then it will not work
	      for(int stIt=0;stIt<=(int)(maxStout/stepStout);stIt++){
		if(stIt>0) gaugeWL.stoutSmearing(gaugeWL,stepStout,rhoStout,3);
		std::string suff="_stout_"+std::to_string(stIt*stepStout);
		su3.absorbDir_device(gaugeWL, WilsDir); 
		WL.setUnit( (std::vector<int>) {0,4,8});
		for(int i = 0 ; i < HGC_totalL[WilsDir]/2;i++){ // HGC_totalL[2] only for z direction
		  corrThrpWL.contractNucleonThrp_wilsonLine(*seqPropOut, *propF, WL, signProps, gammas,i,aux_str);
		  if(signPer < 0) for(int iv = 0 ; iv < corrThrpWL.getTotalSize()*2; iv++) (corrThrpWL.getCorr())[iv] *= signPer;
		  corrThrpWL.writeHDF5( (threep_filename + suff + "_ts_" + std::to_string(tSinks[ts]) + "_Proj_" + proj_str.c_str() ).c_str() );
		  propExchange = propIn; propIn = propF; propF = propExchange;
		  WL.wilsonLineUpdate(su3, tmp, 4+WilsDir); // build Wilson line in the +z direction
		  propF->shift(*propIn, 4+WilsDir);
		}
    
		propF->load();
		suff="_stout_"+std::to_string(stIt*stepStout);
		su3.absorbDir_device(gaugeWL, WilsDir); 
		WL.setUnit( (std::vector<int>) {0,4,8});
		for(int i = 0 ; i < HGC_totalL[WilsDir]/2;i++){ // HGC_totalL[2] only for z direction
		  corrThrpWL.contractNucleonThrp_wilsonLine(*seqPropOut, *propF, WL, signProps, gammas,-i, aux_str);
		  if(signPer < 0) for(int iv = 0 ; iv < corrThrpWL.getTotalSize()*2; iv++) (corrThrpWL.getCorr())[iv] *= signPer;
		  corrThrpWL.writeHDF5( (threep_filename + suff + "_ts_" + std::to_string(tSinks[ts]) + "_Proj_" + proj_str.c_str()).c_str() );
		  propExchange = propIn; propIn = propF; propF = propExchange;
		  WL.wilsonLineUpdate(su3, tmp, WilsDir); // build Wilson line in the +z direction
		  propF->shift(*propIn, WilsDir);
		}
		propF->load();
	      }
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

    PLEGMA_Correlator<float> corr(corr_space, sinkMom_3D);
    corrThrpWL.setFixMomVec(sinkMom_3D);
    corr.contractMesons(*propUP, *propDN);
    corr.writeFile(twop_filename.c_str(), corr_file_format);

    corr.contractBaryons(*propUP, *propDN);
    corr.writeFile(twop_filename.c_str(), corr_file_format);
  }
  delete propUP;
  delete propDN;
  delete propIn;
  delete seqPropOut;
  if(isGPD) delete smearedGauge_sink;

  delete solver;
  
  finalize();
  return 0;
}

