#include <PLEGMA.h>
#include <PLEGMA_utils.h>

using namespace plegma;
using namespace quda;

template<typename T>
std::vector<T> extract_v(std::vector<T> const &v, int m, int n) {
  auto first = v.begin() + m;
  auto last = v.begin() + n + 1;
  std::vector<T> vector(first, last);
  return vector;
}

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

  std::string aux_str ;
  HGC_options->set("which-projector", "Which projector to use for 3pt function", verbosity, aux_str);
  WHICHPROJECTOR which_proj=get_projector(aux_str.c_str());


  /*
  We consider the momenta in the symmetric frame. Both P-momentum and Delta-momentum are vectors
  having a number of components that is a multiple of three. They correspond to lists of 
  three-dimensional momenta. 
  The source momentum is given by delta/2-p, while the sink momentum by delta/2+p.
  All the three-dimensional momenta in the vectors have to share the same sink momentum. Indeed,
  keeping constant the sink momentum (i.e. doing one inversion of the sequential source)
  different combinations of the momentum transfer delta and source momentum can be tested.
  */

  std::vector<int> PMom_v = {0,0,0};
  HGC_options->set("P-momentum", "If added to the momentum transfer delta gives the sink momentum", verbosity, PMom_v);

  std::vector<int> DeltaMom_v = {0,0,0};
  HGC_options->set("Delta-momentum", "Square root of the momentum transfer", verbosity, DeltaMom_v);

  aux_str = "proton";
  HGC_options->set("which-particle", "Choice of the nucleon interpolator to insert in the three point function (neutron,proton)", verbosity, aux_str);
  WHICHPARTICLE nucleon = get_particle(aux_str.c_str());

  //=========================================================================================================//

  initializePLEGMA();

  if(DeltaMom_v.size()%3 != 0 || DeltaMom_v.size()%3 != 0) PLEGMA_error("P-momentum and Delta-momentum size has to be a multiple of three\n");
  if(DeltaMom_v.size()!=PMom_v.size()) PLEGMA_error("PMom has to have the same size of DeltaMom\n");
  if(WilsDir>N_DIMS) PLEGMA_error("The direction of the Wilson line has to be smaller than 3\n");
  if(nucleon!=NEUTRON && nucleon!=PROTON) PLEGMA_error("Only nucleon PDFs have been implemented so far\n");

  // Reading from Lime file and loading to device
  PLEGMA_Gauge<double> gauge;
  gauge.readFile(latfile, LIME_FORMAT);
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

  std::vector<std::vector<int>> DeltaMom = {};
  std::vector<std::vector<int>> PMom = {};
  for(size_t i = 0; i< (int)(DeltaMom_v.size()/3); i++){
    DeltaMom.push_back(extract_v<int>(DeltaMom_v,i*3,(i+1)*3));
    PMom.push_back(extract_v<int>(PMom_v,i*3,(i+1)*3));
  }

  
  std::vector<double> HalfDelta = {0,0,0,0}; 
  std::vector<std::vector<double>> auxMom ;
  std::vector<double> sinkMom = {0,0,0,0};
  std::vector<double> sourceMom = {0,0,0,0};
  std::vector<float> SourcePhases;
  
  /* 
     Compute sink momentum. Checking that each combination
     of PMom and DeltaMom gives the same sinkMom.
  */
  
  for(size_t i=0; i < PMom.size() ; i++){
    std::transform(DeltaMom[i].begin(), DeltaMom[i].end(), HalfDelta.begin(), [](int &c) { return (double)c/(double)(2.);});
    std::transform(PMom[i].begin(), PMom[i].end(), HalfDelta.begin(), sinkMom.begin(), std::plus<double>());
    auxMom.push_back(sinkMom);
  }

  bool constSink =  !std::all_of(auxMom.begin(), auxMom.end(), [auxMom](std::vector<double> x){ return x==auxMom[0]; });
  if(constSink) PLEGMA_error("The different combinations of momenta give incompatible sink momenta\n");
  
  //Momentum smearing: put the momentum phase to smeared gauge field
  std::complex<double> momSmScale[N_DIMS];
  std::complex<double> I(0,1);
  for(int i = 0 ; i < N_DIMS-1; i++) momSmScale[i] = std::exp(-(xiMomSm*2.*PI*sinkMom[i]/HGC_totalL[i])*I);
  momSmScale[i] = 1.;

  smearedGauge.scaleDirWise(momSmScale);

  PLEGMA_Gauge<float> gaugeWL;
  gaugeWL.copy(gauge);

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

  PLEGMA_Correlator<float> corrThrpWL(corr_space,0); 
  
  
  int  isource=0;

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

	  for(size_t np = 0 ; np < DeltaMom.size(); np++)
	    {
	      
	      std::complex<float> Isingle(0,1);

	      std::transform(DeltaMom[np].begin(), DeltaMom[np].end(), HalfDelta.begin(), [](int &c) { return (double)c/(double)(2.);});
	      std::transform(PMom[np].begin(), PMom[np].end(), HalfDelta.begin(), sourceMom.begin(), std::minus<double>());
	      
	      float phase = 2.*PI*(((float) sourceMom[0] * sourcePositions[isource][0])/HGC_totalL[0]
				   + ((float)sourceMom[1] * sourcePositions[isource][1])/HGC_totalL[1]
				   + ((float)sourceMom[2] * sourcePositions[isource][2])/HGC_totalL[2])
				   + (float) (PI/2.);
	      
	      SourcePhases.push_back(phase);
	      
	      if(np==0){
		seqPropOut->apply_gamma(G5);
		seqPropOut->conjugate();
	      }

	      seqPropOut->cscale(std::exp<float>(-phase*Isingle));

	      int signProps = (nucleon == PROTON) ? +1: -1;

	      PLEGMA_Propagator<float> *propF = (nucleon == PROTON) ? propUP : propDN;

	      propF->mulMomentumPhases(DeltaMom[np],1);
	      
	      //!!!!!!!!!!!!!!!!!!!!!!!!! if spatial extent is not multiple of 2 then it will not work
	      for(int stIt=0;stIt<=(int)(maxStout/stepStout);stIt++){
		std::string suff = "_CP2_stout_"+std::to_string(stIt*stepStout)+"_Plus_";
		if(stIt>0) gaugeWL.stoutSmearing(gaugeWL,stepStout,rhoStout,3);
		su3.absorbDir_device(gaugeWL, WilsDir);
		WL.setUnit( (std::vector<int>) {0,4,8});
		for(int i = 0 ; i < HGC_totalL[WilsDir]/2;i++){ 
		  corrThrpWL.contractNucleonThrp_wilsonLine(*seqPropOut, *propF, WL, signProps, gammas, sourcePositions[isource]);
		  if(signPer < 0) for(int iv = 0 ; iv < corrThrpWL.getTotalSize()*2; iv++) (corrThrpWL.getCorr())[iv] *= signPer;
		  corrThrpWL.writeASCII( (threep_filename +  suff + std::to_string(i) + "_ts_" + std::to_string(tSinks[ts]) + "_tSource_" + std::to_string(np)  + ".dat").c_str() ); 
		  propExchange = propIn; propIn = propF; propF = propExchange;
		  WL.wilsonLineUpdate(su3, tmp, 4+WilsDir); 
		  propF->shift(*propIn, 4+WilsDir);
		}

		suff="_CP2_stout_"+std::to_string(stIt*stepStout)+"_Minus_";
		propF->load();
		su3.absorbDir_device(gaugeWL, WilsDir); // only for z direction
		WL.setUnit( (std::vector<int>) {0,4,8});
		for(int i = 0 ; i < HGC_totalL[WilsDir]/2;i++){ // HGC_totalL[2] only for z direction
		  corrThrpWL.contractNucleonThrp_wilsonLine(*seqPropOut, *propF, WL, signProps, gammas, sourcePositions[isource]);
		  if(signPer < 0) for(int iv = 0 ; iv < corrThrpWL.getTotalSize()*2; iv++) (corrThrpWL.getCorr())[iv] *= signPer;
		  corrThrpWL.writeASCII( (threep_filename + suff + std::to_string(i) +  "_ts_" + std::to_string(tSinks[ts]) + "_tSource_" + std::to_string(np) + ".dat").c_str() );
		  propExchange = propIn; propIn = propF; propF = propExchange;
		  WL.wilsonLineUpdate(su3, tmp, WilsDir); // build Wilson line in the +z direction
		  propF->shift(*propIn, WilsDir);
		}
		propF->load();
	      }
	      seqPropOut->cscale(std::exp<float>(phase*Isingle)); 
	      propF->mulMomentumPhases(DeltaMom[np],-1);
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

	  for(size_t np = 0 ; np < DeltaMom.size(); np++)
	    {
	      std::complex<float> Isingle(0,1);
	      	      
	      
	      if(np==0){
		seqPropOut->apply_gamma(G5);
		seqPropOut->conjugate();
	      }
	      seqPropOut->cscale(std::exp<float>(-SourcePhases[np]*Isingle));

	      int signProps = (nucleon == PROTON) ? -1: +1;

	      PLEGMA_Propagator<float> *propF = (nucleon == PROTON) ? propDN : propUP;
	      propF->mulMomentumPhases(DeltaMom[np],1);

	      gaugeWL.copy(gauge);

	      //!!!!!!!!!!!!!!!!!!!!!!!!! if spatial extent is not multiple of 2 then it will not work
	      for(int stIt=0;stIt<=(int)(maxStout/stepStout);stIt++){
		if(stIt>0) gaugeWL.stoutSmearing(gaugeWL,stepStout,rhoStout,3);
		std::string suff="_CP1_stout_"+std::to_string(stIt*stepStout)+"_Plus_";
		su3.absorbDir_device(gaugeWL, WilsDir); 
		WL.setUnit( (std::vector<int>) {0,4,8});
		for(int i = 0 ; i < HGC_totalL[WilsDir]/2;i++){ // HGC_totalL[2] only for z direction
		  corrThrpWL.contractNucleonThrp_wilsonLine(*seqPropOut, *propF, WL, signProps, gammas, sourcePositions[isource]);
		  if(signPer < 0) for(int iv = 0 ; iv < corrThrpWL.getTotalSize()*2; iv++) (corrThrpWL.getCorr())[iv] *= signPer;
		  corrThrpWL.writeASCII( (threep_filename + suff + std::to_string(i) + "_ts_" + std::to_string(tSinks[ts])  + "_tSource_" + std::to_string(np) + ".dat").c_str() );
		  propExchange = propIn; propIn = propF; propF = propExchange;
		  WL.wilsonLineUpdate(su3, tmp, 4+WilsDir); // build Wilson line in the +z direction
		  propF->shift(*propIn, 4+WilsDir);
		}
    
		propF->load();
		suff="_CP1_stout_"+std::to_string(stIt*stepStout)+"_Minus_";
		su3.absorbDir_device(gaugeWL, WilsDir); 
		WL.setUnit( (std::vector<int>) {0,4,8});
		for(int i = 0 ; i < HGC_totalL[WilsDir]/2;i++){ // HGC_totalL[2] only for z direction
		  corrThrpWL.contractNucleonThrp_wilsonLine(*seqPropOut, *propF, WL, signProps, gammas, sourcePositions[isource]);
		  if(signPer < 0) for(int iv = 0 ; iv < corrThrpWL.getTotalSize()*2; iv++) (corrThrpWL.getCorr())[iv] *= signPer;
		  corrThrpWL.writeASCII( (threep_filename + suff + std::to_string(i) + "_ts_" + std::to_string(tSinks[ts])  + "_tSource_" + std::to_string(np) + ".dat").c_str() );
		  propExchange = propIn; propIn = propF; propF = propExchange;
		  WL.wilsonLineUpdate(su3, tmp, WilsDir); // build Wilson line in the +z direction
		  propF->shift(*propIn, WilsDir);
		}
		propF->load();
	      }
	      seqPropOut->cscale(std::exp<float>(+SourcePhases[np]*Isingle));
	      propF->mulMomentumPhases(DeltaMom[np],-1);
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

  PLEGMA_Correlator<float> corr(corr_space, sinkMom);
  corr.contractMesons(*propUP, *propDN, sourcePositions[isource]);
  corr.writeFile(twop_filename.c_str(), corr_file_format);

  //!!!!!!!!!! maybe later we choose the specific momentum when this allows it
  corr.contractBaryons(*propUP, *propDN, sourcePositions[isource]);
  corr.writeFile(twop_filename.c_str(), corr_file_format);
    
  delete propUP;
  delete propDN;
  delete propIn;
  delete seqPropOut;
  delete solver;
  
  finalize();
  return 0;
}

