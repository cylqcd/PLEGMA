#include <PLEGMA.h>
#include <PLEGMA_utils.h>

std::vector<double> runtime;
#define TIME(fnc)  runtime.push_back(MPI_Wtime()); fnc;			\
  PLEGMA_printf("TIME for "#fnc" %f sec\n", MPI_Wtime()-runtime.back()); \
  runtime.pop_back()

std::vector<std::thread> threads;
//#define THREAD(fnc) threads.push_back(std::thread([=]() { TIME(fnc); }))
#define THREAD(fnc) saveTuneCache(false); TIME(fnc)

using namespace plegma;
using namespace quda;

int main(int argc, char **argv)
{

  static std::vector<std::string> listOpt = {"verbosity", "load-gauge", "nsmear-APE", "alpha-APE", "nsmear-gauss", "alpha-gauss",
					     "nsrc", "src-filename", "maxQsq", "twop-filename","threep-filename",  "corr-file-format",
					     "corr-space", "tSinks","xiMomSm","gammas"};

  initializeOptions(argc, argv, true, listOpt);

  //================ Add your options in this between initializeOptions and initializePLEGMA ================//

  size_t WilsDir = DIM_Z;
  HGC_options->set("wilson-direction", "Direction of the wilson line", verbosity, WilsDir);
  

  double rhoStout = 0.;
  HGC_options->set("rho-stout", "Rho parameter stout smearing", verbosity, rhoStout);

  size_t maxStout = 1;
  HGC_options->set("max-stout", "Maximum number of stout smearing steps", verbosity, maxStout);

  size_t stepStout = 5;
  HGC_options->set("stout-steps", "Save the PDFs every step_stout stout smearing step", verbosity, stepStout);

  size_t maxQsq3pt = 5;
  HGC_options->set("maxQsq3pt", "The max Qsq to use in the 3pt contractions", verbosity, maxQsq3pt);
  
  bool calc3pt = true ;
  HGC_options->set("calc3pt", "If true then the 3pt function is computed", verbosity, calc3pt);

  std::string proj ;
  HGC_options->set("which-projector", "Which projector to use for 3pt function", verbosity, proj);
  WHICHPROJECTOR which_proj=get_projector(proj.c_str());


  /*
    We consider the momenta in the symmetric frame. Both P-momentum and Delta-momentum are vectors
    having three components. 
    IF SYMM: The source momentum is given by p-delta/2, while the sink momentum by p+delta/2.
    ELSE: The source momentum is given by p-delta, while the sink momentum by p.
  */

  bool symm = true ;
  HGC_options->set("Delta-symm", "If true delta is applied symmetrically otherwise asymmetrically", verbosity, calc3pt);

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
  
  
  {
  // Reading from Lime file and loading to device
  PLEGMA_Gauge<double> gauge;
  gauge.readFile(latfile, LIME_FORMAT);
  gauge.calculatePlaq();

  // Loading to QUDA and computing plaquette also there
  initGaugeQuda(gauge, true, QUDA_WILSON_LINKS);
  plaqQuda();
 
  
  //Smearing
  PLEGMA_Gauge<double> smearedGauge;
  smearedGauge.APEsmearing(gauge, nsmearAPE, alphaAPE, 3);
  PLEGMA_printf("Plaquette after smearing:\n");
  smearedGauge.calculatePlaq();

  std::vector<int> sinkMom = {0,0,0};
  std::vector<int> sourceMom = {0,0,0};
  float SourcePhase;
  
  // Compute the source and sink momenta

  if(symm) {
    std::vector<int> HalfDelta = {0,0,0}; 
    std::transform(DeltaMom.begin(), DeltaMom.end(), HalfDelta.begin(), [](int &c) { return (int)(c/2);});
    std::transform(PMom.begin(), PMom.end(), HalfDelta.begin(), sinkMom.begin(), std::plus<int>());
    std::transform(PMom.begin(), PMom.end(), HalfDelta.begin(), sourceMom.begin(), std::minus<int>());
  } else {
    std::transform(PMom.begin(), PMom.end(), sinkMom.begin(), [](int &c) { return c;});
    std::transform(PMom.begin(), PMom.end(), DeltaMom.begin(), sourceMom.begin(), std::minus<int>());
  }
  
  PLEGMA_printf("Source Momentum px %d, py %d, pz %d, pt %d\nSink Momentum px %d, py %d, pz %d, pt %d\n",
		sourceMom[0],sourceMom[1],sourceMom[2],sourceMom[3],sinkMom[0],sinkMom[1],sinkMom[2],sinkMom[3]);
  if(isGPD && maxQsq < sourceMom[0]*sourceMom[0]+sourceMom[1]*sourceMom[1]+sourceMom[2]*sourceMom[2])
   PLEGMA_error("maxQsq does not include the source momentum\n");

  PLEGMA_Gauge<double> *smearedGauge_sink;
  if(isGPD){
    smearedGauge_sink = new PLEGMA_Gauge<double>(BOTH);
    smearedGauge_sink->copy(smearedGauge);}
  
  
  //Momentum smearing: put the momentum phase to smeared gauge field
  std::complex<double> momSmScale[N_DIMS];
  std::complex<double> I(0,1);
  for(int i = 0 ; i < N_DIMS; i++) momSmScale[i] = std::exp(-(xiMomSm*2.*PI*sourceMom[i]/HGC_totalL[i])*I);
  TIME(smearedGauge.scaleDirWise(momSmScale));
  
  for(int i = 0 ; i < N_DIMS; i++) momSmScale[i] = std::exp(-(xiMomSm*2.*PI*sinkMom[i]/HGC_totalL[i])*I);
  if(isGPD) smearedGauge_sink->scaleDirWise(momSmScale);

  
  // ensuring mu positive
  if(mu<0)  mu*=-1.;
  QUDA_solver *solver = new QUDA_solver(mu);

  PLEGMA_Propagator<float> *propUP = new PLEGMA_Propagator<float>(BOTH);
  PLEGMA_Propagator<float> *propDN = new PLEGMA_Propagator<float>(BOTH);

  PLEGMA_Propagator<float> *propUP_SL=new PLEGMA_Propagator<float>(tSinks.size()>0 ? BOTH:NONE, FIRST_CORNER);
  PLEGMA_Propagator<float> *propDN_SL=new PLEGMA_Propagator<float>(tSinks.size()>0 ? BOTH:NONE, FIRST_CORNER);

  PLEGMA_Propagator<float> *propIn = new PLEGMA_Propagator<float>(BOTH);  

  PLEGMA_Propagator<float> *seqPropOut = new PLEGMA_Propagator<float>(BOTH);

  PLEGMA_Propagator<float> *propExchange = nullptr;
    
  PLEGMA_Gauge<float> gaugeWL;
  PLEGMA_Gauge<double> *AuxSinkGauge;
  if(isGPD) AuxSinkGauge = smearedGauge_sink;
  else AuxSinkGauge = &smearedGauge;

  for(int isource=0;isource<numSourcePositions;isource++) {
    site& source = sourcePositions[isource];

    PLEGMA_Gauge3D<double> smearedGauge3D;
    smearedGauge3D.absorb(smearedGauge, source[DIM_T]);

    auto computePropagator = [&](PLEGMA_Propagator<float>* prop_SS, PLEGMA_Propagator<float>* prop_SL,
				 double run_mu) {
			       // ensuring mu value
			       if(mu != run_mu) {
				 mu = run_mu;
				 solver->UpdateSolver();
			       }
			       for(int isc = 0 ; isc < 12 ; isc++){
				 PLEGMA_Vector<double> vectorInOut;
				 { // Smearing the source
				   PLEGMA_Vector3D<double> vector1, vector2;
				   vector1.pointSource(source, isc/3, isc%3, DEVICE);
				   TIME(vector2.gaussianSmearing(vector1, smearedGauge3D, nsmearGauss, alphaGauss));
				   vectorInOut.absorb(vector2,source[DIM_T]);
				 }
				 // Inverting
				 PLEGMA_printf("Going to invert %s for component %d\n",
					       run_mu>0 ? "UP" : "DN", isc);
				 TIME(solver->solve(vectorInOut, vectorInOut));
				 if(prop_SL->getAllocation() != NONE) {
				   PLEGMA_Vector<float> vectorAuxF;
				   vectorAuxF.copy(vectorInOut);
				   prop_SL->absorb(vectorAuxF, isc/3, isc%3);
				 }
				 { // Smearing the solution
				   PLEGMA_Vector<double> vectorAuxD;
				   PLEGMA_Vector<float> vectorAuxF;
				   TIME(vectorAuxD.gaussianSmearing(vectorInOut, smearedGauge, nsmearGauss, alphaGauss));
				   vectorAuxF.copy(vectorAuxD);
				   prop_SS->absorb(vectorAuxF, isc/3, isc%3);
				 }
			       }
			     };
    
    TIME(computePropagator(propUP, propUP_SL, mu>0 ? mu : -mu));
    TIME(computePropagator(propDN, propDN_SL, mu<0 ? mu : -mu));
    
    for(int ts=0;ts<tSinks.size();ts++) {
      PLEGMA_Correlator<float> corrThrpWL(corr_space,source,0,tSinks[ts]+1,maxQsq3pt);
      int signPer = (tSinks[ts] + source[3]) >= HGC_totalL[3] ? -1 : +1;
      int global_fixSinkTime = (tSinks[ts] + source[3])%HGC_totalL[3]; 

      // 3D propagators at t_sink
      PLEGMA_Propagator3D<float> propUP3D;
      PLEGMA_Propagator3D<float> propDN3D;
      PLEGMA_Gauge3D<double> smearedGauge3D_sink;
      smearedGauge3D_sink.absorb(*AuxSinkGauge, global_fixSinkTime);


      for(int isc = 0 ; isc < 12 ; isc++){
	PLEGMA_Vector3D<double> vectorAuxD;
	PLEGMA_Vector3D<double> vectorAuxD2;
	PLEGMA_Vector3D<float> vectorAuxF;
	vectorAuxF.absorb(*propUP_SL,global_fixSinkTime,isc/3, isc%3);
	vectorAuxD2.copy(vectorAuxF);

	TIME(vectorAuxD.gaussianSmearing(vectorAuxD2, smearedGauge3D_sink, nsmearGauss, alphaGauss));
	vectorAuxF.copy(vectorAuxD);
	propUP3D.absorb(vectorAuxF,isc/3, isc%3);
      }
      for(int isc = 0 ; isc < 12 ; isc++){
	PLEGMA_Vector3D<double> vectorAuxD;
	PLEGMA_Vector3D<double> vectorAuxD2;
	PLEGMA_Vector3D<float> vectorAuxF;
	vectorAuxF.absorb(*propDN_SL,global_fixSinkTime,isc/3, isc%3);
	vectorAuxD2.copy(vectorAuxF);
	TIME(vectorAuxD.gaussianSmearing(vectorAuxD2, smearedGauge3D_sink, nsmearGauss, alphaGauss));
	vectorAuxF.copy(vectorAuxD);
	propDN3D.absorb(vectorAuxF,isc/3, isc%3);
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
	      PLEGMA_Vector3D<float> vectorAux3D;
              PLEGMA_Vector<float> vectorAuxF;
              PLEGMA_Vector<double> vectorAuxD;
              PLEGMA_Vector<double> vectorOut,vectorIn;

	      if(nucleon == PROTON)
		vectorAux3D.seqSourceNucleon(propUP3D, propDN3D, which_proj, nucleon, nu, c2);
	      else
		vectorAux3D.seqSourceNucleon(propDN3D, propUP3D, which_proj, nucleon, nu, c2);
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
	      PLEGMA_Vector3D<float> vectorAux3D;
              PLEGMA_Vector<double> vectorOut,vectorIn;
              PLEGMA_Vector<double> vectorAuxD;
	      PLEGMA_Vector<float> vectorAuxF;
	      if(nucleon == PROTON)
		vectorAux3D.seqSourceNucleon(propUP3D, which_proj, nucleon, nu, c2);
	      else
		vectorAux3D.seqSourceNucleon(propDN3D, which_proj, nucleon, nu, c2);
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
    
    propUP->rotateToPhysicalBase_device(+1);
    propDN->rotateToPhysicalBase_device(-1);
    propUP->applyBoundaries_device(source[3]);
    propDN->applyBoundaries_device(source[3]);
    
    PLEGMA_Correlator<float> corr(corr_space, source, maxQsq);
    TIME(corr.contractMesons(*propUP, *propDN));
    THREAD(corr.writeFile(twop_filename, corr_file_format));
    
    TIME(corr.contractBaryons(*propUP, *propDN));
    THREAD(corr.writeFile(twop_filename, corr_file_format));
  }
  if(isGPD) delete smearedGauge_sink;
  
  }
  finalize();
  return 0;
}

