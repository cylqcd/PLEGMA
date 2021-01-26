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


    
  std::vector<int> HalfDelta = {0,0,0}; 
  std::vector<int> sinkMom = {0,0,0};
  std::vector<int> sourceMom = {0,0,0};
  float SourcePhase;
  
  // Compute the source and sink momenta
  
  std::transform(DeltaMom.begin(), DeltaMom.end(), HalfDelta.begin(), [](int &c) { return (int)(c/2);});
  std::transform(PMom.begin(), PMom.end(), HalfDelta.begin(), sinkMom.begin(), std::plus<int>());
  std::transform(PMom.begin(), PMom.end(), HalfDelta.begin(), sourceMom.begin(), std::minus<int>());
  
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
  TIME(QUDA_solver solver(mu));

  PLEGMA_Propagator<float> propUP(BOTH);
  PLEGMA_Propagator<float> propDN(BOTH);
  PLEGMA_Propagator<float> propUP_SL(tSinks.size()>0 ? BOTH:NONE, FIRST_CORNER);
  PLEGMA_Propagator<float> propDN_SL(tSinks.size()>0 ? BOTH:NONE, FIRST_CORNER);
  
    
  PLEGMA_Gauge<double> *AuxSinkGauge;
  if(isGPD) AuxSinkGauge = smearedGauge_sink;
  else AuxSinkGauge = &smearedGauge;

  for(int isource=0;isource<numSourcePositions;isource++) {
    site& source = sourcePositions[isource];

    PLEGMA_Gauge3D<double> smearedGauge3D;
    smearedGauge3D.absorb(smearedGauge, source[DIM_T]);

    auto computePropagator = [&](PLEGMA_Propagator<float>& prop_SS, PLEGMA_Propagator<float>& prop_SL,
				 double run_mu) {
			       // ensuring mu value
			       if(mu != run_mu) {
				 mu = run_mu;
				 solver.UpdateSolver();
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
				 TIME(solver.solve(vectorInOut, vectorInOut));
				 if(prop_SL.getAllocation() != NONE) {
				   PLEGMA_Vector<float> vectorAuxF;
				   vectorAuxF.copy(vectorInOut);
				   prop_SL.absorb(vectorAuxF, isc/3, isc%3);
				 }
				 { // Smearing the solution
				   PLEGMA_Vector<double> vectorAuxD;
				   PLEGMA_Vector<float> vectorAuxF;
				   TIME(vectorAuxD.gaussianSmearing(vectorInOut, smearedGauge, nsmearGauss, alphaGauss));
				   vectorAuxF.copy(vectorAuxD);
				   prop_SS.absorb(vectorAuxF, isc/3, isc%3);
				 }
			       }
			     };
    
    TIME(computePropagator(propUP, propUP_SL, mu>0 ? mu : -mu));
    TIME(computePropagator(propDN, propDN_SL, mu<0 ? mu : -mu));
    
    for(int ts=0;ts<tSinks.size();ts++) {
      PLEGMA_Correlator<float> corrThrpWL(corr_space,source,0,tSinks[ts]+1);
      corrThrpWL.setFixMomVec(DeltaMom);
      int signPer = (tSinks[ts] + source[3]) >= HGC_totalL[3] ? -1 : +1;
      int global_fixSinkTime = (tSinks[ts] + source[3])%HGC_totalL[3]; 

      // 3D propagators at t_sink
      PLEGMA_Propagator3D<float> propUP3D;
      PLEGMA_Propagator3D<float> propDN3D;
      PLEGMA_Gauge3D<double> smearedGauge3D_sink;
      smearedGauge3D_sink.absorb(*AuxSinkGauge, global_fixSinkTime);

      propUP3D.absorb(propUP_SL, global_fixSinkTime);
      propDN3D.absorb(propDN_SL, global_fixSinkTime);

      for(int isc = 0 ; isc < 12 ; isc++){
	PLEGMA_Vector3D<double> vectorAuxD;
	PLEGMA_Vector3D<double> vectorAuxD2;
	PLEGMA_Vector3D<float> vectorAuxF;
	vectorAuxF.absorb(propUP3D,isc/3, isc%3);
	vectorAuxD2.copy(vectorAuxF);
	TIME(vectorAuxD.gaussianSmearing(vectorAuxD2, smearedGauge3D_sink, nsmearGauss, alphaGauss));
	vectorAuxF.copy(vectorAuxD);
	propUP3D.absorb(vectorAuxF,isc/3, isc%3);
      }
      for(int isc = 0 ; isc < 12 ; isc++){
	PLEGMA_Vector3D<double> vectorAuxD;
	PLEGMA_Vector3D<double> vectorAuxD2;
	PLEGMA_Vector3D<float> vectorAuxF;
	vectorAuxF.absorb(propDN3D,isc/3, isc%3);
	vectorAuxD2.copy(vectorAuxF);
	TIME(vectorAuxD.gaussianSmearing(vectorAuxD2, smearedGauge3D_sink, nsmearGauss, alphaGauss));
	vectorAuxF.copy(vectorAuxD);
	propDN3D.absorb(vectorAuxF,isc/3, isc%3);
      }
	
      
      
      
      PLEGMA_Su3field<float> su3;
      PLEGMA_Su3field<float> WL;
      PLEGMA_Su3field<float> tmp;
      PLEGMA_Propagator<float> seqPropOut(BOTH, FIRST_CORNER);

      for(std::string proj_str : Projs){
	WHICHPROJECTOR which_proj = get_projector(proj_str);

	auto computeThreep = [&](double run_mu, PLEGMA_Propagator3D<float>& prop1, PLEGMA_Propagator3D<float>& prop2, int signProps, PLEGMA_Propagator<float> *propF, std::string fl_str) {
			       // ensuring mu value
			       if(mu != run_mu) {
				 mu = run_mu;
				 solver.UpdateSolver();
			       }
			       
			       for(int nu = 0 ; nu < 4 ; nu++)
				 for(int c2 = 0 ; c2 < 3 ; c2++){
				   PLEGMA_Vector<double> vectorInOut;
				   {
				     PLEGMA_Vector3D<double> vectorAuxD1,vectorAuxD2;
				     PLEGMA_Vector3D<float> vectorAuxF;
				     if( &prop1 != &prop2 ) {
				       vectorAuxF.seqSourceNucleon(prop1, prop2, which_proj, nucleon, nu, c2);
				     } else {
				       vectorAuxF.seqSourceNucleon(prop1, which_proj, nucleon, nu, c2);
				     }
				     vectorAuxF.mulMomentumPhases(sinkMom,-1); // put momentum at the sink
				     std::complex<float> Isingle(0,1);
				     float phase = 2.*PI*(((float) sourceMom[0] * source[0])/HGC_totalL[0]
							  + ((float) sourceMom[1] * source[1])/HGC_totalL[1]
							  + ((float) sourceMom[2] * source[2])/HGC_totalL[2]);
				     vectorAuxF.cscale(std::exp<float>(+phase*Isingle)); // put momentum from the point source
				     vectorAuxF.conjugate();
				     vectorAuxF.apply_gamma(G5);
				     vectorAuxD1.copy(vectorAuxF);
				     TIME(vectorAuxD2.gaussianSmearing(vectorAuxD1, smearedGauge3D_sink, nsmearGauss, alphaGauss));
				     vectorInOut.absorb(vectorAuxD2, global_fixSinkTime);
				   }
				   double norm = vectorInOut.norm();
				   vectorInOut.scale(1/norm);
				   TIME(solver.solve(vectorInOut, vectorInOut));
				   vectorInOut.scale(norm);
				   PLEGMA_Vector<float> vectorAuxF;
				   vectorAuxF.copy(vectorInOut);
				   seqPropOut.absorb(vectorAuxF, nu, c2);
				 }
			       seqPropOut.apply_gamma(G5);
			       seqPropOut.conjugate();

			       std::string filename = threep_filename + "_" + proj_str + "_dt" + std::to_string(tSinks[ts]) + "_" + fl_str;

			       PLEGMA_Gauge<float> gaugeWL;
			       gaugeWL.copy(gauge);

			       {
				 PLEGMA_Correlator<float> corrQsq(corr_space,source,maxQsq,tSinks[ts]+1);
			       
				 // LOCAL contractions
				 TIME(corrQsq.contractNucleonThrp_local(seqPropOut, *propF, signProps, gammas));
				 if(signPer < 0) for(size_t iv = 0 ; iv < corrQsq.getTotalSize()*2; iv++) corrQsq.H_elem()[iv] *= signPer;      
				 THREAD(corrQsq.writeFile(filename, corr_file_format));
			       
				 // oneD contractions
				 TIME(corrQsq.contractNucleonThrp_oneD(seqPropOut, *propF, gaugeWL, signProps, gammas));
				 if(signPer < 0) for(size_t iv = 0 ; iv < corrQsq.getTotalSize()*2; iv++) corrQsq.H_elem()[iv] *= signPer;
				 THREAD(corrQsq.writeFile( filename, corr_file_format));
			       
				 // noe contractions
				 TIME(corrQsq.contractNucleonThrp_noe(seqPropOut, *propF, gaugeWL, signProps));
				 if(signPer < 0) for(size_t iv = 0 ; iv < corrQsq.getTotalSize()*2; iv++) corrQsq.H_elem()[iv] *= signPer;
				 THREAD(corrQsq.writeFile( filename, corr_file_format));
	      
				 // twoD contractions
				 TIME(corrQsq.contractNucleonThrp_twoD(seqPropOut, *propF, gaugeWL, signProps, gammas));
				 if(signPer < 0) for(size_t iv = 0 ; iv < corrQsq.getTotalSize()*2; iv++) corrQsq.H_elem()[iv] *= signPer;
				 THREAD(corrQsq.writeFile( filename, corr_file_format));
			       }
			       
			       PLEGMA_Propagator<float> *propIn = new PLEGMA_Propagator<float>(BOTH);
			       PLEGMA_Propagator<float> *propExchange = nullptr;
			       propF->unload();
			       
			       //!!!!!!!!!!!!!!!!!!!!!!!!! if spatial extent is not multiple of 2 then it will not work
			       for(int stIt=0;stIt<=(int)(maxStout/stepStout);stIt++){
				 std::string suff = "_stout_"+std::to_string(stIt*stepStout);
				 if(stIt>0) gaugeWL.stoutSmearing(gaugeWL,stepStout,rhoStout,3);
				 su3.absorbDir_device(gaugeWL, WilsDir);
				 WL.setUnit( (std::vector<int>) {0,4,8});
				 for(int i = 0 ; i < HGC_totalL[WilsDir]/2;i++){ 
				   TIME(corrThrpWL.contractNucleonThrp_wilsonLine(seqPropOut, *propF, WL, signProps, gammas, i,fl_str));
				   if(signPer < 0) for(int iv = 0 ; iv < corrThrpWL.getTotalSize()*2; iv++) (corrThrpWL.H_elem())[iv] *= signPer;
				   THREAD(corrThrpWL.writeHDF5(threep_filename + suff + "_ts_" + std::to_string(tSinks[ts])  + "_Proj_" + proj_str)); 
				   propExchange = propIn; propIn = propF; propF = propExchange;
				   TIME(WL.wilsonLineUpdate(su3, tmp, 4+WilsDir)); 
				   TIME(propF->shift(*propIn, 4+WilsDir));
				 }
				 
				 suff="_stout_"+std::to_string(stIt*stepStout);
				 propF->load();
				 su3.absorbDir_device(gaugeWL, WilsDir); // only for z direction
				 WL.setUnit( (std::vector<int>) {0,4,8});
				 for(int i = 0 ; i < HGC_totalL[WilsDir]/2;i++){ // HGC_totalL[2] only for z direction
				   TIME(corrThrpWL.contractNucleonThrp_wilsonLine(seqPropOut, *propF, WL, signProps, gammas ,-i,fl_str));
				   if(signPer < 0) for(int iv = 0 ; iv < corrThrpWL.getTotalSize()*2; iv++) (corrThrpWL.H_elem())[iv] *= signPer;
				   THREAD(corrThrpWL.writeHDF5(threep_filename + suff  +  "_ts_" + std::to_string(tSinks[ts]) + "_Proj_" + proj_str));
				   propExchange = propIn; propIn = propF; propF = propExchange;
				   TIME(WL.wilsonLineUpdate(su3, tmp, WilsDir)); // build Wilson line in the +z direction
				   TIME(propF->shift(*propIn, WilsDir));
				 }
				 
				 propF->load();
			       }
			       delete propIn;
			     };
	if(nucleon == PROTON) {
	  TIME(computeThreep(mu<0 ? mu : -mu, propUP3D, propDN3D, +1, &propUP_SL, "UP"));
	  TIME(computeThreep(mu>0 ? mu : -mu, propUP3D, propUP3D, -1, &propDN_SL, "DOWN"));
	} else {
	  TIME(computeThreep(mu>0 ? mu : -mu, propDN3D, propUP3D, -1, &propDN_SL, "DOWN"));
	  TIME(computeThreep(mu<0 ? mu : -mu, propDN3D, propDN3D, +1, &propUP_SL, "UP"));
	}
      }
    }
    
    propUP.rotateToPhysicalBase_device(+1);
    propDN.rotateToPhysicalBase_device(-1);
    propUP.applyBoundaries_device(source[3]);
    propDN.applyBoundaries_device(source[3]);
    
    PLEGMA_Correlator<float> corr(corr_space, source, maxQsq);
    TIME(corr.contractMesons(propUP, propDN));
    THREAD(corr.writeFile(twop_filename, corr_file_format));
    
    TIME(corr.contractBaryons(propUP, propDN));
    THREAD(corr.writeFile(twop_filename, corr_file_format));
  }
  if(isGPD) delete smearedGauge_sink;
  
  }
  finalize();
  return 0;
}

