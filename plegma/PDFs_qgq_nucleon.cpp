#include <PLEGMA.h>
#include <PLEGMA_utils.h>

using namespace plegma;
using namespace quda;

struct pairedSinged {
  std::pair<int,int> munu;
  int sign;
};
    
static pairedSinged makePairCheck(int mu, int nu){
  if( mu == nu) PLEGMA_error("Cannot choose mu == nu");
  pairedSinged prs;
  if(nu > mu){
    prs.sign = +1;
    prs.munu = std::make_pair(mu,nu);
  }
  else{
    prs.sign = -1;
    prs.munu = std::make_pair(nu,mu);
  }
  return prs;
}


int main(int argc, char **argv)
{

  static std::vector<std::string> listOpt = {"verbosity", "load-gauge", "nsmear-APE", "alpha-APE", "nsmear-gauss", "alpha-gauss",
					     "nsrc", "src-filename", "maxQsq", "twop-filename","threep-filename",  "corr-file-format",
					     "corr-space", "tSinks","xiMomSm","sinkMom","which_particle","gammas"};

  initializeOptions(argc, argv, true, listOpt);

  int WilsDir;
  HGC_options->set("wilson_direction", "Direction of the wilson line", verbosity, WilsDir);
  if(WilsDir>N_DIMS) PLEGMA_error("The direction of the WIlson line has to be smaller than 3");

  double rhoStout;
  HGC_options->set("rho_stout", "Rho parameter stout smearing", verbosity, rhoStout);

  size_t maxStout;
  HGC_options->set("max_stout", "Maximum number of stout smearing steps", verbosity, maxStout);

  size_t stepStout;
  HGC_options->set("step_stout", "Save the PDFs every step_stout stout smearing step", verbosity, stepStout);

  int nStoutFmunu;
  HGC_options->set("nStoutFmunu", "How many steps of stout smearing to apply on the FST", verbosity, nStoutFmunu);
  
  bool calc3pt = true ;
  HGC_options->set("calc3pt", "If true then the 3pt function is computed", verbosity, calc3pt);

  bool debugMode = false;
  HGC_options->set("debugMode", "If enabled then it sets the Fmunu to unit matrices to check with standard PDFs calculation", verbosity, debugMode);
  
  std::string proj ;
  HGC_options->set("which_projector", "Which projector to use for 3pt function", verbosity, proj);
  WHICHPROJECTOR which_proj=get_projector(proj.c_str());

  initializePLEGMA();
  
  // Reading from Lime file and loading to device
  PLEGMA_Gauge<double> gauge;
  gauge.readFile(latfile, LIME_FORMAT);
  gauge.calculatePlaq();
  gauge.unload();


  
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


  // Extracting the spacial sink momentum from the sink 4-momentum  
  std::vector<int> sinkMom_3D(3);
  std::copy(sinkMom.begin(),sinkMom.begin()+3,sinkMom_3D.begin());
  
    
  // ensuring mu positive
  if(mu<0)  mu*=-1.;
  QUDA_solver *solver = new QUDA_solver(mu);

  PLEGMA_Propagator<float> *propUP = new PLEGMA_Propagator<float>(BOTH);
  PLEGMA_Propagator<float> *propDN = new PLEGMA_Propagator<float>(BOTH);
  PLEGMA_Propagator<float> *seqPropOut = new PLEGMA_Propagator<float>(BOTH);
  PLEGMA_Propagator<float> *propIn = new PLEGMA_Propagator<float>(BOTH);
  PLEGMA_Propagator<float> *propExchange = nullptr;
  
  PLEGMA_Su3field<float> su3;
  PLEGMA_Su3field<float> su3_2;
  PLEGMA_Su3field<float> WL;
  PLEGMA_Su3field<float> bWL; // beginning of WL
  PLEGMA_Su3field<float> eWL; // end of WL (between is the fmunu)
  PLEGMA_Gauge<float> gaugeWL;
  
  PLEGMA_Su3field<float> tmp;
  PLEGMA_Fmunu<float> *fmunu = new PLEGMA_Fmunu<float>(BOTH);
  PLEGMA_Fmunu<float> *fmunuIn = new PLEGMA_Fmunu<float>(BOTH);
  PLEGMA_Fmunu<float> *fmunuExchange = nullptr;


  if(! debugMode){
    gauge.stoutSmearing(gauge,nStoutFmunu,rhoStout,4);
    gaugeWL.copy(gauge);
    fmunu->compute_leaves(gaugeWL);
    gauge.load();
  }
  else fmunu->setUnit((std::vector<int>) {0,4,8,9,13,17,18,22,26,27,31,35,36,40,44,45,49,53});  

  fmunu->unload();
  fmunuIn->copy(*fmunu);
  fmunuIn->unload();

  std::vector<int> F_ind0 = {WilsDir,3};
  
  for(int isource = 0; isource < numSourcePositions; isource++){
    site& source = sourcePositions[isource];
    PLEGMA_printf("\n ### Calculations for source-position %d - %02d.%02d.%02d.%02d begin now ###\n\n",
		  isource, source[0], source[1], source[2], source[3]);
    gaugeWL.copy(gauge);
    PLEGMA_Vector<double> vectorIn;
    PLEGMA_Vector<double> vectorOut;
    PLEGMA_Vector<double> vectorAuxD;
    PLEGMA_Vector<float> vectorAuxF;
    PLEGMA_Propagator3D<float> propUP3D;
    PLEGMA_Propagator3D<float> propDN3D;

    propUP->zero_where(BOTH);
    propDN->zero_where(BOTH);
    seqPropOut->zero_where(BOTH);
    propIn->zero_where(BOTH);
    for(int ts=0;ts<tSinks.size();ts++)
      {
	PLEGMA_Correlator<float> corrThrpWL(corr_space, source, 0, tSinks[ts]+1);
      
	int signPer = (tSinks[ts] + sourcePositions[isource][3]) >= HGC_totalL[3] ? -1 : +1;
	int global_fixSinkTime = (tSinks[ts] + sourcePositions[isource][3])%HGC_totalL[3]; 
	int my_fixSinkTime = global_fixSinkTime - HGC_procPosition[3] * HGC_localL[3];
	bool is_myST = (my_fixSinkTime >= 0) && ( my_fixSinkTime < HGC_localL[3] );

	if(mu<0) {
	  mu*=-1.;
	  solver->UpdateSolver();
	}

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


	  propUP->unload();
	  propDN->unload();
	  //seq source part 2Props and contraction block
	  {
	    for(int nu = 0 ; nu < 4 ; nu++)
	      for(int c2 = 0 ; c2 < 3 ; c2++){
		PLEGMA_Vector3D<float> vectorAux3D;
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
	      if(stIt>0){
		gauge.stoutSmearing(gauge,stepStout,rhoStout,3);
		//		PLEGMA_printf("Something here is the problem\n");
		gauge.calculatePlaq();
		gaugeWL.copy(gauge);
		//		gauge.load();
	      }

	      su3.absorbDir_device(gaugeWL, WilsDir);
	      su3_2.copy(su3);
	      su3.unload();
	      su3_2.unload();
	      WL.setUnit( (std::vector<int>) {0,4,8});
	      for(int z2 = 0 ; z2 < HGC_totalL[WilsDir]/2; z2++){
		bWL.setUnit( (std::vector<int>) {0,4,8});
		eWL.copy(WL);
		for(int z1 = 0; z1 <= z2; z1++){
		  for(int mu : F_ind0)
		    for(int nu =0; nu < 4; nu++)
		      if(nu != WilsDir && nu != 3){
			pairedSinged prs = makePairCheck(mu,nu);
			corrThrpWL.contractNucleonThrp_qgq(*seqPropOut, *propF, bWL,*fmunu,
							   prs.munu, eWL, signProps, gammas, z2,z1);
			if(signPer < 0) for(int iv = 0 ; iv < corrThrpWL.getTotalSize()*2; iv++) corrThrpWL.H_elem()[iv] *= signPer*prs.sign;
			corrThrpWL.writeHDF5( (threep_filename +  suff + "FST_"+
						std::to_string(mu)+std::to_string(nu)+"_ts_"
					       + std::to_string(tSinks[ts])  + "_proj_"+proj.c_str()).c_str() );
		      }
		  eWL.Udag();
		  tmp.UxU(eWL,su3_2);
		  eWL.Udag(tmp);
		  bWL.wilsonLineUpdate(su3_2, tmp, 4+WilsDir);
		  fmunuExchange = fmunuIn; fmunuIn = fmunu; fmunu = fmunuExchange;
		  fmunu->shift(*fmunuIn,4+WilsDir);
		}
		fmunu->load();
		su3_2.load();
		propExchange = propIn; propIn = propF; propF = propExchange;
		WL.wilsonLineUpdate(su3, tmp, 4+WilsDir); 
		propF->shift(*propIn, 4+WilsDir);
	      }

	      suff="_CP2_stout_"+std::to_string(stIt*stepStout)+"_Minus_";
	      propF->load();
	      //	    su3.absorbDir_device(gaugeWL, WilsDir); // only for z direction
	      su3.load();
	      WL.setUnit( (std::vector<int>) {0,4,8});
	      for(int z2 = 0 ; z2 < HGC_totalL[WilsDir]/2; z2++){ // HGC_totalL[2] only for z direction
		bWL.setUnit( (std::vector<int>) {0,4,8});
		eWL.copy(WL);
		for(int z1 = 0; z1 <= z2; z1++){
		  for(int mu : F_ind0)
		    for(int nu =0; nu < 4; nu++)
		      if(nu != WilsDir && nu != 3){
			pairedSinged prs = makePairCheck(mu,nu);
			corrThrpWL.contractNucleonThrp_qgq(*seqPropOut, *propF, bWL,*fmunu,
							   prs.munu, eWL, signProps, gammas, z2, z1);
			if(signPer < 0) for(int iv = 0 ; iv < corrThrpWL.getTotalSize()*2; iv++) corrThrpWL.H_elem()[iv] *= signPer*prs.sign;

			corrThrpWL.writeHDF5( (threep_filename +  suff + "FST_"+
					       std::to_string(mu)+std::to_string(nu)+"_ts_"
					       + std::to_string(tSinks[ts])  + "_proj_"+proj.c_str()).c_str() );

		      }
		  bWL.wilsonLineUpdate(su3_2, tmp, WilsDir);
		  tmp.UxU(su3_2,eWL);
		  eWL.copy(tmp);
		  fmunuExchange = fmunuIn; fmunuIn = fmunu; fmunu = fmunuExchange;
		  fmunu->shift(*fmunuIn,WilsDir);
		}
		fmunu->load();
		su3_2.load();
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
	    gauge.unload();
	    gaugeWL.copy(gauge);

	    //!!!!!!!!!!!!!!!!!!!!!!!!! if spatial extent is not multiple of 2 then it will not work
	    for(int stIt=0;stIt<=(int)(maxStout/stepStout);stIt++){
	      if(stIt>0){
		gauge.stoutSmearing(gauge,stepStout,rhoStout,3);
		gauge.calculatePlaq();
		gaugeWL.copy(gauge);
	      }
	      std::string suff="_CP1_stout_"+std::to_string(stIt*stepStout)+"_Plus_";

	      su3.absorbDir_device(gaugeWL, WilsDir);
	      su3_2.copy(su3);
	      su3.unload();
	      su3_2.unload();
	      WL.setUnit( (std::vector<int>) {0,4,8});
	      for(int z2 = 0 ; z2 < HGC_totalL[WilsDir]/2; z2++){
		bWL.setUnit( (std::vector<int>) {0,4,8});
		eWL.copy(WL);
		for(int z1 = 0; z1 <= z2; z1++){
		  for(int mu : F_ind0)
		    for(int nu =0; nu < 4; nu++)
		      if(nu != WilsDir && nu != 3){
			pairedSinged prs = makePairCheck(mu,nu);

			corrThrpWL.contractNucleonThrp_qgq(*seqPropOut, *propF, bWL,*fmunu,
							   prs.munu, eWL, signProps, gammas, z2, z1);
			if(signPer < 0) for(int iv = 0 ; iv < corrThrpWL.getTotalSize()*2; iv++) corrThrpWL.H_elem()[iv] *= signPer*prs.sign;
			corrThrpWL.writeHDF5( (threep_filename +  suff + "FST_"+
					       std::to_string(mu)+std::to_string(nu)+"_ts_"
					       + std::to_string(tSinks[ts])  + "_proj_"+proj.c_str()).c_str() );

		      }
		  eWL.Udag();
		  tmp.UxU(eWL,su3_2);
		  eWL.Udag(tmp);
		  bWL.wilsonLineUpdate(su3_2, tmp, 4+WilsDir);
		  fmunuExchange = fmunuIn; fmunuIn = fmunu; fmunu = fmunuExchange;
		  fmunu->shift(*fmunuIn,4+WilsDir);
		}
		fmunu->load();
		su3_2.load();
		propExchange = propIn; propIn = propF; propF = propExchange;
		WL.wilsonLineUpdate(su3, tmp, 4+WilsDir); 
		propF->shift(*propIn, 4+WilsDir);
	      }
	    
	      propF->load();
	      suff="_CP1_stout_"+std::to_string(stIt*stepStout)+"_Minus_";

	      su3.load();
	      WL.setUnit( (std::vector<int>) {0,4,8});
	      for(int z2 = 0 ; z2 < HGC_totalL[WilsDir]/2; z2++){ 
		bWL.setUnit( (std::vector<int>) {0,4,8});
		eWL.copy(WL);
		for(int z1 = 0; z1 <= z2; z1++){
		  for(int mu : F_ind0)
		    for(int nu =0; nu < 4; nu++)
		      if(nu != WilsDir && nu != 3){
			pairedSinged prs = makePairCheck(mu,nu);
			corrThrpWL.contractNucleonThrp_qgq(*seqPropOut, *propF, bWL,*fmunu,
							   prs.munu, eWL, signProps, gammas, z2, z1);
			if(signPer < 0) for(int iv = 0 ; iv < corrThrpWL.getTotalSize()*2; iv++) corrThrpWL.H_elem()[iv] *= signPer*prs.sign;
			corrThrpWL.writeHDF5( (threep_filename +  suff + "FST_"+
					       std::to_string(mu)+std::to_string(nu)+"_ts_"
					       + std::to_string(tSinks[ts])  + "_proj_"+proj.c_str()).c_str() );
		      }
		  bWL.wilsonLineUpdate(su3_2, tmp, WilsDir);
		  tmp.UxU(su3_2,eWL);
		  eWL.copy(tmp);
		  fmunuExchange = fmunuIn; fmunuIn = fmunu; fmunu = fmunuExchange;
		  fmunu->shift(*fmunuIn,WilsDir);
		}
		fmunu->load();
		su3_2.load();
		propExchange = propIn; propIn = propF; propF = propExchange;
		WL.wilsonLineUpdate(su3, tmp, WilsDir); // build Wilson line in the +z direction
		propF->shift(*propIn, WilsDir);
	      }
	    
	      propF->load();
	    }
	  }
	}
      }

    gauge.unload();
    
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
  }
  
  delete propUP;
  delete propDN;
  delete propIn;
  delete seqPropOut;
  delete fmunu;
  delete fmunuIn;	


  delete solver;
  
  finalize();
  return 0;
}

