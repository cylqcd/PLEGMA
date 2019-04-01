#include <PLEGMA.h>
#include <PLEGMA_utils.h>

using namespace plegma;
using namespace quda;

static std::vector<std::string> listOpt = {"verbosity", "load-gauge", "nsmear-APE", "alpha-APE", "nsmear-gauss", "alpha-gauss",
					   "nsrc", "src-filename", "maxQsq", "twop-filename","threep-filename",  "corr-file-format",
					   "corr-space", "tSinks","Projs"};

int main(int argc, char **argv)
{
  initializeOptions(argc, argv, true, listOpt);
  //================ Add your options in this between initializeOptions and initializePLEGMA ================//
  std::string prOrNt = "neutron";
  HGC_options->set("whichParticle", "Which particle we want to do the 3pf. Options (proton, neutron)", verbosity, prOrNt);
  if(prOrNt != "proton" && prOrNt != "neutron") PLEGMA_error("This exec is only for nucleon, %s is not allowed",prOrNt.c_str());
  //=========================================================================================================//
  initializePLEGMA();

  {
    // Reading from Lime file and loading to device
    PLEGMA_Gauge<double> gauge;
    gauge.readFromLime(latfile.c_str());
    gauge.load();
    gauge.calculatePlaq();

    // Loading to QUDA and computing plaquette also there
    initGaugeQuda(gauge, true, QUDA_WILSON_LINKS);
    plaqQuda();

    PLEGMA_Gauge<float> contractGauge;
    contractGauge.copy(gauge);

    // Smearing
    PLEGMA_Gauge<float> smearedGauge;
    smearedGauge.APEsmearing(contractGauge, nsmearAPE, alphaAPE, 3); 
    PLEGMA_printf("Plaquette after smearing:\n");
    smearedGauge.calculatePlaq();

    // apply boundary conditions since is needed for the covariant derivative
    applyBoundaryConditions(contractGauge,true);

    // ensuring mu positive
    QUDA_solver solver(mu);

    std::string smearType = ((nsmearGauss>0) ? "SS" : "LL");
    std::string smearString = smearType + "_" + "gN" + std::to_string(nsmearGauss) + "a" + convNumToStr(alphaGauss) + "aN" + std::to_string(nsmearAPE) + "a" + convNumToStr(alphaAPE);
    for(int isource = 0 ; isource < numSourcePositions; isource++){
      PLEGMA_printf("\n ### Calculations for source-position %d - %02d.%02d.%02d.%02d begin now ###\n\n",
		    isource, sourcePositions[isource][0], sourcePositions[isource][1],
		    sourcePositions[isource][2], sourcePositions[isource][3]);
      // std::stringstream ss;
      // for(int i = 0 ; i < N_DIMS; i++)
      //   ss << std::setfill('0') << std::setw(3) << sourcePositions[isource][i] << ".";
      // 	//<< sourcePositions[isource][1] << "." << sourcePositions[isource][2] << "." << sourcePositions[isource][3];
      // std::string sourcePosString = "_" + ss.str().substr(0,std::string::npos-1) + "_";
      // ss.str(std::string()); ss.clear();
    
      // Do forward propagators ----------------------------------------------------
      PLEGMA_Propagator<float> propUP;
      // ensuring mu positive
      if(mu<0) {
	mu*=-1.;
	solver.UpdateSolver();
      }
      for(int isc = 0 ; isc < 12 ; isc++){
	PLEGMA_Vector<double> vectorInOut;
	PLEGMA_Vector<float> vectorAux1,vectorAux2;
	vectorAux1.pointSource(sourcePositions[isource], isc/3, isc%3, DEVICE);
	vectorAux2.gaussianSmearing(vectorAux1, smearedGauge, nsmearGauss, alphaGauss);
	vectorInOut.copy(vectorAux2);
      
	PLEGMA_printf("Going to invert UP for component %d\n", isc);
	solver.solve(vectorInOut, vectorInOut);
	vectorAux1.copy(vectorInOut);
	propUP.absorb(vectorAux1, isc/3, isc%3);
      }	
    
      PLEGMA_Propagator<float> propDN;
      // ensuring mu negative
      if(mu>0) {
	mu*=-1.;
	solver.UpdateSolver();
      }
      for(int isc = 0 ; isc < 12 ; isc++){
	PLEGMA_Vector<double> vectorInOut;
	PLEGMA_Vector<float> vectorAux1,vectorAux2;
	vectorAux1.pointSource(sourcePositions[isource], isc/3, isc%3, DEVICE);
	vectorAux2.gaussianSmearing(vectorAux1, smearedGauge, nsmearGauss, alphaGauss);
	vectorInOut.copy(vectorAux2);
      
	PLEGMA_printf("Going to invert DN for component %d\n", isc);
	solver.solve(vectorInOut, vectorInOut);
	vectorAux1.copy(vectorInOut);
	propDN.absorb(vectorAux1, isc/3, isc%3);
      }

      for(size_t its = 0; its < tSinks.size(); its++){
	int tsinkMtsource = tSinks[its];
	if(tsinkMtsource >= HGC_totalL[3]) PLEGMA_error("Provided tsink=%d is >= than temporal extent",tsinkMtsource);
	int signPer = (tsinkMtsource+sourcePositions[isource][3]) >= HGC_totalL[3] ? -1 : +1;
	int global_fixSinkTime = (tsinkMtsource + sourcePositions[isource][3])%HGC_totalL[3]; 

	// Smear the 3D propagators ------------------------
	PLEGMA_Propagator3D<float> propUP3D;
	PLEGMA_Propagator3D<float> propDN3D;
	for(int isc = 0 ; isc < 12 ; isc++){
	  PLEGMA_Vector<float> vectorAux1,vectorAux2;
	  vectorAux1.absorb(propUP,isc/3, isc%3);
	  vectorAux2.gaussianSmearing(vectorAux1, smearedGauge, nsmearGauss, alphaGauss);
	  propUP3D.absorb(vectorAux2, global_fixSinkTime, isc/3, isc%3);

	  vectorAux1.absorb(propDN,isc/3, isc%3);
	  vectorAux2.gaussianSmearing(vectorAux1, smearedGauge, nsmearGauss, alphaGauss);
	  propDN3D.absorb(vectorAux2, global_fixSinkTime, isc/3, isc%3);      
	}
    
	WHICHPARTICLE nucleon = get_particle(prOrNt); 
	std::vector<GAMMAS> gammas = {ONE,G1,G2,G3,G4,G5,G5G1,G5G2,G5G3,G5G4,S12,S13,S23,S41,S42,S43};
	for(size_t iproj = 0; iproj < Projs.size(); iproj++){
	  //seq source part 2Props and contraction block
	  std::string filename = threep_filename + "_" +  smearString + "_" + Projs[iproj] + "_dt" + std::to_string(tsinkMtsource) + "_";
	  {
	    PLEGMA_Propagator<float> seqProp;
	    // ensuring correct sign of mu
	    if((nucleon == PROTON && mu > 0) || (mu < 0)) {
	      mu*=-1.;
	      solver.UpdateSolver();
	    }
	    for(int nu = 0 ; nu < 4 ; nu++)
	      for(int c2 = 0 ; c2 < 3 ; c2++){
		PLEGMA_Vector<double> vectorInOut;
		PLEGMA_Vector<float> vectorAux1, vectorAux2;
		if(nucleon == PROTON)
		  vectorAux1.seqSourceNucleon(propUP3D, propDN3D, get_projector(Projs[iproj]), nucleon, global_fixSinkTime, nu, c2);
		else
		  vectorAux1.seqSourceNucleon(propDN3D, propUP3D, get_projector(Projs[iproj]), nucleon, global_fixSinkTime, nu, c2);
		// put a momentum in the sink later
		vectorAux1.conjugate();
		vectorAux1.apply_gamma(G5);
		vectorAux2.gaussianSmearing(vectorAux1,smearedGauge, nsmearGauss, alphaGauss);
		vectorInOut.copy(vectorAux2);
		// check if we need to normalize the seqsource for mix precision solver
		solver.solve(vectorInOut, vectorInOut);
		// if we normalize the seqsource we have to take it out here
		vectorAux1.copy(vectorInOut);
		seqProp.absorb(vectorAux1, nu, c2);
	      }
	    seqProp.apply_gamma(G5);
	    seqProp.conjugate();
	    int signProps = (nucleon == PROTON) ? +1: -1;
	    std::string partName = (nucleon == PROTON) ? "up" : "dn";
	    std::string preSuf;
	  
	    PLEGMA_Propagator<float> &propF = (nucleon == PROTON) ? propUP : propDN;
	    PLEGMA_Correlator<float> corr(corr_space,maxQsq);
	  
	    // LOCAL contractions
	    corr.contractNucleonThrp_local(seqProp, propF, signProps, gammas, sourcePositions[isource]); // 0 is isource change later 
	    if(signPer < 0) for(size_t iv = 0 ; iv < corr.getTotalSize()*2; iv++) (corr.getCorr())[iv] *= signPer;      
	    preSuf = (corr_file_format == ASCII_FORM || corr_file_format == LIME_FORM) ? "_local" : "";
	    corr.writeFile( (filename+partName+preSuf+get_file_format_suffix(corr_file_format)).c_str(), corr_file_format);

	    // ONED contractions
	    corr.contractNucleonThrp_oneD(seqProp, propF, contractGauge, signProps, gammas, sourcePositions[isource]); // 0 is isource change lat
	    if(signPer < 0) for(size_t iv = 0 ; iv < corr.getTotalSize()*2; iv++) (corr.getCorr())[iv] *= signPer;
	    preSuf = (corr_file_format == ASCII_FORM || corr_file_format == LIME_FORM) ? "_oneD" : "";
	    corr.writeFile( (filename+partName+preSuf+get_file_format_suffix(corr_file_format)).c_str(), corr_file_format);

	    // noe contractions
	    corr.contractNucleonThrp_noe(seqProp, propF, contractGauge, signProps, sourcePositions[isource]); // 0 is isource change lat
	    if(signPer < 0) for(size_t iv = 0 ; iv < corr.getTotalSize()*2; iv++) (corr.getCorr())[iv] *= signPer;
	    preSuf = (corr_file_format == ASCII_FORM || corr_file_format == LIME_FORM) ? "_noe" : "";
	    corr.writeFile( (filename+partName+preSuf+get_file_format_suffix(corr_file_format)).c_str(), corr_file_format);
	  }
  
	  //seq source part 1Prop contraction
	  {
	    PLEGMA_Propagator<float> seqProp;
	    // ensuring correct sign of mu
	    if((nucleon == PROTON && mu < 0) || (mu > 0)) {
	      mu*=-1.;
	      solver.UpdateSolver();
	    }
	    for(int nu = 0 ; nu < 4 ; nu++)
	      for(int c2 = 0 ; c2 < 3 ; c2++){
		PLEGMA_Vector<double> vectorInOut;
		PLEGMA_Vector<float> vectorAux1, vectorAux2;
		if(nucleon == PROTON)
		  vectorAux1.seqSourceNucleon(propUP3D, get_projector(Projs[iproj]), nucleon, global_fixSinkTime, nu, c2); //test case unpolarized proj
		else
		  vectorAux1.seqSourceNucleon(propDN3D, get_projector(Projs[iproj]), nucleon, global_fixSinkTime, nu, c2);
		// put a momentum in the sink later
		vectorAux1.conjugate();
		vectorAux1.apply_gamma(G5);
		vectorAux2.gaussianSmearing(vectorAux1,smearedGauge, nsmearGauss, alphaGauss);
		// check if we need to normalize the seqsource for mix precision solver
		solver.solve(vectorInOut, vectorInOut);
		// if we normalize the seqsource we have to take it out here
		vectorAux1.copy(vectorInOut);
		seqProp.absorb(vectorAux1, nu, c2);
	      }
	    seqProp.apply_gamma(G5);
	    seqProp.conjugate();
	    int signProps = (nucleon == PROTON) ? -1: +1;
	    std::string partName = (nucleon == PROTON) ? "dn" : "up";
	    std::string preSuf;
	  
	    PLEGMA_Propagator<float> &propF = (nucleon == PROTON) ? propDN : propUP;
	    PLEGMA_Correlator<float> corr(corr_space,maxQsq);
	  
	    //LOCAL
	    corr.contractNucleonThrp_local(seqProp, propF, signProps, gammas, sourcePositions[isource]);
	    if(signPer < 0) for(size_t iv = 0 ; iv < corr.getTotalSize()*2; iv++) (corr.getCorr())[iv] *= signPer;
	    preSuf = (corr_file_format == ASCII_FORM || corr_file_format == LIME_FORM) ? "_local" : "";
	    corr.writeFile( (filename+partName+preSuf+get_file_format_suffix(corr_file_format)).c_str(), corr_file_format);

	    //ONED
	    corr.contractNucleonThrp_oneD(seqProp, propF, contractGauge, signProps, gammas, sourcePositions[isource]);
	    if(signPer < 0) for(size_t iv = 0 ; iv < corr.getTotalSize()*2; iv++) (corr.getCorr())[iv] *= signPer;
	    preSuf = (corr_file_format == ASCII_FORM || corr_file_format == LIME_FORM) ? "_oneD" : "";
	    corr.writeFile( (filename+partName+preSuf+get_file_format_suffix(corr_file_format)).c_str(), corr_file_format);

	    //ONED
	    corr.contractNucleonThrp_noe(seqProp, propF, contractGauge, signProps, sourcePositions[isource]);
	    if(signPer < 0) for(size_t iv = 0 ; iv < corr.getTotalSize()*2; iv++) (corr.getCorr())[iv] *= signPer;
	    preSuf = (corr_file_format == ASCII_FORM || corr_file_format == LIME_FORM) ? "_noe" : "";
	    corr.writeFile( (filename+partName+preSuf+get_file_format_suffix(corr_file_format)).c_str(), corr_file_format);
	  }
	}
      }    
      for(int nu = 0 ; nu < 4 ; nu++)
	for(int c2 = 0 ; c2 < 3 ; c2++){
	  PLEGMA_Vector<float> vectorAux1,vectorAux2;
	  vectorAux1.absorb(propUP, nu, c2);
	  vectorAux2.gaussianSmearing(vectorAux1, smearedGauge, nsmearGauss, alphaGauss);
	  propUP.absorb(vectorAux2, nu, c2);

	  vectorAux1.absorb(propDN, nu, c2);
	  vectorAux2.gaussianSmearing(vectorAux1, smearedGauge, nsmearGauss, alphaGauss);
	  propDN.absorb(vectorAux2, nu, c2);
	}
  
      propUP.rotateToPhysicalBase_device(+1);
      propDN.rotateToPhysicalBase_device(-1);
      propUP.applyBoundaries_device(sourcePositions[isource][3]);
      propDN.applyBoundaries_device(sourcePositions[isource][3]);
  
      PLEGMA_Correlator<float> corr(corr_space,maxQsq);
      corr.contractMesons(propUP, propDN, sourcePositions[isource]);
      corr.writeFile((twop_filename + "_" + smearString + get_file_format_suffix(corr_file_format)).c_str(), corr_file_format);

      corr.contractBaryons(propUP, propDN, sourcePositions[isource]);
      corr.writeFile((twop_filename + "_" + smearString + get_file_format_suffix(corr_file_format)).c_str(), corr_file_format);
    }  
  }

  finalize();

  return 0;
}

