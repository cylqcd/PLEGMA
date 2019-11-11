#include <PLEGMA.h>
#include <PLEGMA_utils.h>

double runtime;
#define TIME(fnc)  runtime = MPI_Wtime(); fnc; runtime = MPI_Wtime()-runtime; \
  PLEGMA_printf("TIME for "#fnc" %lf sec\n", runtime)

std::vector<std::thread> threads;
//#define THREAD(fnc) threads.push_back(std::thread([=]() { TIME(fnc); }))
#define THREAD(fnc) TIME(fnc)

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
    std::vector<std::thread> threads;
    // Reading from Lime file and loading to device
    PLEGMA_Gauge<double> gauge;
    gauge.readFile(latfile, LIME_FORMAT);
    gauge.load();
    gauge.calculatePlaq();

    // Loading to QUDA and computing plaquette also there
    initGaugeQuda(gauge, true, QUDA_WILSON_LINKS);
    plaqQuda();

    PLEGMA_Gauge<float> contractGauge;
    contractGauge.copy(gauge);
    // apply boundary conditions since is needed for the covariant derivative
    applyBoundaryConditions(contractGauge,true);

    // Smearing
    PLEGMA_Gauge<double> smearedGauge;
    TIME(smearedGauge.APEsmearing(gauge, nsmearAPE, alphaAPE, 3)); 
    PLEGMA_printf("Plaquette after smearing:\n");
    smearedGauge.calculatePlaq();

    // ensuring mu positive
    TIME(QUDA_solver solver(mu));

    std::string smearType = ((nsmearGauss>0) ? "SS" : "LL");
    std::string smearString = smearType + "_" + "gN" + std::to_string(nsmearGauss) + "a" + convNumToStr(alphaGauss) + "aN" + std::to_string(nsmearAPE) + "a" + convNumToStr(alphaAPE);
    for(int isource = 0 ; isource < numSourcePositions; isource++){
      PLEGMA_printf("\n ### Calculations for source-position %d - %02d.%02d.%02d.%02d begin now ###\n\n",
		    isource, sourcePositions[isource][0], sourcePositions[isource][1],
		    sourcePositions[isource][2], sourcePositions[isource][3]);
    
      // Do forward propagators ----------------------------------------------------
      PLEGMA_Propagator<float> propUP;
      // ensuring mu positive
      if(mu<0) {
	mu*=-1.;
	solver.UpdateSolver();
      }
      for(int isc = 0 ; isc < 12 ; isc++){
	PLEGMA_Vector<double> vectorInOut,vectorAuxD;
	PLEGMA_Vector<float> vectorAuxF;
	vectorAuxD.pointSource(sourcePositions[isource], isc/3, isc%3, DEVICE);
	vectorInOut.gaussianSmearing(vectorAuxD, smearedGauge, nsmearGauss, alphaGauss, sourcePositions[isource][DIM_T]);
      
	PLEGMA_printf("Going to invert UP for component %d\n", isc);
	solver.solve(vectorInOut, vectorInOut);
	vectorAuxF.copy(vectorInOut);
	propUP.absorb(vectorAuxF, isc/3, isc%3);
      }	
    
      PLEGMA_Propagator<float> propDN;
      // ensuring mu negative
      if(mu>0) {
	mu*=-1.;
	solver.UpdateSolver();
      }
      for(int isc = 0 ; isc < 12 ; isc++){
	PLEGMA_Vector<double> vectorInOut,vectorAuxD;
	PLEGMA_Vector<float> vectorAuxF;
	vectorAuxD.pointSource(sourcePositions[isource], isc/3, isc%3, DEVICE);
	TIME(vectorInOut.gaussianSmearing(vectorAuxD, smearedGauge, nsmearGauss, alphaGauss, sourcePositions[isource][DIM_T]));
      
	PLEGMA_printf("Going to invert DN for component %d\n", isc);
	TIME(solver.solve(vectorInOut, vectorInOut));
	vectorAuxF.copy(vectorInOut);
	propDN.absorb(vectorAuxF, isc/3, isc%3);
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
	  PLEGMA_Vector<double> vectorAuxD1,vectorAuxD2;
	  PLEGMA_Vector<float> vectorAuxF;
	  vectorAuxF.absorb(propUP,isc/3, isc%3);
	  vectorAuxD1.copy(vectorAuxF);
	  TIME(vectorAuxD2.gaussianSmearing(vectorAuxD1, smearedGauge, nsmearGauss, alphaGauss, global_fixSinkTime));
	  vectorAuxF.copy(vectorAuxD2);
	  propUP3D.absorb(vectorAuxF, global_fixSinkTime, isc/3, isc%3);

	  vectorAuxF.absorb(propDN,isc/3, isc%3);
	  vectorAuxD1.copy(vectorAuxF);
	  TIME(vectorAuxD2.gaussianSmearing(vectorAuxD1, smearedGauge, nsmearGauss, alphaGauss, global_fixSinkTime));
	  vectorAuxF.copy(vectorAuxD2);
	  propDN3D.absorb(vectorAuxF, global_fixSinkTime, isc/3, isc%3);
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
		PLEGMA_Vector<double> vectorInOut, vectorAuxD;
		PLEGMA_Vector<float> vectorAuxF;
		if(nucleon == PROTON)
		  vectorAuxF.seqSourceNucleon(propUP3D, propDN3D, get_projector(Projs[iproj]), nucleon, global_fixSinkTime, nu, c2);
		else
		  vectorAuxF.seqSourceNucleon(propDN3D, propUP3D, get_projector(Projs[iproj]), nucleon, global_fixSinkTime, nu, c2);
		// put a momentum in the sink later
		vectorAuxF.conjugate();
		vectorAuxF.apply_gamma(G5);
		vectorAuxD.copy(vectorAuxF);
		TIME(vectorInOut.gaussianSmearing(vectorAuxD,smearedGauge, nsmearGauss, alphaGauss, global_fixSinkTime));
		double norm = vectorInOut.norm();
		vectorInOut.cscale(1/norm);
		TIME(solver.solve(vectorInOut, vectorInOut));
		vectorInOut.cscale(norm);
		vectorAuxF.copy(vectorInOut);
		seqProp.absorb(vectorAuxF, nu, c2);
	      }
	    seqProp.apply_gamma(G5);
	    seqProp.conjugate();
	    int signProps = (nucleon == PROTON) ? +1: -1;
	    std::string partName = (nucleon == PROTON) ? "up" : "dn";
	    std::string preSuf;
	  
	    PLEGMA_Propagator<float> &propF = (nucleon == PROTON) ? propUP : propDN;
	    PLEGMA_Correlator<float> corr(corr_space, sourcePositions[isource], maxQsq, tsinkMtsource+1);
	  
	    // LOCAL contractions
	    TIME(corr.contractNucleonThrp_local(seqProp, propF, signProps, gammas));
	    if(signPer < 0) for(size_t iv = 0 ; iv < corr.getTotalSize()*2; iv++) corr.H_elem()[iv] *= signPer;      
	    preSuf = (corr_file_format == ASCII_FORMAT || corr_file_format == LIME_FORMAT) ? "_local" : "";
	    THREAD(corr.writeFile(filename+partName+preSuf+get_file_format_suffix(corr_file_format), corr_file_format));

	    // ONED contractions
	    TIME(corr.contractNucleonThrp_oneD(seqProp, propF, contractGauge, signProps, gammas));
	    if(signPer < 0) for(size_t iv = 0 ; iv < corr.getTotalSize()*2; iv++) corr.H_elem()[iv] *= signPer;
	    preSuf = (corr_file_format == ASCII_FORMAT || corr_file_format == LIME_FORMAT) ? "_oneD" : "";
	    THREAD(corr.writeFile(filename+partName+preSuf+get_file_format_suffix(corr_file_format), corr_file_format));

	    // noe contractions
	    TIME(corr.contractNucleonThrp_noe(seqProp, propF, contractGauge, signProps));
	    if(signPer < 0) for(size_t iv = 0 ; iv < corr.getTotalSize()*2; iv++) corr.H_elem()[iv] *= signPer;
	    preSuf = (corr_file_format == ASCII_FORMAT || corr_file_format == LIME_FORMAT) ? "_noe" : "";
	    THREAD(corr.writeFile(filename+partName+preSuf+get_file_format_suffix(corr_file_format), corr_file_format));
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
		PLEGMA_Vector<double> vectorInOut, vectorAuxD;
		PLEGMA_Vector<float> vectorAuxF;
		if(nucleon == PROTON)
		  vectorAuxF.seqSourceNucleon(propUP3D, get_projector(Projs[iproj]), nucleon, global_fixSinkTime, nu, c2); //test case unpolarized proj
		else
		  vectorAuxF.seqSourceNucleon(propDN3D, get_projector(Projs[iproj]), nucleon, global_fixSinkTime, nu, c2);
		// put a momentum in the sink later
		vectorAuxF.conjugate();
		vectorAuxF.apply_gamma(G5);
		vectorAuxD.copy(vectorAuxF);
		TIME(vectorInOut.gaussianSmearing(vectorAuxD,smearedGauge, nsmearGauss, alphaGauss, global_fixSinkTime));
		double norm = vectorInOut.norm();
		vectorInOut.cscale(1/norm);
		solver.solve(vectorInOut, vectorInOut);
		vectorInOut.cscale(norm);
		vectorAuxF.copy(vectorInOut);
		seqProp.absorb(vectorAuxF, nu, c2);
	      }
	    seqProp.apply_gamma(G5);
	    seqProp.conjugate();
	    int signProps = (nucleon == PROTON) ? -1: +1;
	    std::string partName = (nucleon == PROTON) ? "dn" : "up";
	    std::string preSuf;
	  
	    PLEGMA_Propagator<float> &propF = (nucleon == PROTON) ? propDN : propUP;
	    PLEGMA_Correlator<float> corr(corr_space, sourcePositions[isource], maxQsq, tsinkMtsource+1);
	  
	    //LOCAL
	    TIME(corr.contractNucleonThrp_local(seqProp, propF, signProps, gammas));
	    if(signPer < 0) for(size_t iv = 0 ; iv < corr.getTotalSize()*2; iv++) corr.H_elem()[iv] *= signPer;
	    preSuf = (corr_file_format == ASCII_FORMAT || corr_file_format == LIME_FORMAT) ? "_local" : "";
	    THREAD(corr.writeFile(filename+partName+preSuf+get_file_format_suffix(corr_file_format), corr_file_format));

	    //ONED
	    TIME(corr.contractNucleonThrp_oneD(seqProp, propF, contractGauge, signProps, gammas));
	    if(signPer < 0) for(size_t iv = 0 ; iv < corr.getTotalSize()*2; iv++) corr.H_elem()[iv] *= signPer;
	    preSuf = (corr_file_format == ASCII_FORMAT || corr_file_format == LIME_FORMAT) ? "_oneD" : "";
	    THREAD(corr.writeFile(filename+partName+preSuf+get_file_format_suffix(corr_file_format), corr_file_format));

	    //ONED
	    TIME(corr.contractNucleonThrp_noe(seqProp, propF, contractGauge, signProps));
	    if(signPer < 0) for(size_t iv = 0 ; iv < corr.getTotalSize()*2; iv++) corr.H_elem()[iv] *= signPer;
	    preSuf = (corr_file_format == ASCII_FORMAT || corr_file_format == LIME_FORMAT) ? "_noe" : "";
	    THREAD(corr.writeFile(filename+partName+preSuf+get_file_format_suffix(corr_file_format), corr_file_format));
	  }
	}
      }    
      for(int nu = 0 ; nu < 4 ; nu++)
	for(int c2 = 0 ; c2 < 3 ; c2++){
	  PLEGMA_Vector<double> vectorAuxD1,vectorAuxD2;
	  PLEGMA_Vector<float> vectorAuxF;
	  vectorAuxF.absorb(propUP, nu, c2);
	  vectorAuxD1.copy(vectorAuxF);
	  TIME(vectorAuxD2.gaussianSmearing(vectorAuxD1, smearedGauge, nsmearGauss, alphaGauss));
	  vectorAuxF.copy(vectorAuxD2);
	  propUP.absorb(vectorAuxF, nu, c2);

	  vectorAuxF.absorb(propDN, nu, c2);
	  vectorAuxD1.copy(vectorAuxF);
	  TIME(vectorAuxD2.gaussianSmearing(vectorAuxD1, smearedGauge, nsmearGauss, alphaGauss));
	  vectorAuxF.copy(vectorAuxD2);
	  propDN.absorb(vectorAuxF, nu, c2);
	}
  
      propUP.rotateToPhysicalBase_device(+1);
      propDN.rotateToPhysicalBase_device(-1);
      propUP.applyBoundaries_device(sourcePositions[isource][3]);
      propDN.applyBoundaries_device(sourcePositions[isource][3]);
  
      PLEGMA_Correlator<float> corr(corr_space, sourcePositions[isource], maxQsq);
      TIME(corr.contractMesons(propUP, propDN));
      THREAD(corr.writeFile(twop_filename + "_" + smearString + get_file_format_suffix(corr_file_format), corr_file_format));

      TIME(corr.contractBaryons(propUP, propDN));
      THREAD(corr.writeFile(twop_filename + "_" + smearString + get_file_format_suffix(corr_file_format), corr_file_format));
    }
    while(not threads.empty()) {threads.back().join(); threads.pop_back();}
  }

  finalize();

  return 0;
}

