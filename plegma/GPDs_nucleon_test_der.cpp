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
					     "nsrc", "src-filename", "twop-filename","threep-filename",  "corr-file-format",
					     "corr-space", "tSinks","xiMomSm","gammas", "Projs"};

  initializeOptions(argc, argv, true, listOpt);

  //================ Add your options in this between initializeOptions and initializePLEGMA ================//

  size_t WilsDir = DIM_Z;
  

  double rhoStout = 0.;

  size_t maxStout = 1;

  size_t minStout = 1;

  size_t stepStout = 5;

  bool calc3pt = true ;

  /*
    We consider the momenta in the symmetric frame. Both P-momentum and Delta-momentum are vectors
    having three components. 
    IF SYMM: The source momentum is given by p-delta/2, while the sink momentum by p+delta/2.
    ELSE: The source momentum is given by p-delta, while the sink momentum by p.
  */

  bool symm = true ;

  //std::vector<int> PMom = {0,0,0};
  int PMom = 4;

  //std::vector<int> DeltaMom = {0,0,0};
  int DeltaMom = 0;

  int maxZ = -1;

  std::string aux_str = "proton";

  std::string mom_path = "%+d_%+d_%+d.txt";

  std::string srcInputFile = "./input.src";

  auto add_options = [&](Options& options) {
  options.set("wilson-direction", "Direction of the wilson line", verbosity, WilsDir);
  options.set("rho-stout", "Rho parameter stout smearing", verbosity, rhoStout);
  options.set("max-stout", "Maximum number of stout smearing steps", verbosity, maxStout);
  options.set("min-stout", "Minimum number of stout smearing steps", verbosity, minStout);
  options.set("stout-steps", "Save the PDFs every step_stout stout smearing step", verbosity, stepStout);
  options.set("calc3pt", "If true then the 3pt function is computed", verbosity, calc3pt);
  options.set("Delta-symm", "If true delta is applied symmetrically otherwise asymmetrically", verbosity, symm);
  options.set("P-momentum", "If added to the momentum transfer delta gives the sink momentum", verbosity, PMom);
  options.set("Delta-momentum", "Square root of the momentum transfer", verbosity, DeltaMom);
  options.set("max-z", "Largest z-value", verbosity, maxZ);
  options.set("which-particle", "Choice of the nucleon interpolator to insert in the three point function (neutron,proton)", verbosity, aux_str);
  options.set("mompath", "Path to the list of momenta", verbosity, mom_path);  
  options.set("src-input-file", "Use the file to update option at every source. The file searched is [src-input-file]+str(n) where n is the source (0, 1, ...)", verbosity, srcInputFile);
		     };
  add_options(*HGC_options);

  
  //=========================================================================================================//
  initializePLEGMA();

  if (maxZ<0) maxZ = HGC_totalL[WilsDir]/2;
  if (maxZ%2==1) PLEGMA_error("max-z must be even\n");
  if (symm and DeltaMom%2 != 0) PLEGMA_error("DeltaMom must be even if symmetric\n");
  if (tSinks.size()==0) calc3pt = false;
  
  double mu_ud = mu;
  WHICHPARTICLE nucleon = get_particle(aux_str.c_str());

  //if(DeltaMom.size() > 3 || PMom.size() > 3) PLEGMA_error("DeltaMom and PMom have to be vectors with lenght three\n");
  //if(DeltaMom.size()!=PMom.size()) PLEGMA_error("PMom has to have the same size of DeltaMom\n");
  if(WilsDir>N_DIMS) PLEGMA_error("The direction of the Wilson line has to be smaller than 3\n");
  if(nucleon!=NEUTRON && nucleon!=PROTON) PLEGMA_error("Only nucleon PDFs have been implemented so far\n");

  //bool isGPD = !std::all_of(DeltaMom.begin(), DeltaMom.end(), [](int i) { return i==0; });
  bool isGPD = DeltaMom != 0;

  std::string given_twop_filename = twop_filename;
  std::string given_threep_filename = threep_filename;
  
  {
  // Reading from Lime file and loading to device
  PLEGMA_Gauge<double> gauge;
  gauge.readFile(latfile, LIME_FORMAT);
  gauge.calculatePlaq();

  // Loading to QUDA and computing plaquette also there
  initGaugeQuda(gauge, true, QUDA_WILSON_LINKS);
  plaqQuda();

  // ensuring mu positive
  if(mu<0)  mu*=-1.;
  QUDA_solver solver(mu);

  //Smearing
  PLEGMA_Gauge<double> smearedGauge;
  smearedGauge.APEsmearing(gauge, nsmearAPE, alphaAPE, 3);
  PLEGMA_printf("Plaquette after smearing:\n");
  smearedGauge.calculatePlaq();
  smearedGauge.unload();

  /*int nMoms = PMom>0 ? 2:1;
  if (isGPD) {
    nMoms *= 5;
  }*/
    
  for(int isource=0;isource<numSourcePositions;isource++) {
    site& source = sourcePositions[isource];
    PLEGMA_printf("\n ### Calculations for source-position %d - %02d.%02d.%02d.%02d begin now ###\n\n",
		  isource, source[0], source[1], source[2], source[3]);
    updateOptions(srcInputFile + std::to_string(isource), listOpt, add_options);

    for(int imom=0; imom<1; imom++) {

    int imom2 = imom;
    std::vector<int> sinkMom = {1,1,0};
    std::vector<int> sourceMom = {1,1,0};
    /*if(PMom) {
      sinkMom[WilsDir] = ((imom%2)*2-1) * PMom;
      sourceMom[WilsDir] = ((imom%2)*2-1) * PMom;
      imom2/=2;
    }
  
    // Compute the source and sink momenta
    if(isGPD and imom2>0) {
      int DeltaDir = (WilsDir + (imom2+1)/2) % 3;
      int DeltaSign = ((imom2+1)%2)*2-1;
      if(symm) {
	sinkMom[DeltaDir] = DeltaSign*DeltaMom/2;
	sourceMom[DeltaDir] = -DeltaSign*DeltaMom/2;
      } else {
	sourceMom[DeltaDir] = -DeltaSign*DeltaMom;
      }
    }*/

    char * src_string;
    asprintf(&src_string, "_sx%02dsy%02dsz%02dst%03d", source[0], source[1], source[2], source[3]);
    char * mom_string;
    asprintf(&mom_string, "_px%+dpy%+dpz%+d", sourceMom[0], sourceMom[1], sourceMom[2]);
    twop_filename = given_twop_filename + src_string + mom_string + ".h5";
    threep_filename = given_threep_filename + src_string + mom_string;
    free(src_string);
    free(mom_string);

    
    bool skip=true;
    bool do2pt=true;
    bool do3pt = calc3pt;
    if(isGPD and imom2==0) do3pt = false;
    
    if(access( twop_filename.c_str(), F_OK ) == -1) skip=false;
    else do2pt=false;

    if(do3pt)
      for(size_t its = 0; skip and its < tSinks.size(); its++)
	for(size_t iproj = 0; skip and iproj < Projs.size(); iproj++){
	  std::string filename = threep_filename + "_" + Projs[iproj] + "_dt" + std::to_string(tSinks[its]) + "_up.h5";
	  if(access( filename.c_str(), F_OK ) == -1) skip=false;
	  filename = threep_filename + "_" + Projs[iproj] + "_dt" + std::to_string(tSinks[its]) + "_dn.h5";
	  if(access( filename.c_str(), F_OK ) == -1) skip=false;
	}
    
    if(skip) continue;
    

    
    PLEGMA_printf("Source Momentum px %d, py %d, pz %d\nSink Momentum px %d, py %d, pz %d\n",
		  sourceMom[0],sourceMom[1],sourceMom[2],sinkMom[0],sinkMom[1],sinkMom[2]);

    momenta.clear();
    char * tmp;
    asprintf(&tmp, mom_path.c_str(), sourceMom[0],sourceMom[1],sourceMom[2]);
    pathListMomenta = tmp;
    free(tmp);
    readMomentaList();
      
    std::vector<std::vector<int>> momenta2; 
    std::vector<std::vector<int>> momenta3;
    for(int im=0;im<momenta.size();im++) {
      momentum v1 = momenta[im];
      std::vector<int> v2 = {-v1[0],-v1[1],-v1[2]};
      v2[WilsDir] += sinkMom[WilsDir];
      std::vector<int> v3 = {v1[0],v1[1],v1[2]};
      momenta2.push_back(v2);
      momenta3.push_back(v3);
    }  
  
    //Momentum smearing: put the momentum phase to smeared gauge field
    PLEGMA_Gauge<double> smearedGauge_sink(isGPD ? BOTH:NONE);
    std::complex<double> momSmScale[N_DIMS];
    std::complex<double> I(0,1);
    smearedGauge.load();
    if(isGPD){
      smearedGauge_sink.copy(smearedGauge);
      for(int i = 0 ; i < N_DIMS; i++)
	momSmScale[i] = std::exp(-(xiMomSm*2.*PI*sinkMom[i]/HGC_totalL[i])*I);
      smearedGauge_sink.scaleDirWise(momSmScale);
    } else {
      smearedGauge_sink.D_elem(smearedGauge.D_elem());
    }
    for(int i = 0 ; i < N_DIMS; i++)
      momSmScale[i] = std::exp(-(xiMomSm*2.*PI*sourceMom[i]/HGC_totalL[i])*I);
    smearedGauge.scaleDirWise(momSmScale);
    
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
				 if(prop_SS.getAllocation() != NONE) { // Smearing the solution
				   PLEGMA_Vector<double> vectorAuxD;
				   PLEGMA_Vector<float> vectorAuxF;
				   TIME(vectorAuxD.gaussianSmearing(vectorInOut, smearedGauge, nsmearGauss, alphaGauss));
				   vectorAuxF.copy(vectorAuxD);
				   prop_SS.absorb(vectorAuxF, isc/3, isc%3);
				 }
			       }
			     };

    PLEGMA_Propagator<float> propUP_SL(do3pt ? BOTH:NONE), propDN_SL(do3pt ? BOTH:NONE);

    {
      PLEGMA_Propagator<float> propUP(do2pt ? BOTH:NONE), propDN(do2pt ? BOTH:NONE);
      TIME(computePropagator(propUP, propUP_SL, mu>0 ? mu : -mu));
      TIME(computePropagator(propDN, propDN_SL, mu<0 ? mu : -mu));

      if(do2pt) {
	propUP.rotateToPhysicalBase_device(+1);
	propDN.rotateToPhysicalBase_device(-1);
	propUP.applyBoundaries_device(source[3]);
	propDN.applyBoundaries_device(source[3]);
	
	PLEGMA_Correlator<float> corr(corr_space, source, 0);
	corr.setFixMomList(momenta2);
	
	//TIME(corr.contractMesons(propUP, propDN));
	//THREAD(corr.writeFile(twop_filename, corr_file_format));
	
	TIME(corr.contractBaryons(propUP, propDN));
	THREAD(corr.writeFile(twop_filename, corr_file_format));
      }
    }
    if(do3pt){
      propUP_SL.unload();
      propDN_SL.unload();

      for(int ts=0;ts<tSinks.size();ts++) {
	
	int signPer = (tSinks[ts] + source[3]) >= HGC_totalL[3] ? -1 : +1;
	int global_fixSinkTime = (tSinks[ts] + source[3])%HGC_totalL[3]; 

	// 3D propagators at t_sink
	PLEGMA_Propagator3D<float> propUP3D;
	PLEGMA_Propagator3D<float> propDN3D;
	PLEGMA_Gauge3D<double> smearedGauge3D_sink;
	smearedGauge3D_sink.absorb(smearedGauge_sink, global_fixSinkTime);

	for(int isc = 0 ; isc < 12 ; isc++){
	  PLEGMA_Vector3D<double> vectorAuxD;
	  PLEGMA_Vector3D<double> vectorAuxD2;
	  PLEGMA_Vector3D<float> vectorAuxF;
	  vectorAuxF.absorb(propUP_SL,global_fixSinkTime,isc/3, isc%3);
	  vectorAuxD2.copy(vectorAuxF);

	  TIME(vectorAuxD.gaussianSmearing(vectorAuxD2, smearedGauge3D_sink, nsmearGauss, alphaGauss));
	  vectorAuxF.copy(vectorAuxD);
	  propUP3D.absorb(vectorAuxF,isc/3, isc%3);
	}
	for(int isc = 0 ; isc < 12 ; isc++){
	  PLEGMA_Vector3D<double> vectorAuxD;
	  PLEGMA_Vector3D<double> vectorAuxD2;
	  PLEGMA_Vector3D<float> vectorAuxF;
	  vectorAuxF.absorb(propDN_SL,global_fixSinkTime,isc/3, isc%3);
	  vectorAuxD2.copy(vectorAuxF);
	  TIME(vectorAuxD.gaussianSmearing(vectorAuxD2, smearedGauge3D_sink, nsmearGauss, alphaGauss));
	  vectorAuxF.copy(vectorAuxD);
	  propDN3D.absorb(vectorAuxF,isc/3, isc%3);
	}
	

	//seq source part 2Props and contraction block
	for(size_t iproj = 0; iproj < Projs.size(); iproj++) {
	  WHICHPROJECTOR which_proj = get_projector(Projs[iproj]);

	  auto computeThreep = [&](double run_mu, PLEGMA_Propagator3D<float>& prop13D, PLEGMA_Propagator3D<float>& prop23D, int signProps, PLEGMA_Propagator<float> &propF, std::string fl) {

	  std::string filename = threep_filename + "_" + Projs[iproj] + "_dt" + std::to_string(tSinks[ts]) + "_" + fl +".h5";
	  if(access( filename.c_str(), F_OK ) != -1) return;
				 
	  if(mu != run_mu) {
	    updateOptions(LIGHT);
	    mu = run_mu;
	    solver.UpdateSolver();
	  }

	  PLEGMA_Propagator<float> seqPropOut;

	  for(int nu = 0 ; nu < 4 ; nu++)
	    for(int c2 = 0 ; c2 < 3 ; c2++){
	      PLEGMA_Vector3D<float> vectorAux3D;
	      PLEGMA_Vector3D<double> vectorAux3D_1, vectorAux3D_2;
              PLEGMA_Vector<float> vectorAuxF;
              PLEGMA_Vector<double> vectorInOut;

	      if(&prop13D != &prop23D)
		vectorAux3D.seqSourceNucleon(prop13D, prop23D, which_proj, nucleon, nu, c2);
	      else
		vectorAux3D.seqSourceNucleon(prop13D, which_proj, nucleon, nu, c2);
	      
	      vectorAux3D.mulMomentumPhases(sinkMom,-1); // put momentum at the sink
	      std::complex<float> Isingle(0,1);
	      float phase = 2.*PI*(((float) sinkMom[0] * sourcePositions[isource][0])/HGC_totalL[0]
				   + ((float)sinkMom[1] * sourcePositions[isource][1])/HGC_totalL[1]
				   + ((float)sinkMom[2] * sourcePositions[isource][2])/HGC_totalL[2]);
	      vectorAux3D.cscale(std::exp<float>(+phase*Isingle)); // put momentum from the point source
	      vectorAux3D.conjugate();
	      vectorAux3D.apply_gamma(G5);
	      vectorAux3D_1.copy(vectorAux3D);
	      vectorAux3D_2.gaussianSmearing(vectorAux3D_1, smearedGauge3D_sink, nsmearGauss, alphaGauss);
	      vectorInOut.absorb(vectorAux3D_2, global_fixSinkTime);
	      
	      double norm = vectorInOut.norm();
	      vectorInOut.cscale(1/norm);
	      solver.solve(vectorInOut, vectorInOut);
	      vectorInOut.cscale(norm);    
	      vectorAuxF.copy(vectorInOut);
	      seqPropOut.absorb(vectorAuxF, nu, c2);
	    }
	  
	  seqPropOut.apply_gamma(G5);
	  seqPropOut.conjugate();
    
	  PLEGMA_Gauge<float> gaugeWL;
	  gaugeWL.copy(gauge);

	  PLEGMA_Su3field<float> su3;
	  PLEGMA_Su3field<float> WL;
	  PLEGMA_Su3field<float> tmp;
	  
	  PLEGMA_Propagator<float> propTmp;
	  PLEGMA_Correlator<float> corrThrpWL(corr_space,source,0,tSinks[ts]+1);
	  corrThrpWL.setFixMomList(momenta3);

	  if(minStout>0) gaugeWL.stoutSmearing(gaugeWL,minStout,rhoStout,3);
	  for(int stIt=0; stIt<=(int)((maxStout-minStout)/stepStout);stIt++){
	    if(stIt>0) gaugeWL.stoutSmearing(gaugeWL,stepStout,rhoStout,3);
	    
	    for(int plus=0; plus<2; plus++) {
	      PLEGMA_Propagator<float> *prop1=&propF, *prop2=&propTmp, *propEx;
	      int dir = plus ? (4+WilsDir):WilsDir;
	      
	      su3.absorbDir_device(gaugeWL, WilsDir);
	      WL.setUnit( (std::vector<int>) {0,4,8});
	      for(int i = 0 ; i < maxZ;i++){
		if(not (plus>0 and i==0)) { // doing z=0 only once
		  TIME(corrThrpWL.contractNucleonThrp_wilsonLine(seqPropOut, *prop1, WL, signProps, gammas));
		  if(signPer < 0) for(int iv = 0 ; iv < corrThrpWL.getTotalSize()*2; iv++) corrThrpWL.H_elem()[iv] *= signPer;
		  char *dset;
		  asprintf(&dset, "stout%d/z%+02d", minStout+stepStout*stIt, (2*plus-1)*i);
		  corrThrpWL.setDatasets((std::vector<std::string>) {dset});
		  TIME(corrThrpWL.writeFile( filename, corr_file_format ));
		  free(dset);
		}
		TIME(WL.wilsonLineUpdate(su3, tmp, dir));
		
		TIME(prop2->shift(*prop1, dir));

		// Exchanging pointers between prop1 and prop2
		propEx = prop2; prop2 = prop1; prop1 = propEx;
	      }
	      propF.load();
	    }
	  }
			       };
	  
	  if(nucleon == PROTON) {
	    TIME(computeThreep(-mu_ud, propUP3D, propDN3D, +1, propUP_SL, "up"));
	    TIME(computeThreep( mu_ud, propUP3D, propUP3D, -1, propDN_SL, "dn"));
	  } else {
	    TIME(computeThreep( mu_ud, propDN3D, propUP3D, -1, propDN_SL, "dn"));
	    TIME(computeThreep(-mu_ud, propDN3D, propDN3D, +1, propUP_SL, "up"));
	  }
	}
      }
    }
    }
  }
  }
  finalize();
  return 0;
}

