#include <PLEGMA.h>
#include <PLEGMA_utils.h>

std::vector<double> runtime;
#define TIME(fnc)  runtime.push_back(MPI_Wtime()); fnc;			\
  PLEGMA_printf("TIME for "#fnc" %f sec\n", MPI_Wtime()-runtime.back()); \
  runtime.pop_back()

using namespace plegma;
using namespace quda;

int main(int argc, char **argv)
{

  static std::vector<std::string> listOpt = {"verbosity", "load-gauge", "nsmear-APE", "alpha-APE", "nsmear-gauss", "alpha-gauss",
					     "nsrc", "src-filename", "maxQsq", "twop-filename","threep-filename",  "corr-file-format",
					     "corr-space", "tSinks","xiMomSm","gammas"};

  initializeOptions(argc, argv, true, listOpt);

  if(mu<0){
    mu *= -1.0;
  }
  
  double mu_ud = mu;
  
  double mu_s;
  HGC_options->set("mu-s", "mu for strange quark", verbosity, mu_s);
  
  int L_LEN = 0;
  HGC_options->set("l-len", "length of l", verbosity, L_LEN);

  int L_DIR = 2;
  HGC_options->set("l-dir", "direction of l", verbosity, L_DIR);

  int B_LEN = 0;
  HGC_options->set("b-len", "length of b", verbosity, B_LEN);

  int Z_LEN = 0;
  HGC_options->set("z-len", "length of z", verbosity, Z_LEN);

  int nStout;
  HGC_options->set("n_stout", "n parameter stout smearing", verbosity, nStout);

  double rhoStout;
  HGC_options->set("rho_stout", "Rho parameter stout smearing", verbosity, rhoStout);

  bool calc3pt = true;
  HGC_options->set("calc3pt", "If true then the 3pt function is computed", verbosity, calc3pt);

  std::string conf;
  HGC_options->set("conf", "configuration", verbosity, conf);

  std::string twop_filename_1;
  HGC_options->set("twop-filename-1", "2p filename for kaon +", verbosity, twop_filename_1);
  
  std::string threep_filename_1;
  HGC_options->set("threep-filename-1", "3p filename for kaon +", verbosity, threep_filename_1);

  std::vector<int> sinkMom(4);
  HGC_options->set("sinkMom", "momentum at the sink", verbosity, sinkMom);
  
  initializePLEGMA();

  int B_DIR1, B_DIR2;
  switch(L_DIR) {
    case 0:
      B_DIR1 = 1;
      B_DIR2 = 2;
      break;
    case 1:
      B_DIR1 = 0;
      B_DIR2 = 2;
      break;
    case 2:
      B_DIR1 = 0;
      B_DIR2 = 1;
      break;
    default:
      PLEGMA_error("The direction of the WIlson line has to be smaller than 3");
  }
  
  // Reading from Lime file and loading to device
  PLEGMA_Gauge<double> gauge;
  gauge.readFile(latfile, LIME_FORMAT);
  gauge.calculatePlaq();

  // Loading to QUDA and computing plaquette also there
  initGaugeQuda(gauge, true, QUDA_WILSON_LINKS);
  plaqQuda();
 
  // Smearing
  PLEGMA_Gauge<double> smearedGauge;
  PLEGMA_Gauge<double> smearedGaugeMn;
  smearedGauge.APEsmearing(gauge, nsmearAPE, alphaAPE, 3);
  PLEGMA_printf("Plaquette after smearing:\n");
  smearedGauge.calculatePlaq();
  smearedGaugeMn.copy(smearedGauge);
  
  //Momentum smearing: put the momentum phase to smeared gauge field
  std::complex<double> momSmScale[N_DIMS];
  std::complex<double> momSmScaleMn[N_DIMS];
  std::complex<double> I(0,1);
  for(int i = 0 ; i < N_DIMS; i++){
    momSmScale[i] = std::exp(-(xiMomSm*2.*PI*sinkMom[i]/HGC_totalL[i])*I);
    momSmScaleMn[i] = std::exp(+(xiMomSm*2.*PI*sinkMom[i]/HGC_totalL[i])*I);
  }
  smearedGauge.scaleDirWise(momSmScale);
  smearedGaugeMn.scaleDirWise(momSmScaleMn);

  // Extracting the spacial sink momentum from the sink 4-momentum  
  std::vector<int> sinkMom_3D(3);
  std::copy(sinkMom.begin(),sinkMom.begin()+3,sinkMom_3D.begin());
  
  // ensuring mu positive
  if(mu<0)  mu*=-1.;
  updateOptions(LIGHT);
  TIME(QUDA_solver *solver = new QUDA_solver(mu));

  PLEGMA_Vector<double> vectorIn;
  PLEGMA_Vector<double> vectorOut;
  PLEGMA_Vector<double> vectorAuxD;
  PLEGMA_Vector<double> vectorAuxD21;
  PLEGMA_Vector<float> vectorAuxF;
  
  PLEGMA_Propagator<float> *propUP_smearplus = new PLEGMA_Propagator<float>(BOTH);
  PLEGMA_Propagator<float> *propUP_smearminus = new PLEGMA_Propagator<float>(BOTH);
  PLEGMA_Propagator<float> *propST_smearplus = new PLEGMA_Propagator<float>(BOTH);
  PLEGMA_Propagator<float> *propST_smearminus = new PLEGMA_Propagator<float>(BOTH);
  PLEGMA_Propagator<float> *seqProp_pionUPplusmom = new PLEGMA_Propagator<float>(BOTH);
  PLEGMA_Propagator<float> *seqProp_pionUPminusmom = new PLEGMA_Propagator<float>(BOTH);
  PLEGMA_Propagator<float> *seqProp_kaonUPplusmom = new PLEGMA_Propagator<float>(BOTH);
  PLEGMA_Propagator<float> *seqProp_kaonUPminusmom = new PLEGMA_Propagator<float>(BOTH);
  PLEGMA_Propagator<float> *seqProp_kaonSTplusmom = new PLEGMA_Propagator<float>(BOTH);
  PLEGMA_Propagator<float> *seqProp_kaonSTminusmom = new PLEGMA_Propagator<float>(BOTH);

  PLEGMA_Propagator<float> *propF_UPsmearplus_1 = new PLEGMA_Propagator<float>(BOTH);
  PLEGMA_Propagator<float> *propF_UPsmearminus_1 = new PLEGMA_Propagator<float>(BOTH);
  PLEGMA_Propagator<float> *propF_STsmearplus_1 = new PLEGMA_Propagator<float>(BOTH);
  PLEGMA_Propagator<float> *propF_STsmearminus_1 = new PLEGMA_Propagator<float>(BOTH);
  PLEGMA_Propagator<float> *propF_UPsmearplus_2 = new PLEGMA_Propagator<float>(BOTH);
  PLEGMA_Propagator<float> *propF_UPsmearminus_2 = new PLEGMA_Propagator<float>(BOTH);
  PLEGMA_Propagator<float> *propF_STsmearplus_2 = new PLEGMA_Propagator<float>(BOTH);
  PLEGMA_Propagator<float> *propF_STsmearminus_2 = new PLEGMA_Propagator<float>(BOTH);

  PLEGMA_Propagator<float> *propIn = new PLEGMA_Propagator<float>(BOTH);
  PLEGMA_Propagator<float> *propExchange = nullptr;
  
  PLEGMA_Propagator3D<float> propUP3D_smearplus;
  PLEGMA_Propagator3D<float> propUP3D_smearminus;
  PLEGMA_Propagator3D<float> propST3D_smearplus;
  PLEGMA_Propagator3D<float> propST3D_smearminus;

  PLEGMA_Gauge<float> gaugeWL;
  gaugeWL.copy(gauge);
  gaugeWL.stoutSmearing(gaugeWL,nStout,rhoStout,3);

  PLEGMA_Su3field<float> su3;
  PLEGMA_Su3field<float> *su3_1 = new PLEGMA_Su3field<float>(BOTH);
  PLEGMA_Su3field<float> *su3_2_1 = new PLEGMA_Su3field<float>(BOTH);
  PLEGMA_Su3field<float> *su3_2_2 = new PLEGMA_Su3field<float>(BOTH);
  PLEGMA_Su3field<float> *su3_3_1 = new PLEGMA_Su3field<float>(BOTH);
  PLEGMA_Su3field<float> *su3_3_2 = new PLEGMA_Su3field<float>(BOTH);
  PLEGMA_Su3field<float> *su3_4_1 = new PLEGMA_Su3field<float>(BOTH);
  PLEGMA_Su3field<float> *su3_4_2 = new PLEGMA_Su3field<float>(BOTH);
  PLEGMA_Su3field<float> *WL_1 = new PLEGMA_Su3field<float>(BOTH);
  PLEGMA_Su3field<float> *WL_2_1 = new PLEGMA_Su3field<float>(BOTH);
  PLEGMA_Su3field<float> *WL_3_1 = new PLEGMA_Su3field<float>(BOTH);
  PLEGMA_Su3field<float> *WL_2_2 = new PLEGMA_Su3field<float>(BOTH);
  PLEGMA_Su3field<float> *WL_3_2 = new PLEGMA_Su3field<float>(BOTH);
  PLEGMA_Su3field<float> *WL_4_1 = new PLEGMA_Su3field<float>(BOTH);
  PLEGMA_Su3field<float> *WL_4_2 = new PLEGMA_Su3field<float>(BOTH);
  PLEGMA_Su3field<float> *WL_temp = new PLEGMA_Su3field<float>(BOTH);
  PLEGMA_Su3field<float> *WL_temp1 = new PLEGMA_Su3field<float>(BOTH);
  PLEGMA_Su3field<float> *WL_temp2 = new PLEGMA_Su3field<float>(BOTH);
  PLEGMA_Su3field<float> *su3_in = new PLEGMA_Su3field<float>(BOTH);
  PLEGMA_Su3field<float> *WL1 = new PLEGMA_Su3field<float>(BOTH);
  PLEGMA_Su3field<float> *WL2 = new PLEGMA_Su3field<float>(BOTH);
  PLEGMA_Su3field<float> *su3_exchange = new PLEGMA_Su3field<float>(BOTH);
  PLEGMA_Su3field<float> tmp(BOTH);
  
  for(int isource=0;isource<numSourcePositions;isource++){

  auto computePropagator = [&](PLEGMA_Propagator<float>& prop_SL, double run_mu, WHICHFLAVOR fl,PLEGMA_Gauge<double>& smeared_Gauge, int nsmear) {
				 // ensuring mu value
				 if(mu != run_mu) {
				   //updateOptions(fl);
				   mu = run_mu;
				   solver->UpdateSolver();
				 }
				 
				 PLEGMA_Vector<double> vectorIn1;
				 PLEGMA_Vector<double> vectorOut1;
				 PLEGMA_Vector<double> vectorAuxD1;
				 PLEGMA_Vector<float> vectorAuxF1;

				 for(int isc = 0 ; isc < 12 ; isc++){
				   vectorAuxD1.pointSource(sourcePositions[isource], isc/3, isc%3, DEVICE);
				   vectorIn1.gaussianSmearing(vectorAuxD1, smeared_Gauge, nsmear, alphaGauss);
				   PLEGMA_printf("Going to invert for component %d\n", isc);
				   solver->solve(vectorOut1, vectorIn1);
				   vectorAuxF1.copy(vectorOut1);
				   prop_SL.absorb(vectorAuxF1, isc/3, isc%3);
				 }
			       };

  TIME(computePropagator(*propUP_smearplus, mu_ud, LIGHT, smearedGauge ,nsmearGauss));
  TIME(computePropagator(*propUP_smearminus , mu_ud, LIGHT, smearedGaugeMn ,nsmearGauss));
  TIME(computePropagator(*propST_smearplus, mu_s, STRANGE, smearedGauge ,nsmearGauss));
  TIME(computePropagator(*propST_smearminus , mu_s, STRANGE, smearedGaugeMn ,nsmearGauss));
  
  for(int ts=0;ts<tSinks.size();ts++)
    {

      PLEGMA_Correlator<float> corrThrpWL(corr_space, sourcePositions[isource], 0, tSinks[ts]+1);

      int signPer = (tSinks[ts] + sourcePositions[isource][3]) >= HGC_totalL[3] ? -1 : +1;
      int global_fixSinkTime = (tSinks[ts] + sourcePositions[isource][3])%HGC_totalL[3]; 
      int my_fixSinkTime = global_fixSinkTime - HGC_procPosition[3] * HGC_localL[3];
      bool is_myST = (my_fixSinkTime >= 0) && ( my_fixSinkTime < HGC_localL[3] );

      for(int isc = 0 ; isc < 12 ; isc++){
        vectorAuxF.absorb(*propUP_smearplus,isc/3,isc%3);
        vectorAuxD21.copy(vectorAuxF);
        vectorAuxD.gaussianSmearing(vectorAuxD21, smearedGauge , nsmearGauss, alphaGauss);
        vectorAuxF.copy(vectorAuxD);
        propUP3D_smearplus.absorb(vectorAuxF,global_fixSinkTime, isc/3, isc%3);
      }

      for(int isc = 0 ; isc < 12 ; isc++){
        vectorAuxF.absorb(*propUP_smearminus,isc/3,isc%3);
        vectorAuxD21.copy(vectorAuxF);
        vectorAuxD.gaussianSmearing(vectorAuxD21, smearedGaugeMn , nsmearGauss, alphaGauss);
        vectorAuxF.copy(vectorAuxD);
        propUP3D_smearminus.absorb(vectorAuxF,global_fixSinkTime, isc/3, isc%3);
      }

      for(int isc = 0 ; isc < 12 ; isc++){
        vectorAuxF.absorb(*propST_smearplus,isc/3,isc%3);
        vectorAuxD21.copy(vectorAuxF);
        vectorAuxD.gaussianSmearing(vectorAuxD21, smearedGauge , nsmearGauss, alphaGauss);
        vectorAuxF.copy(vectorAuxD);
        propST3D_smearplus.absorb(vectorAuxF,global_fixSinkTime, isc/3, isc%3);
      }

      for(int isc = 0 ; isc < 12 ; isc++){
        vectorAuxF.absorb(*propST_smearminus,isc/3,isc%3);
        vectorAuxD21.copy(vectorAuxF);
        vectorAuxD.gaussianSmearing(vectorAuxD21, smearedGaugeMn , nsmearGauss, alphaGauss);
        vectorAuxF.copy(vectorAuxD);
        propST3D_smearminus.absorb(vectorAuxF,global_fixSinkTime, isc/3, isc%3);
      }
      
      if(calc3pt){
	
	propUP_smearplus->unload();
	propUP_smearminus->unload();
	propST_smearplus->unload();
        propST_smearminus->unload();
	
	auto computeSequentialPropagator = [&](PLEGMA_Propagator<float>& seqPropOUT,PLEGMA_Propagator3D<float>& prop1, double run_mu, WHICHFLAVOR fl,
					       PLEGMA_Gauge<double>& smeared_Gauge, int nsmear, int sign_mom) {

	                                         PLEGMA_Vector3D<float> vectorAux3D;
						 PLEGMA_Vector<float> vectorAuxF3;
						 PLEGMA_Vector<double> vectorAuxD3;
						 PLEGMA_Vector<double> vectorIn3;
						 PLEGMA_Vector<double> vectorOut3;
	    
	                                         for(int nu = 0 ; nu < 4 ; nu++)
                                                   for(int c2 = 0 ; c2 < 3 ; c2++){
						     vectorAux3D.absorb(prop1,nu,c2);
						     vectorAux3D.mulMomentumPhases(sinkMom,sign_mom); //put momentum at the sink
						     std::complex<float> Isingle(0,1);
						     float phase = 2.*PI*(((float) sinkMom[0] * sourcePositions[isource][0])/HGC_totalL[0]
									  + ((float)sinkMom[1] * sourcePositions[isource][1])/HGC_totalL[1]
									  + ((float)sinkMom[2] * sourcePositions[isource][2])/HGC_totalL[2]);
						     vectorAux3D.cscale(std::exp<float>(-sign_mom*phase*Isingle)); // put momentum from the point source
						     vectorAux3D.apply_gamma(G5);
						     vectorAuxF3.absorb(vectorAux3D, global_fixSinkTime);
						     vectorAuxD3.copy(vectorAuxF3);
						     vectorIn3.gaussianSmearing(vectorAuxD3,smeared_Gauge, nsmear, alphaGauss);
						     // check if we need to normalize the seqsource for mix precision solver
						     if(mu!=run_mu) {
						       //updateOptions(fl);
						       mu=run_mu;
						       solver->UpdateSolver();
						     }
						     double norm = vectorIn3.norm();
						     vectorIn3.cscale(1/norm);
						     solver->solve(vectorOut3, vectorIn3);
						     vectorOut3.cscale(norm);
						     vectorAuxF3.copy(vectorOut3);
						     seqPropOUT.absorb(vectorAuxF3, nu, c2);
						     
						   }
						 seqPropOUT.apply_gamma(G5);
						 seqPropOUT.conjugate();
						   
					       };
	
	TIME(computeSequentialPropagator(*seqProp_pionUPplusmom, propUP3D_smearminus, -mu_ud, LIGHT, smearedGauge, nsmearGauss, +1));
	TIME(computeSequentialPropagator(*seqProp_pionUPminusmom, propUP3D_smearplus, -mu_ud, LIGHT, smearedGaugeMn, nsmearGauss, -1));
        TIME(computeSequentialPropagator(*seqProp_kaonUPplusmom, propST3D_smearminus, -mu_ud, LIGHT, smearedGauge, nsmearGauss, +1));
        TIME(computeSequentialPropagator(*seqProp_kaonUPminusmom, propST3D_smearplus, -mu_ud, LIGHT, smearedGaugeMn, nsmearGauss, -1));
        TIME(computeSequentialPropagator(*seqProp_kaonSTplusmom, propUP3D_smearminus, mu_s, STRANGE, smearedGauge, nsmearGauss, +1));
        TIME(computeSequentialPropagator(*seqProp_kaonSTminusmom, propUP3D_smearplus, mu_s, STRANGE, smearedGaugeMn, nsmearGauss, -1));
	
	propF_UPsmearplus_1->copy(*propUP_smearplus,BOTH);
        propF_UPsmearminus_1->copy(*propUP_smearminus,BOTH);
	propF_STsmearplus_1->copy(*propST_smearplus,BOTH);
        propF_STsmearminus_1->copy(*propST_smearminus,BOTH);
        propF_UPsmearplus_2->copy(*propUP_smearplus,BOTH);
        propF_UPsmearminus_2->copy(*propUP_smearminus,BOTH);
        propF_STsmearplus_2->copy(*propST_smearplus,BOTH);
        propF_STsmearminus_2->copy(*propST_smearminus,BOTH);
	  
	WL_1->setUnit( (std::vector<int>) {0,4,8});
	WL_2_1->setUnit( (std::vector<int>) {0,4,8});
	WL_3_1->setUnit( (std::vector<int>) {0,4,8});
	WL_4_1->setUnit( (std::vector<int>) {0,4,8});
	WL_2_2->setUnit( (std::vector<int>) {0,4,8});
	WL_3_2->setUnit( (std::vector<int>) {0,4,8});
	WL_4_2->setUnit( (std::vector<int>) {0,4,8});

	//absorb the gauge directions for building the 4 branches
	su3_1->absorbDir_device(gaugeWL, L_DIR);
	su3_2_1->absorbDir_device(gaugeWL, B_DIR1);
	su3_2_2->absorbDir_device(gaugeWL, B_DIR2);
	su3_3_1->absorbDir_device(gaugeWL, L_DIR);
	su3_3_2->absorbDir_device(gaugeWL, L_DIR);
	su3_4_1->absorbDir_device(gaugeWL, L_DIR);
	su3_4_2->absorbDir_device(gaugeWL, L_DIR);

	size_t bval, zval;
	bool WL2_rev = true;
	int WL2_dir = 0;
	bool WL4_rev = true;
	int WL4_dir = 1;

	for(int l = 0; l <= L_LEN; l++) {//loop over values of L, building "1" and "3"

	  if(l != 0) {

	    WL_1->wilsonLineUpdate(*su3_1, tmp, 4 + L_DIR);
	    WL_3_1->wilsonLineUpdate(*su3_3_1, tmp, 4 + L_DIR, true);
	    WL_3_2->wilsonLineUpdate(*su3_3_2, tmp, 4 + L_DIR, true);

	    su3_exchange = su3_in; su3_in = su3_2_1; su3_2_1 = su3_exchange;
	    su3_2_1->shift(*su3_in, 4 + L_DIR);

	    su3_exchange = su3_in; su3_in = su3_2_2; su3_2_2 = su3_exchange;
	    su3_2_2->shift(*su3_in, 4 + L_DIR);

	    su3_exchange = su3_in; su3_in = WL_2_1; WL_2_1 = su3_exchange;
	    WL_2_1->shift(*su3_in, 4 + L_DIR);

	    su3_exchange = su3_in; su3_in = WL_2_2; WL_2_2 = su3_exchange;
	    WL_2_2->shift(*su3_in, 4 + L_DIR);

	  }

	  if(WL2_rev) {
	    WL2_rev = false;
	    WL2_dir = 1;
	  }else {
	    WL2_rev = true;
	    WL2_dir = 0;
	  }

	  for(int b = 0; b <= B_LEN; b++) {//loop over values of B, building "2"

	    if(b != 0) {

	      propExchange = propIn; propIn = propF_UPsmearplus_1; propF_UPsmearplus_1 = propExchange;
	      propF_UPsmearplus_1->shift(*propIn, 4*WL2_dir + B_DIR1);

	      propExchange = propIn; propIn = propF_UPsmearminus_1; propF_UPsmearminus_1 = propExchange;
              propF_UPsmearminus_1->shift(*propIn, 4*WL2_dir + B_DIR1);

	      propExchange = propIn; propIn = propF_STsmearplus_1; propF_STsmearplus_1 = propExchange;
              propF_STsmearplus_1->shift(*propIn, 4*WL2_dir + B_DIR1);

              propExchange = propIn; propIn = propF_STsmearminus_1; propF_STsmearminus_1 = propExchange;
              propF_STsmearminus_1->shift(*propIn, 4*WL2_dir + B_DIR1);

	      WL_2_1->wilsonLineUpdate(*su3_2_1, tmp, 4*WL2_dir + B_DIR1);

	      propExchange = propIn; propIn = propF_UPsmearplus_2; propF_UPsmearplus_2 = propExchange;
              propF_UPsmearplus_2->shift(*propIn, 4*WL2_dir + B_DIR2);

              propExchange = propIn; propIn = propF_UPsmearminus_2; propF_UPsmearminus_2 = propExchange;
              propF_UPsmearminus_2->shift(*propIn, 4*WL2_dir + B_DIR2);

              propExchange = propIn; propIn = propF_STsmearplus_2; propF_STsmearplus_2 = propExchange;
              propF_STsmearplus_2->shift(*propIn, 4*WL2_dir + B_DIR2);

              propExchange = propIn; propIn = propF_STsmearminus_2; propF_STsmearminus_2 = propExchange;
              propF_STsmearminus_2->shift(*propIn, 4*WL2_dir + B_DIR2);

	      WL_2_2->wilsonLineUpdate(*su3_2_2, tmp, 4*WL2_dir + B_DIR2);

	      su3_exchange = su3_in; su3_in = su3_3_1; su3_3_1 = su3_exchange;
	      su3_3_1->shift(*su3_in, 4*WL2_dir + B_DIR1);

	      su3_exchange = su3_in; su3_in = su3_3_2; su3_3_2 = su3_exchange;
	      su3_3_2->shift(*su3_in, 4*WL2_dir + B_DIR2);

              su3_exchange = su3_in; su3_in = su3_4_1; su3_4_1 = su3_exchange;
	      su3_4_1->shift(*su3_in, 4*WL2_dir + B_DIR1);

	      su3_exchange = su3_in; su3_in = su3_4_2; su3_4_2 = su3_exchange;
	      su3_4_2->shift(*su3_in, 4*WL2_dir + B_DIR2);

	      su3_exchange = su3_in; su3_in = WL_3_1; WL_3_1 = su3_exchange;
	      WL_3_1->shift(*su3_in, 4*WL2_dir + B_DIR1);

	      su3_exchange = su3_in; su3_in = WL_3_2; WL_3_2 = su3_exchange;
	      WL_3_2->shift(*su3_in, 4*WL2_dir + B_DIR2);

	      su3_exchange = su3_in; su3_in = WL_4_1; WL_4_1 = su3_exchange;
	      WL_4_1->shift(*su3_in, 4*WL2_dir + B_DIR1);

	      su3_exchange = su3_in; su3_in = WL_4_2; WL_4_2 = su3_exchange;
	      WL_4_2->shift(*su3_in, 4*WL2_dir + B_DIR2);

	    }

	    WL_temp->UxU(*WL_1, *WL_2_1);
	    WL_temp1->UxU(*WL_temp, *WL_3_1);

	    WL_temp->UxU(*WL_1, *WL_2_2);
	    WL_temp2->UxU(*WL_temp, *WL_3_2);

	    if(WL4_rev) {
	      WL4_rev = false;
	      WL4_dir = 0;
	    }else {
	      WL4_rev = true;
	      WL4_dir = 1;
	    }

	    for(int z = 0; z <= Z_LEN; z++) {//loop over values of Z, building "4"

	      if(z != 0) {

		propExchange = propIn; propIn = propF_UPsmearplus_1; propF_UPsmearplus_1 = propExchange;
                propF_UPsmearplus_1->shift(*propIn, 4*WL4_dir + L_DIR);

                propExchange = propIn; propIn = propF_UPsmearminus_1; propF_UPsmearminus_1 = propExchange;
                propF_UPsmearminus_1->shift(*propIn, 4*WL4_dir + L_DIR);

                propExchange = propIn; propIn = propF_STsmearplus_1; propF_STsmearplus_1 = propExchange;
                propF_STsmearplus_1->shift(*propIn, 4*WL4_dir + L_DIR);

                propExchange = propIn; propIn = propF_STsmearminus_1; propF_STsmearminus_1 = propExchange;
                propF_STsmearminus_1->shift(*propIn, 4*WL4_dir + L_DIR);

		WL_4_1->wilsonLineUpdate(*su3_4_1, tmp, 4*WL4_dir + L_DIR);

		propExchange = propIn; propIn = propF_UPsmearplus_2; propF_UPsmearplus_2 = propExchange;
                propF_UPsmearplus_2->shift(*propIn, 4*WL4_dir + L_DIR);

                propExchange = propIn; propIn = propF_UPsmearminus_2; propF_UPsmearminus_2 = propExchange;
                propF_UPsmearminus_2->shift(*propIn, 4*WL4_dir + L_DIR);

                propExchange = propIn; propIn = propF_STsmearplus_2; propF_STsmearplus_2 = propExchange;
                propF_STsmearplus_2->shift(*propIn, 4*WL4_dir + L_DIR);

                propExchange = propIn; propIn = propF_STsmearminus_2; propF_STsmearminus_2 = propExchange;
                propF_STsmearminus_2->shift(*propIn, 4*WL4_dir + L_DIR);

		WL_4_2->wilsonLineUpdate(*su3_4_2, tmp, 4*WL4_dir + L_DIR);

	      }

	      WL1->UxU(*WL_temp1, *WL_4_1);
	      WL2->UxU(*WL_temp2, *WL_4_2);

	      if(WL2_rev) bval = B_LEN - b;
	      else bval = b;

	      if(WL4_rev) zval = Z_LEN - z;
	      else zval = z;
	      
	      TIME(corrThrpWL.contractNucleonThrp_staple(*seqProp_pionUPplusmom, *propF_UPsmearplus_1, *WL1, +1, gammas, false, bval, -l, -zval));
	      corrThrpWL.writeFile((threep_filename+"_ts"+std::to_string(tSinks[ts])+"_pionUP_plusmom_B_DIR1_"+conf).c_str(), corr_file_format);

	      TIME(corrThrpWL.contractNucleonThrp_staple(*seqProp_pionUPplusmom, *propF_UPsmearplus_2, *WL2, +1, gammas, false, bval, -l, -zval));
	      corrThrpWL.writeFile((threep_filename+"_ts"+std::to_string(tSinks[ts])+"_pionUP_plusmom_B_DIR2_"+conf).c_str(), corr_file_format);

	      TIME(corrThrpWL.contractNucleonThrp_staple(*seqProp_pionUPminusmom, *propF_UPsmearminus_1, *WL1, +1, gammas, false, bval, -l, -zval));
              corrThrpWL.writeFile((threep_filename+"_ts"+std::to_string(tSinks[ts])+"_pionUP_minusmom_B_DIR1_"+conf).c_str(), corr_file_format);

              TIME(corrThrpWL.contractNucleonThrp_staple(*seqProp_pionUPminusmom, *propF_UPsmearminus_2, *WL2, +1, gammas, false, bval, -l, -zval));
              corrThrpWL.writeFile((threep_filename+"_ts"+std::to_string(tSinks[ts])+"_pionUP_minusmom_B_DIR2_"+conf).c_str(), corr_file_format);

	      TIME(corrThrpWL.contractNucleonThrp_staple(*seqProp_kaonUPplusmom, *propF_UPsmearplus_1, *WL1, +1, gammas, false, bval, -l, -zval));
	      corrThrpWL.writeFile((threep_filename_1+"_ts"+std::to_string(tSinks[ts])+"_kaonUP_plusmom_B_DIR1_"+conf).c_str(), corr_file_format);
	      
	      TIME(corrThrpWL.contractNucleonThrp_staple(*seqProp_kaonUPplusmom, *propF_UPsmearplus_2, *WL2, +1, gammas, false, bval, -l, -zval));
	      corrThrpWL.writeFile((threep_filename_1+"_ts"+std::to_string(tSinks[ts])+"_kaonUP_plusmom_B_DIR2_"+conf).c_str(), corr_file_format);

	      TIME(corrThrpWL.contractNucleonThrp_staple(*seqProp_kaonUPminusmom, *propF_UPsmearminus_1, *WL1, +1, gammas, false, bval, -l, -zval));
              corrThrpWL.writeFile((threep_filename_1+"_ts"+std::to_string(tSinks[ts])+"_kaonUP_minusmom_B_DIR1_"+conf).c_str(), corr_file_format);

              TIME(corrThrpWL.contractNucleonThrp_staple(*seqProp_kaonUPminusmom, *propF_UPsmearminus_2, *WL2, +1, gammas, false, bval, -l, -zval));
              corrThrpWL.writeFile((threep_filename_1+"_ts"+std::to_string(tSinks[ts])+"_kaonUP_minusmom_B_DIR2_"+conf).c_str(), corr_file_format);

	      TIME(corrThrpWL.contractNucleonThrp_staple(*seqProp_kaonSTplusmom, *propF_STsmearplus_1, *WL1, +1, gammas, false, bval, -l, -zval));
              corrThrpWL.writeFile((threep_filename_1+"_ts"+std::to_string(tSinks[ts])+"_kaonST_plusmom_B_DIR1_"+conf).c_str(), corr_file_format);

              TIME(corrThrpWL.contractNucleonThrp_staple(*seqProp_kaonSTplusmom, *propF_STsmearplus_2, *WL2, +1, gammas, false, bval, -l, -zval));
              corrThrpWL.writeFile((threep_filename_1+"_ts"+std::to_string(tSinks[ts])+"_kaonST_plusmom_B_DIR2_"+conf).c_str(), corr_file_format);

              TIME(corrThrpWL.contractNucleonThrp_staple(*seqProp_kaonSTminusmom, *propF_STsmearminus_1, *WL1, +1, gammas, false, bval, -l, -zval));
              corrThrpWL.writeFile((threep_filename_1+"_ts"+std::to_string(tSinks[ts])+"_kaonST_minusmom_B_DIR1_"+conf).c_str(), corr_file_format);

              TIME(corrThrpWL.contractNucleonThrp_staple(*seqProp_kaonSTminusmom, *propF_STsmearminus_2, *WL2, +1, gammas, false, bval, -l, -zval));
              corrThrpWL.writeFile((threep_filename_1+"_ts"+std::to_string(tSinks[ts])+"_kaonST_minusmom_B_DIR2_"+conf).c_str(), corr_file_format);

	    }//loop over values of Z, building "4"
          }//loop over values of B, building "2"
	}//loop over values of L, building "1" and "3"

	propF_UPsmearplus_1->copy(*propUP_smearplus,BOTH);
        propF_UPsmearminus_1->copy(*propUP_smearminus,BOTH);
        propF_STsmearplus_1->copy(*propST_smearplus,BOTH);
        propF_STsmearminus_1->copy(*propST_smearminus,BOTH);
        propF_UPsmearplus_2->copy(*propUP_smearplus,BOTH);
        propF_UPsmearminus_2->copy(*propUP_smearminus,BOTH);
        propF_STsmearplus_2->copy(*propST_smearplus,BOTH);
        propF_STsmearminus_2->copy(*propST_smearminus,BOTH);
	  
	//now repeat the same thing only going in negative boost direction for the staple
	//set the 4 branches as unit
	WL_1->setUnit( (std::vector<int>) {0,4,8});
	WL_2_1->setUnit( (std::vector<int>) {0,4,8});
	WL_3_1->setUnit( (std::vector<int>) {0,4,8});
	WL_4_1->setUnit( (std::vector<int>) {0,4,8});
	WL_2_2->setUnit( (std::vector<int>) {0,4,8});
	WL_3_2->setUnit( (std::vector<int>) {0,4,8});
	WL_4_2->setUnit( (std::vector<int>) {0,4,8});

	//absorb the gauge directions for building the 4 branches
	su3_1->absorbDir_device(gaugeWL, L_DIR);
	su3_2_1->absorbDir_device(gaugeWL, B_DIR1);
	su3_2_2->absorbDir_device(gaugeWL, B_DIR2);
	su3_3_1->absorbDir_device(gaugeWL, L_DIR);
	su3_3_2->absorbDir_device(gaugeWL, L_DIR);
	su3_4_1->absorbDir_device(gaugeWL, L_DIR);
	su3_4_2->absorbDir_device(gaugeWL, L_DIR);

	WL2_rev = true;
	WL2_dir = 0;
	WL4_rev = true;
	WL4_dir = 0;

	propExchange = propIn; propIn = propF_UPsmearplus_1; propF_UPsmearplus_1 = propExchange;
	propF_UPsmearplus_1->shift(*propIn, 4 + L_DIR);

        propExchange = propIn; propIn = propF_UPsmearminus_1; propF_UPsmearminus_1 = propExchange;
        propF_UPsmearminus_1->shift(*propIn, 4 + L_DIR);

        propExchange = propIn; propIn = propF_STsmearplus_1; propF_STsmearplus_1 = propExchange;
        propF_STsmearplus_1->shift(*propIn, 4 + L_DIR);

        propExchange = propIn; propIn = propF_STsmearminus_1; propF_STsmearminus_1 = propExchange;
        propF_STsmearminus_1->shift(*propIn, 4 + L_DIR);

        WL_4_1->wilsonLineUpdate(*su3_4_1, tmp, 4 + L_DIR);

        propExchange = propIn; propIn = propF_UPsmearplus_2; propF_UPsmearplus_2 = propExchange;
        propF_UPsmearplus_2->shift(*propIn, 4 + L_DIR);

        propExchange = propIn; propIn = propF_UPsmearminus_2; propF_UPsmearminus_2 = propExchange;
        propF_UPsmearminus_2->shift(*propIn, 4 + L_DIR);

        propExchange = propIn; propIn = propF_STsmearplus_2; propF_STsmearplus_2 = propExchange;
        propF_STsmearplus_2->shift(*propIn, 4 + L_DIR);

        propExchange = propIn; propIn = propF_STsmearminus_2; propF_STsmearminus_2 = propExchange;
        propF_STsmearminus_2->shift(*propIn, 4 + L_DIR);

        WL_4_2->wilsonLineUpdate(*su3_4_2, tmp, 4 + L_DIR);
	
	for(int l = 0; l <= L_LEN; l++) {//loop over values of L, building "1" and "3"

	  if(l != 0) {

	    WL_1->wilsonLineUpdate(*su3_1, tmp, L_DIR);
	    WL_3_1->wilsonLineUpdate(*su3_3_1, tmp, L_DIR, true);
            WL_3_2->wilsonLineUpdate(*su3_3_2, tmp, L_DIR, true);

	    su3_exchange = su3_in; su3_in = su3_2_1; su3_2_1 = su3_exchange;
	    su3_2_1->shift(*su3_in, L_DIR);

	    su3_exchange = su3_in; su3_in = su3_2_2; su3_2_2 = su3_exchange;
	    su3_2_2->shift(*su3_in, L_DIR);

	    su3_exchange = su3_in; su3_in = WL_2_1; WL_2_1 = su3_exchange;
	    WL_2_1->shift(*su3_in, L_DIR);

	    su3_exchange = su3_in; su3_in = WL_2_2; WL_2_2 = su3_exchange;
	    WL_2_2->shift(*su3_in, L_DIR);

	  }

	  if(WL2_rev) {
	    WL2_rev = false;
	    WL2_dir = 1;
	  }else {
	    WL2_rev = true;
	    WL2_dir = 0;
	  }
	  
	  for(int b = 0; b <= B_LEN; b++) {//loop over values of B, building "2"

	    if(b != 0) {

	      propExchange = propIn; propIn = propF_UPsmearplus_1; propF_UPsmearplus_1 = propExchange;
              propF_UPsmearplus_1->shift(*propIn, 4*WL2_dir + B_DIR1);

              propExchange = propIn; propIn = propF_UPsmearminus_1; propF_UPsmearminus_1 = propExchange;
              propF_UPsmearminus_1->shift(*propIn, 4*WL2_dir + B_DIR1);

              propExchange = propIn; propIn = propF_STsmearplus_1; propF_STsmearplus_1 = propExchange;
              propF_STsmearplus_1->shift(*propIn, 4*WL2_dir + B_DIR1);

              propExchange = propIn; propIn = propF_STsmearminus_1; propF_STsmearminus_1 = propExchange;
              propF_STsmearminus_1->shift(*propIn, 4*WL2_dir + B_DIR1);

	      WL_2_1->wilsonLineUpdate(*su3_2_1, tmp, 4*WL2_dir + B_DIR1);

	      propExchange = propIn; propIn = propF_UPsmearplus_2; propF_UPsmearplus_2 = propExchange;
              propF_UPsmearplus_2->shift(*propIn, 4*WL2_dir + B_DIR2);

              propExchange = propIn; propIn = propF_UPsmearminus_2; propF_UPsmearminus_2 = propExchange;
              propF_UPsmearminus_2->shift(*propIn, 4*WL2_dir + B_DIR2);

              propExchange = propIn; propIn = propF_STsmearplus_2; propF_STsmearplus_2 = propExchange;
              propF_STsmearplus_2->shift(*propIn, 4*WL2_dir + B_DIR2);

              propExchange = propIn; propIn = propF_STsmearminus_2; propF_STsmearminus_2 = propExchange;
              propF_STsmearminus_2->shift(*propIn, 4*WL2_dir + B_DIR2);				

	      WL_2_2->wilsonLineUpdate(*su3_2_2, tmp, 4*WL2_dir + B_DIR2);

	      su3_exchange = su3_in; su3_in = su3_3_1; su3_3_1 = su3_exchange;
	      su3_3_1->shift(*su3_in, 4*WL2_dir + B_DIR1);

	      su3_exchange = su3_in; su3_in = su3_3_2; su3_3_2 = su3_exchange;
	      su3_3_2->shift(*su3_in, 4*WL2_dir + B_DIR2);

	      su3_exchange = su3_in; su3_in = su3_4_1; su3_4_1 = su3_exchange;
	      su3_4_1->shift(*su3_in, 4*WL2_dir + B_DIR1);

	      su3_exchange = su3_in; su3_in = su3_4_2; su3_4_2 = su3_exchange;
	      su3_4_2->shift(*su3_in, 4*WL2_dir + B_DIR2);

	      su3_exchange = su3_in; su3_in = WL_3_1; WL_3_1 = su3_exchange;
	      WL_3_1->shift(*su3_in, 4*WL2_dir + B_DIR1);

	      su3_exchange = su3_in; su3_in = WL_3_2; WL_3_2 = su3_exchange;
	      WL_3_2->shift(*su3_in, 4*WL2_dir + B_DIR2);

	      su3_exchange = su3_in; su3_in = WL_4_1; WL_4_1 = su3_exchange;
	      WL_4_1->shift(*su3_in, 4*WL2_dir + B_DIR1);

	      su3_exchange = su3_in; su3_in = WL_4_2; WL_4_2 = su3_exchange;
	      WL_4_2->shift(*su3_in, 4*WL2_dir + B_DIR2);

	    }

	    WL_temp->UxU(*WL_1, *WL_2_1);
	    WL_temp1->UxU(*WL_temp, *WL_3_1);
	    
	    WL_temp->UxU(*WL_1, *WL_2_2);
	    WL_temp2->UxU(*WL_temp, *WL_3_2);

	    if(WL4_rev) {
	      WL4_rev = false;
	      WL4_dir = 1;
	    }else {
	      WL4_rev = true;
	      WL4_dir = 0;
	    }
	    
	    for(int z = 0; z <= Z_LEN-1; z++) {//loop over values of Z, building "4"

	      if(z != 0) {

		propExchange = propIn; propIn = propF_UPsmearplus_1; propF_UPsmearplus_1 = propExchange;
                propF_UPsmearplus_1->shift(*propIn, 4*WL4_dir + L_DIR);

                propExchange = propIn; propIn = propF_UPsmearminus_1; propF_UPsmearminus_1 = propExchange;
                propF_UPsmearminus_1->shift(*propIn, 4*WL4_dir + L_DIR);

                propExchange = propIn; propIn = propF_STsmearplus_1; propF_STsmearplus_1 = propExchange;
                propF_STsmearplus_1->shift(*propIn, 4*WL4_dir + L_DIR);

                propExchange = propIn; propIn = propF_STsmearminus_1; propF_STsmearminus_1 = propExchange;
                propF_STsmearminus_1->shift(*propIn, 4*WL4_dir + L_DIR);

		WL_4_1->wilsonLineUpdate(*su3_4_1, tmp, 4*WL4_dir + L_DIR);

		propExchange = propIn; propIn = propF_UPsmearplus_2; propF_UPsmearplus_2 = propExchange;
                propF_UPsmearplus_2->shift(*propIn, 4*WL4_dir + L_DIR);

                propExchange = propIn; propIn = propF_UPsmearminus_2; propF_UPsmearminus_2 = propExchange;
                propF_UPsmearminus_2->shift(*propIn, 4*WL4_dir + L_DIR);

                propExchange = propIn; propIn = propF_STsmearplus_2; propF_STsmearplus_2 = propExchange;
                propF_STsmearplus_2->shift(*propIn, 4*WL4_dir + L_DIR);

                propExchange = propIn; propIn = propF_STsmearminus_2; propF_STsmearminus_2 = propExchange;
                propF_STsmearminus_2->shift(*propIn, 4*WL4_dir + L_DIR);

		WL_4_2->wilsonLineUpdate(*su3_4_2, tmp, 4*WL4_dir + L_DIR);

	      }

	      WL1->UxU(*WL_temp1, *WL_4_1);
	      WL2->UxU(*WL_temp2, *WL_4_2);

	      if(WL2_rev) bval = B_LEN - b;
	      else bval = b;

	      if(WL4_rev) zval = Z_LEN - z;
	      else zval = 1 + z;

	      TIME(corrThrpWL.contractNucleonThrp_staple(*seqProp_pionUPplusmom, *propF_UPsmearplus_1, *WL1, +1, gammas, false, bval, l, zval));
              corrThrpWL.writeFile((threep_filename+"_ts"+std::to_string(tSinks[ts])+"_pionUP_plusmom_B_DIR1_"+conf).c_str(), corr_file_format);

	      TIME(corrThrpWL.contractNucleonThrp_staple(*seqProp_pionUPplusmom, *propF_UPsmearplus_2, *WL2, +1, gammas, false, bval, l, zval));
              corrThrpWL.writeFile((threep_filename+"_ts"+std::to_string(tSinks[ts])+"_pionUP_plusmom_B_DIR2_"+conf).c_str(), corr_file_format);

              TIME(corrThrpWL.contractNucleonThrp_staple(*seqProp_pionUPminusmom, *propF_UPsmearminus_1, *WL1, +1, gammas, false, bval, l, zval));
              corrThrpWL.writeFile((threep_filename+"_ts"+std::to_string(tSinks[ts])+"_pionUP_minusmom_B_DIR1_"+conf).c_str(), corr_file_format);

              TIME(corrThrpWL.contractNucleonThrp_staple(*seqProp_pionUPminusmom, *propF_UPsmearminus_2, *WL2, +1, gammas, false, bval, l, zval));
              corrThrpWL.writeFile((threep_filename+"_ts"+std::to_string(tSinks[ts])+"_pionUP_minusmom_B_DIR2_"+conf).c_str(), corr_file_format);

              TIME(corrThrpWL.contractNucleonThrp_staple(*seqProp_kaonUPplusmom, *propF_UPsmearplus_1, *WL1, +1, gammas, false, bval, l, zval));
              corrThrpWL.writeFile((threep_filename_1+"_ts"+std::to_string(tSinks[ts])+"_kaonUP_plusmom_B_DIR1_"+conf).c_str(), corr_file_format);

              TIME(corrThrpWL.contractNucleonThrp_staple(*seqProp_kaonUPplusmom, *propF_UPsmearplus_2, *WL2, +1, gammas, false, bval, l, zval));
              corrThrpWL.writeFile((threep_filename_1+"_ts"+std::to_string(tSinks[ts])+"_kaonUP_plusmom_B_DIR2_"+conf).c_str(), corr_file_format);

              TIME(corrThrpWL.contractNucleonThrp_staple(*seqProp_kaonUPminusmom, *propF_UPsmearminus_1, *WL1, +1, gammas, false, bval, l, zval));
              corrThrpWL.writeFile((threep_filename_1+"_ts"+std::to_string(tSinks[ts])+"_kaonUP_minusmom_B_DIR1_"+conf).c_str(), corr_file_format);

              TIME(corrThrpWL.contractNucleonThrp_staple(*seqProp_kaonUPminusmom, *propF_UPsmearminus_2, *WL2, +1, gammas, false, bval, l, zval));
              corrThrpWL.writeFile((threep_filename_1+"_ts"+std::to_string(tSinks[ts])+"_kaonUP_minusmom_B_DIR2_"+conf).c_str(), corr_file_format);

              TIME(corrThrpWL.contractNucleonThrp_staple(*seqProp_kaonSTplusmom, *propF_STsmearplus_1, *WL1, +1, gammas, false, bval, l, zval));
              corrThrpWL.writeFile((threep_filename_1+"_ts"+std::to_string(tSinks[ts])+"_kaonST_plusmom_B_DIR1_"+conf).c_str(), corr_file_format);

              TIME(corrThrpWL.contractNucleonThrp_staple(*seqProp_kaonSTplusmom, *propF_STsmearplus_2, *WL2, +1, gammas, false, bval, l, zval));
              corrThrpWL.writeFile((threep_filename_1+"_ts"+std::to_string(tSinks[ts])+"_kaonST_plusmom_B_DIR2_"+conf).c_str(), corr_file_format);

              TIME(corrThrpWL.contractNucleonThrp_staple(*seqProp_kaonSTminusmom, *propF_STsmearminus_1, *WL1, +1, gammas, false, bval, l, zval));
              corrThrpWL.writeFile((threep_filename_1+"_ts"+std::to_string(tSinks[ts])+"_kaonST_minusmom_B_DIR1_"+conf).c_str(), corr_file_format);

              TIME(corrThrpWL.contractNucleonThrp_staple(*seqProp_kaonSTminusmom, *propF_STsmearminus_2, *WL2, +1, gammas, false, bval, l, zval));
              corrThrpWL.writeFile((threep_filename_1+"_ts"+std::to_string(tSinks[ts])+"_kaonST_minusmom_B_DIR2_"+conf).c_str(), corr_file_format);
	      
	    }//loop over values of Z, building "4"
	  }//loop over values of B, building "2"			
	}//loop over values of L, building "1" and "3"
      }
    }
  
  propUP_smearplus->load();
  propUP_smearminus->load();
  propST_smearplus->load();
  propST_smearminus->load();
  
  for(int nu = 0 ; nu < 4 ; nu++)
    for(int c2 = 0 ; c2 < 3 ; c2++){
      vectorAuxF.absorb(*propUP_smearplus, nu, c2);
      vectorAuxD.copy(vectorAuxF);
      vectorOut.gaussianSmearing(vectorAuxD, smearedGauge, nsmearGauss, alphaGauss);
      vectorAuxF.copy(vectorOut);
      propUP_smearplus->absorb(vectorAuxF, nu, c2);

      vectorAuxF.absorb(*propUP_smearminus, nu, c2);
      vectorAuxD.copy(vectorAuxF);
      vectorOut.gaussianSmearing(vectorAuxD, smearedGaugeMn, nsmearGauss, alphaGauss);
      vectorAuxF.copy(vectorOut);
      propUP_smearminus->absorb(vectorAuxF, nu, c2);

      vectorAuxF.absorb(*propST_smearplus, nu, c2);
      vectorAuxD.copy(vectorAuxF);
      vectorOut.gaussianSmearing(vectorAuxD, smearedGauge, nsmearGauss, alphaGauss);
      vectorAuxF.copy(vectorOut);
      propST_smearplus->absorb(vectorAuxF, nu, c2);
      
      vectorAuxF.absorb(*propST_smearminus, nu, c2);
      vectorAuxD.copy(vectorAuxF);
      vectorOut.gaussianSmearing(vectorAuxD, smearedGaugeMn, nsmearGauss, alphaGauss);
      vectorAuxF.copy(vectorOut);
      propST_smearminus->absorb(vectorAuxF, nu, c2);

    }
  
  propUP_smearplus->rotateToPhysicalBase_device(+1);
  propUP_smearminus->rotateToPhysicalBase_device(+1);
  propST_smearplus->rotateToPhysicalBase_device(+1);
  propST_smearminus->rotateToPhysicalBase_device(+1);
  propUP_smearplus->applyBoundaries_device(sourcePositions[isource][3]);
  propUP_smearminus->applyBoundaries_device(sourcePositions[isource][3]);
  propST_smearplus->applyBoundaries_device(sourcePositions[isource][3]);
  propST_smearminus->applyBoundaries_device(sourcePositions[isource][3]);
  
  PLEGMA_Correlator<float> corr(corr_space, sourcePositions[isource]);
  corr.setFixMomVec(sinkMom_3D);
  
  TIME(corr.contractMesonsNew(*propUP_smearplus, *propUP_smearminus));
  corr.writeFile((twop_filename+conf).c_str(), corr_file_format);

  TIME(corr.contractMesonsNew(*propUP_smearplus, *propST_smearminus));
  corr.writeFile((twop_filename_1+conf).c_str(), corr_file_format);
  
  }
  
  delete propUP_smearplus;
  delete propUP_smearminus;
  delete propST_smearplus;
  delete propST_smearminus;
  delete propIn;

  delete seqProp_pionUPplusmom;
  delete seqProp_pionUPminusmom;
  delete seqProp_kaonUPplusmom;
  delete seqProp_kaonUPminusmom;
  delete seqProp_kaonSTplusmom;
  delete seqProp_kaonSTminusmom;
  
  delete su3_1;
  delete su3_2_1;
  delete su3_2_2;
  delete su3_3_1;
  delete su3_3_2;
  delete su3_4_1;
  delete su3_4_2;
  delete WL_1;
  delete WL_2_1;
  delete WL_3_1;
  delete WL_2_2;
  delete WL_3_2;
  delete WL_4_1;
  delete WL_4_2;
  delete WL_temp;
  delete WL_temp1;
  delete WL_temp2;
  delete su3_in;
  delete WL1;
  delete WL2;
  
  delete solver;
  
  finalize();
  return 0;
}
