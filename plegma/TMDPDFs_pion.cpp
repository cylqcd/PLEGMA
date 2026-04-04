#include <PLEGMA.h>
#include <PLEGMA_utils.h>

using namespace plegma;
using namespace quda;

int main(int argc, char **argv)
{

  static std::vector<std::string> listOpt = {"verbosity", "load-gauge", "nsmear-APE", "alpha-APE", "nsmear-gauss", "alpha-gauss",
					     "nsrc", "src-filename", "maxQsq", "twop-filename","threep-filename",  "corr-file-format",
					     "corr-space", "tSinks","xiMomSm","sinkMom","gammas"};

  initializeOptions(argc, argv, true, listOpt);

  if(mu<0){
    mu *= -1.0;
  }
  
  double mu_ud = mu;
  
  double mu_s;
  HGC_options->set("mu-s", "mu for strange quark", verbosity, mu_s);
  
  int L_LEN;
  HGC_options->set("l-len", "length of l", verbosity, L_LEN);

  int L_DIR;
  HGC_options->set("l-dir", "direction of l", verbosity, L_DIR);

  int B_LEN;
  HGC_options->set("b-len", "length of b", verbosity, B_LEN);

  int B_DIR;
  HGC_options->set("b-dir", "direction of b", verbosity, B_DIR);

  int z = 0;
  HGC_options->set("z-len", "length of z", verbosity, z);

  int nStout;
  HGC_options->set("n_stout", "n parameter stout smearing", verbosity, nStout);

  double rhoStout;
  HGC_options->set("rho_stout", "Rho parameter stout smearing", verbosity, rhoStout);

  bool calc3pt = true ;
  HGC_options->set("calc3pt", "If true then the 3pt function is computed", verbosity, calc3pt);

  std::string conf;
  HGC_options->set("conf", "configuration", verbosity, conf);

  std::string twop_filename_1;
  HGC_options->set("twop-filename-1", "2p filename for kaon +", verbosity, twop_filename_1);
  
  std::string threep_filename_1;
  HGC_options->set("threep-filename-1", "3p filename for kaon +", verbosity, threep_filename_1);
  
  initializePLEGMA();
  
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
  QUDA_solver *solver = new QUDA_solver(mu);

  PLEGMA_Vector<double> vectorIn;
  PLEGMA_Vector<double> vectorOut;
  PLEGMA_Vector<double> vectorAuxD;
  PLEGMA_Vector<float> vectorAuxF;
  PLEGMA_Propagator<float> *propUP_smearplus = new PLEGMA_Propagator<float>(BOTH);
  PLEGMA_Propagator<float> *propUP_smearminus = new PLEGMA_Propagator<float>(BOTH);
  PLEGMA_Propagator<float> *propST_smearminus = new PLEGMA_Propagator<float>(BOTH);
  PLEGMA_Propagator<float> *seqPropOut1 = new PLEGMA_Propagator<float>(BOTH);
  PLEGMA_Propagator<float> *seqPropOut2 = new PLEGMA_Propagator<float>(BOTH);
  PLEGMA_Propagator<float> *propIn = new PLEGMA_Propagator<float>(BOTH);
  PLEGMA_Propagator<float> *propExchange = nullptr;

  PLEGMA_Gauge<float> gaugeWL;
  gaugeWL.copy(gauge);
  gaugeWL.stoutSmearing(gaugeWL,nStout,rhoStout,3);
  PLEGMA_Su3field<float> *WL = new PLEGMA_Su3field<float>(BOTH);
  PLEGMA_Su3field<float> tmp(BOTH);
  PLEGMA_Su3field<float> *u_s[4];
  for(int idir = 0; idir < 4 ; idir++){
    u_s[idir] = new PLEGMA_Su3field<float>(BOTH);
    u_s[idir]->absorbDir_device(gaugeWL,idir);
  }
  PLEGMA_Su3field<float> *WLIn = new PLEGMA_Su3field<float>(BOTH);
  PLEGMA_Su3field<float> *WLExchange = nullptr;
  
  PLEGMA_Su3field<float> ZE(BOTH);
  PLEGMA_Su3field<float> tmp1(BOTH);
  
  PLEGMA_Propagator3D<float> propUP3D_smearplus;
  PLEGMA_Propagator3D<float> propUP3D_smearminus;
  PLEGMA_Propagator3D<float> propST3D_smearminus;

  for(int isource=0;isource<numSourcePositions;isource++){
    
  PLEGMA_Correlator<float> corrThrpWL1(corr_space, sourcePositions[isource], 0); 
  PLEGMA_Correlator<float> corrThrpWL2(corr_space, sourcePositions[isource], 0);

  for(int ts=0;ts<tSinks.size();ts++)
    {

      int signPer = (tSinks[ts] + sourcePositions[isource][3]) >= HGC_totalL[3] ? -1 : +1;
      int global_fixSinkTime = (tSinks[ts] + sourcePositions[isource][3])%HGC_totalL[3]; 
      int my_fixSinkTime = global_fixSinkTime - HGC_procPosition[3] * HGC_localL[3];
      bool is_myST = (my_fixSinkTime >= 0) && ( my_fixSinkTime < HGC_localL[3] );

      if(mu!=mu_ud) {
	updateOptions(LIGHT);
	mu=mu_ud;
	solver->UpdateSolver();
      }
      for(int isc = 0 ; isc < 12 ; isc++){
	vectorAuxD.pointSource(sourcePositions[isource], isc/3, isc%3, DEVICE);
	vectorIn.gaussianSmearing(vectorAuxD, smearedGauge, nsmearGauss, alphaGauss);
    
	PLEGMA_printf("Going to invert UP for component %d\n", isc);
	solver->solve(vectorOut, vectorIn);
	vectorAuxF.copy(vectorOut);
	propUP_smearplus->absorb(vectorAuxF, isc/3, isc%3);
	vectorAuxD.gaussianSmearing(vectorOut, smearedGauge , nsmearGauss, alphaGauss);
	vectorAuxF.copy(vectorAuxD);
	propUP3D_smearplus.absorb(vectorAuxF,global_fixSinkTime, isc/3, isc%3);
      }

      for(int isc = 0 ; isc < 12 ; isc++){
	vectorAuxD.pointSource(sourcePositions[isource], isc/3, isc%3, DEVICE);
	vectorIn.gaussianSmearing(vectorAuxD, smearedGaugeMn, nsmearGauss, alphaGauss);
    
	PLEGMA_printf("Going to invert UP for component %d\n", isc);
	solver->solve(vectorOut, vectorIn);
	vectorAuxF.copy(vectorOut);
	propUP_smearminus->absorb(vectorAuxF, isc/3, isc%3);
	vectorAuxD.gaussianSmearing(vectorOut, smearedGaugeMn , nsmearGauss, alphaGauss);
	vectorAuxF.copy(vectorAuxD);
	propUP3D_smearminus.absorb(vectorAuxF,global_fixSinkTime, isc/3, isc%3);
      }

      if(mu!=mu_s) {
        updateOptions(STRANGE);
        mu=mu_s;
        solver->UpdateSolver();
      }
      for(int isc = 0 ; isc < 12 ; isc++){
        vectorAuxD.pointSource(sourcePositions[isource], isc/3, isc%3, DEVICE);
        vectorIn.gaussianSmearing(vectorAuxD, smearedGaugeMn, nsmearGauss, alphaGauss);

        PLEGMA_printf("Going to invert STRANGE for component %d\n", isc);
        solver->solve(vectorOut, vectorIn);
        vectorAuxF.copy(vectorOut);
        propST_smearminus->absorb(vectorAuxF, isc/3, isc%3);
        vectorAuxD.gaussianSmearing(vectorOut, smearedGaugeMn , nsmearGauss, alphaGauss);
        vectorAuxF.copy(vectorAuxD);
        propST3D_smearminus.absorb(vectorAuxF,global_fixSinkTime, isc/3, isc%3);
      }
      
      if(calc3pt){

	propUP_smearplus->unload();
	propUP_smearminus->unload();
        propST_smearminus->unload();
	
	//seq source part 2Props and contraction block for Pion + and Kaon +
	{
	  for(int nu = 0 ; nu < 4 ; nu++)
	    for(int c2 = 0 ; c2 < 3 ; c2++){
	      PLEGMA_Vector3D<float> vectorAux3D;
	      vectorAux3D.absorb(propUP3D_smearminus,nu,c2);
	      vectorAux3D.mulMomentumPhases(sinkMom,+1); //put momentum at the sink
	      std::complex<float> Isingle(0,1);
	      float phase = 2.*PI*(((float) sinkMom[0] * sourcePositions[isource][0])/HGC_totalL[0]
				   + ((float)sinkMom[1] * sourcePositions[isource][1])/HGC_totalL[1]
				   + ((float)sinkMom[2] * sourcePositions[isource][2])/HGC_totalL[2]);
	      vectorAux3D.cscale(std::exp<float>(-phase*Isingle)); // put momentum from the point source
	      vectorAux3D.apply_gamma(G5);
	      vectorAuxF.absorb(vectorAux3D, global_fixSinkTime);
	      vectorAuxD.copy(vectorAuxF);
	      vectorIn.gaussianSmearing(vectorAuxD,smearedGauge, nsmearGauss, alphaGauss);
	      // check if we need to normalize the seqsource for mix precision solver
	      if(mu!=-mu_ud) {
		updateOptions(LIGHT);
		mu=-mu_ud;
		solver->UpdateSolver();
	      }
	      double norm = vectorIn.norm();
	      vectorIn.cscale(1/norm);
	      solver->solve(vectorOut, vectorIn);
	      vectorOut.cscale(norm);    
	      vectorAuxF.copy(vectorOut);
	      seqPropOut1->absorb(vectorAuxF, nu, c2);
	    }
	  seqPropOut1->apply_gamma(G5);
	  seqPropOut1->conjugate();

	  for(int nu = 0 ; nu < 4 ; nu++)
            for(int c2 = 0 ; c2 < 3 ; c2++){
              PLEGMA_Vector3D<float> vectorAux3D;
              vectorAux3D.absorb(propST3D_smearminus,nu,c2);
              vectorAux3D.mulMomentumPhases(sinkMom,+1); //put momentum at the sink
              std::complex<float> Isingle(0,1);
              float phase = 2.*PI*(((float) sinkMom[0] * sourcePositions[isource][0])/HGC_totalL[0]
                                   + ((float)sinkMom[1] * sourcePositions[isource][1])/HGC_totalL[1]
                                   + ((float)sinkMom[2] * sourcePositions[isource][2])/HGC_totalL[2]);
              vectorAux3D.cscale(std::exp<float>(-phase*Isingle)); // put momentum from the point source
              vectorAux3D.apply_gamma(G5);
              vectorAuxF.absorb(vectorAux3D, global_fixSinkTime);
              vectorAuxD.copy(vectorAuxF);
              vectorIn.gaussianSmearing(vectorAuxD,smearedGauge, nsmearGauss, alphaGauss);
              // check if we need to normalize the seqsource for mix precision solver
              if(mu!=-mu_ud) {
                updateOptions(LIGHT);
                mu=-mu_ud;
                solver->UpdateSolver();
              }
              double norm = vectorIn.norm();
              vectorIn.cscale(1/norm);
              solver->solve(vectorOut, vectorIn);
              vectorOut.cscale(norm);
              vectorAuxF.copy(vectorOut);
              seqPropOut2->absorb(vectorAuxF, nu, c2);
            }
          seqPropOut2->apply_gamma(G5);
          seqPropOut2->conjugate();

	  PLEGMA_Propagator<float> *propF = propUP_smearplus;
	  
	  for(int z_step=0 ; z_step <= z ; z_step++){
	    
	    if(z_step>0){
	      propExchange = propIn; propIn = propF; propF = propExchange;
	      propF->shift(*propIn, (4+L_DIR)%8);
	    }
  
	    for(int b=0 ; b <= B_LEN ; b++){

	      if(b>0){
		propExchange = propIn; propIn = propF; propF = propExchange;
		propF->shift(*propIn, (4+B_DIR)%8);
	      }
	      
	      for(int l=0 ; l <= L_LEN ; l++){

		WL->setUnit((std::vector<int>) {0,4,8});

		int spath[(2*l)+b+z_step];

		if((l!=0)||(b!=0)||(z_step!=0)){
		  for(int i=0;i<l;i++){
		    spath[i] = (4+L_DIR)%8;
		  }
		  for(int j=l;j<(l+b);j++){
		    spath[j] = B_DIR;
		  }
		  for(int k=(l+b);k<((2*l)+b+z_step);k++){
		    spath[k] = L_DIR;
		  }

		  std::vector<int> vspath(spath,spath+(2*l)+b+z_step);

		  WL->path(vspath, u_s, tmp);

		  for(int k=(l+b);k<((2*l)+b+z_step);k++){
		    WLExchange = WLIn; WLIn = WL; WL = WLExchange;
		    WL->shift(*WLIn,(4+L_DIR)%8);
		  }
		  for(int j=l;j<(l+b);j++){
		    WLExchange = WLIn; WLIn = WL; WL = WLExchange;
		    WL->shift(*WLIn,(4+B_DIR)%8);
		  }
		  for(int i=0;i<l;i++){
		    WLExchange = WLIn; WLIn = WL; WL = WLExchange;
		    WL->shift(*WLIn,L_DIR);
		  }
		}

	        std::string suff = "_CP2_stout_"+std::to_string(nStout);
	        corrThrpWL1.contractNucleonThrp_staple(*seqPropOut1, *propF, *WL, +1, gammas, false, b, l, z_step);
	        if(signPer < 0) for(int iv = 0 ; iv < corrThrpWL1.getTotalSize()*2; iv++) corrThrpWL1.H_elem()[iv] *= signPer;
	        //corrThrpWL.writeASCII( (threep_filename +  suff + "_ts_" + std::to_string(tSinks[ts])  + ".dat").c_str() );
		corrThrpWL1.writeFile((threep_filename+"_CP2").c_str(), corr_file_format);

                corrThrpWL2.contractNucleonThrp_staple(*seqPropOut2, *propF, *WL, +1, gammas, false, b, l, z_step);
                if(signPer < 0) for(int iv = 0 ; iv < corrThrpWL2.getTotalSize()*2; iv++) corrThrpWL2.H_elem()[iv] *= signPer;
                //corrThrpWL.writeASCII( (threep_filename +  suff + "_ts_" + std::to_string(tSinks[ts])  + ".dat").c_str() );
                corrThrpWL2.writeFile((threep_filename_1+"_CP2").c_str(), corr_file_format);
	      }
	    }

            for(int b=0 ; b < B_LEN ; b++){
	      propExchange = propIn; propIn = propF; propF = propExchange;
	      propF->shift(*propIn, B_DIR);
	    }
	      
	  }

	  for(int z_step=0 ; z_step < z ; z_step++){
            propExchange = propIn; propIn = propF; propF = propExchange;
            propF->shift(*propIn, L_DIR);
          }

          for(int z_step=1 ; z_step <= z ; z_step++){

            if(z_step>0){
              propExchange = propIn; propIn = propF; propF = propExchange;
              propF->shift(*propIn, (L_DIR)%8);
            }

            for(int b=0 ; b <= B_LEN ; b++){

              if(b>0){
                propExchange = propIn; propIn = propF; propF = propExchange;
                propF->shift(*propIn, (4+B_DIR)%8);
              }

              for(int l=0 ; l <= L_LEN ; l++){

                WL->setUnit((std::vector<int>) {0,4,8});

                int spath[(2*l)+b+z_step];

                if((l!=0)||(b!=0)||(z_step!=0)){
                  for(int i=0;i<l;i++){
                    spath[i] = L_DIR;
                  }
                  for(int j=l;j<(l+b);j++){
                    spath[j] = B_DIR;
                  }
                  for(int k=(l+b);k<((2*l)+b+z_step);k++){
                    spath[k] = (4+L_DIR)%8;
                  }

                  std::vector<int> vspath(spath,spath+(2*l)+b+z_step);

                  WL->path(vspath, u_s, tmp);

                  for(int k=(l+b);k<((2*l)+b+z_step);k++){
                    WLExchange = WLIn; WLIn = WL; WL = WLExchange;
                    WL->shift(*WLIn,L_DIR);
                  }
                  for(int j=l;j<(l+b);j++){
                    WLExchange = WLIn; WLIn = WL; WL = WLExchange;
                    WL->shift(*WLIn,(4+B_DIR)%8);
                  }
                  for(int i=0;i<l;i++){
                    WLExchange = WLIn; WLIn = WL; WL = WLExchange;
                    WL->shift(*WLIn,(4+L_DIR)%8);
                  }
                }

                std::string suff = "_CP2_stout_"+std::to_string(nStout);
                corrThrpWL1.contractNucleonThrp_staple(*seqPropOut1, *propF, *WL, +1, gammas, false, b, -l, -z_step);
                if(signPer < 0) for(int iv = 0 ; iv < corrThrpWL1.getTotalSize()*2; iv++) corrThrpWL1.H_elem()[iv] *= signPer;
                //corrThrpWL.writeASCII( (threep_filename +  suff + "_ts_" + std::to_string(tSinks[ts])  + ".dat").c_str() );
                corrThrpWL1.writeFile((threep_filename+"_CP2").c_str(), corr_file_format);

                corrThrpWL2.contractNucleonThrp_staple(*seqPropOut2, *propF, *WL, +1, gammas, false, b, -l, -z_step);
                if(signPer < 0) for(int iv = 0 ; iv < corrThrpWL2.getTotalSize()*2; iv++) corrThrpWL2.H_elem()[iv] *= signPer;
                //corrThrpWL.writeASCII( (threep_filename +  suff + "_ts_" + std::to_string(tSinks[ts])  + ".dat").c_str() );
                corrThrpWL2.writeFile((threep_filename_1+"_CP2").c_str(), corr_file_format);

              }
            }

            for(int b=0 ; b < B_LEN ; b++){
              propExchange = propIn; propIn = propF; propF = propExchange;
              propF->shift(*propIn, B_DIR);
            }

          }

	  for(int z_step=0 ; z_step < z ; z_step++){
	    propExchange = propIn; propIn = propF; propF = propExchange;
	    propF->shift(*propIn, (4+L_DIR)%8);
	  }
	  
	}
      }
    }

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

      vectorAuxF.absorb(*propST_smearminus, nu, c2);
      vectorAuxD.copy(vectorAuxF);
      vectorOut.gaussianSmearing(vectorAuxD, smearedGaugeMn, nsmearGauss, alphaGauss);
      vectorAuxF.copy(vectorOut);
      propST_smearminus->absorb(vectorAuxF, nu, c2);

    }
  
  propUP_smearplus->rotateToPhysicalBase_device(+1);
  propUP_smearminus->rotateToPhysicalBase_device(+1);
  propST_smearminus->rotateToPhysicalBase_device(+1);
  propUP_smearplus->applyBoundaries_device(sourcePositions[isource][3]);
  propUP_smearminus->applyBoundaries_device(sourcePositions[isource][3]);
  propST_smearminus->applyBoundaries_device(sourcePositions[isource][3]);
  
  PLEGMA_Correlator<float> corr(corr_space, sourcePositions[isource]);
  corr.setFixMomVec(sinkMom_3D);
  corr.contractMesonsNew(*propUP_smearplus, *propUP_smearminus);
  corr.writeFile(twop_filename.c_str(), corr_file_format);

  PLEGMA_Correlator<float> corr1(corr_space, sourcePositions[isource]);
  corr1.setFixMomVec(sinkMom_3D);
  corr1.contractMesonsNew(*propUP_smearplus, *propST_smearminus);
  corr1.writeFile(twop_filename_1.c_str(), corr_file_format);
  
  }

  /***for(int b=0;b<=B_LEN;b++){
    for(int l=0;l<=L_LEN;l++){

      std::string namefile = "/onyx/qdata/jtarello/TMDPDFs_pion/ZE_b"+std::to_string(b)+"_l"+std::to_string(l)+"_stout"+std::to_string(nStout)+"_"+conf+".h5";
      
      ZE.setUnit((std::vector<int>) {0,4,8});

      if(l!=0 || b!=0){
        int sspath[(4*l)+(2*b)];
	
        for(int j=0;j<b;j++){
          sspath[j] = 4+B_DIR;
        }
        for(int k=b;k<((2*l)+b);k++){
          sspath[k] = 4+L_DIR;
        }
        for(int m=((2*l)+b);m<((2*l)+(2*b));m++){
          sspath[m] = B_DIR;
        }
        for(int n=((2*l)+(2*b));n<((4*l)+(2*b));n++){
          sspath[n] = L_DIR;
        }

        std::vector<int> vsspath(sspath,sspath+(4*l)+(2*b));

        ZE.path(vsspath, u_s, tmp1);
      }

      ZE.conjugate();

      PLEGMA_Field<float> ZEcontracted(BOTH,SCALAR);

      ZEcontracted.SU3Trace(ZE);

      PLEGMA_FT<float> ZEFT((std::vector<int>) {0,0,0},3);

      ZEFT.apply(ZEcontracted);
      
      ZEFT.writeFile(namefile,corr_file_format);
      
    }
    }***/
  
  delete propUP_smearplus;
  delete propUP_smearminus;
  delete propST_smearminus;

  delete propIn;
  delete seqPropOut1;
  delete seqPropOut2;

  for(int idir = 0; idir < 4 ; idir++){
    delete u_s[idir];
  }
  delete WL;
  delete WLIn;
  
  delete solver;
  
  finalize();
  return 0;
}

