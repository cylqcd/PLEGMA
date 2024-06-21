#include <PLEGMA.h>
#include <PLEGMA_utils.h>

std::vector<double> runtime;
#define TIME(fnc) runtime.push_back(MPI_Wtime()); fnc;                     \
  PLEGMA_printf("TIME for "#fnc" %f sec\n", MPI_Wtime()-runtime.back());  \
  runtime.pop_back()

std::vector<std::thread> threads;
//#define THREAD(fnc) threads.push_back(std::thread([=]() { TIME(fnc); }))
#define THREAD(fnc) saveTuneCache(false); TIME(fnc)

using namespace plegma;
using namespace quda;

int main(int argc, char **argv) {

  static std::vector<std::string> listOpt = {"verbosity", "load-gauge", "nsmear-APE", "alpha-APE", "nsmear-gauss", "alpha-gauss",
               "nsrc", "src-filename", "maxQsq", "twop-filename",  "corr-file-format",
               "corr-space", "tSinks","xiMomSm"};

  initializeOptions(argc, argv, true, listOpt);


  if(mu<0){
    mu *= -1.0;
  }
  
  std::string proj;
  HGC_options->set("which_projector", "Which projector to use for 3pt function", verbosity, proj);
  WHICHPROJECTOR which_proj=get_projector(proj.c_str());

  std::vector<int> sinkMom(4);
  HGC_options->set("sinkMom", "momentum at the sink", verbosity, sinkMom);

  std::string which_particle = "proton";
  HGC_options->set("which_particle", "Choice of the nucleon interpolator to insert in the three point function (neutron,proton)", verbosity, which_particle);
  WHICHPARTICLE nucleon = get_particle(which_particle.c_str());
  
  initializePLEGMA();

  char *fname;

  
  
  // Reading from Lime file and loading to device
  PLEGMA_Gauge<double> gauge;
  gauge.readFile(latfile, LIME_FORMAT);
  gauge.calculatePlaq();

  // Loading to QUDA and computing plaquette also there
  initGaugeQuda(gauge, true, QUDA_WILSON_LINKS);
  plaqQuda();
 
  // Smearing
  PLEGMA_Gauge<double> smearedGauge;
  smearedGauge.APEsmearing(gauge, nsmearAPE, alphaAPE, 3);
  PLEGMA_printf("Plaquette after smearing:\n");
  smearedGauge.calculatePlaq();

  PLEGMA_printf("Sink Momentum px %d, py %d, pz %d, pt %d\n",
		sinkMom[0],sinkMom[1],sinkMom[2],sinkMom[3]);
  
  //Momentum smearing: put the momentum phase to smeared gauge field
  std::complex<double> momSmScale[N_DIMS];
  std::complex<double> I(0,1);
  for(int i = 0 ; i < N_DIMS; i++){
    momSmScale[i] = std::exp(-(xiMomSm*2.*PI*sinkMom[i]/HGC_totalL[i])*I);
  }
  smearedGauge.scaleDirWise(momSmScale);

  // Extracting the spacial sink momentum from the sink 4-momentum  
  std::vector<int> sinkMom_3D(3);
  std::copy(sinkMom.begin(),sinkMom.begin()+3,sinkMom_3D.begin());
  
  // ensuring mu positive
  if(mu<0)  mu*=-1.;

  TIME(QUDA_solver *solver = new QUDA_solver(mu));

  PLEGMA_Vector<double> vectorIn;
  PLEGMA_Vector<double> vectorOut;
  PLEGMA_Vector<double> vectorAuxD;
  PLEGMA_Vector<float> vectorAuxF;
  PLEGMA_Propagator<float> *propUP = new PLEGMA_Propagator<float>(BOTH);
  PLEGMA_Propagator<float> *propDN = new PLEGMA_Propagator<float>(BOTH);
  PLEGMA_Propagator<float> *seqPropOut = new PLEGMA_Propagator<float>(BOTH);

  PLEGMA_Propagator3D<float> propUP3D;
  PLEGMA_Propagator3D<float> propDN3D;
  

  for(int isource=0;isource<numSourcePositions;isource++){//loop over source positions

    // UP prop
    if(mu<0){
      mu *= (-1);
      solver->UpdateSolver();
    }

    for(int isc = 0 ; isc < 12 ; isc++) {
      vectorAuxD.pointSource(sourcePositions[isource], isc/3, isc%3, DEVICE);
      vectorIn.gaussianSmearing(vectorAuxD, smearedGauge, nsmearGauss, alphaGauss);
      PLEGMA_printf("Going to invert UP for component %d\n", isc);
      solver->solve(vectorOut, vectorIn);
      vectorAuxF.copy(vectorOut);
      propUP->absorb(vectorAuxF, isc/3, isc%3);
    }

    // DN prop
    if(mu>0){
      mu *= (-1);
      solver->UpdateSolver();
    }

    for(int isc = 0 ; isc < 12 ; isc++) {
      vectorAuxD.pointSource(sourcePositions[isource], isc/3, isc%3, DEVICE);
      vectorIn.gaussianSmearing(vectorAuxD, smearedGauge, nsmearGauss, alphaGauss);
      PLEGMA_printf("Going to invert DN for component %d\n", isc);
      solver->solve(vectorOut, vectorIn);
      vectorAuxF.copy(vectorOut);
      propDN->absorb(vectorAuxF, isc/3, isc%3);
    }
  
  asprintf(&fname, "propUP.px%dpy%dpz%d.sx%02dsy%02dsz%02dst%02d",sinkMom[0],sinkMom[1],sinkMom[2],
            sourcePositions[isource][0],sourcePositions[isource][1],sourcePositions[isource][2],sourcePositions[isource][3]);
  TIME(propUP->writeFile(fname, LIME_FORMAT));
  asprintf(&fname, "propDN.px%dpy%dpz%d.sx%02dsy%02dsz%02dst%02d",sinkMom[0],sinkMom[1],sinkMom[2],
            sourcePositions[isource][0],sourcePositions[isource][1],sourcePositions[isource][2],sourcePositions[isource][3]);
  TIME(propDN->writeFile(fname, LIME_FORMAT));


  for(int ts=0;ts<tSinks.size();ts++) {//loop over tSinks

    int signPer = (tSinks[ts] + sourcePositions[isource][3]) >= HGC_totalL[3] ? -1 : +1;
    int global_fixSinkTime = (tSinks[ts] + sourcePositions[isource][3])%HGC_totalL[3]; 
    int my_fixSinkTime = global_fixSinkTime - HGC_procPosition[3] * HGC_localL[3];
    bool is_myST = (my_fixSinkTime >= 0) && ( my_fixSinkTime < HGC_localL[3] );
    
    for(int nu = 0 ; nu < 4 ; nu++)
    for(int c2 = 0 ; c2 < 3 ; c2++){
      vectorAuxF.absorb(*propUP, nu, c2);
      vectorAuxD.copy(vectorAuxF);
      vectorOut.gaussianSmearing(vectorAuxD, smearedGauge, nsmearGauss, alphaGauss);
      vectorAuxF.copy(vectorOut);
      propUP3D.absorb(vectorAuxF,global_fixSinkTime, nu, c2);

      vectorAuxF.absorb(*propDN, nu, c2);
      vectorAuxD.copy(vectorAuxF);
      vectorOut.gaussianSmearing(vectorAuxD, smearedGauge, nsmearGauss, alphaGauss);
      vectorAuxF.copy(vectorOut);
      propDN3D.absorb(vectorAuxF,global_fixSinkTime, nu, c2);
    }

    //calculate sequential propagators

    //CP1 part
    if(nucleon == PROTON){
      if(mu<0) {
        mu*=-1.;
        solver->UpdateSolver();
      }
    }else {
      if(mu>0) {
        mu*=-1.;
        solver->UpdateSolver();
      }
    }

    for(int nu = 0 ; nu < 4 ; nu++) {//loop over spin
      for(int c2 = 0 ; c2 < 3 ; c2++) {//loop over color

        PLEGMA_Vector3D<float> vectorAux3D;

        if(nucleon == PROTON) vectorAux3D.seqSourceNucleon(propUP3D, which_proj, nucleon, nu, c2);
        else vectorAux3D.seqSourceNucleon(propDN3D, which_proj, nucleon, nu, c2);
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

        double norm = vectorIn.norm();
        vectorIn.cscale(1/norm);
        solver->solve(vectorOut, vectorIn);
        vectorOut.cscale(norm);
        vectorAuxF.copy(vectorOut);
        seqPropOut->absorb(vectorAuxF, nu, c2);
        }//loop over color

      }//loop over spin

    seqPropOut->apply_gamma(G5);
    seqPropOut->conjugate();
    
    asprintf(&fname, "seqPropOut.CP1.px%dpy%dpz%d.sx%02dsy%02dsz%02dst%02d.tS%02d",sinkMom[0],sinkMom[1],sinkMom[2],
            sourcePositions[isource][0],sourcePositions[isource][1],sourcePositions[isource][2],sourcePositions[isource][3],tSinks[ts]);
    TIME(seqPropOut->writeFile(fname, LIME_FORMAT));

    int signProps1 = (nucleon == PROTON) ? -1: +1;

    //repeat for CP2 part
    if(nucleon == PROTON){
      if(mu>0) {
        mu*=-1.;
        solver->UpdateSolver();
      }
    }else {
      if(mu<0) {
        mu*=-1.;
        solver->UpdateSolver();
      }
    }

    for(int nu = 0 ; nu < 4 ; nu++) {//loop over spin
      for(int c2 = 0 ; c2 < 3 ; c2++) {//loop over color

        PLEGMA_Vector3D<float> vectorAux3D;

        if(nucleon == PROTON) vectorAux3D.seqSourceNucleon(propUP3D, propDN3D, which_proj, nucleon, nu, c2);
        else vectorAux3D.seqSourceNucleon(propDN3D, propUP3D, which_proj, nucleon, nu, c2);

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

        double norm = vectorIn.norm();
        vectorIn.cscale(1/norm);
        solver->solve(vectorOut, vectorIn);
        vectorOut.cscale(norm);
        vectorAuxF.copy(vectorOut);
        seqPropOut->absorb(vectorAuxF, nu, c2);

      }//loop over color

    }//loop over spin

    seqPropOut->apply_gamma(G5);
    seqPropOut->conjugate();
    
    asprintf(&fname, "seqPropOut.CP2.px%dpy%dpz%d.sx%02dsy%02dsz%02dst%02d.tS%02d",sinkMom[0],sinkMom[1],sinkMom[2],
            sourcePositions[isource][0],sourcePositions[isource][1],sourcePositions[isource][2],sourcePositions[isource][3],tSinks[ts]);
    TIME(seqPropOut->writeFile(fname, LIME_FORMAT));

  }//loop over tSinks

  //two point correlator
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
  
  PLEGMA_Correlator<float> corr(corr_space, sourcePositions[isource], 0);
  corr.setFixMomVec(sinkMom_3D);
  
  TIME(corr.contractBaryons(*propUP, *propDN));
  THREAD(corr.writeFile((twop_filename).c_str(), corr_file_format));
  
  }//loop over source positions

  delete propUP;
  delete propDN;

  delete solver;

  finalize();
  return 0;

}
