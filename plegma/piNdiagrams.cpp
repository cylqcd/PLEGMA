#include <PLEGMA.h>
#include <PLEGMA_utils.h>

using namespace plegma;
using namespace quda;

std::vector<double> runtime;
#define TIME(fnc)  runtime.push_back(MPI_Wtime()); fnc;                 \
  PLEGMA_printf("TIME for "#fnc" %f sec\n", MPI_Wtime()-runtime.back()); \
  runtime.pop_back()

std::vector<std::thread> threads;
//#define THREAD(fnc) threads.push_back(std::thread([=]() { TIME(fnc); }))
#define THREAD(fnc) TIME(fnc)

extern int device;
static std::vector<std::string> listOpt = {"verbosity", "load-gauge","nsmear-APE","alpha-APE", "nsmear-gauss","alpha-gauss","nsrc","src-filename", "momlist-filename", "time-dilution"};
// Note here sinkMom is used as the momentum insertion in the sequential souce, probably has to be renamed to seqMom

int main(int argc, char **argv)
{
  initializeOptions(argc, argv, true, listOpt);
  //================ Add your options in this between initializeOptions and initializePLEGMA ================//
  double mu_ud = mu;
  double mu_ud_factor[QUDA_MAX_MG_LEVEL];
  for(int i=0;i<QUDA_MAX_MG_LEVEL;i++) mu_ud_factor[i] = mu_factor[i];
  bool timedilution;
  std::string outfile_V="";
  std::string outfile_upS="";
  std::string outfile_dnS="";
  std::string outfile_SEQ="";
//  std::string outfile_V3;
//  std::string outfile_V2;
//  std::string outfile_V4;
//  std::string path_V="";
//  std::string path_P="";

  HGC_options->set("time-dilution", "Flag for switching time-dilution in stochastic propagators", verbosity, timedilution);
  HGC_options->set("outVector", "Path for saving the vector field used", verbosity, outfile_V);
  HGC_options->set("outPropUP", "Path for saving the up propagator used", verbosity, outfile_upS);
  HGC_options->set("outPropDN", "Path for saving the up propagator used", verbosity, outfile_dnS);
  HGC_options->set("outPropSeq", "Path for saving the sequential propagator used", verbosity, outfile_SEQ);
//  HGC_options->set("outV3", "Path for saving the result of V3_reduction", verbosity, outfile_V3);
//  HGC_options->set("outV2", "Path for saving the result of V3_reduction", verbosity, outfile_V2);
//  HGC_options->set("outV4", "Path for saving the result of V3_reduction", verbosity, outfile_V4);
//  HGC_options->set("loadVector", "Path for loading V", verbosity, path_V);
//  HGC_options->set("loadProp", "Path for loading P", verbosity, path_P);

  //=========================================================================================================//
  initializePLEGMA();
  double start_time, tmp_time;
  {
    
    // Reading from Lime file and loading to device
    PLEGMA_Gauge<double> gauge;
    gauge.readFile(latfile, LIME_FORMAT);
    gauge.load();
    gauge.calculatePlaq();

    //Reading the momentum lists
    momList sourcemomentumList(pathListMomenta);


    //List of gammas
    std::vector<GAMMAS_SCATT> glist_source_delta={CG_1,CG_2,CG_3,CG_1_G_4,CG_2_G_4,CG_3_G_4,CG_1_G_4_G_5,CG_2_G_4_G_5,CG_3_G_4_G_5};
    std::vector<GAMMAS_SCATT> glist_sink_delta={CG_1,CG_2,CG_3,CG_1_G_4,CG_2_G_4,CG_3_G_4,CG_1_G_4_G_5,CG_2_G_4_G_5,CG_3_G_4_G_5};
    std::vector<GAMMAS_SCATT> glist_source_nucleon={CG_5,C,CG_5_G_4,CG_4};
    std::vector<GAMMAS_SCATT> glist_sink_nucleon={CG_5,C,CG_5_G_4,CG_4};
    std::vector<GAMMAS_SCATT> glist_source_nucleon_unpaired={ID,G_5};
    std::vector<GAMMAS_SCATT> glist_sink_nucleon_unpaired={ID,G_5};

    std::vector<GAMMAS_SCATT> glist_sink_meson={G_5};
    std::vector<GAMMAS_SCATT> glist_source_meson={G_5};

    std::vector<GAMMAS_SCATT> glist_source_delta_unpaired={ID};
    std::vector<GAMMAS_SCATT> glist_sink_delta_unpaired={ID};

    
    // Loading to QUDA and computing plaquette also there
    initGaugeQuda(gauge, true, QUDA_WILSON_LINKS);
    plaqQuda();

    PLEGMA_Gauge<double> contractGauge;
    contractGauge.copy(gauge);

    // Smearing
    PLEGMA_Gauge<double> smearedGauge;
    smearedGauge.APEsmearing(contractGauge, nsmearAPE, alphaAPE, 3);
    PLEGMA_printf("Plaquette after smearing:\n");
    smearedGauge.calculatePlaq();
   
    // apply boundary conditions since is needed for the covariant derivative
    applyBoundaryConditions(contractGauge,true);
    //ensuring mu positive
    QUDA_solver solver(mu);

    std::string smearType = ((nsmearGauss>0) ? "SS" : "LL");
    std::string smearString = smearType + "_" + "gN" + std::to_string(nsmearGauss) + "a" + convNumToStr(alphaGauss) + "aN" + std::to_string(nsmearAPE) + "a" + convNumToStr(alphaAPE);


    //Computing time-diluted stochastic propagators and stochastic source

    PLEGMA_Vector<float> vectorStoc_source(BOTH);
    PLEGMA_Vector<float> vectorStoc_propag(BOTH);
    PLEGMA_Vector<float> vectorStoc_source_arch(BOTH); 
    PLEGMA_Vector<double> vectorAuxD1(BOTH);
    PLEGMA_Vector<double> vectorAuxD2(BOTH);
    PLEGMA_Vector<double> vectorInOut;
    PLEGMA_printf("Build vector from scratch\n");
    //Note that we replace the f1<-f2 DN propagator with a stochastic one
    //in two steps actually
    //DN(x_f1 <- x_f2 ) = \phihat(x_f2)(x_f1)\xi^{dagger}(x_f2)(x_f2)
    //where x_f2 is the source
    //      x_f1 is the sink
    //so \phihat(x_f2)(x_f1) is the x_f1 coordinate of the stochastic 
    //propagator created at x_f2 for the down quark
    //=gamma_5*U(x_f2 <- x_f1)^dagger*gamma_5
    //=gamma_5*\xi(x_f1)(x_f1)*\phi(x_f2)(x_f1)^dagger*gamma_5
    //Here we compute phi and xi
    int nroots=4;
    //Step(1) Creating the time-diluted stochastic source
    vectorStoc_source.randInit(1234);
    vectorStoc_source.stochastic_Z(nroots);
    

    //Step(2) Smearing all the time slice
    vectorAuxD1.copy(vectorStoc_source);
    TIME(vectorAuxD2.gaussianSmearing(vectorAuxD1, smearedGauge, nsmearGauss, alphaGauss ));
    //Save the smeared source in order to reuse it for oet.
    vectorStoc_source_arch.copy(vectorAuxD2);

    //We rotate the source to the physical basis
    TIME(vectorAuxD1.rotateToPhysicalBasis(vectorAuxD2,+1));

    //In vectorAuxD2 we store the results for the inversion
    vectorAuxD2.scale(0.0);
    
    if (timedilution){
      PLEGMA_printf("#piNdiagramms: Full time dilution is turned on\n");
      for (int timeidx=0; timeidx< HGC_totalL[DIM_T]; ++timeidx){
        //Step(3) pick out a particular timeslice from the source
        vectorInOut.absorbTimeslice(vectorAuxD1, timeidx);
        //Step(4) Solve
        TIME(solver.solve(vectorInOut, vectorInOut));
        //Step(5) absorbing the particular timeslice to a 4d vector
        vectorAuxD2.absorbTimeslice(vectorInOut, timeidx, false);
      }
    }
    else{
      PLEGMA_printf("#piNdiagramms: No time dilution is used n stochastic propagators\n");
      vectorInOut.copy(vectorAuxD1);
      TIME(solver.solve(vectorInOut, vectorInOut));
      vectorAuxD2.copy(vectorInOut);
    } 

    //vectorStoc_source_arch.writeLIME(outfile_V+"globalTfulltimedilution_source");
    //vectorStoc_source_arch.writeHDF5(outfile_V+"globalTfulltimedilution_source");
    vectorAuxD1.copy(vectorStoc_source_arch);
    vectorAuxD1.apply_gamma5();
    vectorStoc_source.copy(vectorAuxD1);

    //Step(6) We rotate back the propagator to the physical basis
    TIME(vectorAuxD1.rotateToPhysicalBasis(vectorAuxD2,+1));

    //Step(7) Smearing all the time slice in the propagator
    TIME(vectorAuxD2.gaussianSmearing(vectorAuxD1, smearedGauge, nsmearGauss, alphaGauss ));
    
    //vectorAuxD1.writeLIME(outfile_V+"globalTfulltimedilution_propagator");
    //vectorAuxD1.writeHDF5(outfile_V+"globalTfulltimedilution_propagator");
    vectorStoc_propag.copy(vectorAuxD1);
    vectorStoc_propag.apply_gamma5();

    //loop over the soure positions
    for(int isource = 0 ; isource < numSourcePositions; isource++){

      PLEGMA_Gauge3D<double> smearedGauge3D;
      smearedGauge3D.absorb(smearedGauge, sourcePositions[isource][DIM_T]);


      int sequential_time_source=sourcePositions[isource][DIM_T];

      PLEGMA_printf("\n ### Calculations for source-position %d - %02d.%02d.%02d.%02d begin now ###\n\n",
                    isource, sourcePositions[isource][0], sourcePositions[isource][1],
                    sourcePositions[isource][2], sourcePositions[isource][3]);

    
      //Create Propagator
      PLEGMA_Propagator<float> propUP(BOTH);
      PLEGMA_Propagator<float> propDN(BOTH);
      PLEGMA_Propagator<float> propUPDN(BOTH);

      //to measure the smearing time
      tmp_time = 0;


      // ensuring mu positive
      if(mu<0) {
        mu*=-1.;
        solver.UpdateSolver();
      }

      for(int isc = 0 ; isc < 12 ; isc++){
        PLEGMA_Vector<double> vectorInOut;
        PLEGMA_Vector<float> vectorAuxF;
        PLEGMA_Vector<double> vectorAuxD;
        { // Smearing the source
          PLEGMA_Vector3D<double> vector1, vector2;
          vector1.pointSource(sourcePositions[isource], isc/3, isc%3, DEVICE);
          TIME(vector2.gaussianSmearing(vector1, smearedGauge3D, nsmearGauss, alphaGauss));
          vectorInOut.absorb(vector2,sourcePositions[isource][DIM_T]);
        }

        //Rotation to the physical basis
        TIME(vectorAuxD.rotateToPhysicalBasis(vectorInOut,+1));       

        //Inversion
        PLEGMA_printf("Going to invert UP for component %d\n", isc);
        TIME(solver.solve(vectorAuxD, vectorAuxD));

        //Rotation to the physical basis
        TIME(vectorInOut.rotateToPhysicalBasis(vectorAuxD,+1));

        //Smearing at the sink
        start_time = MPI_Wtime();
        TIME(vectorAuxD.gaussianSmearing(vectorInOut, smearedGauge, nsmearGauss, alphaGauss));
        tmp_time += MPI_Wtime()-start_time;

        vectorAuxF.copy(vectorAuxD);
        propUP.absorb(vectorAuxF, isc/3, isc%3);
      }
      /*

      if(outfile_upS!="")
        {
          PLEGMA_printf("Save propagator for the up quark\n");
          PLEGMA_Vector<float> vectorAuxPrint(BOTH);
          for(int isc = 0 ; isc < 12 ; isc++){
            std::string spin=std::to_string(isc/3);
            std::string col=std::to_string(isc%3);

            vectorAuxPrint.absorb(propUP,isc/3,isc%3);
            vectorAuxPrint.unload();
            vectorAuxPrint.writeLIME(outfile_upS+"_s"+spin+"_c"+col);
            vectorAuxPrint.writeHDF5(outfile_upS+"_s"+spin+"_c"+col);
          }
        }

      */
      // ensuring mu negative
      if(mu>0) {
        mu*=-1.;
        solver.UpdateSolver();
      }

      for(int isc = 0 ; isc < 12 ; isc++){
        PLEGMA_Vector<double> vectorInOut;
        PLEGMA_Vector<float> vectorAuxF;
        PLEGMA_Vector<double> vectorAuxD;

        { // Smearing the source
          PLEGMA_Vector3D<double> vector1, vector2;
          vector1.pointSource(sourcePositions[isource], isc/3, isc%3, DEVICE);
          TIME(vector2.gaussianSmearing(vector1, smearedGauge3D, nsmearGauss, alphaGauss));
          vectorInOut.absorb(vector2,sourcePositions[isource][DIM_T]);
        }

        //(3 step) rotation to the physical basis
        TIME(vectorAuxD.rotateToPhysicalBasis(vectorInOut,-1));

        //(4 step) doing the inversion
        PLEGMA_printf("Going to invert DN for component %d\n", isc);
        TIME(solver.solve(vectorAuxD, vectorAuxD));

        start_time = MPI_Wtime();
        //(5 step) rotating to the physical base
        TIME(vectorInOut.rotateToPhysicalBasis(vectorAuxD,-1));

        //(6 step) doing the smearing on the propagator
        TIME(vectorAuxD.gaussianSmearing(vectorInOut, smearedGauge, nsmearGauss, alphaGauss));

        vectorAuxF.copy(vectorAuxD);

        propDN.absorb(vectorAuxF, isc/3, isc%3);
      }

      /*
      if(outfile_dnS!="")
        {
          PLEGMA_printf("Save propagator for the d quark\n");
          PLEGMA_Vector<float> vectorAuxPrint(BOTH);
          for(int isc = 0 ; isc < 12 ; isc++){
            std::string spin=std::to_string(isc/3);
            std::string col=std::to_string(isc%3);

            vectorAuxPrint.absorb(propDN,isc/3,isc%3);
            vectorAuxPrint.unload();
            vectorAuxPrint.writeLIME(outfile_dnS+"_s"+spin+"_c"+col);
            vectorAuxPrint.writeHDF5(outfile_dnS+"_s"+spin+"_c"+col);

          }
        }*/

      std::vector<int> mom={0,0,0};
      PLEGMA_ScattCorrelator<float> diagramm(sourcePositions[isource], mom);
 
      site source=site({0,0,0,sourcePositions[isource][3]});

      PLEGMA_ScattCorrelator<float> diagramm_pion(source, mom);
      PLEGMA_ScattCorrelator<float> reductionsT1(source, sourcemomentumList.uniq_p(3));
      PLEGMA_ScattCorrelator<float> reductionsT2(source, sourcemomentumList.uniq_p(3));


      TIME(reductionsT1.T1(glist_source_delta, glist_sink_delta, propUP, propUP, propUP));
      reductionsT1.writeHDF5("T1sourceforD");


      TIME(reductionsT2.T2(glist_source_delta, glist_sink_delta, propUP, propUP, propUP));
      reductionsT2.writeHDF5("T2sourceforD");


      std::string outfilename="Ddiagramm_Antonino" ;
      TIME(diagramm.D_diagramms( reductionsT1, reductionsT2, glist_source_delta_unpaired, glist_sink_delta_unpaired, outfilename));

      //N diagram
      PLEGMA_ScattCorrelator<float> diagramm_nucleon(sourcePositions[isource], mom);
      //diagramm_nucleon.setSource(source);
      
      TIME(reductionsT1.T1(glist_source_nucleon, glist_sink_nucleon, propUP, propDN, propUP));
      reductionsT1.writeHDF5("T1sourceforN");

      TIME(reductionsT2.T2(glist_source_nucleon, glist_sink_nucleon, propUP, propDN, propUP));
      reductionsT2.writeHDF5("T2sourceforN");


      outfilename = "Ndiagramm_Antonino";
      TIME(diagramm_nucleon.N_diagramms( reductionsT1, reductionsT2, glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, outfilename));

      //Constructing list of momenta that contains for all P_tot all the unique combinations of pf1 pf2 that adds up to p_tot
      /*
      momList sourcemomentumList_forTpiNsink;

      //We first have a loop over all unique the source meson momentum p_i2 
      for (auto momentum_tot : sourcemomentumList.uniq_p(3)) {
        //List of momenta corresponding to a fix value of p_i2
        momList filtered_sourcemomentumList(sourcemomentumList.extract(momentum_tot, 3));
        std::vector<std::vector<int>> momlist_0=filtered_sourcemomentumList.pi(0);
        std::vector<std::vector<int>> momlist_1=filtered_sourcemomentumList.pi(1);
        std::vector<std::vector<int>> momlist_2=filtered_sourcemomentumList.pi(2);
        for (int i=0; i< momlist_0.size(); ++i){
          sourcemomentumList_forTpiNsink.add_mom( momlist_0[i], momlist_1[i], momlist_2[i] );
        }
      }      

      PLEGMA_ScattCorrelator<float> reductionsV2_T(source, sourcemomentumList_forTpiNsink.uniq_p(1));
      PLEGMA_ScattCorrelator<float> reductionsV3_T(source, sourcemomentumList_forTpiNsink.uniq_p(2));

      reductionsV3_T.V3( vectorStoc_propag, glist_sink_meson, propUP);
      reductionsV3_T.writeHDF5("V3sourceforTPINSINK");

      reductionsV2_T.V2( vectorStoc_source, glist_sink_nucleon, propUP, propUP);
      reductionsV2_T.writeHDF5("V2sourceforTPINSINK");

      outfilename = "Tdiagramm_piNsinkAntonino";
      diagramm.T_diagramms_piNsink(sourcemomentumList_forTpiNsink, reductionsV2_T, reductionsV3_T, glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_delta, outfilename);*/

      // ensuring mu positive
      if(mu<0) {
        mu*=-1.;
        solver.UpdateSolver();
      }

      PLEGMA_Vector<float> vectorStoc_source_oet;
      PLEGMA_Vector<float> vectortmp1;
      PLEGMA_Vector<float> vectortmp2;
          
      PLEGMA_Vector<float> stochastic_source_spin_diluted_momzero; 
      std::array<PLEGMA_Vector<float>,4> stochastic_propagator_momzero;

      vectorStoc_source_oet.randInit(1234);
      vectorStoc_source_oet.stochastic_Z(nroots);


      vectortmp1.absorbTimeslice(vectorStoc_source_oet, sequential_time_source);
      //Transforming to physical base
      vectortmp2.rotateToPhysicalBasis(vectortmp1,+1);
      stochastic_source_spin_diluted_momzero.dilutespin(vectortmp2,0);

      for (int spinindex=0; spinindex<4; ++spinindex){
        PLEGMA_Vector<double> vectorAuxD;

        //tmp_time += MPI_Wtime()-start_time;       
        //stochastic_source_spin_diluted_momzero.writeLIME(outfile_V+"source_zero_momentum"+std::to_string(spinindex));         
        //Doing the zero momentum stochastic propagator with spin dilution
        vectorInOut.copy(stochastic_source_spin_diluted_momzero);
        //Doing the inversion
        TIME(solver.solve(vectorInOut, vectorInOut));
        //Rotate back immediately to the physical basis
        vectorAuxD.rotateToPhysicalBasis(vectorInOut,+1);
        //Gaussian smearing of the propagator
        TIME(vectorInOut.gaussianSmearing(vectorAuxD, smearedGauge, nsmearGauss, alphaGauss));
        stochastic_propagator_momzero[spinindex].copy(vectorInOut);
        //tmp_time += MPI_Wtime()-start_time;
        //stochastic_propagator_momzero[spinindex].writeLIME(outfile_V+"propagator_zero_momentum"+std::to_string(spinindex));
        //stochastic_propagator_momzero[spinindex].writeHDF5(outfile_V+"propagator_zero_momentum"+std::to_string(spinindex));
        if (spinindex<3){
          vectortmp1.dilutespindisplace(stochastic_source_spin_diluted_momzero,spinindex+1,spinindex);
          stochastic_source_spin_diluted_momzero.copy(vectortmp1);
        }
      }


      //We first have a loop over all unique the source meson momentum p_i2 
      for (auto momentum_i2 : sourcemomentumList.uniq_p(0)) {


        //List of momenta corresponding to a fix value of p_i2
        momList filtered_sourcemomentumList(sourcemomentumList.extract(momentum_i2, 0));

        PLEGMA_ScattCorrelator<float> reductionsV2(source, filtered_sourcemomentumList.uniq_p(1));
        PLEGMA_ScattCorrelator<float> reductionsV3(source, filtered_sourcemomentumList.uniq_p(2));

        //Loop over the different gamma structure for the source meson
        for (auto gamma_i2 : glist_source_meson) {
         
          // Computing sequential propagators f1 <- i_2 <- i_1 
          // so the sequential source source time is fixed
          // and the momentum is also fixed to be momentum_i2

          //smearing the 3D propagators
          PLEGMA_Propagator3D<float> propDN3D;      
          for(int isc = 0 ; isc < 12 ; isc++){
            PLEGMA_Vector<double> vectorAuxD;
            PLEGMA_Vector<float> vectorAuxF;
            PLEGMA_Vector<double> vectorAuxD2;
            vectorAuxF.absorb(propDN,isc/3, isc%3);
            vectorAuxD.copy(vectorAuxF);
            start_time = MPI_Wtime();

            //Performing the smearing
            // Smearing the source
            {
              PLEGMA_Vector3D<double> vector1, vector2;
              vectorAuxF.absorb(propDN, isc/3, isc%3);
              vectorAuxD.copy(vectorAuxF);
              vector1.absorb( vectorAuxD, sequential_time_source );
              TIME(vector2.gaussianSmearing(vector1, smearedGauge3D, nsmearGauss, alphaGauss));
              vectorAuxD2.absorb(vector2,sourcePositions[isource][DIM_T]);
            }

            //Perform multiplication with gamma_i2
            vectorAuxD2.apply_gamma_scatt(gamma_i2);

            //Perform rotation to the physical basis
            TIME(vectorAuxD.rotateToPhysicalBasis(vectorAuxD2,+1));

            
            vectorAuxF.copy(vectorAuxD);
            propDN3D.absorb(vectorAuxF, sequential_time_source, isc/3, isc%3);
          }
 
          propDN3D.mulMomentumPhases(momentum_i2,1);

          //Computing sequential propagators UD T_fii with insertion
          //gamma_i2 and momentum SinkMom
          for(int isc = 0 ; isc < 12 ; isc++){
            PLEGMA_Vector<double> vectorInOut;
            PLEGMA_Vector<float> vectorAuxF;
            PLEGMA_Vector<double> vectorAuxD;
          
            vectorAuxF.absorb(propDN3D, sequential_time_source, isc/3, isc%3);
            vectorInOut.copy(vectorAuxF);
            PLEGMA_printf("Going to invert UP for sequential propagator DN  for component %d\n", isc);
            //performing the inversion
            TIME(solver.solve(vectorInOut, vectorInOut));
            start_time = MPI_Wtime();

            //performing rotation to physical base
            TIME(vectorAuxD.rotateToPhysicalBasis(vectorInOut,+1));

            //performing smearing
            TIME(vectorInOut.gaussianSmearing(vectorAuxD, smearedGauge, nsmearGauss, alphaGauss));
            tmp_time += MPI_Wtime()-start_time;

            vectorAuxF.copy(vectorInOut);
            propUPDN.absorb(vectorAuxF, isc/3, isc%3);
          }

          /*
          if(outfile_SEQ!="")
          {
             PLEGMA_printf("Save sequential propagator for the ud \n");
             PLEGMA_Vector<float> vectorAuxPrint(BOTH);
             for(int isc = 0 ; isc < 12 ; isc++){
               std::string spin=std::to_string(isc/3);
               std::string col=std::to_string(isc%3);
               vectorAuxPrint.absorb(propUPDN,isc/3,isc%3);
               vectorAuxPrint.unload();
               vectorAuxPrint.writeLIME(outfile_SEQ+"_s"+spin+"_c"+col);
               vectorAuxPrint.writeHDF5(outfile_SEQ+"_s"+spin+"_c"+col);
             }
          }*/

          PLEGMA_printf("Smearing time %lf sec\n",tmp_time);

          //Compute triangle diagramms          
          

          PLEGMA_ScattCorrelator<float> reductionsT1triangle(source, filtered_sourcemomentumList.uniq_p(3));
         
          PLEGMA_ScattCorrelator<float> reductionsT3triangle(source, filtered_sourcemomentumList.uniq_p(3));

          PLEGMA_ScattCorrelator<float> reductionsT5triangle(source, filtered_sourcemomentumList.uniq_p(3));
          TIME(reductionsT1triangle.T1(glist_source_nucleon, glist_sink_delta, propUPDN, propUP  , propUP));
          reductionsT1triangle.writeHDF5("T1sourceforT");
          TIME(reductionsT3triangle.T1(glist_source_nucleon, glist_sink_delta, propUP  , propUPDN, propUP));
          reductionsT3triangle.writeHDF5("T3sourceforT");
          TIME(reductionsT5triangle.T2(glist_source_nucleon, glist_sink_delta, propUP  , propUP, propUPDN));
          reductionsT5triangle.writeHDF5("T5sourceforT");

          outfilename="Tdiagramm_Antonino";
          TIME(diagramm.T_diagramms(filtered_sourcemomentumList, reductionsT1triangle, reductionsT3triangle, reductionsT5triangle, gamma_i2, glist_source_nucleon_unpaired, glist_sink_delta_unpaired,  outfilename));


          //Compute Diagram B1 and B2 

          TIME(reductionsV3.V3( vectorStoc_propag, glist_sink_meson, propUPDN));
          reductionsV3.writeHDF5("V3sourceforB1");

          TIME(reductionsV2.V2( vectorStoc_source, glist_sink_nucleon, propUP, propUP));
          reductionsV2.writeHDF5("V2sourceforB1");
 
          outfilename="Bdiagramm_Antonino" ;
          TIME(diagramm.B_diagramms(filtered_sourcemomentumList, reductionsV3, reductionsV2, glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, gamma_i2, glist_source_nucleon, outfilename, 1));
	  
          TIME(diagramm.B_diagramms(filtered_sourcemomentumList, reductionsV3, reductionsV2, glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, gamma_i2, glist_source_nucleon, outfilename, 2));

          //Compute Diagram W1,W2
          
          TIME(reductionsV3.V3( vectorStoc_propag, glist_sink_meson, propUP));
          reductionsV3.writeHDF5("V3sourceforW12");
          TIME(reductionsV2.V2( vectorStoc_source, glist_sink_nucleon, propUP, propUPDN));
          reductionsV2.writeHDF5("V2sourceforW12");

          outfilename= "Wdiagramm_Antonino";
          TIME(diagramm.W_diagramms(filtered_sourcemomentumList, reductionsV3, reductionsV2, glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, gamma_i2, glist_source_nucleon, outfilename, 1));

          TIME(diagramm.W_diagramms(filtered_sourcemomentumList, reductionsV3, reductionsV2, glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, gamma_i2, glist_source_nucleon, outfilename, 2));


          //Compute Diagram W3,W4
          //
          TIME(reductionsV2.V2( vectorStoc_source, glist_sink_nucleon, propUPDN, propUP));
          reductionsV2.writeHDF5("V2sourceforW34");

          TIME(diagramm.W_diagramms(filtered_sourcemomentumList, reductionsV3, reductionsV2, glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, gamma_i2, glist_source_nucleon, outfilename, 3));

          TIME(diagramm.W_diagramms(filtered_sourcemomentumList, reductionsV3, reductionsV2, glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, gamma_i2, glist_source_nucleon, outfilename, 4));

       } //loop over gamma i2
         
       std::string outfilename="Zdiagramm_Antonino" ;

      
       //Producing spin diluted stochastic propagators for diagram Z1,Z2,Z3,Z4
              
       std::array<PLEGMA_Vector<float>,4> stochastic_propagator_momp_i2;

       std::array<PLEGMA_ScattCorrelator<float> ,4> reductionsV3_diluted = {
           PLEGMA_ScattCorrelator<float>(source, filtered_sourcemomentumList.uniq_p(2)),
           PLEGMA_ScattCorrelator<float>(source, filtered_sourcemomentumList.uniq_p(2)),
           PLEGMA_ScattCorrelator<float>(source, filtered_sourcemomentumList.uniq_p(2)),
           PLEGMA_ScattCorrelator<float>(source, filtered_sourcemomentumList.uniq_p(2))
         };

       std::array<PLEGMA_ScattCorrelator<float>,4> reductionsV2_diluted = {
           PLEGMA_ScattCorrelator<float>(source, filtered_sourcemomentumList.uniq_p(1)),
           PLEGMA_ScattCorrelator<float>(source, filtered_sourcemomentumList.uniq_p(1)),
           PLEGMA_ScattCorrelator<float>(source, filtered_sourcemomentumList.uniq_p(1)),
           PLEGMA_ScattCorrelator<float>(source, filtered_sourcemomentumList.uniq_p(1))
       };


       PLEGMA_Vector<float> stochastic_source_spin_diluted_momp_i2; 
       //Using the already generated stochastic source and project it to a time-slice
       //Smearing was already performed
       //Creating oet time-slice source
       vectortmp1.absorbTimeslice(vectorStoc_source_arch, sequential_time_source);
       //Transforming to physical base
       TIME(vectortmp2.rotateToPhysicalBasis(vectortmp1,+1));
       
       //Multiplying by the appropriate momentum phase

       std::vector<int> tmp_4Dmom= momentum_i2 ; 
       tmp_4Dmom.push_back(0);
       vectortmp2.mulMomentumPhases(tmp_4Dmom,-1);
       stochastic_source_spin_diluted_momp_i2.dilutespin(vectortmp2,0);
 
       
       for (int spinindex=0; spinindex<4; ++spinindex){
         PLEGMA_Vector<double> vectorAuxD;

         //tmp_time += MPI_Wtime()-start_time;       
         //stochastic_source_spin_diluted_momp_i2.writeLIME(outfile_V+"source_fini_momentum"+std::to_string(spinindex));
       
 
         vectorInOut.copy(stochastic_source_spin_diluted_momp_i2);
         //Doing the inversion
         TIME(solver.solve(vectorInOut, vectorInOut));

         //Rotate back immediately to the physical basis
         TIME(vectorAuxD.rotateToPhysicalBasis(vectorInOut,+1));

         //performing smearing
         TIME(vectorInOut.gaussianSmearing(vectorAuxD, smearedGauge, nsmearGauss, alphaGauss));

         //Saving the propagator
         stochastic_propagator_momp_i2[spinindex].copy(vectorInOut);

         //tmp_time += MPI_Wtime()-start_time;         
         //stochastic_propagator_momp_i2[spinindex].writeLIME(outfile_V+"propagator_fini_momentum"+std::to_string(spinindex));
         
         if (spinindex<3){
           vectortmp1.dilutespindisplace(stochastic_source_spin_diluted_momp_i2,spinindex+1,spinindex);
           stochastic_source_spin_diluted_momp_i2.copy(vectortmp1);
         }

       }      

       //Diagram Z1,Z2
       std::vector<GAMMAS_SCATT> gamma_5_t_sinkmeson=apply_gamma5_scatt_gamma(glist_sink_meson,LEFT);       
       std::vector<GAMMAS_SCATT>  sourcemeson_t_gamma_5=apply_gamma5_scatt_gamma(glist_source_meson,RIGHT);
       for (int i=0; i< 4; ++i){
         TIME(reductionsV3_diluted[i].V3( stochastic_propagator_momp_i2[i], gamma_5_t_sinkmeson, propUP));
         reductionsV3_diluted[i].writeHDF5("V3sourceforZ"+std::to_string(i));

         TIME(reductionsV2_diluted[i].V4( stochastic_propagator_momzero[i], glist_sink_nucleon, propDN, propUP));
         reductionsV2_diluted[i].writeHDF5("V4sourceforZ"+std::to_string(i));
       }

       TIME(diagramm.Z_diagramms(filtered_sourcemomentumList, reductionsV3_diluted, reductionsV2_diluted, glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, sourcemeson_t_gamma_5, glist_source_nucleon, outfilename, 1));


       TIME(diagramm.Z_diagramms(filtered_sourcemomentumList, reductionsV3_diluted, reductionsV2_diluted, glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, sourcemeson_t_gamma_5, glist_source_nucleon, outfilename, 2));
 
      
       //Diagram Z3,Z4	
       for (int i=0; i< 4; ++i){

         TIME(reductionsV2_diluted[i].V2( stochastic_propagator_momzero[i], glist_sink_nucleon, propDN, propUP));
         reductionsV2_diluted[i].writeHDF5("V2sourceforZ"+std::to_string(i));

       }

       TIME(diagramm.Z_diagramms(filtered_sourcemomentumList, reductionsV3_diluted, reductionsV2_diluted, glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, sourcemeson_t_gamma_5, glist_source_nucleon,  outfilename, 3));


       TIME(diagramm.Z_diagramms(filtered_sourcemomentumList, reductionsV3_diluted, reductionsV2_diluted, glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, sourcemeson_t_gamma_5, glist_source_nucleon,  outfilename, 4));



       //M diagram N.B. I still need Phi_0, Phi_1 here! So even if we decide to enclose Phi's plegma_vectors in a smaller scope, we need to move this diagram too.
       
       std::vector<GAMMAS_SCATT> glist_sourcemeson_g5 = apply_gamma5_scatt_gamma(glist_source_meson,RIGHT);       
       std::vector<GAMMAS_SCATT> glist_sinkmeson_g5 = apply_gamma5_scatt_gamma(glist_sink_meson,LEFT);
       outfilename = "Pdiagramm_Antonino";

       TIME(diagramm_pion.P_diagramms( momentum_i2, glist_sourcemeson_g5, glist_sinkmeson_g5, stochastic_propagator_momzero, stochastic_propagator_momp_i2, outfilename));
       
       outfilename = "Mdiagramm_Antonino";
       TIME(diagramm.M_diagramms( sourcemomentumList, filtered_sourcemomentumList, diagramm_nucleon, glist_sourcemeson_g5, glist_sinkmeson_g5, stochastic_propagator_momzero, stochastic_propagator_momp_i2, outfilename));


      }//loop over unique set of momenta for p_i2

    } //loop over source position

  } 
  finalize();
  
  return 0;
}
