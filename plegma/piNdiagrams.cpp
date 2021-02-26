
#include <PLEGMA.h>
#include <PLEGMA_utils.h>
#include <omp.h>

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
static std::vector<std::string> listOpt = {"verbosity", "load-gauge","nsmear-APE","alpha-APE", "nsmear-gauss","alpha-gauss","nsrc","src-filename", "momlist-filename", "readStochSamples","time-dilution","nstochSamples","confnumber"};
// Note here sinkMom is used as the momentum insertion in the sequential souce, probably has to be renamed to seqMom

int main(int argc, char **argv)
{
  initializeOptions(argc, argv, true, listOpt);
  //================ Add your options in this between initializeOptions and initializePLEGMA ================//
  double mu_ud = mu;
  double mu_ud_factor[QUDA_MAX_MG_LEVEL];
  for(int i=0;i<QUDA_MAX_MG_LEVEL;i++) mu_ud_factor[i] = mu_factor[i];
  bool timedilution;
  bool readstochastic;
  int n_coherent_source;
  int n_stochastic_samples;
  int nroots=4;
  int confnumber_int;
  int rand_seed1=1234;
  int rand_seed2=1234;
  std::string outfile_V="";
  std::string outfile_upS="";
  std::string outfile_dnS="";
  std::string outfile_SEQ="";
  std::string outdiagramPrefix="";
  std::string outfile_V3;
  std::string outfile_V2;
  std::string outfile_V4;
  HGC_options->set("confnumber", "Integer determining the index of the gauge configuration", verbosity, confnumber_int);
  HGC_options->set("readStochSamples", "Flag for switching read/building stochastic propagators", verbosity, readstochastic);
  HGC_options->set("time-dilution", "Flag for switching time-dilution in stochastic propagators", verbosity, timedilution);
  HGC_options->set("n_coherent_source", "Flag for switching time-dilution in stochastic propagators", verbosity, n_coherent_source);
  HGC_options->set("outVector", "Path for saving the vector field used", verbosity, outfile_V);
  HGC_options->set("outPropUP", "Path for saving the up propagator used", verbosity, outfile_upS);
  HGC_options->set("outPropDN", "Path for saving the dn propagator used", verbosity, outfile_dnS);
  HGC_options->set("outPropSeq", "Path for saving the sequential propagator used", verbosity, outfile_SEQ);
  HGC_options->set("outdiagramPrefix", "Prefix of the resulting diagrams", verbosity, outdiagramPrefix);
  HGC_options->set("nstochSamples", "Number of stochastic samples", verbosity, n_stochastic_samples);
  HGC_options->set("outV3", "Path for saving the result of V2_reduction", verbosity, outfile_V3);
  HGC_options->set("outV2", "Path for saving the result of V3_reduction", verbosity, outfile_V2);
  HGC_options->set("outV4", "Path for saving the result of V4_reduction", verbosity, outfile_V4);
  HGC_options->set("seed1", "Seed for initialization of stochastic sources for the oet", verbosity, rand_seed1);
  HGC_options->set("seed2", "Seed for intiialization of stochastic sources", verbosity, rand_seed2);

  //=========================================================================================================//
  initializePLEGMA();
  {

    //Storing only the smeared gauge
    PLEGMA_Gauge<double> smearedGauge;

    {
      // Reading from Lime file and loading to device
      PLEGMA_Gauge<double> gauge;
      gauge.readFile(latfile, LIME_FORMAT);
      gauge.calculatePlaq();

      // Loading to QUDA and computing plaquette also there
      initGaugeQuda(gauge, true);
      plaqQuda();

      // Smearing
      TIME(smearedGauge.APEsmearing(gauge, nsmearAPE, alphaAPE, 3));
      PLEGMA_printf("Plaquette after smearing:\n");
      smearedGauge.calculatePlaq();
    }

    updateOptions(LIGHT);
    TIME(QUDA_solver solver(mu));


    //Get the confnumber for latfile
    char *ssource;
    asprintf(&ssource,"%04d", confnumber_int);
    std::string confnumber= ssource;
    free(ssource);

    //Reading the momentum lists
    PLEGMA_printf("###Momentum list read from : %s", pathListMomenta.c_str());
    momList sourcemomentumList(3,pathListMomenta,{1,2});
    PLEGMA_printf("N momenta in sourcemomentumList: %d",sourcemomentumList.size());

    if(sourcemomentumList.empty())
      PLEGMA_error("momentumList empty");


    //List of gammas
    std::vector<GAMMAS_SCATT> glist_source_delta={CG_1,CG_2,CG_3};
    std::vector<GAMMAS_SCATT> glist_sink_delta={CG_1,CG_2,CG_3};
    std::vector<GAMMAS_SCATT> glist_source_nucleon={CG_5};
    std::vector<GAMMAS_SCATT> glist_sink_nucleon={CG_5};
    std::vector<GAMMAS_SCATT> glist_source_nucleon_unpaired={ID};
    std::vector<GAMMAS_SCATT> glist_sink_nucleon_unpaired={ID};
    std::vector<GAMMAS_SCATT> glist_source_meson_T={ID};

    std::vector<GAMMAS_SCATT> glist_sink_meson={G_5};
    std::vector<GAMMAS_SCATT> glist_source_meson={G_5};

    std::vector<GAMMAS_SCATT> glist_source_delta_unpaired={ID};
    std::vector<GAMMAS_SCATT> glist_sink_delta_unpaired={ID};

    
    std::string smearType = ((nsmearGauss>0) ? "SS" : "LL");
    std::string smearString = smearType + "_" + "gN" + std::to_string(nsmearGauss) + "a" + convNumToStr(alphaGauss) + "aN" + std::to_string(nsmearAPE) + "a" + convNumToStr(alphaAPE);


    //Computing time-diluted stochastic propagators and stochastic source

    std::vector<PLEGMA_Vector<double>*> stochastic_sources;
    std::vector<PLEGMA_Vector<double>*> stochastic_propags;

    for(int i=0; i< n_stochastic_samples; ++i) {
      stochastic_sources.push_back(new PLEGMA_Vector<double>(HOST));
      stochastic_propags.push_back(new PLEGMA_Vector<double>(HOST));
    }

    PLEGMA_Vector<double> vectorStoc_source_oet;
    vectorStoc_source_oet.randInit(rand_seed1);

    PLEGMA_printf("Start producing stochastic vectors and propagators\n");
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
    //Producing the stochastic source
    if (readstochastic==0){
      
      PLEGMA_Vector<double> vectorAuxD1(BOTH);
      PLEGMA_Vector<double> vectorAuxD2(BOTH);
      PLEGMA_Vector<double> vectorInOut;
      PLEGMA_Vector<double> vectorSource(BOTH);
      vectorSource.randInit(rand_seed2);
      for (int i=0; i<n_stochastic_samples; ++i){
        //Step(1) Creating the time-diluted stochastic source
        vectorSource.stochastic_Z(nroots);
      
        //Step(2) Save it on the host memory
        {
          PLEGMA_Vector<float> vectorAuxF;
          vectorAuxF.copy(vectorSource);
          vectorAuxF.unload();
          vectorAuxF.writeLIME(outfile_V+"globalTfulltimedilution_source_nstoch"+std::to_string(i)+"_"+confnumber);
        }
        vectorAuxD1.copy(vectorSource);
        vectorAuxD1.apply_gamma5();

        vectorAuxD1.unload();
        stochastic_sources[i]->copy(vectorAuxD1,HOST);
        vectorAuxD1.load();
        vectorAuxD1.copy(vectorSource);
     
        //Step(3) Smearing all the time slice
        TIME(vectorAuxD2.gaussianSmearing(vectorAuxD1, smearedGauge, nsmearGauss, alphaGauss ));
 
        //Step(4) We rotate the source to the physical basis
        TIME(vectorAuxD1.rotateToPhysicalBasis(vectorAuxD2,+1));

        //In vectorAuxD2 we store the results for the inversion
        vectorAuxD2.scale(0.0);
    
        if (timedilution){
          PLEGMA_printf("#piNdiagramms: Full time dilution is turned on\n");
          for (int timeidx=0; timeidx< HGC_totalL[DIM_T]; ++timeidx){
            //Step(5) pick out a particular timeslice from the source
            vectorInOut.absorbTimeslice(vectorAuxD1, timeidx);
            //Step(6) Solve
            TIME(solver.solve(vectorInOut, vectorInOut));
            //Step(7) absorbing the particular timeslice to a 4d vector
            vectorAuxD2.absorbTimeslice(vectorInOut, timeidx, false);
          }
        }
        else{
          PLEGMA_printf("#piNdiagramms: No time dilution is used n stochastic propagators\n");
          vectorInOut.copy(vectorAuxD1);
          TIME(solver.solve(vectorInOut, vectorInOut));
          vectorAuxD2.copy(vectorInOut);
        } 
    

        //Step(6) We rotate back the propagator to the physical basis
        TIME(vectorAuxD1.rotateToPhysicalBasis(vectorAuxD2,+1));

        //Step(7) Smearing all the time slice in the propagator
        TIME(vectorAuxD2.gaussianSmearing(vectorAuxD1, smearedGauge, nsmearGauss, alphaGauss ));

        //Step(8) Save the propagator to the disk
        {
          PLEGMA_Vector<float> vectorAuxF;
          vectorAuxF.copy(vectorAuxD2);
          vectorAuxF.unload();
          vectorAuxF.writeLIME(outfile_V+"globalTfulltimedilution_propagator_nstoch"+std::to_string(i)+"_"+confnumber);
        }

        vectorAuxD2.apply_gamma5();

        //Step(9) Save the propagator to the host memory
        vectorAuxD2.unload();
        stochastic_propags[i]->copy(vectorAuxD2,HOST);
        vectorAuxD2.load();
        
      } //loop over the stochastic samples

    }
    else{
      for (int i=0; i<n_stochastic_samples; ++i){
        std::string inputfilename=outfile_V+"globalTfulltimedilution_source_nstoch"+std::to_string(i)+"_"+confnumber;
        PLEGMA_printf("Read stochastic source from: %s\n",inputfilename.c_str());
        PLEGMA_Vector<float> vectorRead(BOTH);
        vectorRead.readFile(inputfilename,LIME_FORMAT);
        vectorRead.load();
        vectorRead.apply_gamma5();
        vectorRead.unload();
        stochastic_sources[i]->copy(vectorRead,HOST);
        inputfilename=outfile_V+"globalTfulltimedilution_propagator_nstoch"+std::to_string(i)+"_"+confnumber;
        PLEGMA_printf("Read propagator from: %s\n",inputfilename.c_str());
        vectorRead.readFile(inputfilename,LIME_FORMAT);
        vectorRead.load();
        vectorRead.apply_gamma5();
        vectorRead.unload();
        stochastic_propags[i]->copy(vectorRead,HOST);
      }
    }

    //loop over the soure positions
    for(int isource = 0 ; isource < numSourcePositions; isource++){

      //Creating look up tables for the coherent time-slice sources
      int *coherent_source_table=NULL;
      int *coherent_source_table_timeslice=NULL;
      int **coherent_look_up_table=NULL;
      //if (n_coherent_source > 1){

      coherent_source_table=(int *)malloc(sizeof(int)*n_coherent_source);
      for (int i_coherent_source=0; i_coherent_source < n_coherent_source; ++i_coherent_source){
        coherent_source_table[i_coherent_source]=(sourcePositions[isource][DIM_T]+i_coherent_source*HGC_totalL[DIM_T]/n_coherent_source)%HGC_totalL[DIM_T];
      }
      coherent_source_table_timeslice=(int *)malloc(sizeof(int)*HGC_totalL[DIM_T]);
      for (int i_coherent_source=0; i_coherent_source < n_coherent_source; ++i_coherent_source){
        for (int i=0; i<=HGC_totalL[DIM_T]/(2*n_coherent_source); ++i){
          coherent_source_table_timeslice[(coherent_source_table[i_coherent_source]+i)%HGC_totalL[DIM_T]]=i_coherent_source;
        }
        for (int i=1; i<(HGC_totalL[DIM_T]/(2*n_coherent_source));++i){
	  coherent_source_table_timeslice[(coherent_source_table[i_coherent_source]-i+HGC_totalL[DIM_T])%HGC_totalL[DIM_T]]=i_coherent_source;
        }
      }
      coherent_look_up_table=(int **)malloc(sizeof(int*)*n_coherent_source);
      for (int i=0; i<n_coherent_source; ++i){
        int k=0;
        coherent_look_up_table[i]=(int *)malloc(sizeof(int)*HGC_totalL[DIM_T]/n_coherent_source);
        for (int j=0; j<=HGC_totalL[DIM_T]/(2*n_coherent_source); ++j){
	  coherent_look_up_table[i][k]=(coherent_source_table[i]+j)%HGC_totalL[DIM_T];
	  k++;
	}
	for (int j=1;j<(HGC_totalL[DIM_T]/(2*n_coherent_source));++j){
          coherent_look_up_table[i][k]=(coherent_source_table[i]-j+HGC_totalL[DIM_T])%HGC_totalL[DIM_T];
          k++;    
	}
      }

      //Calculations for source-position Calculations for source-position 
      PLEGMA_printf("\n ### Calculations for source-position %d - %02d.%02d.%02d.%02d begin now ###\n\n",
                    isource, sourcePositions[isource][0], sourcePositions[isource][1],
                    sourcePositions[isource][2], sourcePositions[isource][3]);
      for(int icoherentsource; icoherentsource < n_coherent_source; ++icoherentsource){
	PLEGMA_printf("\n ### Calculations for coherent-source-numbedr %d - timeslice %03d begin now ###\n\n",
                    icoherentsource, sourcePositions[isource][3]+icoherentsource*HGC_totalL[DIM_T]/n_coherent_source);
      }
 
      //Create Propagator
      PLEGMA_Propagator<float> propUP(BOTH); //To be saved for all the coherent sources.
      PLEGMA_Propagator<float> propDN(BOTH);

      std::vector<std::vector<int>> mpf1 = sourcemomentumList.uniq_p(1);
      momList list_mpf1(1,{mpf1,},{0,});
      PLEGMA_ScattCorrelator<float> corrN(sourcePositions[isource], list_mpf1 );
      TIME(corrN.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_sink_nucleon,"N"));


      for(int icoherentsource=0;icoherentsource < n_coherent_source; ++icoherentsource){

        PLEGMA_Propagator<float> propUP_coherent;
        PLEGMA_Propagator<float> propDN_coherent;

        asprintf(&ssource,"sx%02dsy%02dsz%02dst%03d", sourcePositions[isource][0], sourcePositions[isource][1], sourcePositions[isource][2], coherent_source_table[icoherentsource]);
        std::string sourcepositiontext= (std::string)"_" + ssource; 
        free(ssource);

        PLEGMA_Gauge3D<double> smearedGauge3D;
        smearedGauge3D.absorb(smearedGauge, coherent_source_table[icoherentsource]);

        // ensuring mu positive
        if(mu<0) {
          mu*=-1.;
          solver.UpdateSolver();
        }

        site src;
        src=site({sourcePositions[isource][0],
                  sourcePositions[isource][1],
	          sourcePositions[isource][2],
	          coherent_source_table[icoherentsource]});
        for(int isc = 0 ; isc < 12 ; isc++){
          PLEGMA_Vector<double> vectorInOut;
          PLEGMA_Vector<float>  vectorAuxF;
          PLEGMA_Vector<double> vectorAuxD;
          { // Smearing the source
            PLEGMA_Vector3D<double> vector1, vector2;
            vector1.pointSource(src, isc/3, isc%3, DEVICE);
            TIME(vector2.gaussianSmearing(vector1, smearedGauge3D, nsmearGauss, alphaGauss));
            vectorInOut.absorb(vector2,coherent_source_table[icoherentsource]);
          }

          //Rotation to the physical basis
          TIME(vectorAuxD.rotateToPhysicalBasis(vectorInOut,+1));       

          //Inversion
          PLEGMA_printf("Going to invert UP for component %d\n", isc);
          TIME(solver.solve(vectorAuxD, vectorAuxD));

          //Rotation to the physical basis
          TIME(vectorInOut.rotateToPhysicalBasis(vectorAuxD,+1));

          //Smearing at the sink
          TIME(vectorAuxD.gaussianSmearing(vectorInOut, smearedGauge, nsmearGauss, alphaGauss));

          vectorAuxF.copy(vectorAuxD);
          propUP_coherent.absorb(vectorAuxF, isc/3, isc%3);

        }
        for (int timeslice=0; timeslice<HGC_totalL[DIM_T]/n_coherent_source; ++timeslice){
          propUP.absorbTimeslice(propUP_coherent, coherent_look_up_table[icoherentsource][timeslice], false);
        }
        // ensuring mu negative
        if(mu>0) {
          mu*=-1.;
          solver.UpdateSolver();
        }

        for(int isc = 0 ; isc < 12 ; isc++){
          PLEGMA_Vector<double> vectorInOut;
          PLEGMA_Vector<float>  vectorAuxF;
          PLEGMA_Vector<double> vectorAuxD;

          {  // Smearing the source
            PLEGMA_Vector3D<double> vector1, vector2;
            vector1.pointSource(src, isc/3, isc%3, DEVICE);
            TIME(vector2.gaussianSmearing(vector1, smearedGauge3D, nsmearGauss, alphaGauss));
            vectorInOut.absorb(vector2,coherent_source_table[icoherentsource]);
          }

          //(3 step) rotation to the physical basis
          TIME(vectorAuxD.rotateToPhysicalBasis(vectorInOut,-1));

          //(4 step) doing the inversion
          PLEGMA_printf("Going to invert DN for component %d\n", isc);
          TIME(solver.solve(vectorAuxD, vectorAuxD));

          //(5 step) rotating to the physical base
          TIME(vectorInOut.rotateToPhysicalBasis(vectorAuxD,-1));

          //(6 step) doing the smearing on the propagator
          TIME(vectorAuxD.gaussianSmearing(vectorInOut, smearedGauge, nsmearGauss, alphaGauss));

          vectorAuxF.copy(vectorAuxD);

          propDN_coherent.absorb(vectorAuxF, isc/3, isc%3);
        }
        for (int timeslice=0; timeslice<HGC_totalL[DIM_T]/n_coherent_source; ++timeslice){
	  propDN.absorbTimeslice(propDN_coherent, coherent_look_up_table[icoherentsource][timeslice], false);
        } 

        std::vector<int> mom={0,0,0};
      
        site source=site({0,0,0, coherent_source_table[icoherentsource]});
        std::string outfilename;

        //D diagram
        {
	  std::vector<std::vector<int>> mtot = sourcemomentumList.uniq_p(3);
	  momList list_mtot(1,{mtot,},{0,});
	  PLEGMA_ScattCorrelator<float> corrD(src,list_mtot);

	  //initialize diagram
	  corrD.initialize_diagram( glist_source_delta_unpaired, glist_sink_delta_unpaired,glist_source_delta, glist_sink_delta,"D");
      
	  PLEGMA_ScattCorrelator<float> reductionsT1(source, mtot);
	  PLEGMA_ScattCorrelator<float> reductionsT2(source, mtot);

	  TIME(reductionsT1.T1(glist_source_delta, glist_sink_delta, propUP_coherent, propUP_coherent, propUP_coherent));

	  TIME(reductionsT2.T2(glist_source_delta, glist_sink_delta, propUP_coherent, propUP_coherent, propUP_coherent));

	  //write D
	  outfilename=outdiagramPrefix+confnumber+ sourcepositiontext+"_D";
	
	  TIME( corrD.D_diagramms( reductionsT1, reductionsT2 ));
	  TIME( corrD.apply_phase() );
	  TIME( corrD.apply_sign("D") );
	  TIME( corrD.applyBoundaryConditions( true ) );
	  TIME( corrD.writeHDF5(outfilename) );

        }
      
      //T diagram piN sink
/*
      {
	momList list_pf1pf2comb = sourcemomentumList.extract({0,0,0}, 0);
        PLEGMA_ScattCorrelator<float> corrT_piNsink(sourcePositions[isource], list_pf1pf2comb);

        corrT_piNsink.initialize_diagram(glist_source_delta_unpaired, glist_sink_nucleon_unpaired, glist_source_delta, glist_sink_nucleon,  glist_sink_meson, "T1"); 
 
        PLEGMA_ScattCorrelator<float> reductionsV2(source, list_pf1pf2comb.uniq_p(1));
        PLEGMA_ScattCorrelator<float> reductionsV3(source, list_pf1pf2comb.uniq_p(2));
 
        for (int i=0; i<n_stochastic_samples; ++i){
          PLEGMA_Vector<float> stochastic_propagator;
          PLEGMA_Vector<float> stochastic_source;

          stochastic_propagator.copy(*stochastic_propags[i],HOST);
          stochastic_source.copy(*stochastic_sources[i],HOST);

          stochastic_propagator.load();
          stochastic_source.load();

          TIME(reductionsV3.V3( stochastic_propagator, glist_sink_meson,   propUP));

          TIME(reductionsV2.V2( stochastic_source,     glist_sink_nucleon, propUP, propUP));

          TIME(corrT_piNsink.T_diagramms_piNsink(reductionsV3, reductionsV2, true));

        }                
        outfilename=outdiagramPrefix+confnumber+sourcepositiontext+"_TpiNsink";

        TIME(corrT_piNsink.apply_phase());
        TIME(corrT_piNsink.apply_sign("T1"));
        TIME(corrT_piNsink.applyBoundaryConditions( true ));
        TIME(corrT_piNsink.normalize_nstoch(n_stochastic_samples));
        TIME(corrT_piNsink.writeHDF5( outfilename ));

      }

  */    
        //N diagram
        std::vector<std::vector<int>> mpf1 = sourcemomentumList.uniq_p(1);
        momList list_mpf1(1,{mpf1,},{0,});
        PLEGMA_ScattCorrelator<float> corrN_coherent(src, list_mpf1 );

        //initialize diagram
        TIME(corrN_coherent.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_sink_nucleon,"N"));
      
        //Computing T reductions+recombination
        { 
          PLEGMA_ScattCorrelator<float> reductionsT1N(source, mpf1);
          PLEGMA_ScattCorrelator<float> reductionsT2N(source, mpf1);
      
          TIME(reductionsT1N.T1(glist_source_nucleon, glist_sink_nucleon, propUP_coherent, propDN_coherent, propUP_coherent));

          TIME(reductionsT2N.T2(glist_source_nucleon, glist_sink_nucleon, propUP_coherent, propDN_coherent, propUP_coherent));

          TIME(corrN_coherent.N_diagramms( reductionsT1N, reductionsT2N ));

          outfilename=outdiagramPrefix+confnumber+ sourcepositiontext+"_N";

          for (int timeslice=0; timeslice<HGC_totalL[DIM_T]/n_coherent_source; ++timeslice){
            corrN.absorbTimeslice(corrN_coherent, coherent_look_up_table[icoherentsource][timeslice], false);
          }
          TIME( corrN_coherent.apply_phase() );
          TIME( corrN_coherent.apply_sign("N") );
          TIME( corrN_coherent.applyBoundaryConditions( true ) );
          TIME( corrN_coherent.writeHDF5(outfilename) ); 
        } //End computing N diagramm

     } //End of loop on coherent sources

      //P diagram
      std::vector<std::vector<int>> mpi2 = sourcemomentumList.uniq_p(0);
      momList list_mpi2(1,{mpi2,},{0,});
      PLEGMA_ScattCorrelator<float> corrP(sourcePositions[isource], list_mpi2);
      corrP.initialize_diagram(glist_source_meson, glist_sink_meson, "P");

      // ensuring mu positive
      if(mu<0) {
        mu*=-1.;
        solver.UpdateSolver();
      }



      //We draw a different random vector for every source position
      vectorStoc_source_oet.stochastic_Z(nroots);
      
      //Store zero momentum oet propagators: also for the oet propagators we produce coherent sources
      std::array<PLEGMA_Vector<float>,4> stochastic_propagator_momzero;
      {
         PLEGMA_Vector<double> vectortmp1;
         PLEGMA_Vector<double> vectortmp2;          
	 PLEGMA_Vector<double> vectortmp3;
 

         {  // Smearing the source
            
            PLEGMA_Vector3D<double> vector1, vector2;
	    for (int i_coherent_source=0; i_coherent_source < n_coherent_source; ++i_coherent_source){

              vectortmp2.absorbTimeslice(vectorStoc_source_oet, coherent_source_table[i_coherent_source]);
              vector1.absorb(vectortmp2, coherent_source_table[i_coherent_source]);

	      PLEGMA_Gauge3D<double> smearedGauge3D;
              smearedGauge3D.absorb(smearedGauge, coherent_source_table[i_coherent_source]);

              TIME(vector2.gaussianSmearing(vector1, smearedGauge3D, nsmearGauss, alphaGauss));
              vectortmp1.absorb(vector2,coherent_source_table[i_coherent_source]);
              vectortmp2.absorbTimeslice(vectortmp1,coherent_source_table[i_coherent_source],false);

	    }

         }
 
         //Transforming to physical base
         vectortmp1.rotateToPhysicalBasis(vectortmp2,+1);
 
          //Dilution     
         vectortmp2.dilutespin(vectortmp1,0);

         //Save the smeared,transformed and diluted source for non-zero momentum oet.
         vectorStoc_source_oet.copy(vectortmp2);

         for (int spinindex=0; spinindex<4; ++spinindex){
           vectortmp2.copy(vectorStoc_source_oet);
           //stochastic_source_spin_diluted_momzero.writeLIME(outfile_V+"source_zero_momentum"+std::to_string(spinindex));         
           //Doing the zero momentum stochastic propagator with spin dilution
           //Doing the inversion
           TIME(solver.solve(vectortmp2, vectortmp2));
           //Rotate back immediately to the physical basis
           vectortmp1.rotateToPhysicalBasis(vectortmp2,+1);
           //Gaussian smearing of the propagator
           TIME(vectortmp2.gaussianSmearing(vectortmp1, smearedGauge, nsmearGauss, alphaGauss));
           stochastic_propagator_momzero[spinindex].copy(vectortmp2);
           //stochastic_propagator_momzero[spinindex].writeLIME(outfile_V+confnumber+"propagator_"+sourcepositiontext+"mompi2_0_0_0_s"+std::to_string(spinindex));
           if (spinindex<3){
             vectortmp1.diluteSpinDisplace(vectorStoc_source_oet,spinindex+1,spinindex);
             vectorStoc_source_oet.copy(vectortmp1);
           }
         }
         
         vectortmp1.diluteSpinDisplace(vectorStoc_source_oet,0,3);
         vectorStoc_source_oet.copy(vectortmp1);
      }

      PLEGMA_Propagator<float> propUPDN(BOTH);


      std::string outfilename;
      asprintf(&ssource,"sx%02dsy%02dsz%02dst%03d", sourcePositions[isource][0], sourcePositions[isource][1], sourcePositions[isource][2], sourcePositions[isource][3] );
      std::string sourcepositiontext= (std::string)"_" + ssource;
      free(ssource);

      //We first have a loop over all unique the source meson momentum p_i2 
      for (int i_mpi2=0; i_mpi2<mpi2.size(); ++i_mpi2){

	auto &momentum_i2 =  mpi2[i_mpi2];
	//List of momenta corresponding to a fix value of p_i2
        momList filtered_sourcemomentumList = sourcemomentumList.extract(momentum_i2, 0);


        std::string pi2x=std::to_string(momentum_i2[0]);
        std::string pi2y=std::to_string(momentum_i2[1]);
        std::string pi2z=std::to_string(momentum_i2[2]);

 
	// 4pt diagrams
	
	PLEGMA_ScattCorrelator<float> corrB1(sourcePositions[isource], filtered_sourcemomentumList);
	PLEGMA_ScattCorrelator<float> corrB2(sourcePositions[isource], filtered_sourcemomentumList);
	PLEGMA_ScattCorrelator<float> corrW1(sourcePositions[isource], filtered_sourcemomentumList);
	PLEGMA_ScattCorrelator<float> corrW2(sourcePositions[isource], filtered_sourcemomentumList);
	PLEGMA_ScattCorrelator<float> corrW3(sourcePositions[isource], filtered_sourcemomentumList);
	PLEGMA_ScattCorrelator<float> corrW4(sourcePositions[isource], filtered_sourcemomentumList);
	PLEGMA_ScattCorrelator<float> corrZ1(sourcePositions[isource], filtered_sourcemomentumList);
	PLEGMA_ScattCorrelator<float> corrZ2(sourcePositions[isource], filtered_sourcemomentumList);
	PLEGMA_ScattCorrelator<float> corrZ3(sourcePositions[isource], filtered_sourcemomentumList);
	PLEGMA_ScattCorrelator<float> corrZ4(sourcePositions[isource], filtered_sourcemomentumList);
	PLEGMA_ScattCorrelator<float> corrM(sourcePositions[isource], filtered_sourcemomentumList);
	
	//T diagrams
	std::vector<std::vector<int>> mptot_filt = filtered_sourcemomentumList.uniq_p(3);
	std::vector<std::vector<int>> mpi2_filt;
	mpi2_filt.assign(mptot_filt.size(),momentum_i2);
	momList list_mpi2ptot(2,{mpi2_filt,mptot_filt},{1,});
	PLEGMA_ScattCorrelator<float> corrT(sourcePositions[isource], list_mpi2ptot);
	corrT.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_delta_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_delta, "T");	

	//initialize diagrams
	
	corrB1.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "B1");


	corrB2.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "B2");
	corrW1.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "W1");
	corrW2.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "W2");
	corrW3.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "W3");
	corrW4.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "W4");
	corrZ1.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "Z1");
	corrZ2.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "Z2");
	corrZ3.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "Z3");
	corrZ4.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "Z4");
	corrM.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "M");
	
        site source=site({0,0,0,sourcePositions[isource][DIM_T]});

        PLEGMA_ScattCorrelator<float> reductionsV2(source, filtered_sourcemomentumList.uniq_p(1));
        PLEGMA_ScattCorrelator<float> reductionsV3(source, filtered_sourcemomentumList.uniq_p(2));

        //Loop over the different gamma structure for the source meson
        for (int i_gamma_i2=0; i_gamma_i2<glist_source_meson.size(); ++i_gamma_i2) {
	  GAMMAS_SCATT gamma_i2 = glist_source_meson[i_gamma_i2];
          // Computing sequential propagators f1 <- i_2 <- i_1 
          // so the sequential source source time is fixed
          // and the momentum is also fixed to be momentum_i2

          //smearing the 3D propagators
	  
          for(int isc = 0 ; isc < 12 ; isc++){
            PLEGMA_Vector<double> vectorAuxD;
            PLEGMA_Vector<float> vectorAuxF;
            PLEGMA_Vector<double> vectorAuxD2;
            vectorAuxF.absorb(propDN,isc/3, isc%3);
            vectorAuxD.copy(vectorAuxF);

            //Performing the smearing
            // Smearing the source
	    for (int icoherentsource=0; icoherentsource < n_coherent_source;++icoherentsource)
	    {
              PLEGMA_Gauge3D<double> smearedGauge3D;
              smearedGauge3D.absorb(smearedGauge, coherent_source_table[icoherentsource]);
                
	      PLEGMA_Vector3D<double> vector1, vector2;
              vectorAuxF.absorb(propDN, isc/3, isc%3);
              vectorAuxD.copy(vectorAuxF);
              vector1.absorb( vectorAuxD, coherent_source_table[icoherentsource]);
              TIME(vector2.gaussianSmearing(vector1, smearedGauge3D, nsmearGauss, alphaGauss));

              vector2.mulMomentumPhases(momentum_i2,1);
	        
	      vectorAuxD.absorb(vector2,coherent_source_table[icoherentsource]);
              vectorAuxD2.absorbTimeslice(vectorAuxD,coherent_source_table[icoherentsource], false);

            }//loop over coherent source
	     
            //Perform multiplication with gamma_i2
            vectorAuxD2.apply_gamma_scatt(gamma_i2);

            //Perform rotation to the physical basis
            TIME(vectorAuxD.rotateToPhysicalBasis(vectorAuxD2,+1));


            //Computing sequential propagators UD T_fii with insertion
            //gamma_i2 and momentum SinkMom
            PLEGMA_printf("Going to invert UP for sequential propagator DN  for component %d\n", isc);
            //performing the inversion
            TIME(solver.solve(vectorAuxD, vectorAuxD));

            //performing rotation to physical base
            TIME(vectorAuxD2.rotateToPhysicalBasis(vectorAuxD,+1));

            //performing smearing
            TIME(vectorAuxD.gaussianSmearing(vectorAuxD2, smearedGauge, nsmearGauss, alphaGauss));

            vectorAuxF.copy(vectorAuxD);
            propUPDN.absorb(vectorAuxF, isc/3, isc%3);
	    
          } //loop over isc 


          //Compute triangle diagramms          
          
          PLEGMA_ScattCorrelator<float> reductionsT1triangle(source,  mptot_filt);
         
          PLEGMA_ScattCorrelator<float> reductionsT3triangle(source,  mptot_filt);

          PLEGMA_ScattCorrelator<float> reductionsT5triangle(source,  mptot_filt);
          TIME(reductionsT1triangle.T1(glist_source_nucleon, glist_sink_delta, propUPDN, propUP  , propUP));
          TIME(reductionsT3triangle.T1(glist_source_nucleon, glist_sink_delta, propUP  , propUPDN, propUP));
          TIME(reductionsT5triangle.T2(glist_source_nucleon, glist_sink_delta, propUP  , propUP, propUPDN));
	  
	  //Compute Diagram T 
          TIME(corrT.T_diagramms(reductionsT1triangle, reductionsT3triangle, reductionsT5triangle, i_gamma_i2));


          for (int i=0; i<n_stochastic_samples; ++i){
            //Compute Diagram B1 and B2
            PLEGMA_Vector<float> stochastic_propagator;
            PLEGMA_Vector<float> stochastic_source;

            stochastic_propagator.copy(*stochastic_propags[i],HOST);
            stochastic_source.copy(*stochastic_sources[i],HOST);

            stochastic_propagator.load();
            stochastic_source.load();
 
	  
            TIME(reductionsV3.V3( stochastic_propagator, glist_sink_meson,   propUPDN));

            TIME(reductionsV2.V2( stochastic_source,     glist_sink_nucleon, propUP, propUP));

	    TIME(corrB1.B_diagramms(reductionsV3, reductionsV2, i_gamma_i2, 1, true));
	  
	    TIME(corrB2.B_diagramms(reductionsV3, reductionsV2, i_gamma_i2, 2, true));
          
            //Compute Diagram W1,W2
          
            TIME(reductionsV3.V3( stochastic_propagator, glist_sink_meson,   propUP));
            TIME(reductionsV2.V2( stochastic_source,     glist_sink_nucleon, propUP, propUPDN));

            TIME(corrW1.W_diagramms( reductionsV3, reductionsV2, i_gamma_i2, 1, true));
	    TIME(corrW2.W_diagramms( reductionsV3, reductionsV2, i_gamma_i2, 2, true));
          
            //Compute Diagram W3,W4
          
            TIME(reductionsV2.V2( stochastic_source, glist_sink_nucleon, propUPDN, propUP));

            TIME(corrW3.W_diagramms( reductionsV3, reductionsV2, i_gamma_i2, 3, true));
	    TIME(corrW4.W_diagramms( reductionsV3, reductionsV2, i_gamma_i2, 4, true));

          } //loop over stochastic samples
	  
       } //loop over gamma i2
         
	  
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



       {
          PLEGMA_Vector<double> vectortmp1;
          PLEGMA_Vector<double> vectortmp2;
          PLEGMA_Vector<double> vectorSource_finite_mom;

          //Multiplying by the appropriate momentum phase
          vectorSource_finite_mom.copy(vectorStoc_source_oet);
          std::vector<int> tmp_4Dmom= momentum_i2 ; 
          tmp_4Dmom.push_back(0);
          vectorSource_finite_mom.mulMomentumPhases(tmp_4Dmom,-1);
          
          if ((momentum_i2[0] != 0) || (momentum_i2[1] != 0) || (momentum_i2[2] != 0)){
       
            for (int spinindex=0; spinindex<4; ++spinindex){
       
              vectortmp1.copy(vectorSource_finite_mom);
              //Doing the inversion
              TIME(solver.solve(vectortmp1, vectortmp1));

              //Rotate back immediately to the physical basis
              TIME(vectortmp2.rotateToPhysicalBasis(vectortmp1,+1));

              //performing smearing
              TIME(vectortmp1.gaussianSmearing(vectortmp2, smearedGauge, nsmearGauss, alphaGauss));

              //Saving the propagator
              stochastic_propagator_momp_i2[spinindex].copy(vectortmp1);
         
              if (spinindex<3){
                vectortmp1.diluteSpinDisplace(vectorSource_finite_mom,spinindex+1,spinindex);
                vectorSource_finite_mom.copy(vectortmp1);
              }
            }
          }
       }
         
       //Diagram Z1,Z2
       std::vector<GAMMAS_SCATT> gamma_5_t_sinkmeson=apply_gamma5_scatt_gamma(glist_sink_meson,LEFT);
       
       for (int i=0; i< 4; ++i){
         if ((momentum_i2[0] != 0) || (momentum_i2[1] != 0) || (momentum_i2[2] != 0)){
  
           TIME(reductionsV3_diluted[i].V3( stochastic_propagator_momp_i2[i], gamma_5_t_sinkmeson, propUP));

         }
         else{
           TIME(reductionsV3_diluted[i].V3( stochastic_propagator_momzero[i], gamma_5_t_sinkmeson, propUP));
         }

         TIME(reductionsV2_diluted[i].V4( stochastic_propagator_momzero[i], glist_sink_nucleon, propDN, propUP));
       }

       TIME(corrZ1.Z_diagramms( reductionsV3_diluted, reductionsV2_diluted, 1 ));
       TIME(corrZ2.Z_diagramms( reductionsV3_diluted, reductionsV2_diluted, 2 ));
 
      
       //Diagram Z3,Z4	
       for (int i=0; i< 4; ++i){

         TIME(reductionsV2_diluted[i].V2( stochastic_propagator_momzero[i], glist_sink_nucleon, propDN, propUP));

       }

       TIME(corrZ3.Z_diagramms( reductionsV3_diluted, reductionsV2_diluted, 3 ));
       TIME(corrZ4.Z_diagramms( reductionsV3_diluted, reductionsV2_diluted, 4 ));

       //M diagram N.B. I still need Phi_0, Phi_1 here! So even if we decide to enclose Phi's plegma_vectors in a smaller scope, we need to move this diagram too.
       if ((momentum_i2[0] != 0) || (momentum_i2[1] != 0) || (momentum_i2[2] != 0)){
         TIME(corrP.P_diagramms( stochastic_propagator_momzero, stochastic_propagator_momp_i2, i_mpi2));
       
       
         TIME(corrM.M_diagramms( corrN, stochastic_propagator_momzero, stochastic_propagator_momp_i2 ));
       }
       else{
         TIME(corrP.P_diagramms( stochastic_propagator_momzero, stochastic_propagator_momzero, i_mpi2));


         TIME(corrM.M_diagramms( corrN, stochastic_propagator_momzero, stochastic_propagator_momzero ));
       }
	

       //write everything
       //## T

       outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"_T";
     
       TIME(corrT.apply_phase());
       TIME(corrT.apply_sign("T"));
       TIME(corrT.applyBoundaryConditions( true, n_coherent_source, coherent_source_table_timeslice));

       TIME(corrT.writeHDF5(outfilename));
       
       //## B
       
       outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"_B";
       TIME(corrB1.apply_phase());
       TIME(corrB1.apply_sign("B"));
       TIME(corrB1.applyBoundaryConditions( true, n_coherent_source, coherent_source_table_timeslice));
       TIME(corrB1.normalize_nstoch(n_stochastic_samples));
       TIME(corrB1.writeHDF5( outfilename ));
       TIME(corrB2.apply_phase());
       TIME(corrB2.apply_sign("B"));
       TIME(corrB2.applyBoundaryConditions( true, n_coherent_source, coherent_source_table_timeslice));
       TIME(corrB2.normalize_nstoch(n_stochastic_samples));
       TIME(corrB2.writeHDF5( outfilename ));


       //## W
       outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"_W";

       TIME(corrW1.apply_phase());
       TIME(corrW1.apply_sign("W"));
       TIME(corrW1.applyBoundaryConditions( true, n_coherent_source, coherent_source_table_timeslice));
       TIME(corrW1.normalize_nstoch(n_stochastic_samples));
       TIME(corrW1.writeHDF5(outfilename));
       TIME(corrW2.apply_phase());
       TIME(corrW2.apply_sign("W"));
       TIME(corrW2.applyBoundaryConditions( true, n_coherent_source, coherent_source_table_timeslice));
       TIME(corrW2.normalize_nstoch(n_stochastic_samples));
       TIME(corrW2.writeHDF5(outfilename));
       TIME(corrW3.apply_phase());
       TIME(corrW3.apply_sign("W"));
       TIME(corrW3.applyBoundaryConditions( true, n_coherent_source, coherent_source_table_timeslice));
       TIME(corrW3.normalize_nstoch(n_stochastic_samples));
       TIME(corrW3.writeHDF5(outfilename));
       TIME(corrW4.apply_phase());
       TIME(corrW4.apply_sign("W"));
       TIME(corrW4.applyBoundaryConditions( true, n_coherent_source, coherent_source_table_timeslice));
       TIME(corrW4.normalize_nstoch(n_stochastic_samples));
       TIME(corrW4.writeHDF5(outfilename));
       //## Z

       outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"_Z";

       TIME(corrZ1.apply_phase());
       TIME(corrZ1.apply_sign("Z"));
       TIME(corrZ1.applyBoundaryConditions( true, n_coherent_source, coherent_source_table_timeslice));
       TIME(corrZ1.writeHDF5( outfilename ));
       TIME(corrZ2.apply_phase());
       TIME(corrZ2.apply_sign("Z"));
       TIME(corrZ2.applyBoundaryConditions( true, n_coherent_source, coherent_source_table_timeslice));
       TIME(corrZ2.writeHDF5( outfilename  ));
       TIME(corrZ3.apply_phase());
       TIME(corrZ3.apply_sign("Z"));
       TIME(corrZ3.applyBoundaryConditions( true, n_coherent_source, coherent_source_table_timeslice));
       TIME(corrZ3.writeHDF5( outfilename ));
       TIME(corrZ4.apply_phase());
       TIME(corrZ4.apply_sign("Z"));
       TIME(corrZ4.applyBoundaryConditions( true, n_coherent_source, coherent_source_table_timeslice));
       TIME(corrZ4.writeHDF5( outfilename ));

       //## M
       outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"_M";
       TIME(corrM.apply_phase());
       TIME(corrM.apply_sign("M"));
       TIME(corrM.applyBoundaryConditions( true,  n_coherent_source, coherent_source_table_timeslice));
       TIME(corrM.writeHDF5( outfilename ));

      }//loop over unique set of momenta for p_i2

      
      //write P

      outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"_P";
      TIME(corrP.apply_sign("P"));
      TIME(corrP.writeHDF5( outfilename ));

      free(coherent_source_table);
      free(coherent_source_table_timeslice);
      for (int i=0; i<n_coherent_source;++i)
        free(coherent_look_up_table[i]);
      free(coherent_look_up_table);

    } //loop over source position

    for(int i=0; i< n_stochastic_samples; ++i) {
      stochastic_sources.pop_back();
      stochastic_propags.pop_back();
    }

  } 
  finalize();
  
  return 0;
}
