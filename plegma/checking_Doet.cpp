
#include <PLEGMA.h>
#include <PLEGMA_utils.h>
#include <omp.h>

using namespace plegma;
using namespace quda;

std::vector<double> runtime_inside;
#define TIME_INSIDE(fnc)  runtime_inside.push_back(MPI_Wtime()); fnc; \
  PLEGMA_printf("TIME for "#fnc" %f sec\n", MPI_Wtime()-runtime_inside.back()); \
  runtime_inside.pop_back()

std::vector<double> runtime;
#define TIME(fnc,isospin)  runtime.push_back(MPI_Wtime()); fnc;                 \
  PLEGMA_printf("TIME "#isospin" for "#fnc" %f sec\n", MPI_Wtime()-runtime.back()); \
  runtime.pop_back()

//std::vector<std::thread> threads;
//#define THREAD(fnc) threads.push_back(std::thread([=]() { TIME(fnc); }))
//#define THREAD(fnc) TIME(fnc)

extern int device;
static std::vector<std::string> listOpt = {"verbosity", "load-gauge","nsmear-APE","alpha-APE", "nsmear-gauss","alpha-gauss","nsrc","src-filename", "momlist-filename", "readStochSamples","time-dilution","nstochSamples","confnumber"};
// Note here sinkMom is used as the momentum insertion in the sequential souce, probably has to be renamed to seqMom

void produceOutput( PLEGMA_ScattCorrelator<float> source,
                    std::string outputFilename,
                    std::string diagram_name,
                    int n_stochastic_samples,
                    int n_coherent_source,
                    int *coherent_source_table_timeslice ){
  TIME_INSIDE(source.apply_phase());
  TIME_INSIDE(source.apply_sign(diagram_name));
  TIME_INSIDE(source.applyBoundaryConditions( true ,  n_coherent_source, coherent_source_table_timeslice));
  TIME_INSIDE(source.normalize_nstoch(n_stochastic_samples));
  TIME_INSIDE(source.writeHDF5( outputFilename ));

}

void produceOutput( PLEGMA_ScattCorrelator<float> source,
                    std::string outputFilename,
                    std::string diagram_name,
                    int n_coherent_source, 
                    int *coherent_source_table_timeslice 
                  ){
  TIME_INSIDE(source.apply_phase());
  TIME_INSIDE(source.apply_sign(diagram_name));
  TIME_INSIDE(source.applyBoundaryConditions( true, n_coherent_source, coherent_source_table_timeslice ));
  TIME_INSIDE(source.writeHDF5( outputFilename ));

}
void produceOutput( PLEGMA_ScattCorrelator<float> source,
                    std::string outputFilename,
                    std::string diagram_name
                  ){
  TIME_INSIDE(source.apply_phase());
  TIME_INSIDE(source.apply_sign(diagram_name));
  TIME_INSIDE(source.applyBoundaryConditions( true ));
  TIME_INSIDE(source.writeHDF5( outputFilename ));
}

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
  std::string outfilename;
  std::string outfile_V="";
  std::string outfile_upS="";
  std::string outfile_dnS="";
  std::string outfile_SEQ="";
  std::string outdiagramPrefix="";
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
  HGC_options->set("seed1", "Seed for initialization of stochastic sources for the oet", verbosity, rand_seed1);
  HGC_options->set("seed2", "Seed for intiialization of stochastic sources", verbosity, rand_seed2);

  //=========================================================================================================//
  initializePLEGMA();
  {
    /*******************************************************************************
     *
     *
     *  Initialization: (1) reading the sourcepositions and the source momentum list 
     *                  (2) reading the gaugefield uploading the quda
     *                      produce the smeared gauge field
     *                      only the smeared gauge field will be stored in PLEGMA
     *                  (3) setting up the list of gammas
     *                      (1) we use three triplets for the delta Cgi, Cgigt, Cgigtg5
     *                      (2) we use C,Cg5, Cg4,Cg5g4 for the nucleon
     *                      (3) we use g5 for meson(pion)
     *
     *******************************************************************************/
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
      TIME(smearedGauge.APEsmearing(gauge, nsmearAPE, alphaAPE, 3),"ISOSPIN32");
      PLEGMA_printf("Plaquette after smearing:\n");
      smearedGauge.calculatePlaq();
    }

    updateOptions(LIGHT);
    TIME(QUDA_solver solver(mu),"ISOSPIN32");



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

    std::vector<GAMMAS_SCATT> glist_sink_meson={G_5};
    std::vector<GAMMAS_SCATT> glist_source_meson={G_5};

    std::vector<GAMMAS_SCATT> glist_source_delta_unpaired={ID};
    std::vector<GAMMAS_SCATT> glist_sink_delta_unpaired={ID};

    
    std::string smearType = ((nsmearGauss>0) ? "SS" : "LL");
    std::string smearString = smearType + "_" + "gN" + std::to_string(nsmearGauss) + "a" + convNumToStr(alphaGauss) + "aN" + std::to_string(nsmearAPE) + "a" + convNumToStr(alphaAPE);

/******************************************************************************************
*
*  In the first part of the code we compute 
*      (1) the stochastic propagators for nstoch random volume sources using full-time dilution
*      (2) allocate space for zero momentum (u,d) and finite momentum (u) oet propagators
*      (3) compute loops for the pi0 using the stochastic propagators and sources produced by 
*          time dilution, note that this is needed for the 
*  Note that in both cases we store a standard vector of PLEGMA_Vectors on the host, 
*  and need to load to the device in case we need them
*
*
*******************************************************************************************/
    std::vector<PLEGMA_Vector<float>*> stochastic_oet_prop_u_zero_mom;
    std::vector<PLEGMA_Vector<float>*> stochastic_oet_prop_u_fini_mom;
#ifdef PLEGMA_SCATTERING_SPIN12
    std::vector<PLEGMA_Vector<float>*> stochastic_oet_prop_d_zero_mom;
#endif
    for(int i=0; i< 4; ++i) {
      stochastic_oet_prop_u_fini_mom.push_back(new PLEGMA_Vector<float>(HOST));
      stochastic_oet_prop_u_zero_mom.push_back(new PLEGMA_Vector<float>(HOST));
#ifdef PLEGMA_SCATTERING_SPIN12
      stochastic_oet_prop_d_zero_mom.push_back(new PLEGMA_Vector<float>(HOST));
#endif
    }

    PLEGMA_Vector<double> vectorStoc_source_oet;
    vectorStoc_source_oet.randInit(rand_seed1);




/********************************************************************************************
*
*
*   In the second part of the code we compute point source propagators for
*   UP and DN flavour and perform all the contractions necessary for I=3/2 and 1/2
*   that does not require sequential source propagator and allocate space for the 
*   sequential propagator
*   Producing diagrams (1) N (Nucleon 2pt) (including both N+ with U,D,U and N- with D,U,D
*                      (2) D (Delta 2pt) 
*                      (3) delta -->> pi + N (2pt)
*   Producing factors requiring only D or U for momenta pf1,pf2 or {0,0,0}
*
*********************************************************************************************/


    //loop over the soure positions
    for(int isource = 0 ; isource < numSourcePositions; isource++){


      asprintf(&ssource,"sx%02dsy%02dsz%02dst%03d", sourcePositions[isource][0], sourcePositions[isource][1], sourcePositions[isource][2],  sourcePositions[isource][3]);
      std::string sourcepositiontext= (std::string)"_" + ssource;  
      free(ssource);

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
#if 1 
      //Create Propagator
      PLEGMA_Propagator<float> propUP(BOTH); //To be saved for all the coherent sources.
      PLEGMA_Propagator<float> propDN(BOTH);


      std::vector<std::vector<int>> mpf1 = sourcemomentumList.uniq_p(1);
      momList list_mpf1(1,{mpf1,},{0,});
      PLEGMA_ScattCorrelator<float> corrNP(sourcePositions[isource], list_mpf1 );
      TIME(corrNP.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_sink_nucleon,"N"),"ISOSPIN32");

#ifdef PLEGMA_SCATTERING_SPIN12
      PLEGMA_ScattCorrelator<float> corrN0(sourcePositions[isource], list_mpf1 );
      TIME(corrN0.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_sink_nucleon,"N"),"ISOSPIN12");
#endif


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
            TIME(vector2.gaussianSmearing(vector1, smearedGauge3D, nsmearGauss, alphaGauss),"ISOSPIN32");
            vectorInOut.absorb(vector2,coherent_source_table[icoherentsource]);
          }

          //Rotation to the physical basis
          TIME(vectorAuxD.rotateToPhysicalBasis(vectorInOut,+1),"ISOSPIN32");       

          //Inversion
          PLEGMA_printf("Going to invert UP for component %d\n", isc);
          TIME(solver.solve(vectorAuxD, vectorAuxD),"ISOSPIN32");

          //Rotation to the physical basis
          TIME(vectorInOut.rotateToPhysicalBasis(vectorAuxD,+1),"ISOSPIN32");

          //Smearing at the sink
          TIME(vectorAuxD.gaussianSmearing(vectorInOut, smearedGauge, nsmearGauss, alphaGauss),"ISOSPIN32");

          vectorAuxF.copy(vectorAuxD);
          propUP_coherent.absorb(vectorAuxF, isc/3, isc%3);

        }
        for (int timeslice=0; timeslice<HGC_totalL[DIM_T]/n_coherent_source; ++timeslice){
          propUP.absorbTimeslice(propUP_coherent, coherent_look_up_table[icoherentsource][timeslice], false);
        }
        if(outfile_upS!="")
        {
          PLEGMA_printf("Save propagator for the up quark\n");
          PLEGMA_Vector<float> vectorAuxPrint(BOTH);
          for(int isc = 0 ; isc < 12 ; isc++){
            std::string spin=std::to_string(isc/3);
            std::string col=std::to_string(isc%3);

            vectorAuxPrint.absorb(propUP,isc/3,isc%3);
            vectorAuxPrint.unload();
            vectorAuxPrint.writeLIME(outfile_upS+confnumber+sourcepositiontext+"_s"+spin+"_c"+col);
            //vectorAuxPrint.writeHDF5(outfile_upS+confnumber+sourcepositiontext+"_s"+spin+"_c"+col);
          }
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
            TIME(vector2.gaussianSmearing(vector1, smearedGauge3D, nsmearGauss, alphaGauss),"ISOSPIN32");
            vectorInOut.absorb(vector2,coherent_source_table[icoherentsource]);
          }

          //(3 step) rotation to the physical basis
          TIME(vectorAuxD.rotateToPhysicalBasis(vectorInOut,-1),"ISOSPIN32");

          //(4 step) doing the inversion
          PLEGMA_printf("Going to invert DN for component %d\n", isc);
          TIME(solver.solve(vectorAuxD, vectorAuxD),"ISOSPIN32");

          //(5 step) rotating to the physical base
          TIME(vectorInOut.rotateToPhysicalBasis(vectorAuxD,-1),"ISOSPIN32");

          //(6 step) doing the smearing on the propagator
          TIME(vectorAuxD.gaussianSmearing(vectorInOut, smearedGauge, nsmearGauss, alphaGauss),"ISOSPIN32");

          vectorAuxF.copy(vectorAuxD);

          propDN_coherent.absorb(vectorAuxF, isc/3, isc%3);
        }
        for (int timeslice=0; timeslice<HGC_totalL[DIM_T]/n_coherent_source; ++timeslice){
	  propDN.absorbTimeslice(propDN_coherent, coherent_look_up_table[icoherentsource][timeslice], false);
        } 
        if(outfile_dnS!="")
        {
          PLEGMA_printf("Save propagator for the up quark\n");
          PLEGMA_Vector<float> vectorAuxPrint(BOTH);
          for(int isc = 0 ; isc < 12 ; isc++){
            std::string spin=std::to_string(isc/3);
            std::string col=std::to_string(isc%3);

            vectorAuxPrint.absorb(propUP,isc/3,isc%3);
            vectorAuxPrint.unload();
            vectorAuxPrint.writeLIME(outfile_dnS+confnumber+sourcepositiontext+"_s"+spin+"_c"+col);
            //vectorAuxPrint.writeHDF5(outfile_upS+confnumber+sourcepositiontext+"_s"+spin+"_c"+col);
          }
        }
      }
      site& source = sourcePositions[isource];
      PLEGMA_Correlator<float> corr(corr_space, source, maxQsq);
      TIME(corr.contractMesonsNew(propUP, propDN),"ISOSPIN32"); 
      char *dset;
      asprintf(&dset, "twop_mesons_new_u[%+1.1e]d[%+1.1e]", mu_ud, -1*mu_ud);
      corr.setDatasets((std::vector<std::string>) {dset});
      free(dset);
      corr.writeFile(twop_filename, corr_file_format);

      TIME(corr.contractMesonsNew(propUP, propUP),"ISOSPIN32");
      asprintf(&dset, "twop_mesons_new_u[%+1.1e]u[%+1.1e]", mu_ud, mu_ud);
      corr.setDatasets((std::vector<std::string>) {dset});
      free(dset);
      corr.writeFile(twop_filename, corr_file_format);

      TIME(corr.contractMesonsNew(propDN, propDN),"ISOSPIN32");
      asprintf(&dset, "twop_mesons_new_d[%+1.1e]d[%+1.1e]", -mu_ud, -mu_ud);
      corr.setDatasets((std::vector<std::string>) {dset});
      free(dset);
      corr.writeFile(twop_filename, corr_file_format);

	
      TIME(corr.contractBaryons(propUP, propDN),"ISOSPIN32");
      corr.writeFile(twop_filename, corr_file_format);
#endif
      std::vector<int> mom={0,0,0};
      std::string outfilename;

      //P diagram
      std::vector<std::vector<int>> mpi2 = sourcemomentumList.uniq_p(0);
      momList list_mpi2(1,{mpi2,},{0,});
      PLEGMA_ScattCorrelator<float> corrP(sourcePositions[isource], list_mpi2);
      corrP.initialize_diagram(glist_source_meson, glist_sink_meson, "P");

#ifdef PLEGMA_SCATTERING_SPIN12

      //We draw a different random vector for every source position
      vectorStoc_source_oet.stochastic_Z(nroots);
      
      //Store zero momentum oet propagators: also for the oet propagators we produce coherent sources
      std::array<PLEGMA_Vector<float>,4> stochastic_propagator_momzero;
      {

         //Doing for +mu for the UP propagator spin dilution oet
         if(mu<0) {
           mu*=-1.;
           solver.UpdateSolver();
         }

         PLEGMA_Vector<double> vectortmp1;
         PLEGMA_Vector<double> vectortmp2;          
	 PLEGMA_Vector<double> vectorSave_diluted;
 

         {  // Smearing the source
            

	    for (int i_coherent_source=0; i_coherent_source < n_coherent_source; ++i_coherent_source){
              vectortmp1.absorbTimeslice(vectorStoc_source_oet, coherent_source_table[i_coherent_source]);
              PLEGMA_Vector3D<double> vector1, vector2;
              vector1.absorb(vectortmp1, coherent_source_table[i_coherent_source]);

	      PLEGMA_Gauge3D<double> smearedGauge3D;
              smearedGauge3D.absorb(smearedGauge, coherent_source_table[i_coherent_source]);

              TIME(vector2.gaussianSmearing(vector1, smearedGauge3D, nsmearGauss, alphaGauss),"ISOSPIN32");
              vectortmp1.absorb(vector2,coherent_source_table[i_coherent_source]);
              vectortmp2.absorbTimeslice(vectortmp1,coherent_source_table[i_coherent_source],false);

	    }

         }

         //Save the smeared,transformed and diluted source for non-zero momentum oet.
         vectorStoc_source_oet.copy(vectortmp2);
         // vectorStoc_source_oet.writeLIME(outfile_V+confnumber+"oet_source"+sourcepositiontext);
  
          //Dilution     
         vectortmp1.dilutespin(vectortmp2,0);

         //Save the smeared,transformed and diluted source for non-zero momentum oet.
         vectorSave_diluted.copy(vectortmp1);
         vectorStoc_source_oet.writeLIME(outfile_V+confnumber+"source_oet"+sourcepositiontext+"mompi2_0_0_0_s");

         for (int spinindex=0; spinindex<4; ++spinindex){

           //Transforming to physical base for the UP quark
           vectortmp2.rotateToPhysicalBasis(vectorSave_diluted,+1); 
 
           //stochastic_source_spin_diluted_momzero.writeLIME(outfile_V+"source_zero_momentum"+std::to_string(spinindex));         
           //Doing the zero momentum stochastic propagator with spin dilution
           //Doing the inversion
           TIME(solver.solve(vectortmp2, vectortmp2), "ISOSPIN32");
           //Rotate back immediately to the physical basis
           vectortmp1.rotateToPhysicalBasis(vectortmp2,+1);
           //Gaussian smearing of the propagator
           TIME(vectortmp2.gaussianSmearing(vectortmp1, smearedGauge, nsmearGauss, alphaGauss), "ISOSPIN32");
           vectortmp2.unload();
           stochastic_oet_prop_u_zero_mom[spinindex]->copy(vectortmp2,HOST);
           vectortmp2.load();

           vectortmp2.writeLIME(outfile_V+confnumber+"propagator_up"+sourcepositiontext+"mompi2_0_0_0_s"+std::to_string(spinindex));
           if (spinindex<3){

             vectortmp1.diluteSpinDisplace(vectorSave_diluted,spinindex+1,spinindex);
             vectorSave_diluted.copy(vectortmp1);
           }
         }
         
         //vectortmp1.diluteSpinDisplace(vectortmp3,0,3);
         //vectortmp3.copy(vectortmp1);
          
         //Doing for -mu for the DN propagator spin dilution oet
         if(mu>0) {
           mu*=-1.;
           solver.UpdateSolver();
         }


         //Dilution     
         vectortmp1.dilutespin(vectortmp2,0);

         //Save the smeared,transformed and diluted source for non-zero momentum oet.
         vectorSave_diluted.copy(vectortmp1);

         for (int spinindex=0; spinindex<4; ++spinindex){
           vectortmp2.rotateToPhysicalBasis(vectorSave_diluted,-1);
           //stochastic_source_spin_diluted_momzero.writeLIME(outfile_V+"source_zero_momentum"+std::to_string(spinindex));         
           //Doing the zero momentum stochastic propagator with spin dilution
           //Doing the inversion
           TIME(solver.solve(vectortmp2, vectortmp2), "ISOSPIN12");
           //Rotate back immediately to the physical basis
           vectortmp1.rotateToPhysicalBasis(vectortmp2,-1);
           //Gaussian smearing of the propagator
           TIME(vectortmp2.gaussianSmearing(vectortmp1, smearedGauge, nsmearGauss, alphaGauss), "ISOSPIN12");
           vectortmp2.unload();
           stochastic_oet_prop_d_zero_mom[spinindex]->copy(vectortmp2,HOST);
           vectortmp2.load();
           vectortmp2.writeLIME(outfile_V+confnumber+"propagator_dn"+sourcepositiontext+"mompi2_0_0_0_s"+std::to_string(spinindex));
           if (spinindex<3){
             vectortmp1.diluteSpinDisplace(vectorSave_diluted,spinindex+1,spinindex);
             vectorSave_diluted.copy(vectortmp1);
           }
         }
      }


      //We need the pi-N 4pt functions only at zero pion momentum and non-zero nucleon momentum
      //therefore we filter further the momentumlist corresponding to pi2==0 to also pf2==0
      std::vector<int> filter={0,0,0};
      momList filtered_sourcemomentumList_pi20 = sourcemomentumList.extract(filter, 0);
      momList filtered_sourcemomentumList_pi20pf20 = filtered_sourcemomentumList_pi20.extract(filter,2);
      std::vector<std::vector<int>> mpi2_pizero = filtered_sourcemomentumList_pi20pf20.uniq_p(0);

      

      //Ensuring mu is positive
      if(mu<0) {
        mu*=-1.;
        solver.UpdateSolver();
      }


      ///We first have a loop over all unique the source meson momentum p_i2 

      //We first have a loop over all unique the source meson momentum p_i2 
      for (int i_mpi2=0; i_mpi2<mpi2.size(); ++i_mpi2){

	auto &momentum_i2 =  mpi2[i_mpi2];
	//List of momenta corresponding to a fix value of p_i2
        momList filtered_sourcemomentumList = sourcemomentumList.extract(momentum_i2, 0);

        std::string pi2x=std::to_string(momentum_i2[0]);
        std::string pi2y=std::to_string(momentum_i2[1]);
        std::string pi2z=std::to_string(momentum_i2[2]);
 
	// 4pt diagrams
	
	
	//T diagrams
	std::vector<std::vector<int>> mptot_filt = filtered_sourcemomentumList.uniq_p(3);
	std::vector<std::vector<int>> mpi2_filt;
	mpi2_filt.assign(mptot_filt.size(),momentum_i2);
	momList list_mpi2ptot(2,{mpi2_filt,mptot_filt},{1,});
	
	
	
        site source=site({0,0,0,sourcePositions[isource][DIM_T]});


       if  ((momentum_i2[0] == 0) && (momentum_i2[1] == 0) && (momentum_i2[2] == 0)){

         momList list_mpi2_pizero(1,{mpi2_pizero,},{0,});


         PLEGMA_ScattCorrelator<float> corrP0UP(sourcePositions[isource], list_mpi2_pizero);
         PLEGMA_ScattCorrelator<float> corrP0DN(sourcePositions[isource], list_mpi2_pizero);
         PLEGMA_ScattCorrelator<float> corrPPDN(sourcePositions[isource], list_mpi2_pizero);
         PLEGMA_ScattCorrelator<float> corrPPUP(sourcePositions[isource], list_mpi2_pizero);


         corrP0UP.initialize_diagram(glist_source_meson, glist_sink_meson, "P0UP");
         corrP0DN.initialize_diagram(glist_source_meson, glist_sink_meson, "P0DN");
         corrPPUP.initialize_diagram(glist_source_meson, glist_sink_meson, "PPUP");
         corrPPDN.initialize_diagram(glist_source_meson, glist_sink_meson, "PPDN");



         TIME(corrP0UP.P_diagramms( stochastic_oet_prop_d_zero_mom, stochastic_oet_prop_u_zero_mom, i_mpi2),"ISOSPIN32");
         TIME(corrP0DN.P_diagramms( stochastic_oet_prop_u_zero_mom, stochastic_oet_prop_d_zero_mom, i_mpi2),"ISOSPIN32");
         TIME(corrPPUP.P_diagramms( stochastic_oet_prop_u_zero_mom, stochastic_oet_prop_u_zero_mom, i_mpi2),"ISOSPIN32");
         TIME(corrPPDN.P_diagramms( stochastic_oet_prop_d_zero_mom, stochastic_oet_prop_d_zero_mom, i_mpi2),"ISOSPIN32");


         outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"_P";
         TIME(corrP0UP.apply_sign("P"),"ISOSPIN32");
         TIME(corrP0UP.writeHDF5( outfilename ),"ISOSPIN32");
         TIME(corrP0DN.apply_sign("P"),"ISOSPIN32");
         TIME(corrP0DN.writeHDF5( outfilename ),"ISOSPIN32");
         TIME(corrPPUP.apply_sign("P"),"ISOSPIN32");
         TIME(corrPPUP.writeHDF5( outfilename ),"ISOSPIN32");
         TIME(corrPPDN.apply_sign("P"),"ISOSPIN32");
         TIME(corrPPDN.writeHDF5( outfilename ),"ISOSPIN32");



       }//end of if()

      }//end of for pi2


      free(coherent_source_table);
      free(coherent_source_table_timeslice);
      for (int i=0; i<n_coherent_source;++i)
        free(coherent_look_up_table[i]);
      free(coherent_look_up_table);
#endif

    } //end of loop over source position

//#endif



    for(int i=0; i< 4; ++i) {
#ifdef PLEGMA_SCATTERING_SPIN12
      stochastic_oet_prop_d_zero_mom.pop_back();
#endif
      stochastic_oet_prop_u_zero_mom.pop_back();
      stochastic_oet_prop_u_fini_mom.pop_back();
    }


  } //initialize plegma

  finalize();
  
  return 0;
}
