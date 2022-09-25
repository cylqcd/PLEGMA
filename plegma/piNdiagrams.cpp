
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
static std::vector<std::string> listOpt = {"verbosity", "load-gauge","nsmear-APE","alpha-APE", "nsmear-gauss","alpha-gauss","nsrc","src-filename", "momlist-filename", "readStochSamples","time-dilution","nstochSamples","confnumber","contractionstoch","contractionstd","contractionoet"};
// Note here sinkMom is used as the momentum insertion in the sequential souce, probably has to be renamed to seqMom

void produceOutput( PLEGMA_ScattCorrelator<float> source,
                    std::string outputFilename,
                    std::string diagram_name,
                    int n_stochastic_samples){
  TIME_INSIDE(source.apply_phase());
  TIME_INSIDE(source.apply_sign(diagram_name));
  TIME_INSIDE(source.applyBoundaryConditions( true ));
  TIME_INSIDE(source.normalize_nstoch(n_stochastic_samples));
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
  bool do_contraction_std;
  bool do_stochastic; 
  bool do_stochastic_oet;
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
  HGC_options->set("contractionstoch", "We are performing stochastic contraction for the piN-piN diagrams B,W,D1ii,D1ff,T", verbosity, do_stochastic);
  HGC_options->set("contractionstd", "We are performing std contractions for nucleon and delta in both isospin channels", verbosity, do_contraction_std);
  HGC_options->set("contractionoet", "We are performing oet contraction for Z diagrams and for pions", verbosity, do_stochastic_oet);
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
    std::vector<PLEGMA_Vector<float>*> stochastic_oet_prop_d_zero_mom;
    std::vector<PLEGMA_Vector<float>*> stochastic_oet_prop_d_fini_mom;
    for(int i=0; i< 4; ++i) {

      stochastic_oet_prop_u_fini_mom.push_back(new PLEGMA_Vector<float>(HOST));
      stochastic_oet_prop_u_zero_mom.push_back(new PLEGMA_Vector<float>(HOST));
      stochastic_oet_prop_d_zero_mom.push_back(new PLEGMA_Vector<float>(HOST));
      stochastic_oet_prop_d_fini_mom.push_back(new PLEGMA_Vector<float>(HOST));

    }


    //Computing time-diluted stochastic propagators and stochastic source

    std::vector<PLEGMA_Vector<float>*> stochastic_sources;
    std::vector<PLEGMA_Vector<float>*> stochastic_propags;

    for(int i=0; i< n_stochastic_samples; ++i) {
      stochastic_sources.push_back(new PLEGMA_Vector<float>(HOST));
      stochastic_propags.push_back(new PLEGMA_Vector<float>(HOST));
    }

    PLEGMA_Vector<double> vectorStoc_source_oet;
    vectorStoc_source_oet.randInit(rand_seed1);

    if (do_stochastic==true){

      PLEGMA_printf("Start producing stochastic vectors and propagators\n");
      //Note that we replace the f1<-f2 DN propagator with a stochastic one
      //in two steps actually
      //DN(x_f1 <- x_f2 ) = \phihat(x_f2)(x_f1)\xi^{dagger}(x_f2)(x_f2)
      //where x_f2 is the source
      //      x_f1 is the sink
      //      phihat is the DN propagator obtained by acting on xi
      //      xi is the stochastic source
      //In practice however we invert for the UP type flavour and use 
      //the gamma_5 trick:
      //DN(x_f1 <- x_f2) =gamma_5*U(x_f2 <- x_f1)^dagger*gamma_5
      //-->>
      //gamma_5*\xi(x_f1)(x_f1)*\phi(x_f2)(x_f1)^dagger*gamma_5
      //In the following lines we compute phi and xi
      //Note that in the following we do not apply gamma_5 to xi and
      //phi, because we also construct U(x_f1, x_f2) for the I=1/2 case
      //Producing the stochastic source
      if (readstochastic==0){
      
        PLEGMA_Vector<double> vectorAuxD1(BOTH);//For storing the source (rotated and smeared)
        PLEGMA_Vector<double> vectorAuxD2(BOTH);//For storing the propagotor for the time-slices
        PLEGMA_Vector<double> vectorInOut; //temporary vector using in solve
        PLEGMA_Vector<double> vectorSource(BOTH);//For storing the source 
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

          vectorAuxD1.unload();
          stochastic_sources[i]->copy(vectorAuxD1,HOST);
          vectorAuxD1.load();
          vectorAuxD1.copy(vectorSource);
     
          //Step(3) Smearing all the time slice
          TIME(vectorAuxD2.gaussianSmearing(vectorAuxD1, smearedGauge, nsmearGauss, alphaGauss ),"ISOSPIN32");
 
          //Step(4) We rotate the source to the physical basis
          TIME(vectorAuxD1.rotateToPhysicalBasis(vectorAuxD2,+1),"ISOSPIN32");

          //In vectorAuxD2 we store the results for the inversion
          vectorAuxD2.scale(0.0);
    
          if (timedilution){
            PLEGMA_printf("#piNdiagrams: Full time dilution is turned on\n");
            for (int timeidx=0; timeidx< HGC_totalL[DIM_T]; ++timeidx){
              //Step(5) pick out a particular timeslice from the source
              vectorInOut.absorbTimeslice(vectorAuxD1, timeidx);
              //Step(6) Solve
              TIME(solver.solve(vectorInOut, vectorInOut),"ISOSPIN32");
              //Step(7) absorbing the particular timeslice to a 4d vector
              vectorAuxD2.absorbTimeslice(vectorInOut, timeidx, false);
            }
          }
          else{
            PLEGMA_printf("#piNdiagrams: No time dilution is used n stochastic propagators\n");
            vectorInOut.copy(vectorAuxD1);
            TIME(solver.solve(vectorInOut, vectorInOut),"ISOSPIN32");
            vectorAuxD2.copy(vectorInOut);
          } 
    

          //Step(6) We rotate back the propagator to the physical basis
          TIME(vectorAuxD1.rotateToPhysicalBasis(vectorAuxD2,+1),"ISOSPIN32");

          //Step(7) Smearing all the time slice in the propagator
          TIME(vectorAuxD2.gaussianSmearing(vectorAuxD1, smearedGauge, nsmearGauss, alphaGauss ),"ISOSPIN32");

          //Step(8) Save the propagator to the disk
          {
            PLEGMA_Vector<float> vectorAuxF;
            vectorAuxF.copy(vectorAuxD2);
            vectorAuxF.unload();
            vectorAuxF.writeLIME(outfile_V+"globalTfulltimedilution_propagator_nstoch"+std::to_string(i)+"_"+confnumber);
          }

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
          stochastic_sources[i]->copy(vectorRead,HOST);
          inputfilename=outfile_V+"globalTfulltimedilution_propagator_nstoch"+std::to_string(i)+"_"+confnumber;
          PLEGMA_printf("Read propagator from: %s\n",inputfilename.c_str());
          vectorRead.readFile(inputfilename,LIME_FORMAT);
          stochastic_propags[i]->copy(vectorRead,HOST);
        }
      }
    } //end of if (do_stochastic)
#if 0
    //Creating loops for zero momentum
    //for the I=1/2 case we consider only momentum for the nucleon
    //and not for the pion, so compute the pi0 loops for only the zero momentum case


    site source_stoch=site({0,0,0,0});
    std::vector<int> zero_mom_list_pion={0,0,0};
    momList piN12_zeropion(1,{zero_mom_list_pion,},{0,});

 
    PLEGMA_ScattCorrelator<float> Loop_UPDN(source_stoch, piN12_zeropion);

    if (do_stochastic==true){
      Loop_UPDN.initialize_diagram( glist_sink_meson, "L");

      for (int i=0; i<n_stochastic_samples; ++i){
       
        TIME(Loop_UPDN.Loop_diagrams( stochastic_propags[i], stochastic_sources[i], 0, false, true),"ISOSPIN12");

      }

      TIME(Loop_UPDN.normalize_nstoch(n_stochastic_samples),"ISOSPIN12");

      outfilename = outdiagramPrefix+confnumber+"_LoopUPDN_UP";
      TIME(produceOutput(Loop_UPDN, outfilename,"L"),"ISOSPIN12");

#endif

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

      site& source_full = source_full;

      site source_zero=site({0,0,0,source_full[DIM_T]});

      int sequential_time_source=source_full[DIM_T];


      //Calculations for source-position Calculations for source-position 
      PLEGMA_printf("\n ### Calculations for source-position %d - %02d.%02d.%02d.%02d begin now ###\n\n",
                    isource, source_full[0], source_full[1],
                    source_full[2], source_full[3]);
 
      //Create Propagator
      PLEGMA_Propagator<float> propUP(BOTH); //To be saved for all the coherent sources.
      PLEGMA_Propagator<float> propDN(BOTH);

      PLEGMA_Propagator<float> propUP_SL(NONE);
      PLEGMA_Propagator<float> propDN_SL(NONE);


      PLEGMA_Propagator<float> propTS(BOTH);

      std::vector<std::vector<int>> mpf1 = sourcemomentumList.uniq_p(1);
      momList list_mpf1(1,{mpf1,},{0,});
      PLEGMA_ScattCorrelator<float> corrNP(source_full, list_mpf1 );
      TIME(corrNP.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_sink_nucleon,"N"),"ISOSPIN32");

      PLEGMA_ScattCorrelator<float> corrN0(source_full, list_mpf1 );
      TIME(corrN0.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_sink_nucleon,"N"),"ISOSPIN12");



      asprintf(&ssource,"sx%02dsy%02dsz%02dst%03d", source_full[0], source_full[1], source_full[2], source_full[3]);
      std::string sourcepositiontext= (std::string)"_" + ssource; 
      free(ssource);

      PLEGMA_Gauge3D<double> smearedGauge3D;
      smearedGauge3D.absorb(smearedGauge, sequential_time_source);

      auto computePropagator = [&](PLEGMA_Propagator<float>& prop_SS, PLEGMA_Propagator<float>& prop_SL,
              double run_mu, WHICHFLAVOR fl, int nSmear,  bool finalize) {
                                 // ensuring mu value
                                 if(mu != run_mu) {
                                   updateOptions(fl);
                                   mu = run_mu;
                                   solver.UpdateSolver();
                                 }
                                 for(int isc = 0 ; isc < 12 ; isc++){
                                   PLEGMA_Vector<double> vectorInOut;
                                   { // Smearing the source
                                     PLEGMA_Vector3D<double> vector1, vector2;
                                     vector1.pointSource(source_full, isc/3, isc%3, DEVICE);
                                     TIME(vector2.gaussianSmearing(vector1, smearedGauge3D, nSmear, alphaGauss),"ISOSPIN32");
                                     vectorInOut.absorb(vector2,source_full[DIM_T]);
                                   }
                                   // Inverting
                                   PLEGMA_printf("Going to invert %s for component %d\n",
                                                 fl==LIGHT ? "LIGHT" : (fl == STRANGE ? "STRANGE" : "CHARM"), isc);
                                   {
                                    PLEGMA_Vector<double> vectorAuxD;
                                    TIME(vectorAuxD.rotateToPhysicalBasis(vectorInOut,run_mu/abs(run_mu)),"ISOSPIN32");
                                    TIME(vectorInOut.copy(vectorAuxD),"ISOSPIN32");
                                   }

                                   TIME(solver.solve(vectorInOut, vectorInOut),"ISOSPIN32");
                                   {
                                    PLEGMA_Vector<double> vectorAuxD;
                                    TIME(vectorAuxD.rotateToPhysicalBasis(vectorInOut,run_mu/abs(run_mu)),"ISOSPIN32");
                                    TIME(vectorInOut.copy(vectorAuxD),"ISOSPIN32");
                                   }

                                   if(prop_SL.getAllocation() != NONE) {
                                     PLEGMA_Vector<float> vectorAuxF;
                                     vectorAuxF.copy(vectorInOut);
                                     prop_SL.absorb(vectorAuxF, isc/3, isc%3);
                                   }
                                   { // Smearing the solution
                                     PLEGMA_Vector<double> vectorAuxD;
                                     PLEGMA_Vector<float> vectorAuxF;
                                     TIME(vectorAuxD.gaussianSmearing(vectorInOut, smearedGauge, nSmear, alphaGauss),"ISOSPIN32");
                                     vectorAuxF.copy(vectorAuxD);
                                     prop_SS.absorb(vectorAuxF, isc/3, isc%3);
                                   }
                                 }
                                 if(finalize) {
                                   prop_SS.rotateToPhysicalBase_device(run_mu/abs(run_mu));
                                   prop_SS.applyBoundaries_device(source_full[DIM_T]);
                                 }
                               };

      TIME(computePropagator(propUP, propUP_SL, mu, LIGHT, nsmearGauss, false), "ISOSPIN32");
      TIME(computePropagator(propDN, propDN_SL, -mu, LIGHT, nsmearGauss, false),"ISOSPIN32");

        /*if(outfile_upS!="")
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
        }*/
      std::string outfilename;

         

     //N,D diagram
      if (do_contraction_std == true)
      {
        std::vector<std::vector<int>> mtot = sourcemomentumList.uniq_p(3);
        momList list_mtot(1,{mtot,},{0,});
        PLEGMA_ScattCorrelator<float> corrD(source_full,list_mtot);

        //initialize diagram
        corrD.initialize_diagram( glist_source_delta_unpaired, glist_sink_delta_unpaired,glist_source_delta, glist_sink_delta,"D");
      
        PLEGMA_ScattCorrelator<float> reductionsT1(source_zero, mtot);
        PLEGMA_ScattCorrelator<float> reductionsT2(source_zero, mtot);

        TIME(reductionsT1.T1(glist_source_delta, glist_sink_delta, propUP, propUP, propUP),"ISOSPIN32");

        TIME(reductionsT2.T2(glist_source_delta, glist_sink_delta, propUP, propUP, propUP),"ISOSPIN32");

        //write D
        outfilename=outdiagramPrefix+confnumber+ sourcepositiontext+"_D";
	
        TIME( corrD.D_diagrams( reductionsT1, reductionsT2 ),"ISOSPIN32");
        TIME( corrD.apply_phase(),"ISOSPIN32" );
        TIME( corrD.apply_sign("D") ,"ISOSPIN32");
        TIME( corrD.applyBoundaryConditions( true ),"ISOSPIN32" );
        TIME( corrD.writeHDF5(outfilename),"ISOSPIN32" );


        std::vector<std::vector<int>> mpf1 = sourcemomentumList.uniq_p(1);
        momList list_mpf1(1,{mpf1,},{0,});
        PLEGMA_ScattCorrelator<float> corrNP(source_full, list_mpf1 );
        PLEGMA_ScattCorrelator<float> corrN0(source_full, list_mpf1 );

        //initialize diagram
       
        corrNP.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_sink_nucleon,"NP");
        corrN0.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_sink_nucleon,"N0");


      
        //Computing T reductions+recombination
        { 
          PLEGMA_ScattCorrelator<float> reductionsT1N(source_zero, mpf1);
          PLEGMA_ScattCorrelator<float> reductionsT2N(source_zero, mpf1);
          //First we compute N+ (proton) (we need for M diagram (N+p+)) and for spin half (N+ pi_0)
          TIME(reductionsT1N.T1(glist_source_nucleon, glist_sink_nucleon, propUP, propDN, propUP), "ISOSPIN32");

          TIME(reductionsT2N.T2(glist_source_nucleon, glist_sink_nucleon, propUP, propDN, propUP), "ISOSPIN32");

          TIME(corrNP.N_diagrams( reductionsT1N, reductionsT2N ),"ISOSPIN32");

          //Secondly compute N_0 (neutron)(we need for M diagram (N0p+))
          TIME(reductionsT1N.T1(glist_source_nucleon, glist_sink_nucleon, propDN, propUP, propDN), "ISOSPIN12");
        
          TIME(reductionsT2N.T2(glist_source_nucleon, glist_sink_nucleon, propDN, propUP, propDN), "ISOSPIN12");

          TIME(corrN0.N_diagrams( reductionsT1N, reductionsT2N ),"ISOSPIN12");
        }

      }//end of do_contraction_std

#if 0
          //For I=1/2 I_3=+1/2
          outfilename=outdiagramPrefix+confnumber+ sourcepositiontext+"_DELTA_DNUPUP_T2";
          TIME(reductionsT2.T2(glist_source_delta, glist_sink_delta, propDN_coherent, propUP_coherent, propUP_coherent), "ISOSPIN12");
          TIME( corrD.convertTreductiontoDiagram( reductionsT2, -1 ), "ISOSPIN12"); //The argument -1 indicates the corrD does not contain any mesonic 
          //gamma structure
          TIME( produceOutput(corrD, outfilename, "D" ) ,"ISOSPIN12" );

          outfilename=outdiagramPrefix+confnumber+ sourcepositiontext+"_DELTA_DNUPUP_T1";
          TIME(reductionsT1.T1(glist_source_delta, glist_sink_delta, propDN_coherent, propUP_coherent, propUP_coherent), "ISOSPIN12");
          TIME( corrD.convertTreductiontoDiagram( reductionsT1, -1 ), "ISOSPIN12");
          TIME( produceOutput(corrD, outfilename, "D" ) ,"ISOSPIN12" );

          outfilename=outdiagramPrefix+confnumber+ sourcepositiontext+"_DELTA_UPUPDN_T1";
          TIME(reductionsT1.T1(glist_source_delta, glist_sink_delta, propUP_coherent, propUP_coherent, propDN_coherent), "ISOSPIN12");
          TIME( corrD.convertTreductiontoDiagram( reductionsT1,-1 ), "ISOSPIN12");
          TIME( produceOutput(corrD, outfilename, "D" ) ,"ISOSPIN12" );

          outfilename=outdiagramPrefix+confnumber+ sourcepositiontext+"_DELTA_UPDNUP_T1";
          TIME(reductionsT1.T1(glist_source_delta, glist_sink_delta, propUP_coherent, propDN_coherent, propUP_coherent), "ISOSPIN12");
          TIME( corrD.convertTreductiontoDiagram( reductionsT1,-1 ), "ISOSPIN12");
          TIME( produceOutput(corrD, outfilename, "D" ) ,"ISOSPIN12" );

          outfilename=outdiagramPrefix+confnumber+ sourcepositiontext+"_DELTA_UPDNUP_T2";
          TIME(reductionsT2.T2(glist_source_delta, glist_sink_delta, propUP_coherent, propUP_coherent, propDN_coherent), "ISOSPIN12");
          TIME( corrD.convertTreductiontoDiagram( reductionsT2,-1 ), "ISOSPIN12");
          TIME( produceOutput(corrD, outfilename, "D" ) ,"ISOSPIN12" );
#endif 
      
      //P diagram
      std::vector<std::vector<int>> mpi2 = sourcemomentumList.uniq_p(0);
      momList list_mpi2(1,{mpi2,},{0,});
      PLEGMA_ScattCorrelator<float> corrP0UP(source_full, list_mpi2);
      PLEGMA_ScattCorrelator<float> corrP0DN(source_full, list_mpi2);
      PLEGMA_ScattCorrelator<float> corrPPUP(source_full, list_mpi2);
      PLEGMA_ScattCorrelator<float> corrPPDN(source_full, list_mpi2);

      corrP0UP.initialize_diagram(glist_source_meson, glist_sink_meson, "P0UP");
      corrP0DN.initialize_diagram(glist_source_meson, glist_sink_meson, "P0DN");
      corrPPUP.initialize_diagram(glist_source_meson, glist_sink_meson, "PPUP");
      corrPPDN.initialize_diagram(glist_source_meson, glist_sink_meson, "PPDN");



      //We can compute V2 contractions for B and V3 contraction for W first
      //without having to compute it for all the iterations in the loop
      //over the sequential momentum
      //Provided we have the same pf1,pf2 pairs for all pi2 sequential momentum
      //For the saved V2 and V3 reductions we have to use all possible unique pf1 and pf2
     
      //Here the prefix UU means that reduction is based phi and xi, without the gamma_5
      //We replace U(x_f2,x_f1) with phi(x_f2) xi^dagger(x_f1)
      //phi goes to V2 reduction and xi goes to V3 reduction 
      
      std::vector<PLEGMA_ScattCorrelator<float>*> reductions_UU_V2_GAMMAF1D_U;//implemented
      std::vector<PLEGMA_ScattCorrelator<float>*> reductions_UU_V4_GAMMAF1U_D;//implemented
      std::vector<PLEGMA_ScattCorrelator<float>*> reductions_UU_V3_GAMMAF2U;//implemented
     
      //Here the prefix DD means that reduction is based phi*g5 and xi*g5
      //We replace D(x_f2,x_f1) with xi(x_f2)*gamma_5* phi^dagger(x_f1) *gamma_5
      //phi goes to V3 reduction and xi goes to V2 reduction 
      
      std::vector<PLEGMA_ScattCorrelator<float>*> reductions_DD_V3_GAMMAF2D;//implemented
      std::vector<PLEGMA_ScattCorrelator<float>*> reductions_DD_V3_GAMMAF2U;//implemented

      std::vector<PLEGMA_ScattCorrelator<float>*> reductions_DD_V2_GAMMAF1U_D;//implemented
      std::vector<PLEGMA_ScattCorrelator<float>*> reductions_DD_V4_GAMMAF1U_D;//implemented

      std::vector<PLEGMA_ScattCorrelator<float>*> reductions_DD_V2_GAMMAF1U_U;//implemented
        
      if (do_stochastic==true){
        for(int i=0; i< n_stochastic_samples; ++i) {
          try
          {
            reductions_DD_V2_GAMMAF1U_U.push_back(new PLEGMA_ScattCorrelator<float>(source_zero, sourcemomentumList.uniq_p(1)));
            reductions_UU_V2_GAMMAF1D_U.push_back(new PLEGMA_ScattCorrelator<float>(source_zero, sourcemomentumList.uniq_p(1)));
            reductions_UU_V4_GAMMAF1U_D.push_back(new PLEGMA_ScattCorrelator<float>(source_zero, sourcemomentumList.uniq_p(1)));

            reductions_UU_V3_GAMMAF2U.push_back(new PLEGMA_ScattCorrelator<float>(source_zero, sourcemomentumList.uniq_p(2)));
            reductions_DD_V3_GAMMAF2U.push_back(new PLEGMA_ScattCorrelator<float>(source_zero, sourcemomentumList.uniq_p(2)));
            reductions_DD_V3_GAMMAF2D.push_back(new PLEGMA_ScattCorrelator<float>(source_zero, sourcemomentumList.uniq_p(2)));

            reductions_DD_V4_GAMMAF1U_D.push_back(new PLEGMA_ScattCorrelator<float>(source_zero, sourcemomentumList.uniq_p(1)));
            reductions_DD_V2_GAMMAF1U_D.push_back(new PLEGMA_ScattCorrelator<float>(source_zero, sourcemomentumList.uniq_p(1)));
          }
          catch(std::bad_alloc&){
            PLEGMA_printf("Memory allocation fails to store factors");
            exit(1); 
          }

        }

        for (int i=0; i<n_stochastic_samples; ++i){

          PLEGMA_Vector<float> stochastic_propagator;
          PLEGMA_Vector<float> stochastic_source;
            
          stochastic_propagator.copy(*stochastic_propags[i],HOST);
          stochastic_source.copy(*stochastic_sources[i],HOST);
            
          stochastic_propagator.load();
          stochastic_source.load();

          //For U(xf1,xf2)
          //B3,B5,B9,B11
          //D1ii1,D1ii3,D1ii5,D1ii7
          TIME(reductions_UU_V2_GAMMAF1D_U[i]->V2( stochastic_propagator,     glist_sink_nucleon, propDN, propUP, false), "ISOSPIN12");

          //B4,B6
          //D1ii2,D1ii4,D1ii6,D1ii8
          //B10,B12
          TIME(reductions_UU_V4_GAMMAF1U_D[i]->V4( stochastic_propagator,     glist_sink_nucleon, propUP, propDN, false), "ISOSPIN12");
 
          //W5,W6,W7,W8
          //D1ii1,D1ii2,D1ii3,D1ii4
          //W9,W10,W11,W12,W17,W18,W19,W20
          //D1ii5,D1ii6,D1ii7,D1ii8
          TIME(reductions_UU_V3_GAMMAF2U[i]->V3( stochastic_source, glist_sink_meson,   propUP, true), "ISOSPIN12");

          //For D(xf1,xf2)

          stochastic_propagator.apply_gamma5(); 

	  //W1,W2,W3,W4
          //W25,W26,W27,W28
          //W29,W30,W31,W32
          //D1ii13,D1ii14
          //D1ii15,D1ii16
          //W33,W34,W35,W36
          //D1ii17,D1ii18
          //D1ii19,D1ii20
          TIME(reductions_DD_V3_GAMMAF2U[i]->V3( stochastic_propagator, glist_sink_meson,   propUP, true), "ISOSPIN32");

          //W13,W14,W15,W16
          //W21,W22,W23,W24
          //D1ii9,D1ii10
          //D1ii11,D1ii12
          TIME(reductions_DD_V3_GAMMAF2D[i]->V3( stochastic_propagator, glist_sink_meson,   propDN, true), "ISOSPIN12");


          stochastic_source.apply_gamma5();
          //D1ii9,D1ii10
          //D1ii11,D1ii12
          //B7,B8
          //B1,B2
          TIME(reductions_DD_V2_GAMMAF1U_U[i]->V2( stochastic_source, glist_sink_nucleon, propUP, propUP, false), "ISOSPIN32");

          //D1ii13,14,B13,B14,B17,B18
          TIME(reductions_DD_V4_GAMMAF1U_D[i]->V4( stochastic_source, glist_sink_nucleon, propUP, propDN, false), "ISOSPIN12");

        //D1ii15,D1ii16,B15,16,B19,B20
        TIME(reductions_DD_V2_GAMMAF1U_D[i]->V2( stochastic_source, glist_sink_nucleon, propUP, propDN, false), "ISOSPIN12");


        } //end of for stochastic_sample
      } //end of if(do_stochastic)

      //We draw a different random vector for every source position
      vectorStoc_source_oet.stochastic_Z(nroots);
      
      //Store zero momentum oet propagators: also for the oet propagators we produce coherent sources
      std::array<PLEGMA_Vector<float>,4> stochastic_propagator_momzero;
      if (do_stochastic_oet==true)
      {

         //Doing for +mu for the UP propagator spin dilution oet
         if(mu<0) {
           mu*=-1.;
           solver.UpdateSolver();
         }

         PLEGMA_Vector<double> vectortmp1;
         PLEGMA_Vector<double> vectortmp2;          
	 PLEGMA_Vector<double> vectorSave_diluted;


         vectortmp1.absorbTimeslice(vectorStoc_source_oet, sequential_time_source);

         {  // Smearing the source

            PLEGMA_Vector3D<double> vector1, vector2;
            vector1.absorb(vectortmp1, sequential_time_source);
            TIME(vector2.gaussianSmearing(vector1, smearedGauge3D, nsmearGauss, alphaGauss),"ISOSPIN32");
            vectortmp1.absorb(vector2, sequential_time_source);

         }
 
          //Dilution     
         vectortmp2.dilutespin(vectortmp1,0);

         //Save the smeared,transformed and diluted source for non-zero momentum oet.
         vectorSave_diluted.copy(vectortmp2);

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

          // vectortmp2.writeLIME(outfile_V+confnumber+"propagator_up"+sourcepositiontext+"mompi2_0_0_0_s"+std::to_string(spinindex));
           if (spinindex<3){

             vectortmp1.diluteSpinDisplace(vectorSave_diluted,spinindex+1,spinindex);
             vectorSave_diluted.copy(vectortmp1);
           }
         }
         
         //Doing for -mu for the DN propagator spin dilution oet
         if(mu>0) {
           mu*=-1.;
           solver.UpdateSolver();
         }

         //Dilution     
         vectortmp1.diluteSpinDisplace(vectorSave_diluted,0,3);

         //Save the smeared,transformed and diluted source for non-zero momentum oet.
         vectorSave_diluted.copy(vectortmp1);

         for (int spinindex=0; spinindex<4; ++spinindex){
           //Transforming to physical base for the DN quark
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
           //vectortmp2.writeLIME(outfile_V+confnumber+"propagator_dn"+sourcepositiontext+"mompi2_0_0_0_s"+std::to_string(spinindex));
           if (spinindex<3){
             vectortmp1.diluteSpinDisplace(vectorSave_diluted,spinindex+1,spinindex);
             vectorSave_diluted.copy(vectortmp1);
           }
         }
      }//end of do_stochastic_oet

      //zero momentum oet contractions
      std::array<PLEGMA_ScattCorrelator<float>,4> reductionsV3_diluted = {
         PLEGMA_ScattCorrelator<float>(source_zero, sourcemomentumList.uniq_p(2)),
         PLEGMA_ScattCorrelator<float>(source_zero, sourcemomentumList.uniq_p(2)),
         PLEGMA_ScattCorrelator<float>(source_zero, sourcemomentumList.uniq_p(2)),
         PLEGMA_ScattCorrelator<float>(source_zero, sourcemomentumList.uniq_p(2))
      };

      std::array<PLEGMA_ScattCorrelator<float> ,4> reductionsV3_diluted_STOCHU_DN = {
         PLEGMA_ScattCorrelator<float>(source_zero, sourcemomentumList.uniq_p(2)),
         PLEGMA_ScattCorrelator<float>(source_zero, sourcemomentumList.uniq_p(2)),
         PLEGMA_ScattCorrelator<float>(source_zero, sourcemomentumList.uniq_p(2)),
         PLEGMA_ScattCorrelator<float>(source_zero, sourcemomentumList.uniq_p(2))
      };
      std::array<PLEGMA_ScattCorrelator<float> ,4> reductionsV3_diluted_STOCHD_UP = {
         PLEGMA_ScattCorrelator<float>(source_zero, sourcemomentumList.uniq_p(2)),
         PLEGMA_ScattCorrelator<float>(source_zero, sourcemomentumList.uniq_p(2)),
         PLEGMA_ScattCorrelator<float>(source_zero, sourcemomentumList.uniq_p(2)),
         PLEGMA_ScattCorrelator<float>(source_zero, sourcemomentumList.uniq_p(2))
      };

      std::array<PLEGMA_ScattCorrelator<float> ,4> reductionsV3_diluted_STOCHU_UP = {
         PLEGMA_ScattCorrelator<float>(source_zero, sourcemomentumList.uniq_p(2)),
         PLEGMA_ScattCorrelator<float>(source_zero, sourcemomentumList.uniq_p(2)),
         PLEGMA_ScattCorrelator<float>(source_zero, sourcemomentumList.uniq_p(2)),
         PLEGMA_ScattCorrelator<float>(source_zero, sourcemomentumList.uniq_p(2))
      };


      std::array<PLEGMA_ScattCorrelator<float>,4> reductionsV4_diluted_STOCHU_DN_UP = {
         PLEGMA_ScattCorrelator<float>(source_zero, sourcemomentumList.uniq_p(1)),
         PLEGMA_ScattCorrelator<float>(source_zero, sourcemomentumList.uniq_p(1)),
         PLEGMA_ScattCorrelator<float>(source_zero, sourcemomentumList.uniq_p(1)),
         PLEGMA_ScattCorrelator<float>(source_zero, sourcemomentumList.uniq_p(1))
      };

      std::array<PLEGMA_ScattCorrelator<float>,4> reductionsV2_diluted_STOCHU_DN_UP = {
         PLEGMA_ScattCorrelator<float>(source_zero, sourcemomentumList.uniq_p(1)),
         PLEGMA_ScattCorrelator<float>(source_zero, sourcemomentumList.uniq_p(1)),
         PLEGMA_ScattCorrelator<float>(source_zero, sourcemomentumList.uniq_p(1)),
         PLEGMA_ScattCorrelator<float>(source_zero, sourcemomentumList.uniq_p(1))
      };
      std::array<PLEGMA_ScattCorrelator<float>,4> reductionsV2_diluted_STOCHD_UP_UP = {
         PLEGMA_ScattCorrelator<float>(source_zero, sourcemomentumList.uniq_p(1)),
         PLEGMA_ScattCorrelator<float>(source_zero, sourcemomentumList.uniq_p(1)),
         PLEGMA_ScattCorrelator<float>(source_zero, sourcemomentumList.uniq_p(1)),
         PLEGMA_ScattCorrelator<float>(source_zero, sourcemomentumList.uniq_p(1))
      };
      std::array<PLEGMA_ScattCorrelator<float>,4> reductionsV2_diluted_STOCHU_DN_DN = {
         PLEGMA_ScattCorrelator<float>(source_zero, sourcemomentumList.uniq_p(1)),
         PLEGMA_ScattCorrelator<float>(source_zero, sourcemomentumList.uniq_p(1)),
         PLEGMA_ScattCorrelator<float>(source_zero, sourcemomentumList.uniq_p(1)),
         PLEGMA_ScattCorrelator<float>(source_zero, sourcemomentumList.uniq_p(1))
      };
      std::array<PLEGMA_ScattCorrelator<float>,4> reductionsV2_diluted_STOCHD_UP_DN = {
         PLEGMA_ScattCorrelator<float>(source_zero, sourcemomentumList.uniq_p(1)),
         PLEGMA_ScattCorrelator<float>(source_zero, sourcemomentumList.uniq_p(1)),
         PLEGMA_ScattCorrelator<float>(source_zero, sourcemomentumList.uniq_p(1)),
         PLEGMA_ScattCorrelator<float>(source_zero, sourcemomentumList.uniq_p(1))
      };
      std::array<PLEGMA_ScattCorrelator<float>,4> reductionsV4_diluted_STOCHD_UP_DN = {
         PLEGMA_ScattCorrelator<float>(source_zero, sourcemomentumList.uniq_p(1)),
         PLEGMA_ScattCorrelator<float>(source_zero, sourcemomentumList.uniq_p(1)),
         PLEGMA_ScattCorrelator<float>(source_zero, sourcemomentumList.uniq_p(1)),
         PLEGMA_ScattCorrelator<float>(source_zero, sourcemomentumList.uniq_p(1))
      };


      if (do_stochastic_oet==true){
        for (int i=0; i< 4; ++i){
          PLEGMA_Vector<float> st_oet_u_zero;
          PLEGMA_Vector<float> st_oet_d_zero;

          st_oet_u_zero.copy(*stochastic_oet_prop_u_zero_mom[i],HOST);
          st_oet_u_zero.load();
          st_oet_d_zero.copy(*stochastic_oet_prop_d_zero_mom[i],HOST);
          st_oet_d_zero.load();


          TIME(reductionsV2_diluted_STOCHU_DN_DN[i].V2( st_oet_u_zero, glist_sink_nucleon, propDN, propDN, false),"ISOSPIN12");
          TIME(reductionsV2_diluted_STOCHU_DN_UP[i].V2( st_oet_u_zero, glist_sink_nucleon, propDN, propUP, false),"ISOSPIN32");
          TIME(reductionsV2_diluted_STOCHD_UP_UP[i].V2( st_oet_d_zero, glist_sink_nucleon, propUP, propUP, false),"ISOSPIN12");
          TIME(reductionsV2_diluted_STOCHD_UP_DN[i].V2( st_oet_d_zero, glist_sink_nucleon, propUP, propDN, false),"ISOSPIN12");
          TIME(reductionsV4_diluted_STOCHD_UP_DN[i].V4( st_oet_d_zero, glist_sink_nucleon, propUP, propDN, false),"ISOSPIN12");
          TIME(reductionsV4_diluted_STOCHU_DN_UP[i].V4( st_oet_u_zero, glist_sink_nucleon, propDN, propUP, false),"ISOSPIN32");

          st_oet_u_zero.apply_gamma5();

          TIME(reductionsV3_diluted_STOCHU_UP[i].V3( st_oet_u_zero, glist_sink_meson, propUP, true),"ISOSPIN32");
          TIME(reductionsV3_diluted_STOCHU_DN[i].V3( st_oet_u_zero, glist_sink_meson, propDN, true),"ISOSPIN12");

          st_oet_d_zero.apply_gamma5();
          TIME(reductionsV3_diluted_STOCHD_UP[i].V3( st_oet_d_zero, glist_sink_meson, propUP, true),"ISOSPIN12");

        }
      }

#if 0
      //T diagram piN sink
      //Details eq 51-56 in Marcus's notes
      {

        PLEGMA_printf("Start calculating T piN sink\n");
        //extract all the momenta that corresponds to pi2==(0,0,0)
	momList list_pf1pf2comb = sourcemomentumList.extract({0,0,0}, 0);
        //for the I=1/2 case we compute only at zero pion momentum
        momList list_pf        = list_pf1pf2comb.extract({0,0,0}, 2);

        //udu- dbaru - ubarubarubar: N+pi+ <- Delta++
        PLEGMA_ScattCorrelator<float> corrT_piNsink_1(source_full, list_pf1pf2comb);
        PLEGMA_ScattCorrelator<float> corrT_piNsink_3(source_full, list_pf1pf2comb);
        PLEGMA_ScattCorrelator<float> corrT_piNsink_5(source_full, list_pf1pf2comb);

        //udu- ubaru - dbarubarubar: N_+pi0 <- Delta_1/2 1
        PLEGMA_ScattCorrelator<float> corrT_piNsink_7(source_full, list_pf);
        PLEGMA_ScattCorrelator<float> corrT_piNsink_8(source_full, list_pf);
        PLEGMA_ScattCorrelator<float> corrT_piNsink_9(source_full, list_pf);
        PLEGMA_ScattCorrelator<float> corrT_piNsink_10(source_full, list_pf);

        //udu- dbard - dbarubarubar: N_+pi0 <- Delta_1/2 1
        PLEGMA_ScattCorrelator<float> corrT_piNsink_11(source_full, list_pf);
        PLEGMA_ScattCorrelator<float> corrT_piNsink_12(source_full, list_pf);

        //udu- ubaru - dbarubarubar: N_+pi0 <- Delta_1/2 2
        PLEGMA_ScattCorrelator<float> corrT_piNsink_13(source_full, list_pf);
        PLEGMA_ScattCorrelator<float> corrT_piNsink_15(source_full, list_pf);
        //udu- dbard - dbarubarubar: N_+pi0 <- Delta_1/2 2
        PLEGMA_ScattCorrelator<float> corrT_piNsink_17(source_full, list_pf);

        //dud- dbaru - dbarubarubar: N_0pi+ <- Delta_1/2 1
        PLEGMA_ScattCorrelator<float> corrT_piNsink_19(source_full, list_pf);
        PLEGMA_ScattCorrelator<float> corrT_piNsink_20(source_full, list_pf);
        PLEGMA_ScattCorrelator<float> corrT_piNsink_21(source_full, list_pf);
        PLEGMA_ScattCorrelator<float> corrT_piNsink_22(source_full, list_pf);

        //dud- dbaru - dbarubarubar: N_0pi+ <- Delta_1/2 2
        PLEGMA_ScattCorrelator<float> corrT_piNsink_23(source_full, list_pf);
        PLEGMA_ScattCorrelator<float> corrT_piNsink_25(source_full, list_pf);

        corrT_piNsink_1.initialize_diagram(glist_source_delta_unpaired, glist_sink_nucleon_unpaired, glist_source_delta, glist_sink_nucleon,  glist_sink_meson,  "32", "T1"); 
        corrT_piNsink_3.initialize_diagram(glist_source_delta_unpaired, glist_sink_nucleon_unpaired, glist_source_delta, glist_sink_nucleon,  glist_sink_meson,  "32", "T3");
        corrT_piNsink_5.initialize_diagram(glist_source_delta_unpaired, glist_sink_nucleon_unpaired, glist_source_delta, glist_sink_nucleon,  glist_sink_meson,  "32", "T5");
        corrT_piNsink_7.initialize_diagram(glist_source_delta_unpaired, glist_sink_nucleon_unpaired, glist_source_delta, glist_sink_nucleon,  glist_sink_meson,  "12", "T7");
        corrT_piNsink_8.initialize_diagram(glist_source_delta_unpaired, glist_sink_nucleon_unpaired, glist_source_delta, glist_sink_nucleon,  glist_sink_meson,  "12", "T8");
        corrT_piNsink_9.initialize_diagram(glist_source_delta_unpaired, glist_sink_nucleon_unpaired, glist_source_delta, glist_sink_nucleon,  glist_sink_meson,  "12", "T9");
        corrT_piNsink_10.initialize_diagram(glist_source_delta_unpaired, glist_sink_nucleon_unpaired, glist_source_delta, glist_sink_nucleon,  glist_sink_meson, "12", "T10");
        corrT_piNsink_11.initialize_diagram(glist_source_delta_unpaired, glist_sink_nucleon_unpaired, glist_source_delta, glist_sink_nucleon,  glist_sink_meson, "12", "T11");
        corrT_piNsink_12.initialize_diagram(glist_source_delta_unpaired, glist_sink_nucleon_unpaired, glist_source_delta, glist_sink_nucleon,  glist_sink_meson, "12", "T12");
        corrT_piNsink_13.initialize_diagram(glist_source_delta_unpaired, glist_sink_nucleon_unpaired, glist_source_delta, glist_sink_nucleon,  glist_sink_meson, "12", "T13");
        corrT_piNsink_15.initialize_diagram(glist_source_delta_unpaired, glist_sink_nucleon_unpaired, glist_source_delta, glist_sink_nucleon,  glist_sink_meson, "12", "T15");
        corrT_piNsink_17.initialize_diagram(glist_source_delta_unpaired, glist_sink_nucleon_unpaired, glist_source_delta, glist_sink_nucleon,  glist_sink_meson, "12", "T17");
        corrT_piNsink_19.initialize_diagram(glist_source_delta_unpaired, glist_sink_nucleon_unpaired, glist_source_delta, glist_sink_nucleon,  glist_sink_meson, "12", "T19");
        corrT_piNsink_20.initialize_diagram(glist_source_delta_unpaired, glist_sink_nucleon_unpaired, glist_source_delta, glist_sink_nucleon,  glist_sink_meson, "12", "T20");
        corrT_piNsink_21.initialize_diagram(glist_source_delta_unpaired, glist_sink_nucleon_unpaired, glist_source_delta, glist_sink_nucleon,  glist_sink_meson, "12", "T21");
        corrT_piNsink_22.initialize_diagram(glist_source_delta_unpaired, glist_sink_nucleon_unpaired, glist_source_delta, glist_sink_nucleon,  glist_sink_meson, "12", "T22");
        corrT_piNsink_23.initialize_diagram(glist_source_delta_unpaired, glist_sink_nucleon_unpaired, glist_source_delta, glist_sink_nucleon,  glist_sink_meson, "12", "T23");
        corrT_piNsink_25.initialize_diagram(glist_source_delta_unpaired, glist_sink_nucleon_unpaired, glist_source_delta, glist_sink_nucleon,  glist_sink_meson, "12", "T25");


        PLEGMA_printf("Initialization done\n");

        //we have already all the factors computed
  
        for (int i=0; i<n_stochastic_samples; ++i){

          TIME(corrT_piNsink_1.T_diagrams_piNsink(*reductions_DD_V3_GAMMAF2U[i], *reductions_DD_V2_GAMMAF1U_U[i], 1, true), "ISOSPIN32");
          TIME(corrT_piNsink_3.T_diagrams_piNsink(*reductions_DD_V3_GAMMAF2U[i], *reductions_DD_V2_GAMMAF1U_U[i], 3, true), "ISOSPIN32");
          TIME(corrT_piNsink_5.T_diagrams_piNsink(*reductions_DD_V3_GAMMAF2U[i], *reductions_DD_V2_GAMMAF1U_U[i], 5, true), "ISOSPIN32");
         TIME(corrT_piNsink_7.T_diagrams_piNsink(*reductions_UU_V3_GAMMAF2U_zero_mom[i], *reductions_UU_V4_GAMMAF1U_D[i], 7, true), "ISOSPIN12");
          TIME(corrT_piNsink_8.T_diagrams_piNsink(*reductions_UU_V3_GAMMAF2U_zero_mom[i], *reductions_UU_V4_GAMMAF1U_D[i], 8, true), "ISOSPIN12");
          TIME(corrT_piNsink_9.T_diagrams_piNsink(*reductions_UU_V3_GAMMAF2U_zero_mom[i], *reductions_UU_V2_GAMMAF1D_U[i], 9, true), "ISOSPIN12");
          TIME(corrT_piNsink_10.T_diagrams_piNsink(*reductions_UU_V3_GAMMAF2U_zero_mom[i], *reductions_UU_V2_GAMMAF1D_U[i], 10, true), "ISOSPIN12");
          TIME(corrT_piNsink_11.T_diagrams_piNsink(*reductions_DD_V3_GAMMAF2D_zero_mom[i], *reductions_DD_V2_GAMMAF1U_U[i], 11, true), "ISOSPIN12");
          TIME(corrT_piNsink_12.T_diagrams_piNsink(*reductions_DD_V3_GAMMAF2D_zero_mom[i], *reductions_DD_V2_GAMMAF1U_U[i], 12, true), "ISOSPIN12");
          TIME(corrT_piNsink_13.T_diagrams_piNsink(*reductions_UU_V3_GAMMAF2U_zero_mom[i], *reductions_UU_V4_GAMMAF1U_D[i], 13, true), "ISOSPIN12");
          TIME(corrT_piNsink_15.T_diagrams_piNsink(*reductions_UU_V3_GAMMAF2U_zero_mom[i], *reductions_UU_V2_GAMMAF1D_U[i], 15, true), "ISOSPIN12");
          TIME(corrT_piNsink_17.T_diagrams_piNsink(*reductions_DD_V3_GAMMAF2D_zero_mom[i], *reductions_DD_V2_GAMMAF1U_U[i], 17, true), "ISOSPIN12");
          TIME(corrT_piNsink_19.T_diagrams_piNsink(*reductions_DD_V3_GAMMAF2U_zero_mom[i], *reductions_DD_V4_GAMMAF1U_D[i], 19, true), "ISOSPIN12");
          TIME(corrT_piNsink_20.T_diagrams_piNsink(*reductions_DD_V3_GAMMAF2U_zero_mom[i], *reductions_DD_V4_GAMMAF1U_D[i], 20, true), "ISOSPIN12");
          TIME(corrT_piNsink_21.T_diagrams_piNsink(*reductions_DD_V3_GAMMAF2U_zero_mom[i], *reductions_DD_V2_GAMMAF1U_D[i], 21, true), "ISOSPIN12");
          TIME(corrT_piNsink_22.T_diagrams_piNsink(*reductions_DD_V3_GAMMAF2U_zero_mom[i], *reductions_DD_V2_GAMMAF1U_D[i], 22, true), "ISOSPIN12");
          TIME(corrT_piNsink_23.T_diagrams_piNsink(*reductions_DD_V3_GAMMAF2U_zero_mom[i], *reductions_DD_V4_GAMMAF1U_D[i], 23, true), "ISOSPIN12");
          TIME(corrT_piNsink_25.T_diagrams_piNsink(*reductions_DD_V3_GAMMAF2U_zero_mom[i], *reductions_DD_V2_GAMMAF1U_D[i], 25, true), "ISOSPIN12");

        }                
        outfilename=outdiagramPrefix+confnumber+sourcepositiontext+"_TpiNsink";
        TIME(produceOutput(corrT_piNsink_1, outfilename,  "T1",n_stochastic_samples), "ISOSPIN32");
        TIME(produceOutput(corrT_piNsink_3, outfilename,  "T1",n_stochastic_samples), "ISOSPIN32");
        TIME(produceOutput(corrT_piNsink_5, outfilename,  "T1",n_stochastic_samples), "ISOSPIN32");
        TIME(produceOutput(corrT_piNsink_7, outfilename,  "T1",n_stochastic_samples), "ISOSPIN12");
        TIME(produceOutput(corrT_piNsink_8, outfilename,  "T1",n_stochastic_samples), "ISOSPIN12");
        TIME(produceOutput(corrT_piNsink_9, outfilename,  "T1",n_stochastic_samples), "ISOSPIN12");
        TIME(produceOutput(corrT_piNsink_10, outfilename, "T1",n_stochastic_samples), "ISOSPIN12");
        TIME(produceOutput(corrT_piNsink_11, outfilename, "T1",n_stochastic_samples), "ISOSPIN12");
        TIME(produceOutput(corrT_piNsink_12, outfilename, "T1",n_stochastic_samples), "ISOSPIN12");
        TIME(produceOutput(corrT_piNsink_13, outfilename, "T1",n_stochastic_samples), "ISOSPIN12");
        TIME(produceOutput(corrT_piNsink_15, outfilename, "T1",n_stochastic_samples), "ISOSPIN12");
        TIME(produceOutput(corrT_piNsink_17, outfilename, "T1",n_stochastic_samples), "ISOSPIN12");
        TIME(produceOutput(corrT_piNsink_19, outfilename, "T1",n_stochastic_samples), "ISOSPIN12");
        TIME(produceOutput(corrT_piNsink_20, outfilename, "T1",n_stochastic_samples), "ISOSPIN12");
        TIME(produceOutput(corrT_piNsink_21, outfilename, "T1",n_stochastic_samples), "ISOSPIN12");
        TIME(produceOutput(corrT_piNsink_22, outfilename, "T1",n_stochastic_samples), "ISOSPIN12");
        TIME(produceOutput(corrT_piNsink_23, outfilename, "T1",n_stochastic_samples), "ISOSPIN12");
        TIME(produceOutput(corrT_piNsink_25, outfilename, "T1",n_stochastic_samples), "ISOSPIN12");
     }
#endif
      //We need the pi-N 4pt functions only at zero pion momentum and non-zero nucleon momentum
      //therefore we filter further the momentumlist corresponding to pi2==0 to also pf2==0
      //momList filtered_sourcemomentumList.uniq_p(1) = filtered_sourcemomentumList_pi20.extract(filter,2);
      
#if 0 
      //Loop diagram at the source
      if (do_stochastic){

        PLEGMA_ScattCorrelator<float> corrD1ii1(source_full, filtered_sourcemomentumList.uniq_p(1));
        PLEGMA_ScattCorrelator<float> corrD1ii2(source_full, filtered_sourcemomentumList.uniq_p(1));
        PLEGMA_ScattCorrelator<float> corrD1ii3(source_full, filtered_sourcemomentumList.uniq_p(1));
        PLEGMA_ScattCorrelator<float> corrD1ii4(source_full, filtered_sourcemomentumList.uniq_p(1));

        PLEGMA_ScattCorrelator<float> corrD1ii9(source_full, filtered_sourcemomentumList.uniq_p(1));
        PLEGMA_ScattCorrelator<float> corrD1ii10(source_full, filtered_sourcemomentumList.uniq_p(1));

        PLEGMA_ScattCorrelator<float> corrD1ii13(source_full, filtered_sourcemomentumList.uniq_p(1));
        PLEGMA_ScattCorrelator<float> corrD1ii14(source_full, filtered_sourcemomentumList.uniq_p(1));
        PLEGMA_ScattCorrelator<float> corrD1ii15(source_full, filtered_sourcemomentumList.uniq_p(1));
        PLEGMA_ScattCorrelator<float> corrD1ii16(source_full, filtered_sourcemomentumList.uniq_p(1));


        corrD1ii1.initialize_diagram(  glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "D1ii1");
        corrD1ii2.initialize_diagram(  glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "D1ii2");
        corrD1ii3.initialize_diagram(  glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "D1ii3");
        corrD1ii4.initialize_diagram(  glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "D1ii4");

        corrD1ii9.initialize_diagram(  glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "D1ii9");
        corrD1ii10.initialize_diagram(  glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "D1ii10");

        corrD1ii13.initialize_diagram(  glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "D1ii13");
        corrD1ii14.initialize_diagram(  glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "D1ii14");
        corrD1ii15.initialize_diagram(  glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "D1ii15");
        corrD1ii16.initialize_diagram(  glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "D1ii16");


        float *Loop_UPDN_source=(float *)malloc(2*sizeof(float)*glist_source_meson.size());


        for (int j=0; j< 2*glist_source_meson.size(); ++j){
          Loop_UPDN_source[j]=0.0;
        }


        for (int i=0; i<n_stochastic_samples; ++i){


          float *Loop_UPDN_sp;
          Loop_UPDN_sp=Loop_UPDN.get_source_time_slice();

          for (int j=0; j< glist_source_meson.size(); ++j){
            Loop_UPDN_source[2*j+0]+=Loop_UPDN_sp[2*j+0];
            Loop_UPDN_source[2*j+1]+=Loop_UPDN_sp[2*j+1];
          }
          free(Loop_UPDN_sp);


          TIME(corrD1ii1.D1ii_diagrams(*reductions_UU_V3_GAMMAF2U_zero_mom[i], *reductions_UU_V2_GAMMAF1D_U[i], Loop_UPDN_source, 0, 1, true),"ISOSPIN12");          
          TIME(corrD1ii2.D1ii_diagrams(*reductions_UU_V3_GAMMAF2U_zero_mom[i], *reductions_UU_V4_GAMMAF1U_D[i], Loop_UPDN_source, 0, 2, true),"ISOSPIN12");          
          TIME(corrD1ii3.D1ii_diagrams(*reductions_UU_V3_GAMMAF2U_zero_mom[i], *reductions_UU_V2_GAMMAF1D_U[i], Loop_UPDN_source, 0, 3, true),"ISOSPIN12");
          TIME(corrD1ii4.D1ii_diagrams(*reductions_UU_V3_GAMMAF2U_zero_mom[i], *reductions_UU_V4_GAMMAF1U_D[i], Loop_UPDN_source, 0, 4, true),"ISOSPIN12"); 

          TIME(corrD1ii9.D1ii_diagrams( *reductions_DD_V3_GAMMAF2D_zero_mom[i], *reductions_DD_V2_GAMMAF1U_U[i],Loop_UPDN_source, 0, 9, true),"ISOSPIN12");
          TIME(corrD1ii10.D1ii_diagrams(*reductions_DD_V3_GAMMAF2D_zero_mom[i], *reductions_DD_V2_GAMMAF1U_U[i],Loop_UPDN_source, 0, 10, true),"ISOSPIN12");

          TIME(corrD1ii13.D1ii_diagrams(*reductions_DD_V3_GAMMAF2U_zero_mom[i], *reductions_DD_V4_GAMMAF1U_D[i],Loop_UPDN_source, 0, 13, true),"ISOSPIN12");          
          TIME(corrD1ii14.D1ii_diagrams(*reductions_DD_V3_GAMMAF2U_zero_mom[i], *reductions_DD_V4_GAMMAF1U_D[i],Loop_UPDN_source, 0, 14, true),"ISOSPIN12");
          TIME(corrD1ii15.D1ii_diagrams(*reductions_DD_V3_GAMMAF2U_zero_mom[i], *reductions_DD_V2_GAMMAF1U_D[i],Loop_UPDN_source, 0, 15, true),"ISOSPIN12");
          TIME(corrD1ii16.D1ii_diagrams(*reductions_DD_V3_GAMMAF2U_zero_mom[i], *reductions_DD_V2_GAMMAF1U_D[i],Loop_UPDN_source, 0, 16, true),"ISOSPIN12");

        } //stochastic samples
        
        free(Loop_UPDN_source);

        asprintf(&ssource,"sx%02dsy%02dsz%02dst%03d", source_full[0], source_full[1], source_full[2], source_full[3]);
        std::string sourcepositiontext= (std::string)"_" + ssource;
        free(ssource);

        outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"_D1ii";

        TIME(produceOutput(corrD1ii1, outfilename, "4pt", n_stochastic_samples,n_coherent_source, coherent_source_table_timeslice),"ISOSPIN12");
        TIME(produceOutput(corrD1ii2, outfilename, "4pt", n_stochastic_samples,n_coherent_source, coherent_source_table_timeslice),"ISOSPIN12");
        TIME(produceOutput(corrD1ii3, outfilename, "4pt", n_stochastic_samples,n_coherent_source, coherent_source_table_timeslice),"ISOSPIN12");
        TIME(produceOutput(corrD1ii4, outfilename, "4pt", n_stochastic_samples,n_coherent_source, coherent_source_table_timeslice),"ISOSPIN12");

        TIME(produceOutput(corrD1ii9, outfilename, "4pt", n_stochastic_samples,n_coherent_source, coherent_source_table_timeslice),"ISOSPIN12");
        TIME(produceOutput(corrD1ii10, outfilename,"4pt", n_stochastic_samples,n_coherent_source, coherent_source_table_timeslice),"ISOSPIN12");

        TIME(produceOutput(corrD1ii13, outfilename,"4pt", n_stochastic_samples,n_coherent_source, coherent_source_table_timeslice),"ISOSPIN12");//Because of V4
        TIME(produceOutput(corrD1ii14, outfilename,"4pt", n_stochastic_samples,n_coherent_source, coherent_source_table_timeslice),"ISOSPIN12");//Because of V4
        TIME(produceOutput(corrD1ii15, outfilename,"4pt", n_stochastic_samples,n_coherent_source, coherent_source_table_timeslice),"ISOSPIN12");
        TIME(produceOutput(corrD1ii16, outfilename,"4pt", n_stochastic_samples,n_coherent_source, coherent_source_table_timeslice),"ISOSPIN12");

      } //diagrams containing loop at the source
#endif
/**********************************************************************************************
*
*
*      In the third section we compute sequential propagators (uu,dd, ud)
*      and all the diagrams needed for I=1/2 I_z=1/2 and I=3/2 I_z=3/2
*      For the I=1/2 case we only consider zero momentum for the pion, 
*      this means the sequential momentum and the momentum in V3 reduction is 
*      set to zero, however we have all the possible momenta for pf1
*      In the I=3/2 case we consider all the momentum for the pion as well
*
*
***********************************************************************************************/

      if (do_stochastic==true)
      //uu case
      {

        //We first have a loop over all unique the source meson momentum p_i2
        for (int i_mpi2=0; i_mpi2<mpi2.size(); ++i_mpi2){

	  auto &momentum_i2 =  mpi2[i_mpi2];
	  //List of momenta corresponding to a fix value of p_i2
          momList filtered_sourcemomentumList = sourcemomentumList.extract(momentum_i2, 0);


          std::string pi2x=std::to_string(momentum_i2[0]);
          std::string pi2y=std::to_string(momentum_i2[1]);
          std::string pi2z=std::to_string(momentum_i2[2]);


          //Ensuring that mu is positive
          if(mu<0) {
            mu*=-1.;
            solver.UpdateSolver();
          }
 
          std::vector<std::vector<int>> mpi2_filt;
          std::vector<std::vector<int>> mptot_filt = sourcemomentumList.uniq_p(3);
          mpi2_filt.assign(mptot_filt.size(),momentum_i2);
          momList list_mpi2ptot(2,{mpi2_filt,mptot_filt},{1,});

//        PLEGMA_ScattCorrelator<float> corrT15(source_full, list_mpi2ptot);
//        PLEGMA_ScattCorrelator<float> corrT17(source_full, list_mpi2ptot);
          PLEGMA_ScattCorrelator<float> corrT21(source_full, list_mpi2ptot);
          PLEGMA_ScattCorrelator<float> corrT22(source_full, list_mpi2ptot);
          PLEGMA_ScattCorrelator<float> corrT23(source_full, list_mpi2ptot);
          PLEGMA_ScattCorrelator<float> corrT24(source_full, list_mpi2ptot);

//        corrT15.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_delta, "12", "Tseq15");
//        corrT17.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_delta, "12", "Tseq17");
          corrT21.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, "12", "Tseq21");
          corrT22.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, "12", "Tseq22");
          corrT23.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, "12", "Tseq23");
          corrT24.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, "12", "Tseq24");



          PLEGMA_ScattCorrelator<float> reductionsV3_12(source_zero, filtered_sourcemomentumList.uniq_p(2));
          PLEGMA_ScattCorrelator<float> reductionsV2_12(source_zero, filtered_sourcemomentumList.uniq_p(1));

          PLEGMA_ScattCorrelator<float> corrB3(source_full, filtered_sourcemomentumList.uniq_p(1));
          PLEGMA_ScattCorrelator<float> corrB4(source_full, filtered_sourcemomentumList.uniq_p(1));
          PLEGMA_ScattCorrelator<float> corrB5(source_full, filtered_sourcemomentumList.uniq_p(1));
          PLEGMA_ScattCorrelator<float> corrB6(source_full, filtered_sourcemomentumList.uniq_p(1));
/*
        PLEGMA_ScattCorrelator<float> corrB17(source_full, filtered_sourcemomentumList.uniq_p(1));
        PLEGMA_ScattCorrelator<float> corrB18(source_full, filtered_sourcemomentumList.uniq_p(1));
        PLEGMA_ScattCorrelator<float> corrB19(source_full, filtered_sourcemomentumList.uniq_p(1));
        PLEGMA_ScattCorrelator<float> corrB20(source_full, filtered_sourcemomentumList.uniq_p(1));
*/
          PLEGMA_ScattCorrelator<float> corrW5(source_full, filtered_sourcemomentumList.uniq_p(1));
          PLEGMA_ScattCorrelator<float> corrW6(source_full, filtered_sourcemomentumList.uniq_p(1));
          PLEGMA_ScattCorrelator<float> corrW7(source_full, filtered_sourcemomentumList.uniq_p(1));
          PLEGMA_ScattCorrelator<float> corrW8(source_full, filtered_sourcemomentumList.uniq_p(1));

          PLEGMA_ScattCorrelator<float> corrW13(source_full, filtered_sourcemomentumList.uniq_p(1));
          PLEGMA_ScattCorrelator<float> corrW14(source_full, filtered_sourcemomentumList.uniq_p(1));
          PLEGMA_ScattCorrelator<float> corrW15(source_full, filtered_sourcemomentumList.uniq_p(1));
          PLEGMA_ScattCorrelator<float> corrW16(source_full, filtered_sourcemomentumList.uniq_p(1));

/*      PLEGMA_ScattCorrelator<float> corrW29(source_full, filtered_sourcemomentumList.uniq_p(1));
        PLEGMA_ScattCorrelator<float> corrW30(source_full, filtered_sourcemomentumList.uniq_p(1));
        PLEGMA_ScattCorrelator<float> corrW31(source_full, filtered_sourcemomentumList.uniq_p(1));
        PLEGMA_ScattCorrelator<float> corrW32(source_full, filtered_sourcemomentumList.uniq_p(1));*/

/*      PLEGMA_ScattCorrelator<float> corrD1ff1389(source_full, filtered_sourcemomentumList.uniq_p(1));
        PLEGMA_ScattCorrelator<float> corrD1ff24710(source_full, filtered_sourcemomentumList.uniq_p(1));


        corrD1ff24710.initialize_diagram(  glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson,"12", "D1ff2-4-7-10");
        corrD1ff1389.initialize_diagram(  glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "D1ff1-3-8-9");*/


          corrB3.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "B3");
          corrB4.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "B4");
          corrB5.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "B5");
          corrB6.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "B6");

/*      corrB17.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "B17");
        corrB18.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "B18");
        corrB19.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "B19");
        corrB20.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "B20");*/

          corrW5.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "W5");
          corrW6.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "W6");
          corrW7.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "W7");
          corrW8.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "W8");


          corrW13.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "W13");
          corrW14.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "W14");
          corrW15.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "W15");
          corrW16.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "W16");

/*      corrW29.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "W29");
        corrW30.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "W30");
        corrW31.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "W31");
        corrW32.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "W32");*/

          for (int i_gamma_i2=0; i_gamma_i2<glist_source_meson.size(); ++i_gamma_i2) {
            GAMMAS_SCATT gamma_i2 = glist_source_meson[i_gamma_i2];

            PLEGMA_Propagator3D<float> propTS3D;
            for(int isc = 0 ; isc < 12 ; isc++){
              PLEGMA_Vector<double> vectorAuxD;
              PLEGMA_Vector<float> vectorAuxF;
              PLEGMA_Vector<double> vectorAuxD2;
              //Performing the smearing
              

              PLEGMA_Vector3D<double> vector1, vector2;
              vectorAuxF.absorb(propUP, isc/3, isc%3);
              vectorAuxD.copy(vectorAuxF);
              vector1.absorb( vectorAuxD, sequential_time_source);
              TIME(vector2.gaussianSmearing(vector1, smearedGauge3D, nsmearGauss, alphaGauss),"ISOSPIN12");

              vector2.mulMomentumPhases(momentum_i2,1);

              vectorAuxD.absorb(vector2,sequential_time_source);
              vectorAuxD2.absorbTimeslice(vectorAuxD,sequential_time_source, false);

              //Perform multiplication with gamma_i2
              vectorAuxD2.apply_gamma_scatt(gamma_i2);
              //Perform rotation to the physical basis
              vectorAuxD.rotateToPhysicalBasis(vectorAuxD2,+1);
        
              //Computing sequential propagators UU T_fii with insertion
              //gamma_i2=gamma_5 and momentum momentum_i2
              PLEGMA_printf("Going to invert UP for sequential propagator UP  for component %d\n", isc);
              //performing the inversion
              TIME(solver.solve(vectorAuxD, vectorAuxD),"ISOSPIN12");
              //performing rotation to physical base
              vectorAuxD2.rotateToPhysicalBasis(vectorAuxD,+1);
              //performing smearing
              TIME(vectorAuxD.gaussianSmearing(vectorAuxD2, smearedGauge, nsmearGauss, alphaGauss),"ISOSPIN12");
              vectorAuxF.copy(vectorAuxD);
	      propTS.absorb(vectorAuxF, isc/3, isc%3);
            }
/*
        if(outfile_SEQ!="")
        {
          PLEGMA_printf("Save propagator for the UPUP quark\n");
          PLEGMA_Vector<float> vectorAuxPrint(BOTH);
          for(int isc = 0 ; isc < 12 ; isc++){
            std::string spin=std::to_string(isc/3);
            std::string col=std::to_string(isc%3);

            vectorAuxPrint.absorb(propTS,isc/3,isc%3);
            vectorAuxPrint.unload();
            vectorAuxPrint.writeLIME(outfile_SEQ+"UPUP"+confnumber+sourcepositiontext+"_s"+spin+"_c"+col);
          }
        }

*/
            for (int i=0; i<n_stochastic_samples; ++i){
              //Computing diagrams containing loops first

              PLEGMA_Vector<float> stochastic_propagator;
              PLEGMA_Vector<float> stochastic_source;
              stochastic_propagator.copy(*stochastic_propags[i],HOST);
              stochastic_source.copy(*stochastic_sources[i],HOST);
              stochastic_propagator.load();
              stochastic_source.load();

            //For computing the B diagrams we compute the V3 factor
            //using the sequential propagator for U(xf1,xf2) type
              TIME(reductionsV3_12.V3( stochastic_source, glist_sink_meson,   propTS, true),"ISOSPIN12");//checked

              TIME(corrB3.B_diagrams(reductionsV3_12, *reductions_UU_V2_GAMMAF1D_U[i], i_gamma_i2, 3, true),"ISOSPIN12");
              TIME(corrB4.B_diagrams(reductionsV3_12, *reductions_UU_V4_GAMMAF1U_D[i], i_gamma_i2, 4, true),"ISOSPIN12");
              TIME(corrB5.B_diagrams(reductionsV3_12, *reductions_UU_V2_GAMMAF1D_U[i], i_gamma_i2, 5, true),"ISOSPIN12");
              TIME(corrB6.B_diagrams(reductionsV3_12, *reductions_UU_V4_GAMMAF1U_D[i], i_gamma_i2, 6, true),"ISOSPIN12");

          
              TIME(reductionsV2_12.V4( stochastic_propagator, glist_sink_nucleon, propDN, propTS, false),"ISOSPIN12");//checked

              TIME(corrW5.W_diagrams( *reductions_UU_V3_GAMMAF2U[i], reductionsV2_12, i_gamma_i2, 5, true),"ISOSPIN12");
              TIME(corrW7.W_diagrams( *reductions_UU_V3_GAMMAF2U[i], reductionsV2_12, i_gamma_i2, 7, true),"ISOSPIN12");

              TIME(reductionsV2_12.V2( stochastic_propagator, glist_sink_nucleon, propDN, propTS, false),"ISOSPIN12");//checked

              TIME(corrW6.W_diagrams( *reductions_UU_V3_GAMMAF2U[i], reductionsV2_12, i_gamma_i2, 6, true),"ISOSPIN12");
              TIME(corrW8.W_diagrams( *reductions_UU_V3_GAMMAF2U[i], reductionsV2_12, i_gamma_i2, 8, true),"ISOSPIN12");


            //for D(xf1,xf2) type
//          stochastic_propagator.apply_gamma5();
//          TIME(reductionsV3_12.V3( stochastic_propagator, glist_sink_meson,   propTS, true),"ISOSPIN12");//checked
/*
            TIME(corrB17.B_diagrams(reductionsV3_12, *reductions_DD_V4_GAMMAF1U_D[i], i_gamma_i2, 17, true),"ISOSPIN12");
            TIME(corrB18.B_diagrams(reductionsV3_12, *reductions_DD_V4_GAMMAF1U_D[i], i_gamma_i2, 18, true),"ISOSPIN12");
            TIME(corrB19.B_diagrams(reductionsV3_12, *reductions_DD_V2_GAMMAF1U_D[i], i_gamma_i2, 19, true),"ISOSPIN12");
            TIME(corrB20.B_diagrams(reductionsV3_12, *reductions_DD_V2_GAMMAF1U_D[i], i_gamma_i2, 20, true),"ISOSPIN12");*/

            //For computing the W diagrams we compute the V2 factor using the sequential
              stochastic_source.apply_gamma5();
              TIME(reductionsV2_12.V2( stochastic_source, glist_sink_nucleon, propTS, propUP, false),"ISOSPIN12");//checked

              TIME(corrW13.W_diagrams( *reductions_DD_V3_GAMMAF2D[i], reductionsV2_12, i_gamma_i2, 13, true),"ISOSPIN12");
              TIME(corrW15.W_diagrams( *reductions_DD_V3_GAMMAF2D[i], reductionsV2_12, i_gamma_i2, 15, true),"ISOSPIN12");

              TIME(reductionsV2_12.V2( stochastic_source, glist_sink_nucleon, propUP, propTS, false),"ISOSPIN12");//checked

              TIME(corrW14.W_diagrams( *reductions_DD_V3_GAMMAF2D[i], reductionsV2_12, i_gamma_i2, 14, true),"ISOSPIN12");
              TIME(corrW16.W_diagrams( *reductions_DD_V3_GAMMAF2D[i], reductionsV2_12, i_gamma_i2, 16, true),"ISOSPIN12");

	    /*
            TIME(reductionsV2_12.V2( stochastic_source, glist_sink_nucleon, propTS, propDN, false),"ISOSPIN12");//checked

            TIME(corrW31.W_diagrams( *reductions_DD_V3_GAMMAF2U[i], reductionsV2_12, i_gamma_i2, 31, true),"ISOSPIN12");
            TIME(corrW32.W_diagrams( *reductions_DD_V3_GAMMAF2U[i], reductionsV2_12, i_gamma_i2, 32, true),"ISOSPIN12");

            TIME(reductionsV2_12.V4( stochastic_source, glist_sink_nucleon, propTS, propDN, false),"ISOSPIN12");//checked

            TIME(corrW29.W_diagrams( *reductions_DD_V3_GAMMAF2U[i], reductionsV2_12, i_gamma_i2, 29, true),"ISOSPIN12");
            TIME(corrW30.W_diagrams( *reductions_DD_V3_GAMMAF2U[i], reductionsV2_12, i_gamma_i2, 30, true),"ISOSPIN12");
	    */

            }//loop over stochastic samples


            //D1ff type diagrams
            PLEGMA_ScattCorrelator<float> reductionsT1(source_zero,  list_mpi2ptot.uniq_p(1));
            PLEGMA_ScattCorrelator<float> reductionsT2(source_zero,  list_mpi2ptot.uniq_p(1));

/*          TIME(reductionsT1.T1(glist_source_nucleon, glist_sink_nucleon, propTS, propDN, propUP),"ISOSPIN12");
          TIME(reductionsT2.T2(glist_source_nucleon, glist_sink_nucleon, propTS, propDN, propUP),"ISOSPIN12");

          TIME(corrD1ff24710.LT_diagrams( reductionsT1, reductionsT2, Loop_UPDN ),"ISOSPIN12");

          TIME(reductionsT1.T1(glist_source_nucleon, glist_sink_nucleon, propUP, propDN, propTS),"ISOSPIN12");
          TIME(reductionsT2.T2(glist_source_nucleon, glist_sink_nucleon, propUP, propDN, propTS),"ISOSPIN12");

          TIME(corrD1ff1389.LT_diagrams( reductionsT1, reductionsT2, Loop_UPDN ),"ISOSPIN12");*/

          //Triangle diagrams
/*          TIME(reductionsT1.T1(glist_source_nucleon, glist_sink_delta, propDN, propTS, propUP),"ISOSPIN12");
          TIME(corrT15.convertTreductiontoDiagram( reductionsT1, i_gamma_i2, false, false, false ),"ISOSPIN12");

          TIME(reductionsT1.T1(glist_source_nucleon, glist_sink_delta, propDN, propUP, propTS),"ISOSPIN12");
          TIME( corrT17.convertTreductiontoDiagram( reductionsT1,i_gamma_i2, false, false, false ),"ISOSPIN12");*/

            TIME(reductionsT1.T1(glist_source_nucleon, glist_sink_nucleon, propTS, propDN, propUP),"ISOSPIN12");
            TIME( corrT21.convertTreductiontoDiagram( reductionsT1,i_gamma_i2, false, true, true ),"ISOSPIN12");

            TIME(reductionsT1.T1(glist_source_nucleon, glist_sink_nucleon, propUP, propDN, propTS),"ISOSPIN12");
            TIME( corrT23.convertTreductiontoDiagram( reductionsT1,i_gamma_i2, false, true, true ),"ISOSPIN12");

            TIME(reductionsT2.T2(glist_source_nucleon, glist_sink_nucleon, propTS, propDN, propUP),"ISOSPIN12");
            TIME( corrT22.convertTreductiontoDiagram( reductionsT2,i_gamma_i2, false, true, true ),"ISOSPIN12");

            TIME(reductionsT2.T2(glist_source_nucleon, glist_sink_nucleon, propUP, propDN, propTS),"ISOSPIN12");
            TIME( corrT24.convertTreductiontoDiagram( reductionsT2,i_gamma_i2, false, true, true ),"ISOSPIN12");

	  } //loop over gamma_i2

          asprintf(&ssource,"sx%02dsy%02dsz%02dst%03d", source_full[0], source_full[1], source_full[2], source_full[3]);
          std::string sourcepositiontext= (std::string)"_" + ssource;
          free(ssource);

          outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"_T";
        
/*          TIME(produceOutput(corrT15, outfilename, "T",n_coherent_source, coherent_source_table_timeslice),"ISOSPIN12");
          TIME(produceOutput(corrT17, outfilename, "T",n_coherent_source, coherent_source_table_timeslice),"ISOSPIN12");*/
          TIME(produceOutput(corrT21, outfilename, "T"),"ISOSPIN12");
          TIME(produceOutput(corrT22, outfilename, "T"),"ISOSPIN12");
          TIME(produceOutput(corrT23, outfilename, "T"),"ISOSPIN12");
          TIME(produceOutput(corrT24, outfilename, "T"),"ISOSPIN12");
        
          outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"_B";

          TIME(produceOutput(corrB3, outfilename, "4pt", n_stochastic_samples),"ISOSPIN12");
          TIME(produceOutput(corrB4, outfilename, "4pt", n_stochastic_samples),"ISOSPIN12");
          TIME(produceOutput(corrB5, outfilename, "4pt", n_stochastic_samples),"ISOSPIN12");
          TIME(produceOutput(corrB6, outfilename, "4pt", n_stochastic_samples),"ISOSPIN12");
 
/*          TIME(produceOutput(corrB17, outfilename, "4pt", n_stochastic_samples),"ISOSPIN12");
          TIME(produceOutput(corrB18, outfilename, "4pt", n_stochastic_samples),"ISOSPIN12");
          TIME(produceOutput(corrB19, outfilename, "4pt", n_stochastic_samples),"ISOSPIN12");
          TIME(produceOutput(corrB20, outfilename, "4pt", n_stochastic_samples),"ISOSPIN12");*/

          outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"_W";

          TIME(produceOutput(corrW5, outfilename, "4pt", n_stochastic_samples),"ISOSPIN12");
          TIME(produceOutput(corrW6, outfilename, "4pt", n_stochastic_samples),"ISOSPIN12");
          TIME(produceOutput(corrW7, outfilename, "4pt", n_stochastic_samples),"ISOSPIN12");
          TIME(produceOutput(corrW8, outfilename, "4pt", n_stochastic_samples),"ISOSPIN12");
          TIME(produceOutput(corrW13, outfilename,"4pt", n_stochastic_samples),"ISOSPIN12");
          TIME(produceOutput(corrW14, outfilename,"4pt", n_stochastic_samples),"ISOSPIN12");
          TIME(produceOutput(corrW15, outfilename,"4pt", n_stochastic_samples),"ISOSPIN12");
          TIME(produceOutput(corrW16, outfilename,"4pt", n_stochastic_samples),"ISOSPIN12");
/*          TIME(produceOutput(corrW29, outfilename,"4pt", n_stochastic_samples),"ISOSPIN12");
          TIME(produceOutput(corrW30, outfilename,"4pt", n_stochastic_samples),"ISOSPIN12");
          TIME(produceOutput(corrW31, outfilename,"4pt", n_stochastic_samples),"ISOSPIN12");
          TIME(produceOutput(corrW32, outfilename,"4pt", n_stochastic_samples),"ISOSPIN12");*/

/*          outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"_D1ff";
          TIME(produceOutput(corrD1ff1389, outfilename,  "4pt"),"ISOSPIN12");
          TIME(produceOutput(corrD1ff24710, outfilename, "4pt"),"ISOSPIN12");*/
   
        }//end of loop for momenta pi2

      }//end of loop for sequential UU

      //start for DD
      if (do_stochastic==true)
      {
        for (int i_mpi2=0; i_mpi2<mpi2.size(); ++i_mpi2){

          auto &momentum_i2 =  mpi2[i_mpi2];
          //List of momenta corresponding to a fix value of p_i2
          momList filtered_sourcemomentumList = sourcemomentumList.extract(momentum_i2, 0);

          // ensuring mu negative
          if(mu>0) {
            mu*=-1.;
            solver.UpdateSolver();
          }

          std::vector<std::vector<int>> mpi2_filt;
          std::vector<std::vector<int>> mptot_filt = sourcemomentumList.uniq_p(3);
          mpi2_filt.assign(mptot_filt.size(),momentum_i2);
          momList list_mpi2ptot(2,{mpi2_filt,mptot_filt},{1,});

          //PLEGMA_ScattCorrelator<float> corrT19(source_full, list_mpi2ptot);
          PLEGMA_ScattCorrelator<float> corrT25(source_full, list_mpi2ptot);
          PLEGMA_ScattCorrelator<float> corrT26(source_full, list_mpi2ptot);

          PLEGMA_ScattCorrelator<float> reductionsV3_12(source_zero, filtered_sourcemomentumList.uniq_p(2));
          PLEGMA_ScattCorrelator<float> reductionsV2_12(source_zero, filtered_sourcemomentumList.uniq_p(1));

          PLEGMA_ScattCorrelator<float> corrB7(source_full, filtered_sourcemomentumList);
          PLEGMA_ScattCorrelator<float> corrB8(source_full, filtered_sourcemomentumList);

          PLEGMA_ScattCorrelator<float> corrW9(source_full, filtered_sourcemomentumList);
          PLEGMA_ScattCorrelator<float> corrW10(source_full,filtered_sourcemomentumList);
          PLEGMA_ScattCorrelator<float> corrW11(source_full, filtered_sourcemomentumList);
          PLEGMA_ScattCorrelator<float> corrW12(source_full, filtered_sourcemomentumList);

/*      PLEGMA_ScattCorrelator<float> corrW33(source_full, filtered_sourcemomentumList.uniq_p(1));
        PLEGMA_ScattCorrelator<float> corrW34(source_full, filtered_sourcemomentumList.uniq_p(1));
        PLEGMA_ScattCorrelator<float> corrW35(source_full, filtered_sourcemomentumList.uniq_p(1));
        PLEGMA_ScattCorrelator<float> corrW36(source_full, filtered_sourcemomentumList.uniq_p(1));*/

//      PLEGMA_ScattCorrelator<float> corrD1ff561112(source_full, filtered_sourcemomentumList.uniq_p(1));


//      corrT19.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_delta_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_delta, "12", "Tseq19");
          corrT25.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, "12", "Tseq25");
          corrT26.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, "12", "Tseq26");



//        corrD1ff561112.initialize_diagram(  glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "D1ff5-6-11-12");


          corrB7.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "B7");
          corrB8.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "B8");

          corrW9.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "W9");
          corrW10.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson,"12", "W10");
          corrW11.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson,"12", "W11");
          corrW12.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson,"12", "W12");

/*        corrW33.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson,"12", "W33");
        corrW34.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson,"12", "W34");
        corrW35.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson,"12", "W35");
        corrW36.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson,"12", "W36");*/

          for (int i_gamma_i2=0; i_gamma_i2<glist_source_meson.size(); ++i_gamma_i2) {
            GAMMAS_SCATT gamma_i2 = glist_source_meson[i_gamma_i2];

            PLEGMA_Propagator3D<float> propTS3D;
            for(int isc = 0 ; isc < 12 ; isc++){
              PLEGMA_Vector<double> vectorAuxD;
              PLEGMA_Vector<float> vectorAuxF;
              PLEGMA_Vector<double> vectorAuxD2;
              //Performing the smearing
              // Smearing the source

              PLEGMA_Vector3D<double> vector1, vector2;
              vectorAuxF.absorb(propDN, isc/3, isc%3);
              vectorAuxD.copy(vectorAuxF);
              vector1.absorb( vectorAuxD, sequential_time_source);
              TIME(vector2.gaussianSmearing(vector1, smearedGauge3D, nsmearGauss, alphaGauss),"ISOSPIN12");

              vector2.mulMomentumPhases(momentum_i2,1);

              vectorAuxD.absorb(vector2,sequential_time_source);
              vectorAuxD2.absorbTimeslice(vectorAuxD,sequential_time_source, false);

              //Perform multiplication with gamma_i2
              vectorAuxD2.apply_gamma_scatt(gamma_i2);
              //Perform rotation to the physical basis
              vectorAuxD.rotateToPhysicalBasis(vectorAuxD2,-1);
        
              //Computing sequential propagators DD T_fii with insertion
              //gamma_i2=gamma_5 and momentum (0,0,0)
              PLEGMA_printf("Going to invert DN for sequential propagator DN  for component %d\n", isc);
              //performing the inversion
              TIME(solver.solve(vectorAuxD, vectorAuxD),"ISOSPIN12");
              //performing rotation to physical base
              vectorAuxD2.rotateToPhysicalBasis(vectorAuxD,-1);
              //performing smearing
              TIME(vectorAuxD.gaussianSmearing(vectorAuxD2, smearedGauge, nsmearGauss, alphaGauss),"ISOSPIN12");
              vectorAuxF.copy(vectorAuxD);
              propTS.absorb(vectorAuxF, isc/3, isc%3);
	    }
	
/*
        if(outfile_SEQ!="")
        {
          PLEGMA_printf("Save propagator for the DNDN quark\n");
          PLEGMA_Vector<float> vectorAuxPrint(BOTH);
          for(int isc = 0 ; isc < 12 ; isc++){
            std::string spin=std::to_string(isc/3);
            std::string col=std::to_string(isc%3);

            vectorAuxPrint.absorb(propTS,isc/3,isc%3);
            vectorAuxPrint.unload();
            vectorAuxPrint.writeLIME(outfile_SEQ+"DNDN"+confnumber+sourcepositiontext+"_s"+spin+"_c"+col);
          }
        }
*/

            PLEGMA_ScattCorrelator<float> reductionsT1(source_zero,  list_mpi2ptot.uniq_p(1));
            PLEGMA_ScattCorrelator<float> reductionsT2(source_zero,  list_mpi2ptot.uniq_p(1));

/*          TIME(reductionsT1.T1(glist_source_nucleon, glist_sink_nucleon, propUP, propTS, propUP),"ISOSPIN12");
          TIME(reductionsT2.T2(glist_source_nucleon, glist_sink_nucleon, propUP, propTS, propUP),"ISOSPIN12");

          TIME(corrD1ff561112.LT_diagrams( reductionsT1, reductionsT2, Loop_UPDN ),"ISOSPIN12");

          //Triangle diagrams
          TIME(reductionsT1.T1(glist_source_nucleon, glist_sink_delta, propTS, propUP, propUP),"ISOSPIN12");
          TIME( corrT19.convertTreductiontoDiagram( reductionsT1,i_gamma_i2, false, false, false ),"ISOSPIN12");
*/
            TIME(reductionsT1.T1(glist_source_nucleon, glist_sink_nucleon, propUP, propTS, propUP),"ISOSPIN12");
            TIME( corrT25.convertTreductiontoDiagram( reductionsT1,i_gamma_i2 ),"ISOSPIN12");

            TIME(reductionsT2.T2(glist_source_nucleon, glist_sink_nucleon, propUP, propTS, propUP),"ISOSPIN12");
            TIME( corrT26.convertTreductiontoDiagram( reductionsT2, i_gamma_i2 ),"ISOSPIN12");



            for (int i=0; i<n_stochastic_samples; ++i){
              PLEGMA_Vector<float> stochastic_propagator;
              PLEGMA_Vector<float> stochastic_source;
              stochastic_propagator.copy(*stochastic_propags[i],HOST);
              stochastic_source.copy(*stochastic_sources[i],HOST);
              stochastic_propagator.load();
              stochastic_source.load();

              TIME(reductionsV2_12.V2( stochastic_propagator, glist_sink_nucleon, propTS, propUP, false),"ISOSPIN12");

              TIME( corrW9.W_diagrams( *reductions_UU_V3_GAMMAF2U[i], reductionsV2_12, i_gamma_i2,  9, true),"ISOSPIN12");
              TIME(corrW11.W_diagrams( *reductions_UU_V3_GAMMAF2U[i], reductionsV2_12, i_gamma_i2, 11, true),"ISOSPIN12");

              TIME(reductionsV2_12.V4( stochastic_propagator, glist_sink_nucleon, propTS, propUP, false),"ISOSPIN12");

              TIME(corrW10.W_diagrams( *reductions_UU_V3_GAMMAF2U[i], reductionsV2_12, i_gamma_i2, 10, true),"ISOSPIN12");
              TIME(corrW12.W_diagrams( *reductions_UU_V3_GAMMAF2U[i], reductionsV2_12, i_gamma_i2, 12, true),"ISOSPIN12");


            //For computing the B diagrams we compute the V3 factor
            //using the sequential propagator
              stochastic_propagator.apply_gamma5();
              TIME(reductionsV3_12.V3( stochastic_propagator, glist_sink_meson,   propTS, true),"ISOSPIN12");

              TIME(corrB7.B_diagrams(reductionsV3_12, *reductions_DD_V2_GAMMAF1U_U[i], i_gamma_i2, 7, true),"ISOSPIN12");
              TIME(corrB8.B_diagrams(reductionsV3_12, *reductions_DD_V2_GAMMAF1U_U[i], i_gamma_i2, 8, true),"ISOSPIN12");

            //For computing the W diagrams we compute the V2 factor using the sequential
	    /*
            stochastic_source.apply_gamma5();
            TIME(reductionsV2_12.V4( stochastic_source, glist_sink_nucleon, propUP, propTS, false),"ISOSPIN12");
          
            TIME(corrW33.W_diagrams( *reductions_DD_V3_GAMMAF2U_zero_mom[i], reductionsV2_12, i_gamma_i2, 33, true),"ISOSPIN12");
            TIME(corrW34.W_diagrams( *reductions_DD_V3_GAMMAF2U_zero_mom[i], reductionsV2_12, i_gamma_i2, 34, true),"ISOSPIN12");

            TIME(reductionsV2_12.V2( stochastic_source, glist_sink_nucleon, propUP, propTS, false),"ISOSPIN12");
 
            TIME(corrW35.W_diagrams( *reductions_DD_V3_GAMMAF2U_zero_mom[i], reductionsV2_12, i_gamma_i2, 35, true),"ISOSPIN12");
            TIME(corrW36.W_diagrams( *reductions_DD_V3_GAMMAF2U_zero_mom[i], reductionsV2_12, i_gamma_i2, 36, true),"ISOSPIN12");
           */
            }//loop over stochastic samples

          } //end of loop i_gamma_i2
          asprintf(&ssource,"sx%02dsy%02dsz%02dst%03d", source_full[0], source_full[1], source_full[2], source_full[3]);
          std::string sourcepositiontext= (std::string)"_" + ssource;
          free(ssource);

          outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"_T";

//        TIME(produceOutput(corrT19, outfilename, "T",n_coherent_source, coherent_source_table_timeslice),"ISOSPIN12");
          TIME(produceOutput(corrT25, outfilename, "T"),"ISOSPIN12");
          TIME(produceOutput(corrT26, outfilename, "T"),"ISOSPIN12");

          outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"_B";
          TIME(produceOutput(corrB7, outfilename, "4pt", n_stochastic_samples),"ISOSPIN12");
          TIME(produceOutput(corrB8, outfilename, "4pt", n_stochastic_samples),"ISOSPIN12");

          outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"_W";
          TIME(produceOutput(corrW9, outfilename,"4pt", n_stochastic_samples),"ISOSPIN12");//because of V4
          TIME(produceOutput(corrW10, outfilename,"4pt", n_stochastic_samples),"ISOSPIN12");//because of V4
          TIME(produceOutput(corrW11, outfilename,"4pt", n_stochastic_samples),"ISOSPIN12");
          TIME(produceOutput(corrW12, outfilename,"4pt", n_stochastic_samples),"ISOSPIN12");
/*
          TIME(produceOutput(corrW33, outfilename,"4pt", n_stochastic_samples,n_coherent_source, coherent_source_table_timeslice),"ISOSPIN12");//because of V4
          TIME(produceOutput(corrW34, outfilename,"4pt", n_stochastic_samples,n_coherent_source, coherent_source_table_timeslice),"ISOSPIN12");//because of V4
          TIME(produceOutput(corrW35, outfilename,"4pt", n_stochastic_samples,n_coherent_source, coherent_source_table_timeslice),"ISOSPIN12");
          TIME(produceOutput(corrW36, outfilename,"4pt", n_stochastic_samples,n_coherent_source, coherent_source_table_timeslice),"ISOSPIN12");

          outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"_D1ff";
          TIME(produceOutput(corrD1ff561112, outfilename, "4pt",n_coherent_source, coherent_source_table_timeslice),"ISOSPIN12");*/

        }//end of loop over momenta pi2
      }//end for DD do_stochastic==true


      //Ensuring mu is positive
      if(mu<0) {
        mu*=-1.;
        solver.UpdateSolver();
      }

      ///We first have a loop over all unique the source meson momentum p_i2 

      asprintf(&ssource,"sx%02dsy%02dsz%02dst%03d", source_full[0], source_full[1], source_full[2], source_full[3] );
      sourcepositiontext= (std::string)"_" + ssource;
      free(ssource);

      //We first have a loop over all unique the source meson momentum p_i2 
      for (int i_mpi2=0; i_mpi2<mpi2.size(); ++i_mpi2){

	auto &momentum_i2 =  mpi2[i_mpi2];
	//List of momenta corresponding to a fix value of p_i2
        momList filtered_sourcemomentumList = sourcemomentumList.extract(momentum_i2, 0);

        std::string pi2x=std::to_string(momentum_i2[0]);
        std::string pi2y=std::to_string(momentum_i2[1]);
        std::string pi2z=std::to_string(momentum_i2[2]);
 
	//case sequential UD for spin1/2
	std::vector<std::vector<int>> mptot_filt = filtered_sourcemomentumList.uniq_p(3);
	std::vector<std::vector<int>> mpi2_filt;
	mpi2_filt.assign(mptot_filt.size(),momentum_i2);
	momList list_mpi2ptot(2,{mpi2_filt,mptot_filt},{1,});

	//initialize diagrams
	
        PLEGMA_ScattCorrelator<float> reductionsV2(source_zero, filtered_sourcemomentumList.uniq_p(1));
        PLEGMA_ScattCorrelator<float> reductionsV3(source_zero, filtered_sourcemomentumList.uniq_p(2));

	//T diagrams
        PLEGMA_ScattCorrelator<float> corrT(source_full, list_mpi2ptot);
    
        //PLEGMA_ScattCorrelator<float> corrT7(source_full, list_mpi2ptot);
        //PLEGMA_ScattCorrelator<float> corrT9(source_full, list_mpi2ptot);
        PLEGMA_ScattCorrelator<float> corrT11(source_full,list_mpi2ptot);
        PLEGMA_ScattCorrelator<float> corrT12(source_full,list_mpi2ptot);
        PLEGMA_ScattCorrelator<float> corrT13(source_full,list_mpi2ptot);
        PLEGMA_ScattCorrelator<float> corrT14(source_full,list_mpi2ptot);

	// 4pt diagrams
	PLEGMA_ScattCorrelator<float> corrB1(source_full, filtered_sourcemomentumList);
	PLEGMA_ScattCorrelator<float> corrB2(source_full, filtered_sourcemomentumList);
 /*         PLEGMA_ScattCorrelator<float> corrB9(source_full,  filtered_sourcemomentumList.uniq_p(1));
          PLEGMA_ScattCorrelator<float> corrB10(source_full, filtered_sourcemomentumList.uniq_p(1));
          PLEGMA_ScattCorrelator<float> corrB11(source_full, filtered_sourcemomentumList.uniq_p(1));
          PLEGMA_ScattCorrelator<float> corrB12(source_full, filtered_sourcemomentumList.uniq_p(1));*/

        PLEGMA_ScattCorrelator<float> corrB13(source_full, filtered_sourcemomentumList);
        PLEGMA_ScattCorrelator<float> corrB14(source_full, filtered_sourcemomentumList);
        PLEGMA_ScattCorrelator<float> corrB15(source_full, filtered_sourcemomentumList);
        PLEGMA_ScattCorrelator<float> corrB16(source_full, filtered_sourcemomentumList);


	PLEGMA_ScattCorrelator<float> corrW1(source_full, filtered_sourcemomentumList);
	PLEGMA_ScattCorrelator<float> corrW2(source_full, filtered_sourcemomentumList);
	PLEGMA_ScattCorrelator<float> corrW3(source_full, filtered_sourcemomentumList);
        PLEGMA_ScattCorrelator<float> corrW4(source_full, filtered_sourcemomentumList);
          
/*
          PLEGMA_ScattCorrelator<float> corrW17(source_full, filtered_sourcemomentumList.uniq_p(1));
          PLEGMA_ScattCorrelator<float> corrW18(source_full, filtered_sourcemomentumList.uniq_p(1));
          PLEGMA_ScattCorrelator<float> corrW19(source_full, filtered_sourcemomentumList.uniq_p(1));
	  PLEGMA_ScattCorrelator<float> corrW20(source_full, filtered_sourcemomentumList.uniq_p(1));*/

/*          PLEGMA_ScattCorrelator<float> corrW21(source_full, filtered_sourcemomentumList.uniq_p(1));
            PLEGMA_ScattCorrelator<float> corrW22(source_full, filtered_sourcemomentumList.uniq_p(1));
            PLEGMA_ScattCorrelator<float> corrW23(source_full, filtered_sourcemomentumList.uniq_p(1));
            PLEGMA_ScattCorrelator<float> corrW24(source_full, filtered_sourcemomentumList.uniq_p(1));*/

        PLEGMA_ScattCorrelator<float> corrW25(source_full, filtered_sourcemomentumList);
        PLEGMA_ScattCorrelator<float> corrW26(source_full, filtered_sourcemomentumList);
        PLEGMA_ScattCorrelator<float> corrW27(source_full, filtered_sourcemomentumList);
        PLEGMA_ScattCorrelator<float> corrW28(source_full, filtered_sourcemomentumList);

//            PLEGMA_ScattCorrelator<float> corrD1ff13141718(source_full, filtered_sourcemomentumList.uniq_p(1));
//            PLEGMA_ScattCorrelator<float> corrD1ff15161920(source_full, filtered_sourcemomentumList.uniq_p(1));
//

        PLEGMA_ScattCorrelator<float> corrZ1(source_full, filtered_sourcemomentumList);
        PLEGMA_ScattCorrelator<float> corrZ2(source_full, filtered_sourcemomentumList);
        PLEGMA_ScattCorrelator<float> corrZ3(source_full, filtered_sourcemomentumList);
        PLEGMA_ScattCorrelator<float> corrZ4(source_full, filtered_sourcemomentumList);

        PLEGMA_ScattCorrelator<float> corrZ5(source_full, filtered_sourcemomentumList);
        PLEGMA_ScattCorrelator<float> corrZ6(source_full, filtered_sourcemomentumList);
        PLEGMA_ScattCorrelator<float> corrZ7(source_full, filtered_sourcemomentumList);
        PLEGMA_ScattCorrelator<float> corrZ8(source_full, filtered_sourcemomentumList);

        PLEGMA_ScattCorrelator<float> corrZ9(source_full,  filtered_sourcemomentumList);
        PLEGMA_ScattCorrelator<float> corrZ10(source_full, filtered_sourcemomentumList);

/*       PLEGMA_ScattCorrelator<float> corrZ11(source_full, filtered_sourcemomentumList.uniq_p(1));
       PLEGMA_ScattCorrelator<float> corrZ12(source_full, filtered_sourcemomentumList.uniq_p(1));
       PLEGMA_ScattCorrelator<float> corrZ13(source_full, filtered_sourcemomentumList.uniq_p(1));
       PLEGMA_ScattCorrelator<float> corrZ14(source_full, filtered_sourcemomentumList.uniq_p(1));

       PLEGMA_ScattCorrelator<float> corrZ15(source_full, filtered_sourcemomentumList.uniq_p(1));
       PLEGMA_ScattCorrelator<float> corrZ16(source_full, filtered_sourcemomentumList.uniq_p(1));

       PLEGMA_ScattCorrelator<float> corrZ17(source_full, filtered_sourcemomentumList.uniq_p(1));
       PLEGMA_ScattCorrelator<float> corrZ18(source_full, filtered_sourcemomentumList.uniq_p(1));
       PLEGMA_ScattCorrelator<float> corrZ19(source_full, filtered_sourcemomentumList.uniq_p(1));
       PLEGMA_ScattCorrelator<float> corrZ20(source_full, filtered_sourcemomentumList.uniq_p(1));*/


	PLEGMA_ScattCorrelator<float> corrM(source_full, filtered_sourcemomentumList);
        PLEGMA_ScattCorrelator<float> corrD1if12(source_full, filtered_sourcemomentumList);
        PLEGMA_ScattCorrelator<float> corrD1if34(source_full, filtered_sourcemomentumList);
        PLEGMA_ScattCorrelator<float> corrD1if56(source_full, filtered_sourcemomentumList);


        PLEGMA_ScattCorrelator<float> corrP0UP(source_full, filtered_sourcemomentumList);
        PLEGMA_ScattCorrelator<float> corrP0DN(source_full, filtered_sourcemomentumList);
        PLEGMA_ScattCorrelator<float> corrPPDN(source_full, filtered_sourcemomentumList);
        PLEGMA_ScattCorrelator<float> corrPPUP(source_full, filtered_sourcemomentumList);

	corrT.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_delta_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_delta, "32", "Tseq");	

//            corrT7.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_delta_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_delta, "12", "Tseq7");
//            corrT9.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_delta_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_delta, "12", "Tseq9");
        corrT11.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon,"12", "Tseq11");
        corrT12.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon,"12", "Tseq12");
        corrT13.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon,"12", "Tseq13");
        corrT14.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon,"12", "Tseq14");


	corrB1.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "32", "B1");
	corrB2.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "32", "B2");

/*            corrB9.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "B9");
            corrB10.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson,"12", "B10");
            corrB11.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson,"12", "B11");
            corrB12.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson,"12", "B12");
*/
        corrB13.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson,"12", "B13");
        corrB14.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson,"12", "B14");
        corrB15.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson,"12", "B15");
        corrB16.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson,"12", "B16");

        corrW1.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "32", "W1");
	corrW2.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "32", "W2");
	corrW3.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "32", "W3");
	corrW4.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "32", "W4"); 

  

  /*          corrW17.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson,"12", "W17");
            corrW18.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson,"12", "W18");
            corrW19.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson,"12", "W19");
            corrW20.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson,"12", "W20");

            corrW21.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "W21");
            corrW22.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "W22");
            corrW23.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "W23");
            corrW24.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "W24");*/

        corrW25.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "W25");
        corrW26.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "W26");
        corrW27.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "W27");
        corrW28.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "W28");

//            corrD1ff13141718.initialize_diagram(  glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson,"12", "D1ff13-14-17-18");
//            corrD1ff15161920.initialize_diagram(  glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson,"12", "D1ff15-16-19-20");



	corrZ1.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "32", "Z1");
	corrZ2.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "32", "Z2");
	corrZ3.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "32", "Z3");
	corrZ4.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "32", "Z4");

        corrZ5.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "Z5");
        corrZ6.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "Z6");
        corrZ7.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "Z7");
        corrZ8.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "Z8");
        corrZ9.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "Z9");
        corrZ10.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson,"12", "Z10");


/*       corrZ11.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson,"12", "Z11");
       corrZ12.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson,"12", "Z12");
       corrZ13.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson,"12", "Z13");
       corrZ14.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson,"12", "Z14");
       corrZ15.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson,"12", "Z15");
       corrZ16.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson,"12", "Z16");
       corrZ17.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson,"12", "Z17");
       corrZ18.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson,"12", "Z18");
       corrZ19.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson,"12", "Z19");
       corrZ20.initialize_diagram( glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson,"12", "Z20");       */

	corrM.initialize_diagram(  glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "32", "MNPPP");
        corrD1if34.initialize_diagram(  glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson,"12", "MNPP01");
        corrD1if12.initialize_diagram(  glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson,"12", "MNPP02");
        corrD1if56.initialize_diagram(  glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson,"12", "MN0PP");



        corrP0UP.initialize_diagram(glist_source_meson, glist_sink_meson, "P0UP");
        corrP0DN.initialize_diagram(glist_source_meson, glist_sink_meson, "P0DN");
        corrPPUP.initialize_diagram(glist_source_meson, glist_sink_meson, "PPUP");
        corrPPDN.initialize_diagram(glist_source_meson, glist_sink_meson, "PPDN");


        //Loop over the different gamma structure for the source meson
        for (int i_gamma_i2=0; i_gamma_i2<glist_source_meson.size(); ++i_gamma_i2) {
	  GAMMAS_SCATT gamma_i2 = glist_source_meson[i_gamma_i2];
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

            //Performing the smearing
            // Smearing the source
            {
              PLEGMA_Vector3D<double> vector1, vector2;
              vectorAuxF.absorb(propDN, isc/3, isc%3);
              vectorAuxD.copy(vectorAuxF);
              vector1.absorb( vectorAuxD, sequential_time_source );
              TIME(vector2.gaussianSmearing(vector1, smearedGauge3D, nsmearGauss, alphaGauss),"ISOSPIN32");
              vectorAuxD2.absorb(vector2,sourcePositions[isource][DIM_T]);
            }

            //Perform multiplication with gamma_i2
            vectorAuxD2.apply_gamma_scatt(gamma_i2);

            //Perform rotation to the physical basis
            TIME(vectorAuxD.rotateToPhysicalBasis(vectorAuxD2,+1),"ISOSPIN32");

            
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
            TIME(solver.solve(vectorInOut, vectorInOut),"ISOSPIN32");

            //performing rotation to physical base
            TIME(vectorAuxD.rotateToPhysicalBasis(vectorInOut,+1),"ISOSPIN32");

            //performing smearing
            TIME(vectorInOut.gaussianSmearing(vectorAuxD, smearedGauge, nsmearGauss, alphaGauss),"ISOSPIN32");

            vectorAuxF.copy(vectorInOut);
            propTS.absorb(vectorAuxF, isc/3, isc%3);
          } 

          PLEGMA_ScattCorrelator<float> reductionsT1(source_zero,  mptot_filt);
          PLEGMA_ScattCorrelator<float> reductionsT2(source_zero,  mptot_filt);


          PLEGMA_ScattCorrelator<float> reductionsT1triangle(source_zero,  mptot_filt);
         
	  PLEGMA_ScattCorrelator<float> reductionsT3triangle(source_zero,  mptot_filt);
	  
	  PLEGMA_ScattCorrelator<float> reductionsT5triangle(source_zero,  mptot_filt);

          TIME(reductionsT1triangle.T1(glist_source_nucleon, glist_sink_delta, propTS, propUP, propUP),"ISOSPIN32");
          //reductionsT1triangle.writeHDF5("T1sourceforT_pi2"+pi2x+"_"+pi2y+"_"+pi2z);
          TIME(reductionsT3triangle.T1(glist_source_nucleon, glist_sink_delta, propUP, propTS, propUP),"ISOSPIN32");
          //reductionsT3triangle.writeHDF5("T3sourceforT_pi2"+pi2x+"_"+pi2y+"_"+pi2z);
          TIME(reductionsT5triangle.T2(glist_source_nucleon, glist_sink_delta, propUP, propUP, propTS),"ISOSPIN32");
          //reductionsT5triangle.writeHDF5("T5sourceforT_pi2"+pi2x+"_"+pi2y+"_"+pi2z);
	  
	  //Compute Diagram T 
          TIME(corrT.T_diagrams(reductionsT1triangle, reductionsT3triangle, reductionsT5triangle, i_gamma_i2),"ISOSPIN32");

/*            TIME(reductionsT1.T1(glist_source_nucleon, glist_sink_nucleon, propUP, propTS, propDN),"ISOSPIN12");
            TIME(reductionsT2.T1(glist_source_nucleon, glist_sink_nucleon, propTS, propUP, propDN),"ISOSPIN12");

            TIME(corrD1ff13141718.LT_diagrams( reductionsT1, reductionsT2, Loop_UPDN ),"ISOSPIN12");

            TIME(reductionsT1.T1(glist_source_nucleon, glist_sink_nucleon, propUP, propDN, propTS),"ISOSPIN12");
            TIME(reductionsT2.T2(glist_source_nucleon, glist_sink_nucleon, propTS, propDN, propUP),"ISOSPIN12");

            TIME(corrD1ff15161920.LT_diagrams( reductionsT1, reductionsT2, Loop_UPDN ),"ISOSPIN12");

            TIME(reductionsT1.T1(glist_source_nucleon,glist_sink_delta, propDN, propUP, propTS),"ISOSPIN12");
            TIME( corrT7.convertTreductiontoDiagram( reductionsT1, false, true, false ),"ISOSPIN12");

            TIME(reductionsT2.T2(glist_source_nucleon,glist_sink_delta, propDN, propUP, propTS),"ISOSPIN12");
            TIME( corrT9.convertTreductiontoDiagram( reductionsT2, false, true, false ),"ISOSPIN12");*/

          TIME(reductionsT1.T1(glist_source_nucleon,glist_sink_nucleon, propTS, propUP, propDN),"ISOSPIN12");
          TIME( corrT11.convertTreductiontoDiagram( reductionsT1, false, true, false ),"ISOSPIN12");
            
          TIME(reductionsT2.T2(glist_source_nucleon,glist_sink_nucleon, propTS, propDN, propUP),"ISOSPIN12");
          TIME( corrT12.convertTreductiontoDiagram( reductionsT2, false, false, true ),"ISOSPIN12");

          TIME(reductionsT1.T1(glist_source_nucleon,glist_sink_nucleon, propUP, propTS, propDN),"ISOSPIN12");
          TIME( corrT13.convertTreductiontoDiagram( reductionsT1, false, false, false ),"ISOSPIN12");

          TIME(reductionsT1.T1(glist_source_nucleon,glist_sink_nucleon, propUP, propDN, propTS),"ISOSPIN12");
          TIME( corrT14.convertTreductiontoDiagram( reductionsT1, false, false, true),"ISOSPIN12");


          for (int i=0; i<n_stochastic_samples; ++i){
            PLEGMA_Vector<float> stochastic_propagator;
            PLEGMA_Vector<float> stochastic_source;
            stochastic_propagator.copy(*stochastic_propags[i],HOST);
            stochastic_source.copy(*stochastic_sources[i],HOST);
            stochastic_propagator.load();
            stochastic_source.load();
              /*
              TIME(reductionsV3.V3( stochastic_source, glist_sink_meson,   propTS, true ),"ISOSPIN12");

              TIME( corrB9.B_diagrams(reductionsV3, *reductions_UU_V2_GAMMAF1D_U[i], i_gamma_i2,  9, true),"ISOSPIN12");
              TIME(corrB10.B_diagrams(reductionsV3, *reductions_UU_V4_GAMMAF1U_D[i], i_gamma_i2, 10, true),"ISOSPIN12");
              TIME(corrB11.B_diagrams(reductionsV3, *reductions_UU_V2_GAMMAF1D_U[i], i_gamma_i2, 11, true),"ISOSPIN12");
              TIME(corrB12.B_diagrams(reductionsV3, *reductions_UU_V4_GAMMAF1U_D[i], i_gamma_i2, 12, true),"ISOSPIN12");

              TIME(reductionsV2.V4( stochastic_propagator, glist_sink_nucleon, propDN, propTS, false),"ISOSPIN12");

              TIME(corrW17.W_diagrams( *reductions_UU_V3_GAMMAF2U_zero_mom[i], reductionsV2, i_gamma_i2, 17, true),"ISOSPIN12");
              TIME(corrW19.W_diagrams( *reductions_UU_V3_GAMMAF2U_zero_mom[i], reductionsV2, i_gamma_i2, 19, true),"ISOSPIN12");

              TIME(reductionsV2.V2( stochastic_propagator, glist_sink_nucleon, propDN, propTS, false),"ISOSPIN12");

              TIME(corrW18.W_diagrams( *reductions_UU_V3_GAMMAF2U_zero_mom[i], reductionsV2, i_gamma_i2, 18, true),"ISOSPIN12");
              TIME(corrW20.W_diagrams( *reductions_UU_V3_GAMMAF2U_zero_mom[i], reductionsV2, i_gamma_i2, 20, true),"ISOSPIN12");*/

            stochastic_propagator.apply_gamma5();
            TIME(reductionsV3.V3( stochastic_propagator, glist_sink_meson,   propTS, true),"ISOSPIN12");


            TIME(corrB1.B_diagrams(reductionsV3, *reductions_DD_V2_GAMMAF1U_U[i], i_gamma_i2, 1, true),"ISOSPIN32");
            TIME(corrB2.B_diagrams(reductionsV3, *reductions_DD_V2_GAMMAF1U_U[i], i_gamma_i2, 2, true),"ISOSPIN32");

            TIME(corrB13.B_diagrams(reductionsV3, *reductions_DD_V4_GAMMAF1U_D[i], i_gamma_i2,13, true),"ISOSPIN12");
            TIME(corrB14.B_diagrams(reductionsV3, *reductions_DD_V4_GAMMAF1U_D[i], i_gamma_i2,14, true),"ISOSPIN12");
            TIME(corrB15.B_diagrams(reductionsV3, *reductions_DD_V2_GAMMAF1U_D[i], i_gamma_i2,15, true),"ISOSPIN12");
            TIME(corrB16.B_diagrams(reductionsV3, *reductions_DD_V2_GAMMAF1U_D[i], i_gamma_i2,16, true),"ISOSPIN12");

            stochastic_source.apply_gamma5();

            TIME(reductionsV2.V2( stochastic_source, glist_sink_nucleon, propUP, propTS, false),"ISOSPIN32");

            TIME(corrW1.W_diagrams( *reductions_DD_V3_GAMMAF2U[i], reductionsV2, i_gamma_i2, 1, true),"ISOSPIN32");
            TIME(corrW2.W_diagrams( *reductions_DD_V3_GAMMAF2U[i], reductionsV2, i_gamma_i2, 2, true),"ISOSPIN32");
/*
              TIME(corrW22.W_diagrams( *reductions_DD_V3_GAMMAF2D_zero_mom[i], reductionsV2, i_gamma_i2, 22, true),"ISOSPIN12");
              TIME(corrW24.W_diagrams( *reductions_DD_V3_GAMMAF2D_zero_mom[i], reductionsV2, i_gamma_i2, 24, true),"ISOSPIN12");*/

            TIME(reductionsV2.V2( stochastic_source, glist_sink_nucleon, propTS, propUP, false),"ISOSPIN32");

            TIME(corrW3.W_diagrams( *reductions_DD_V3_GAMMAF2U[i], reductionsV2, i_gamma_i2, 3, true),"ISOSPIN32");
            TIME(corrW4.W_diagrams( *reductions_DD_V3_GAMMAF2U[i], reductionsV2, i_gamma_i2, 4, true),"ISOSPIN32");
/*
              TIME(corrW21.W_diagrams( *reductions_DD_V3_GAMMAF2D_zero_mom[i], reductionsV2, i_gamma_i2, 21, true),"ISOSPIN12");
              TIME(corrW23.W_diagrams( *reductions_DD_V3_GAMMAF2D_zero_mom[i], reductionsV2, i_gamma_i2, 23, true),"ISOSPIN12");*/

            TIME(reductionsV2.V4( stochastic_source, glist_sink_nucleon, propTS, propDN, false),"ISOSPIN12");

            TIME(corrW25.W_diagrams( *reductions_DD_V3_GAMMAF2U[i], reductionsV2, i_gamma_i2, 25, true),"ISOSPIN12");
            TIME(corrW26.W_diagrams( *reductions_DD_V3_GAMMAF2U[i], reductionsV2, i_gamma_i2, 26, true),"ISOSPIN12");

            TIME(reductionsV2.V2( stochastic_source, glist_sink_nucleon, propTS, propDN, false),"ISOSPIN12");

            TIME(corrW27.W_diagrams( *reductions_DD_V3_GAMMAF2U[i], reductionsV2, i_gamma_i2, 27, true),"ISOSPIN12");
            TIME(corrW28.W_diagrams( *reductions_DD_V3_GAMMAF2U[i], reductionsV2, i_gamma_i2, 28, true),"ISOSPIN12");

          } //loop over stochastic samples

	} //loop over gamma_i2

 
       
        //Producing spin diluted stochastic propagators for diagram Z1,Z2,Z3,Z4
             
        //We need V3 reductions for all the possible pf2 for diagrams Z1,Z2,Z3,Z4 

        //We need V2 reductions for all the possible nucleon momenta pf1
        if (do_stochastic_oet==1){
          if ((momentum_i2[0] != 0) || (momentum_i2[1] != 0) || (momentum_i2[2] != 0)){
            PLEGMA_Vector<double> vectortmp1;
            PLEGMA_Vector<double> vectortmp2;
            PLEGMA_Vector<double> vectorSource_finite_mom;

            //Dilution     
            vectortmp1.dilutespin(vectorStoc_source_oet,0);


            //Multiplying by the appropriate momentum phase
            vectorSource_finite_mom.copy(vectortmp1);
            std::vector<int> tmp_4Dmom= momentum_i2 ; 
            tmp_4Dmom.push_back(0);
            vectorSource_finite_mom.mulMomentumPhases(tmp_4Dmom,-1);

	    {  // Smearing the source

              PLEGMA_Vector3D<double> vector1, vector2;
              vector1.absorb(vectorSource_finite_mom, sequential_time_source);
              TIME(vector2.gaussianSmearing(vector1, smearedGauge3D, nsmearGauss, alphaGauss),"ISOSPIN32");
              vectorSource_finite_mom.absorb(vector2, sequential_time_source);

            }

	    //Doing for +mu for the UP propagator spin dilution oet
            if(mu<0) {
              mu*=-1.;
              solver.UpdateSolver();
            }
          
            for (int spinindex=0; spinindex<4; ++spinindex){

              //Transforming to physical base for the UP quark
              vectortmp2.rotateToPhysicalBasis(vectorSource_finite_mom,+1);
       
              vectortmp1.copy(vectorSource_finite_mom);
              //Doing the inversion
              TIME(solver.solve(vectortmp1, vectortmp1),"ISOSPIN32");

              //Rotate back immediately to the physical basis
              vectortmp2.rotateToPhysicalBasis(vectortmp1,+1);

              //performing smearing
              TIME(vectortmp1.gaussianSmearing(vectortmp2, smearedGauge, nsmearGauss, alphaGauss),"ISOSPIN32");

              //Saving the propagator
              vectortmp1.unload();
              stochastic_oet_prop_u_fini_mom[spinindex]->copy(vectortmp1,HOST);
              vectortmp1.load();

              //vectortmp1.writeLIME(outfile_V+confnumber+"propagator_up"+sourcepositiontext+"mompi2_"+pi2x+"_"+pi2y+"_"+pi2z+"_s"+std::to_string(spinindex));
         
              if (spinindex<3){
                vectortmp1.diluteSpinDisplace(vectorSource_finite_mom,spinindex+1,spinindex);
                vectorSource_finite_mom.copy(vectortmp1);
              }

	    } // end of spin dilution
            
            //Dilution
            vectortmp1.diluteSpinDisplace(vectorSource_finite_mom,0,3);

	    //Doing for -mu for the DN propagator spin dilution oet
            if(mu>0) {
              mu*=-1.;
              solver.UpdateSolver();
	    }

	    for (int spinindex=0; spinindex<4; ++spinindex){

	      //Transforming to physical base for the DN quark
              vectortmp2.rotateToPhysicalBasis(vectortmp1,-1);

              //Doing the inversion
              TIME(solver.solve(vectortmp2, vectortmp2),"ISOSPIN32");

              //Rotate back immediately to the physical basis
              vectortmp1.rotateToPhysicalBasis(vectortmp2,-1);

              //performing smearing
              TIME(vectortmp2.gaussianSmearing(vectortmp1, smearedGauge, nsmearGauss, alphaGauss),"ISOSPIN32");

              //Saving the propagator
              vectortmp2.unload();
              stochastic_oet_prop_d_fini_mom[spinindex]->copy(vectortmp2,HOST);
              vectortmp2.load();
              //vectortmp1.writeLIME(outfile_V+confnumber+"propagator_dn"+sourcepositiontext+"mompi2_"+pi2x+"_"+pi2y+"_"+pi2z+"_s"+std::to_string(spinindex));

              if (spinindex<3){
                vectortmp1.diluteSpinDisplace(vectorSource_finite_mom,spinindex+1,3);
              //vectorSource_finite_mom.copy(vectortmp1);
            
              }

            } //end of spin dilution

            for (int i=0; i< 4; ++i){
              PLEGMA_Vector<float> st_oet_u_fini;

              st_oet_u_fini.copy(*stochastic_oet_prop_u_fini_mom[i],HOST);
              st_oet_u_fini.load();

              st_oet_u_fini.apply_gamma5();

              TIME(reductionsV3_diluted[i].V3( st_oet_u_fini, glist_sink_meson  , propUP, true),"ISOSPIN32");

            }

            TIME(corrZ1.Z_diagrams( reductionsV3_diluted, reductionsV4_diluted_STOCHU_DN_UP, 1 ),"ISOSPIN32");
            TIME(corrZ2.Z_diagrams( reductionsV3_diluted, reductionsV4_diluted_STOCHU_DN_UP, 2 ),"ISOSPIN32");

            TIME(corrZ3.Z_diagrams( reductionsV3_diluted, reductionsV2_diluted_STOCHU_DN_UP, 3 ),"ISOSPIN32");
            TIME(corrZ4.Z_diagrams( reductionsV3_diluted, reductionsV2_diluted_STOCHU_DN_UP, 4 ),"ISOSPIN32");

	    for (int i=0; i< 4; ++i){
              PLEGMA_Vector<float> st_oet_u_fini;

             st_oet_u_fini.copy(*stochastic_oet_prop_u_fini_mom[i],HOST);
             st_oet_u_fini.load();

             st_oet_u_fini.apply_gamma5();

             TIME(reductionsV3_diluted[i].V3( st_oet_u_fini, glist_sink_meson  , propDN, true),"ISOSPIN32");

           }

           TIME(corrZ9.Z_diagrams(  reductionsV3_diluted, reductionsV2_diluted_STOCHD_UP_UP, 9 ),"ISOSPIN12");
           TIME(corrZ10.Z_diagrams( reductionsV3_diluted, reductionsV2_diluted_STOCHD_UP_UP, 10 ),"ISOSPIN12");


           for (int i=0; i< 4; ++i){
             PLEGMA_Vector<float> st_oet_d_fini;

             st_oet_d_fini.copy(*stochastic_oet_prop_d_fini_mom[i],HOST);
             st_oet_d_fini.load();

             st_oet_d_fini.apply_gamma5();

             TIME(reductionsV3_diluted[i].V3( st_oet_d_fini, glist_sink_meson  , propUP, true),"ISOSPIN32");


            }

	    TIME(corrZ6.Z_diagrams( reductionsV3_diluted, reductionsV4_diluted_STOCHU_DN_UP, 6 ),"ISOSPIN12");
            TIME(corrZ8.Z_diagrams( reductionsV3_diluted, reductionsV4_diluted_STOCHU_DN_UP, 8 ),"ISOSPIN12");

            TIME(corrZ5.Z_diagrams( reductionsV3_diluted, reductionsV2_diluted_STOCHU_DN_UP, 5 ),"ISOSPIN12");
            TIME(corrZ7.Z_diagrams( reductionsV3_diluted, reductionsV2_diluted_STOCHU_DN_UP, 7 ),"ISOSPIN12");


	    TIME(corrPPUP.P_diagrams( stochastic_oet_prop_u_zero_mom, stochastic_oet_prop_u_fini_mom, i_mpi2),"ISOSPIN32");
            TIME(corrPPDN.P_diagrams( stochastic_oet_prop_d_zero_mom, stochastic_oet_prop_d_fini_mom, i_mpi2),"ISOSPIN32");
            TIME(corrP0UP.P_diagrams( stochastic_oet_prop_d_zero_mom, stochastic_oet_prop_u_fini_mom, i_mpi2),"ISOSPIN32");
            TIME(corrP0DN.P_diagrams( stochastic_oet_prop_u_zero_mom, stochastic_oet_prop_d_fini_mom, i_mpi2),"ISOSPIN32");

            TIME(corrM.M_diagrams( corrNP, stochastic_oet_prop_u_zero_mom, stochastic_oet_prop_u_fini_mom ),"ISOSPIN32");
            TIME(corrD1if56.M_diagrams( corrN0, stochastic_oet_prop_u_zero_mom, stochastic_oet_prop_u_fini_mom),"ISOSPIN12");
            TIME(corrD1if34.M_diagrams( corrNP, stochastic_oet_prop_d_zero_mom, stochastic_oet_prop_u_fini_mom),"ISOSPIN12");
            TIME(corrD1if12.M_diagrams( corrNP, stochastic_oet_prop_u_zero_mom, stochastic_oet_prop_d_fini_mom),"ISOSPIN12");
          } //end of finite momentum
          else{

         
	    TIME(corrM.M_diagrams( corrNP, stochastic_oet_prop_u_zero_mom, stochastic_oet_prop_u_zero_mom ),"ISOSPIN32");

            TIME(corrD1if56.M_diagrams( corrN0, stochastic_oet_prop_u_zero_mom, stochastic_oet_prop_u_zero_mom),"ISOSPIN12");
            TIME(corrD1if34.M_diagrams( corrNP, stochastic_oet_prop_d_zero_mom, stochastic_oet_prop_u_zero_mom),"ISOSPIN12");
            TIME(corrD1if12.M_diagrams( corrNP, stochastic_oet_prop_u_zero_mom, stochastic_oet_prop_d_zero_mom),"ISOSPIN12");

            TIME(corrP0UP.P_diagrams( stochastic_oet_prop_d_zero_mom, stochastic_oet_prop_u_zero_mom, i_mpi2),"ISOSPIN32");
            TIME(corrP0DN.P_diagrams( stochastic_oet_prop_u_zero_mom, stochastic_oet_prop_d_zero_mom, i_mpi2),"ISOSPIN32");
            TIME(corrPPUP.P_diagrams( stochastic_oet_prop_u_zero_mom, stochastic_oet_prop_u_zero_mom, i_mpi2),"ISOSPIN32");
            TIME(corrPPDN.P_diagrams( stochastic_oet_prop_d_zero_mom, stochastic_oet_prop_d_zero_mom, i_mpi2),"ISOSPIN32");
       
	    //Z diagrams: we calculate I=1/2 as well
            //Diagram Z1,Z2
            //std::vector<GAMMAS_SCATT> gamma_5_t_sinkmeson=apply_gamma5_scatt_gamma(glist_sink_meson,LEFT);
      
            //We need V3 reductions only for pf2={0,0,0} for diagrams Z5--Z20
            TIME(corrZ1.Z_diagrams( reductionsV3_diluted_STOCHU_UP, reductionsV4_diluted_STOCHU_DN_UP, 1 ),"ISOSPIN32");
            TIME(corrZ2.Z_diagrams( reductionsV3_diluted_STOCHU_UP, reductionsV4_diluted_STOCHU_DN_UP, 2 ),"ISOSPIN32");

            //Diagram Z6,Z8
            TIME(corrZ6.Z_diagrams( reductionsV3_diluted_STOCHD_UP, reductionsV4_diluted_STOCHU_DN_UP, 6 ),"ISOSPIN12");
            TIME(corrZ8.Z_diagrams( reductionsV3_diluted_STOCHD_UP, reductionsV4_diluted_STOCHU_DN_UP, 8 ),"ISOSPIN12");

      
           //Diagram Z12,Z14
/*         TIME(corrZ12.Z_diagrams( reductionsV3_diluted_STOCHU_DN, reductionsV4_diluted_STOCHU_DN_UP, 12 ),"ISOSPIN12");
           TIME(corrZ14.Z_diagrams( reductionsV3_diluted_STOCHU_DN, reductionsV4_diluted_STOCHU_DN_UP, 14 ),"ISOSPIN12");*/

            //Diagram Z3,Z4	
            TIME(corrZ3.Z_diagrams( reductionsV3_diluted_STOCHU_UP, reductionsV2_diluted_STOCHU_DN_UP, 3 ),"ISOSPIN32");
            TIME(corrZ4.Z_diagrams( reductionsV3_diluted_STOCHU_UP, reductionsV2_diluted_STOCHU_DN_UP, 4 ),"ISOSPIN32");
         
            //Diagram Z5,Z7
            TIME(corrZ5.Z_diagrams( reductionsV3_diluted_STOCHD_UP, reductionsV2_diluted_STOCHU_DN_UP, 5 ),"ISOSPIN12");
            TIME(corrZ7.Z_diagrams( reductionsV3_diluted_STOCHD_UP, reductionsV2_diluted_STOCHU_DN_UP, 7 ),"ISOSPIN12");
       
           //Diagram Z11,Z13
/*         TIME(corrZ11.Z_diagrams( reductionsV3_diluted_STOCHU_DN_zerof2, reductionsV2_diluted_STOCHU_DN_UP, 11 ),"ISOSPIN12");
           TIME(corrZ13.Z_diagrams( reductionsV3_diluted_STOCHU_DN_zerof2, reductionsV2_diluted_STOCHU_DN_UP, 13 ),"ISOSPIN12");*/

            //Diagram Z9,Z10
            TIME(corrZ9.Z_diagrams(  reductionsV3_diluted_STOCHU_DN, reductionsV2_diluted_STOCHD_UP_UP, 9 ),"ISOSPIN12");
            TIME(corrZ10.Z_diagrams( reductionsV3_diluted_STOCHU_DN, reductionsV2_diluted_STOCHD_UP_UP, 10 ),"ISOSPIN12");

          } //end of if zero momentum

        } //do_stochastic_oet

       //Diagram Z15,Z16
/*       TIME(corrZ15.Z_diagrams( reductionsV3_diluted_STOCHU_UP_zerof2, reductionsV2_diluted_STOCHU_DN_DN, 15 ),"ISOSPIN12");
       TIME(corrZ16.Z_diagrams( reductionsV3_diluted_STOCHU_UP_zerof2, reductionsV2_diluted_STOCHU_DN_DN, 16 ),"ISOSPIN12");*/

       //Diagram Z17,Z18
/*       TIME(corrZ17.Z_diagrams( reductionsV3_diluted_STOCHU_UP_zerof2, reductionsV2_diluted_STOCHD_UP_DN, 17 ),"ISOSPIN12");
       TIME(corrZ18.Z_diagrams( reductionsV3_diluted_STOCHU_UP_zerof2, reductionsV2_diluted_STOCHD_UP_DN, 18 ),"ISOSPIN12");*/

       //Diagram Z19,Z20
/*       TIME(corrZ19.Z_diagrams( reductionsV3_diluted_STOCHU_UP_zerof2, reductionsV4_diluted_STOCHD_UP_DN, 19 ),"ISOSPIN12");
       TIME(corrZ20.Z_diagrams( reductionsV3_diluted_STOCHU_UP_zerof2, reductionsV4_diluted_STOCHD_UP_DN, 20 ),"ISOSPIN12");*/

	//write everything
	
        outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"_T";

//            TIME(produceOutput(corrT7, outfilename,  "T",n_coherent_source, coherent_source_table_timeslice),"ISOSPIN12");
//            TIME(produceOutput(corrT9, outfilename,  "T",n_coherent_source, coherent_source_table_timeslice),"ISOSPIN12");
        TIME(produceOutput(corrT11, outfilename, "T"),"ISOSPIN12");
        TIME(produceOutput(corrT12, outfilename, "T"),"ISOSPIN12");
        TIME(produceOutput(corrT13, outfilename, "T"),"ISOSPIN12");
        TIME(produceOutput(corrT14, outfilename, "T"),"ISOSPIN12");
 
        TIME(corrT.apply_phase(),"ISOSPIN32");
        TIME(corrT.apply_sign("T"),"ISOSPIN32");
        TIME(corrT.applyBoundaryConditions( true),"ISOSPIN32");
        TIME(corrT.writeHDF5(outfilename),"ISOSPIN32");


        outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"_B";

        TIME(produceOutput(corrB1, outfilename, "4pt", n_stochastic_samples),"ISOSPIN32");
        TIME(produceOutput(corrB2, outfilename, "4pt", n_stochastic_samples),"ISOSPIN32");
/*     
 *       TIME(produceOutput(corrB9,  outfilename, "4pt",n_stochastic_samples,n_coherent_source, coherent_source_table_timeslice),"ISOSPIN12");
         TIME(produceOutput(corrB10, outfilename, "4pt",n_stochastic_samples,n_coherent_source, coherent_source_table_timeslice),"ISOSPIN12");
         TIME(produceOutput(corrB11, outfilename, "4pt",n_stochastic_samples,n_coherent_source, coherent_source_table_timeslice),"ISOSPIN12");
         TIME(produceOutput(corrB12, outfilename, "4pt",n_stochastic_samples,n_coherent_source, coherent_source_table_timeslice),"ISOSPIN12");*///because of V4

        TIME(produceOutput(corrB13, outfilename, "4pt",n_stochastic_samples),"ISOSPIN12");
        TIME(produceOutput(corrB14, outfilename, "4pt",n_stochastic_samples),"ISOSPIN12");
        TIME(produceOutput(corrB15, outfilename, "4pt",n_stochastic_samples),"ISOSPIN12");
        TIME(produceOutput(corrB16, outfilename, "4pt",n_stochastic_samples),"ISOSPIN12");

        outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"_W";
        
        TIME(produceOutput(corrW1, outfilename, "4pt", n_stochastic_samples),"ISOSPIN32");
        TIME(produceOutput(corrW2, outfilename, "4pt", n_stochastic_samples),"ISOSPIN32");
        TIME(produceOutput(corrW3, outfilename, "4pt", n_stochastic_samples),"ISOSPIN32");
        TIME(produceOutput(corrW4, outfilename, "4pt", n_stochastic_samples),"ISOSPIN32"); 
/*
       TIME(produceOutput(corrW17, outfilename, "4pt", n_stochastic_samples,n_coherent_source, coherent_source_table_timeslice),"ISOSPIN12");
       TIME(produceOutput(corrW18, outfilename, "4pt", n_stochastic_samples,n_coherent_source, coherent_source_table_timeslice),"ISOSPIN12");
       TIME(produceOutput(corrW19, outfilename, "4pt", n_stochastic_samples,n_coherent_source, coherent_source_table_timeslice),"ISOSPIN12");
       TIME(produceOutput(corrW20, outfilename, "4pt", n_stochastic_samples,n_coherent_source, coherent_source_table_timeslice),"ISOSPIN12");

       TIME(produceOutput(corrW21, outfilename, "4pt", n_stochastic_samples,n_coherent_source, coherent_source_table_timeslice),"ISOSPIN12");
       TIME(produceOutput(corrW22, outfilename, "4pt", n_stochastic_samples,n_coherent_source, coherent_source_table_timeslice),"ISOSPIN12");
       TIME(produceOutput(corrW23, outfilename, "4pt", n_stochastic_samples,n_coherent_source, coherent_source_table_timeslice),"ISOSPIN12");
       TIME(produceOutput(corrW24, outfilename, "4pt", n_stochastic_samples,n_coherent_source, coherent_source_table_timeslice),"ISOSPIN12");*/

        TIME(produceOutput(corrW25, outfilename, "4pt", n_stochastic_samples),"ISOSPIN12");
        TIME(produceOutput(corrW26, outfilename, "4pt", n_stochastic_samples),"ISOSPIN12");
        TIME(produceOutput(corrW27, outfilename, "4pt", n_stochastic_samples),"ISOSPIN12");
        TIME(produceOutput(corrW28, outfilename, "4pt", n_stochastic_samples),"ISOSPIN12");
	

        outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"_Z";


        TIME(produceOutput(corrZ1, outfilename, "4pt"),"ISOSPIN32");
        TIME(produceOutput(corrZ2, outfilename, "4pt"),"ISOSPIN32");
        TIME(produceOutput(corrZ3, outfilename, "4pt"),"ISOSPIN32");
        TIME(produceOutput(corrZ4, outfilename, "4pt"),"ISOSPIN32");
 
        TIME(produceOutput(corrZ5, outfilename, "4pt"),"ISOSPIN12");
        TIME(produceOutput(corrZ6, outfilename, "4pt"),"ISOSPIN12");
        TIME(produceOutput(corrZ7, outfilename, "4pt"),"ISOSPIN12");
        TIME(produceOutput(corrZ8, outfilename, "4pt"),"ISOSPIN12");

        TIME(produceOutput(corrZ9, outfilename, "4pt"),"ISOSPIN12");
        TIME(produceOutput(corrZ10, outfilename,"4pt"),"ISOSPIN12");

/*       TIME(produceOutput(corrZ11, outfilename,"4pt",n_coherent_source, coherent_source_table_timeslice),"ISOSPIN12");
       TIME(produceOutput(corrZ12, outfilename,"4pt",n_coherent_source, coherent_source_table_timeslice),"ISOSPIN12");
       TIME(produceOutput(corrZ13, outfilename,"4pt",n_coherent_source, coherent_source_table_timeslice),"ISOSPIN12");
       TIME(produceOutput(corrZ14, outfilename,"4pt",n_coherent_source, coherent_source_table_timeslice),"ISOSPIN12");

       TIME(produceOutput(corrZ15, outfilename,"4pt",n_coherent_source, coherent_source_table_timeslice),"ISOSPIN12");
       TIME(produceOutput(corrZ16, outfilename,"4pt",n_coherent_source, coherent_source_table_timeslice),"ISOSPIN12");

       TIME(produceOutput(corrZ17, outfilename,"4pt",n_coherent_source, coherent_source_table_timeslice),"ISOSPIN12");
       TIME(produceOutput(corrZ18, outfilename,"4pt",n_coherent_source, coherent_source_table_timeslice),"ISOSPIN12");
       TIME(produceOutput(corrZ19, outfilename,"4pt",n_coherent_source, coherent_source_table_timeslice),"ISOSPIN12");
       TIME(produceOutput(corrZ20, outfilename,"4pt",n_coherent_source, coherent_source_table_timeslice),"ISOSPIN12");*/


        outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"_M";
        TIME(produceOutput(corrM, outfilename,"4pt"),"ISOSPIN12");

        TIME(produceOutput(corrD1if12, outfilename,"4pt"),"ISOSPIN12");
        TIME(produceOutput(corrD1if34, outfilename,"4pt"),"ISOSPIN12");
        TIME(produceOutput(corrD1if56, outfilename,"4pt"),"ISOSPIN12");

      }//loop over mpi_i2
     


      outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"_P";
      TIME(corrP0UP.apply_sign("P"),"ISOSPIN32");
      TIME(corrP0UP.writeHDF5( outfilename ),"ISOSPIN32");
      TIME(corrP0DN.apply_sign("P"),"ISOSPIN32");
      TIME(corrP0DN.writeHDF5( outfilename ),"ISOSPIN32");
      TIME(corrPPUP.apply_sign("P"),"ISOSPIN32");
      TIME(corrPPUP.writeHDF5( outfilename ),"ISOSPIN32");
      TIME(corrPPDN.apply_sign("P"),"ISOSPIN32");
      TIME(corrPPDN.writeHDF5( outfilename ),"ISOSPIN32");         

      if (do_contraction_std==true){
        //## N
        outfilename = outdiagramPrefix+confnumber+sourcepositiontext+"_N";

        TIME( corrNP.apply_phase(),"ISOSPIN32" );
        TIME( corrNP.apply_sign("N"),"ISOSPIN32" );
        TIME( corrNP.applyBoundaryConditions( true ),"ISOSPIN32" );
        TIME( corrNP.writeHDF5(outfilename),"ISOSPIN32" );

        TIME( corrN0.apply_phase(),"ISOSPIN12" );
        TIME( corrN0.apply_sign("N"),"ISOSPIN12" );
        TIME( corrN0.applyBoundaryConditions( true ),"ISOSPIN12" );
        TIME( corrN0.writeHDF5(outfilename) ,"ISOSPIN12");
 
      }
      for(int i=0; i< n_stochastic_samples; ++i) {

        reductions_UU_V2_GAMMAF1D_U.pop_back();
        reductions_UU_V4_GAMMAF1U_D.pop_back();
        reductions_UU_V3_GAMMAF2U.pop_back();

        reductions_DD_V3_GAMMAF2U.pop_back();
        reductions_DD_V2_GAMMAF1U_U.pop_back();
        reductions_DD_V3_GAMMAF2D.pop_back();
        reductions_DD_V2_GAMMAF1U_D.pop_back();
        reductions_DD_V4_GAMMAF1U_D.pop_back();

      }


    } //end of loop over source position


    for(int i=0; i< 4; ++i) {
      stochastic_oet_prop_d_fini_mom.pop_back();
      stochastic_oet_prop_d_zero_mom.pop_back();
      stochastic_oet_prop_u_zero_mom.pop_back();
      stochastic_oet_prop_u_fini_mom.pop_back();
    }

    for(int i=0; i< n_stochastic_samples; ++i) {
      stochastic_sources.pop_back();
      stochastic_propags.pop_back();
    }

  }  

  finalize();
  
  return 0;
}
