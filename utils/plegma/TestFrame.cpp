#include <PLEGMA.h>
#include <PLEGMA_utils.h>
#include <string>
#include <fstream>



std::vector<double> runtime;
#define TIME(fnc)  runtime.push_back(MPI_Wtime()); fnc;			\
  PLEGMA_printf("TIME for "#fnc" %f sec\n", MPI_Wtime()-runtime.back()); \
  runtime.pop_back()

std::vector<std::thread> threads;
//#define THREAD(fnc) threads.push_back(std::thread([=]() { TIME(fnc); }))
#define THREAD(fnc) saveTuneCache(false); TIME(fnc)

using namespace plegma;
using namespace quda;
static std::vector<std::string> listOpt = { "verbosity", "load-gauge", "nsmear-APE", "alpha-APE", "nsmear-gauss", "alpha-gauss",
					    "nsrc", "src-filename", "maxQsq", "twop-filename", "corr-file-format", "corr-space"/*, "tSinks","Projs", "threep-filename",*/};
                        
                        
                        
site ReadAndDeleteBottomFileName(const std::string &filename_file, bool &continuing){
    
 int rank = 0;
 site source;
 std::string sourceLine;
  
 MPI_Initialized(&rank);
 if(rank) {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if(rank==0) {
     
    std::string line;
 
    std::ifstream fin;
    fin.open(filename_file);
    std::string tempfile_name = "tempFiles/" + filename_file.substr(filename_file.find_last_of('/') + 1)  + "_tmp";
    std::ofstream temp;
    temp.open(tempfile_name);


    if(getline(fin,sourceLine)){
    
        while (getline(fin,line)){
            temp << line << std::endl;
        } 
    
    temp.close();
    fin.close();
    remove(filename_file.c_str());
    rename(tempfile_name.c_str(), filename_file.c_str());

    continuing=true;
    }
    else{
        continuing= false;
        temp.close();
        remove(tempfile_name.c_str());
    }
    }
 
 MPI_Bcast(&continuing, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);
 int line_size = sourceLine.size();
 MPI_Bcast(&line_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
 if (rank != 0){        
    sourceLine.resize(line_size);}
 MPI_Bcast(const_cast<char*>(sourceLine.data()), line_size, MPI_CHAR, 0, MPI_COMM_WORLD);
 }
 else{
      PLEGMA_error("Only single mode, reading not done!");
 }
  //   std::cout << "rank " << rank << "  continuing " << continuing << std::endl;
  //   std::cout << "rank " << rank << "  sourceLine " << sourceLine << std::endl;
 
 if (continuing == false){
    return source;
 }
       
    size_t pos = 0;
    int i=0;
    std::string token;
    while ((pos = sourceLine.find(" ")) != std::string::npos) {
    token = sourceLine.substr(0, pos);
     source[i] = atoi(token.c_str());
     i++;
    sourceLine.erase(0, pos + 1);
    }

    source[i] = atoi(sourceLine.c_str());  
    
    MPI_Barrier(MPI_COMM_WORLD);
    
return source;
}                         
                        
                        
  

void DeleteBottomPropagatorFile(const std::string &bottom_prop_filename){
    
    int rank = 0;
  
    MPI_Initialized(&rank);
    if(rank) {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if(rank==0) { 
          remove(bottom_prop_filename.c_str());  
        }
    }
}
                        
                        
                        
                        
                        
                        
  
int main(int argc, char **argv) {
  initializeOptions(argc, argv, true, listOpt);
  //================ Add your options in this between initializeOptions and initializePLEGMA ================//
//   std::vector<double> mu_s;
//   std::vector<double> mu_c;
//  double mu_ud = mu;
//  double mu_ud_factor[QUDA_MAX_MG_LEVEL];
//  for(int i=0;i<QUDA_MAX_MG_LEVEL;i++) mu_ud_factor[i] = mu_factor[i];
//   int nsmearGauss_s = nsmearGauss/2;
//   int nsmearGauss_c = 0;
  int startSource = 0;
  std::string light_prop_fileprefix;
//   std::string light_prop_filename;
  std::string bottom_prop_fileprefix;
  std::string bottom_prop_filename;
  std::string srcInputFile = "./input.src";
  auto add_options = [&](Options& options) {
//     options.set("mu-s", "List of mu_s to run for the strange quark in baryons", verbosity, mu_s);
//     options.set("mu-c", "List of mu_c to run for the charm quark in baryons", verbosity, mu_c);
//     options.set("nsmear-gauss-s", "Number of Gaussian smearing step for the strange quark propagator", verbosity, nsmearGauss_s);
//     options.set("nsmear-gauss-c", "Number of Gaussian smearing step for the charm quark propagator", verbosity, nsmearGauss_c);
    options.set("src-input-file", "Use the file to get finished bottom propagators", verbosity, srcInputFile);
    options.set("bottom-prop-filename", "Filename of bottom propagator to be read in", verbosity, bottom_prop_fileprefix);
//     options.set("light-prop-filename", "Filename of light propagator to be read in", verbosity, light_prop_fileprefix);
    options.set("start-src", "The index of the source position where to start the calculation", verbosity, startSource);
		     };
  add_options(*HGC_options);
  //=========================================================================================================//
  initializePLEGMA();
  
  
  std::string openContr_filename = twop_filename + std::string("_openContractions_")+ ((nsmearGauss>0) ? "SS" : "LL") +
    "_gN" + std::to_string(nsmearGauss) + "a" + convNumToStr(alphaGauss) +
    "_aN" + std::to_string(nsmearAPE) + "a" + convNumToStr(alphaAPE);
  
  


  twop_filename += std::string("_") + ((nsmearGauss>0) ? "SS" : "LL") +
    "_gN" + std::to_string(nsmearGauss) + "a" + convNumToStr(alphaGauss) +
    "_aN" + std::to_string(nsmearAPE) + "a" + convNumToStr(alphaAPE);

  {
    PLEGMA_Gauge<double> smearedGauge(BOTH);
 //   PLEGMA_Gauge<float> contractGauge(BOTH);
    {
      // Reading from Lime file and loading to device
      PLEGMA_Gauge<double> gauge;
      gauge.readFile(latfile, LIME_FORMAT);
      gauge.load();
      gauge.calculatePlaq();
      
      // Loading to QUDA and computing plaquette also there
      initGaugeQuda(gauge, true);
      plaqQuda();
      
      // Smearing
      TIME(smearedGauge.APEsmearing(gauge, nsmearAPE, alphaAPE, 3));
      PLEGMA_printf("Plaquette after smearing:\n");
      smearedGauge.calculatePlaq();

      // Gauge for contractions
   //   contractGauge.copy(gauge);
      // apply boundary conditions since is needed for the covariant derivative
  //    applyBoundaryConditions(contractGauge,true);
   }

    updateOptions(LIGHT);
    TIME(QUDA_solver solver(0));

    std::string given_twop_filename = twop_filename;
//     std::string given_threep_filename = threep_filename;
    
    bool running = true;
    int isource=0;
    /*while (running){
        isource++;
//     for(int isource = startSource; isource < numSourcePositions; isource++){
//         site& source = sourcePositions[isource];
//         PLEGMA_printf("\n ### Calculations for source-position %d - %02d.%02d.%02d.%02d begin now ###\n\n",
//         isource, source[0], source[1], source[2], source[3]);

      // XXX  necessary???  
  //    updateOptions(srcInputFile + std::to_string(isource), listOpt, add_options);
        
    site source = ReadAndDeleteBottomFileName(srcInputFile, running);
        
    if (running == false){
        break;
    }
    PLEGMA_printf("\n ### Calculations for source-position %d - %02d.%02d.%02d.%02d begin now ###\n\n",
    isource, source[0], source[1], source[2], source[3]);
    
    
    PLEGMA_Propagator<float> propLT;
    PLEGMA_Propagator<float> propBT;
    
    char * prop_string;
    asprintf(&prop_string, "x%02dy%02dz%02dt%03d", source[0], source[1], source[2], source[3]);
    bottom_prop_filename = bottom_prop_fileprefix + prop_string +"_lime";
    free(prop_string);
    
       
    TIME(propBT.readFile(bottom_prop_filename, LIME_FORMAT));
    MPI_Barrier(MPI_COMM_WORLD);
    DeleteBottomPropagatorFile(bottom_prop_filename);

    
    PLEGMA_Gauge3D<double> smearedGauge3D;
    smearedGauge3D.absorb(smearedGauge, source[DIM_T]);

    
    auto computePropagator = [&](PLEGMA_Propagator<float>& prop, int nSmear) {
            for(int isc = 0 ; isc < 12 ; isc++){
            PLEGMA_Vector<double> vectorInOut;
            { // Smearing the source
                PLEGMA_Vector3D<double> vector1, vector2;
                vector1.pointSource(source, isc/3, isc%3, DEVICE);
                TIME(vector2.gaussianSmearing(vector1, smearedGauge3D, nSmear, alphaGauss));
                vectorInOut.absorb(vector2,source[DIM_T]);
            }
            // Inverting
            PLEGMA_printf("Going to invert component %d\n", isc);
            TIME(solver.solve(vectorInOut, vectorInOut));
            { // Smearing the solution
                PLEGMA_Vector<double> vectorAuxD;
                PLEGMA_Vector<float> vectorAuxF;
                TIME(vectorAuxD.gaussianSmearing(vectorInOut, smearedGauge, nSmear, alphaGauss));
                vectorAuxF.copy(vectorAuxD);
                prop.absorb(vectorAuxF, isc/3, isc%3);
            }
            }
             prop.applyBoundaries_device(source[DIM_T]);
            };
      

        {
   
       TIME(computePropagator(propLT, nsmearGauss));
        

        PLEGMA_Propagator<float> none(NONE);
        PLEGMA_Correlator<double> corr(corr_space, source, maxQsq);
        char * group;
        
        
        TIME(corr.contractTetraquarks(propLT, none, none, propBT));
        asprintf(&group, "contractTetraquarks");
        corr.setGroups(group);
        free(group);
        THREAD(corr.writeFile(twop_filename, corr_file_format));
        
         
        TIME(corr.contractTetraquarksStochastic(propLT, propLT, propBT, propBT)); 
        asprintf(&group, "contractTetraquarksStochastic");
        corr.setGroups(group);
        free(group);
        THREAD(corr.writeFile(twop_filename, corr_file_format));  
        



      }
    }*/
    
    
    PLEGMA_Propagator<float> propLT1;
    PLEGMA_Propagator<float> propLT2;
    PLEGMA_Propagator<float> propBT1;
    PLEGMA_Propagator<float> propBT2;
    
    std::string stoch_prefix="/nvme/h/de21mp1/runs/hl_tetraquark/Testing/comparing_correlators/stochContractions_NRQCD/props/";
    TIME(propLT1.readFile(stoch_prefix + "LightPropStochastic.6.t095.r00_lime", LIME_FORMAT));
    TIME(propLT2.readFile(stoch_prefix + "LightPropStochastic.6.t095.r01_lime", LIME_FORMAT));
    TIME(propBT1.readFile(stoch_prefix + "BottomPropStochastic.6.t095.r00_lime", LIME_FORMAT));
    TIME(propBT2.readFile(stoch_prefix + "BottomPropStochastic.6.t095.r01_lime", LIME_FORMAT));
    
    std::string filename_prefix = twop_filename + "_stochCombinations";
    
    site sourceStoch;
     for(int i=0;i<3; i++){
        sourceStoch[i]=0;
    }
    sourceStoch[3] = 95;
    
    PLEGMA_Correlator<double> corr(corr_space, sourceStoch, maxQsq);
    char * group;
    
    
    TIME(corr.contractTetraquarksStochastic(propLT1, propLT1, propLT1, propLT1, propBT1, propBT1));
    asprintf(&group, "TetraquarksStochastic_r0000");
    corr.setGroups(group);
    free(group);
    THREAD(corr.writeFile(filename_prefix, corr_file_format));

    TIME(corr.contractTetraquarksStochastic(propLT1, propLT1, propLT1, propLT1, propBT1, propBT2));
    asprintf(&group, "TetraquarksStochastic_r0001");
    corr.setGroups(group);
    free(group);
    THREAD(corr.writeFile(filename_prefix, corr_file_format));

    TIME(corr.contractTetraquarksStochastic(propLT1, propLT1, propLT1, propLT1, propBT2, propBT1));
    asprintf(&group, "TetraquarksStochastic_r0010");
    corr.setGroups(group);
    free(group);
    THREAD(corr.writeFile(filename_prefix, corr_file_format));

    TIME(corr.contractTetraquarksStochastic(propLT1, propLT2, propLT1, propLT2, propBT1, propBT1));
    asprintf(&group, "TetraquarksStochastic_r0100");
    corr.setGroups(group);
    free(group);
    THREAD(corr.writeFile(filename_prefix, corr_file_format));



    TIME(corr.contractTetraquarksStochastic(propLT2, propLT1, propLT2, propLT1, propBT1, propBT1));
    asprintf(&group, "TetraquarksStochastic_r1000");
    corr.setGroups(group);
    free(group);
    THREAD(corr.writeFile(filename_prefix, corr_file_format));

    TIME(corr.contractTetraquarksStochastic(propLT1, propLT1, propLT1, propLT1 propBT2, propBT2));
	asprintf(&group, "TetraquarksStochastic_r0011");
	corr.setGroups(group);
	free(group);
	THREAD(corr.writeFile(filename_prefix, corr_file_format));

	TIME(corr.contractTetraquarksStochastic(propLT1, propLT2, propLT1, propLT2, propBT1, propBT2));
	asprintf(&group, "TetraquarksStochastic_r0101");
	corr.setGroups(group);
	free(group);
	THREAD(corr.writeFile(filename_prefix, corr_file_format));

	TIME(corr.contractTetraquarksStochastic(propLT2, propLT1, propLT2, propLT1, propBT1, propBT2));
	asprintf(&group, "TetraquarksStochastic_r1001");
	corr.setGroups(group);
	free(group);
	THREAD(corr.writeFile(filename_prefix, corr_file_format));



	TIME(corr.contractTetraquarksStochastic(propLT1, propLT2, propLT1, propLT2, propBT2, propBT1));
	asprintf(&group, "TetraquarksStochastic_r0110");
	corr.setGroups(group);
	free(group);
	THREAD(corr.writeFile(filename_prefix, corr_file_format));

	TIME(corr.contractTetraquarksStochastic(propLT2, propLT1, propLT2, propLT1, propBT2, propBT1));
	asprintf(&group, "TetraquarksStochastic_r1010");
	corr.setGroups(group);
	free(group);
	THREAD(corr.writeFile(filename_prefix, corr_file_format));

	TIME(corr.contractTetraquarksStochastic(propLT2, propLT2, propLT2, propLT2, propBT1, propBT1));
	asprintf(&group, "TetraquarksStochastic_r1100");
	corr.setGroups(group);
	free(group);
	THREAD(corr.writeFile(filename_prefix, corr_file_format));

	TIME(corr.contractTetraquarksStochastic(propLT1, propLT2, propLT1, propLT2, propBT2, propBT2));
	asprintf(&group, "TetraquarksStochastic_r0111");
	corr.setGroups(group);
	free(group);
	THREAD(corr.writeFile(filename_prefix, corr_file_format));



	TIME(corr.contractTetraquarksStochastic(propLT2, propLT1, propLT2, propLT1, propBT2, propBT2));
	asprintf(&group, "TetraquarksStochastic_r1011");
	corr.setGroups(group);
	free(group);
	THREAD(corr.writeFile(filename_prefix, corr_file_format));

	TIME(corr.contractTetraquarksStochastic(propLT2, propLT2, propLT2, propLT2, propBT1, propBT2));
	asprintf(&group, "TetraquarksStochastic_r1101");
	corr.setGroups(group);
	free(group);
	THREAD(corr.writeFile(filename_prefix, corr_file_format));

	TIME(corr.contractTetraquarksStochastic(propLT2, propLT2, propLT2, propLT2, propBT2, propBT1));
	asprintf(&group, "TetraquarksStochastic_r1110");
	corr.setGroups(group);
	free(group);
	THREAD(corr.writeFile(filename_prefix, corr_file_format));

	TIME(corr.contractTetraquarksStochastic(propLT2, propLT2, propLT2, propLT2, propBT2, propBT2));
	asprintf(&group, "TetraquarksStochastic_r1111");
	corr.setGroups(group);
	free(group);
	THREAD(corr.writeFile(filename_prefix, corr_file_format));

    
    PLEGMA_Propagator<float> propLT;
    PLEGMA_Propagator<float> propBT;

    site source[4];

    source[0][0]=06;
    source[0][1]=23;
    source[0][2]=31;
    source[0][3]=39;

    source[1][0]=8;
    source[1][1]=03;
    source[1][2]=01;
    source[1][3]=89;

    source[2][0]=04;
    source[2][1]=20;
    source[2][2]=8;
    source[2][3]=9;

    source[3][0]=05;
    source[3][1]=22;
    source[3][2]=23;
    source[3][3]=69;


    for (isource=0; isource<4; isource++){

    	char * prop_string;
    	asprintf(&prop_string, "x%02dy%02dz%02dt%03d", source[isource][0], source[isource][1], source[isource][2], source[isource][3]);
    	std::string path = "/nvme/h/de21mp1/runs/hl_tetraquark/Testing/comparing_correlators/pointContractions_NRQCD/props/";
        std::string bottom_filename = path + "BottomProp.6." + prop_string +"_lime";
        std::string light_filename = path +"LightProp.6."+ prop_string +"_lime";
        free(prop_string);

    	TIME(propLT.readFile(light_filename, LIME_FORMAT));
    	TIME(propBT.readFile(bottom_filename, LIME_FORMAT));


    	PLEGMA_Propagator<float> none(NONE);
		PLEGMA_Correlator<double> corr(corr_space, source[isource], maxQsq);
		char * group;


		TIME(corr.contractTetraquarks(propLT, none, none, propBT));
		asprintf(&group, "contractTetraquarks");
		corr.setGroups(group);
		free(group);
		THREAD(corr.writeFile(twop_filename, corr_file_format));


		TIME(corr.contractTetraquarksStochastic(propLT, propLT, propBT, propBT));
		asprintf(&group, "contractTetraquarksStochastic");
		corr.setGroups(group);
		free(group);
		THREAD(corr.writeFile(twop_filename, corr_file_format));


    }


    
    
    
    
    
    
    while(not threads.empty()) {threads.back().join(); threads.pop_back();}
  }

  finalize();
  return 0;
}

