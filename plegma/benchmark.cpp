#include <PLEGMA.h>
#include <PLEGMA_utils.h>
#include <numeric>
#include <iostream>
#include <string>

using namespace plegma;
using namespace quda;

template<typename out,class T,class T1,class ...types, class ...types1>
void PLEGMA_benchmark(T *obj, out (T1::*function)(types1...),std::string name, types&&... kArgs){

  double t1;
  std::vector<double> timing;

  (obj->*function)(kArgs...);  // Calling once outside for performing tuning
  for(int t=0;t<n_benchmark;t++) {
    t1=MPI_Wtime();
    (obj->*function)(kArgs...);
    timing.push_back(MPI_Wtime()-t1);
  }

  double sum = std::accumulate(timing.begin(), timing.end(), 0.0);
  double mean = sum / timing.size();

  double sq_sum = std::inner_product(timing.begin(), timing.end(), timing.begin(), 0.0);
  double stdev = std::sqrt(sq_sum / timing.size() - mean * mean);

  std::vector<double>::iterator min = std::min_element(std::begin(timing), std::end(timing));
  std::vector<double>::iterator max = std::max_element(std::begin(timing), std::end(timing));
  

 
  PLEGMA_printf("############################## BENCHMARK ##############################\n");
  PLEGMA_printf("Function name: \t %s \n", name.c_str());
  PLEGMA_printf("Total number of repetitions:\t %d \n",n_benchmark);
  PLEGMA_printf("Average time time:\t %.16f \n", mean);
  PLEGMA_printf("Standard deviation:\t %.16lf \n", stdev);
  PLEGMA_printf("Maximum time:\t %.16lf\n", timing[std::distance(std::begin(timing), max)]);
  PLEGMA_printf("Minimum time:\t %.16lf\n",timing[std::distance(std::begin(timing), min)]);
  PLEGMA_printf("#######################################################################\n");
}

std::string kind = "all";
inline bool run(std::vector<std::string> run_for) {
  if (kind.find("all") != std::string::npos)
    return true;
  
  for(auto str : run_for)
    if (kind.find(str) != std::string::npos)
      return true;
  
  return false;
}

int main(int argc, char **argv) {
  
  static std::vector<std::string> listOpt = {"verbosity", "n_benchmark", "corr-space", "maxQsq"};

  initializeOptions(argc, argv, true, listOpt);

  HGC_options->set("kind", "Kind of application to benchmark. Multiple options allowed. Options (all, twop, threep, PDFs, qLoops, etc...)", verbosity, kind);
  
  initializePLEGMA();  

  if(run({"twop","threep","PDFs","smearing"})) {
    PLEGMA_Gauge<double> gauge_a, gauge_b;
    PLEGMA_Vector<double> vector_a, vector_b;

    // Benchmark of the APEsmearing function 
    PLEGMA_benchmark(&gauge_b,&PLEGMA_Gauge<double>::APEsmearing, "APEsmearing (1 iter)", gauge_a, 1, 0.5, 3);

    // Benchmark gaussian smearing 
    PLEGMA_benchmark(&vector_a,&PLEGMA_Vector<double>::gaussianSmearing,"Gaussian Smearing (1 iter)",vector_b, gauge_a, 1, 0.2);
  }

  if(run({"twop","threep","PDFs"})) {
    PLEGMA_Propagator<float> prop;
    PLEGMA_Correlator<float> corr(corr_space,maxQsq);
    int sources[4] = {1,0,1,0};
    
    // Benchmark Meson contration 
    PLEGMA_benchmark(&corr,&PLEGMA_Correlator<float>::contractMesons,"Contraction mesons",prop, prop, sources);

    // Benchmark Baryons contractions 
    PLEGMA_benchmark(&corr,&PLEGMA_Correlator<float>::contractBaryons,"Contraction Baryons",prop, prop, sources);
  }

  if(run({"threep","PDFs"})) {
    PLEGMA_Propagator3D<float> prop;
    PLEGMA_Vector<float> vector;

    // Since seqSourceNucleon is overloaded we need to select one version of it
    void (PLEGMA_Vector<float>::*seqSourceNucleon)(PLEGMA_Propagator3D<float> &, PLEGMA_Propagator3D<float> &, WHICHPROJECTOR, WHICHPARTICLE, int, int, int) = &PLEGMA_Vector<float>::seqSourceNucleon;
    // Benchmark Sequential source 
    for(int i=0; i<(int) N_PROJS; i++) {
      PLEGMA_benchmark(&vector, seqSourceNucleon, "Sequential source proton P=" + std::to_string(i), prop, prop, (WHICHPROJECTOR) i, PROTON, 1, 0, 0);
      PLEGMA_benchmark(&vector, seqSourceNucleon, "Sequential source neutron P=" + std::to_string(i), prop, prop, (WHICHPROJECTOR) i, NEUTRON, 1, 0, 0);
    }
  }

  if(run({"threep"})) {
    PLEGMA_Gauge<float> gauge;
    PLEGMA_Propagator<float> prop;
    PLEGMA_Correlator<float> corr(corr_space,maxQsq);
    std::vector<GAMMAS> gammas = {ONE,G1,G2,G3,G4,G5,G5G1,G5G2,G5G3,G5G4,S12,S13,S23,S41,S42,S43};
    int sources[4] = {1,0,1,0};
    
    // Benchmark three point functions 
    PLEGMA_benchmark(&corr,&PLEGMA_Correlator<float>::contractNucleonThrp_local,"Contraction local",prop, prop, +1, gammas, sources);
    PLEGMA_benchmark(&corr,&PLEGMA_Correlator<float>::contractNucleonThrp_oneD,"Contraction one derivative",prop, prop, gauge, +1, gammas, sources);
    PLEGMA_benchmark(&corr,&PLEGMA_Correlator<float>::contractNucleonThrp_noe,"Contraction Noether",prop, prop, gauge, +1, sources);
  }
  
  if(run({"PDFs"})) {
    PLEGMA_Gauge<double> gauge;
    PLEGMA_Su3field<float> su3,su3_a,su3_b;
    PLEGMA_Propagator<float> prop;
    PLEGMA_Correlator<float> corr(MOMENTUM_SPACE,0);
    std::complex<double> momSmScale[N_DIMS] = {1,1,1,1};
    momSmScale[0] = {0.7071, 0.7071};
    std::vector<GAMMAS> gammas = {G3,};
    int sources[4] = {1,0,1,0};
    
    // Benchmark of the momentum smearing 
    PLEGMA_benchmark(&gauge,&PLEGMA_Gauge<double>::scaleDirWise,"Momentum smearing (scale 1 dir)",momSmScale);

    // Benchmark contraction  3pt
    PLEGMA_benchmark(&corr,&PLEGMA_Correlator<float>::contractNucleonThrp_wilsonLine,"Contraction 3pt",prop, prop, su3, +1, gammas, sources);

    // Benchmark Wilson line update 
    PLEGMA_benchmark(&su3,&PLEGMA_Su3field<float>::wilsonLineUpdate,"Update of the Wilson line",su3_a,su3_b,4+2);

    // Benchmark shift routine 
    PLEGMA_benchmark(&prop,&PLEGMA_Field<float>::shift,"Shift routine",prop,2);
  }

  finalize();
  return 0;
}



