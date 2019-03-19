#include <PLEGMA.h>
#include <PLEGMA_utils.h>
#include <numeric>
#include <iostream>




using namespace plegma;
using namespace quda;



template<typename out,class T,class ...types,class ...types1>
__inline__ void PLEGMA_benchmark(T* obj, out (T::*function)(types1...),std::string name, types... kArgs){

  double t1;
  std::vector<double> timing;

  for(int t=0;t<n_benchmark;t++)
    {
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
  PLEGMA_printf("Minimum time:\t %.16lf",timing[std::distance(std::begin(timing), min)]);
  PLEGMA_printf("\n#######################################################################\n");
}




int main(int argc, char **argv)
{

  static std::vector<std::string> listOpt = {"load-gauge","n_benchmark"};

  initializeOptions(argc, argv, true, listOpt);
  initializePLEGMA();  

  
  PLEGMA_Gauge<double> gauge_a;
  gauge_a.zero_host();
  gauge_a.zero_device();

  /* Benchmark of the APEsmearing function */
  PLEGMA_Gauge<double> gauge_b;
  PLEGMA_benchmark(&gauge_b,&PLEGMA_Gauge<double>::APEsmearing,"APEsmearing",gauge_a,nsmearAPE,alphaAPE,3);
  
  /* Benchmark of the momentum smearing */
  std::complex<double> momSmScale[N_DIMS];
  std::complex<double> I(0,1);
  double xi=0.6;
  std::vector<int> sinkM={1,0,0,0};
  for(int i = 0 ; i < N_DIMS; i++) momSmScale[i] = std::exp(-(xi*2.*PI*sinkM[i]/HGC_totalL[i])*I);
  PLEGMA_benchmark(&gauge_b,&PLEGMA_Gauge<double>::scaleDirWise,"Momentum smearing",momSmScale);
  
  /* Benchmark gaussian smearing */
  PLEGMA_Vector<double> vector_a;
  PLEGMA_benchmark(&vector_a,&PLEGMA_Vector<double>::gaussianSmearing,"Gaussian Smearing",vector_a,gauge_b, nsmearGauss, alphaGauss);
    
  /* Benchmark contraction  3pt*/
  PLEGMA_Correlator<float> corr(corr_space,0); 
  PLEGMA_Propagator<float> prop_a(BOTH);
  PLEGMA_Propagator<float> prop_b(BOTH);
  PLEGMA_Su3field<float> su3_a;
  std::vector<GAMMAS> gammas = {G3,};
  int (*sources);
  hostMalloc(sources, N_DIMS*sizeof(int));
  sources[0]=0;
  sources[1]=1;
  sources[2]=0;
  sources[3]=0;
  PLEGMA_benchmark(&corr,&PLEGMA_Correlator<float>::contractNucleonThrp_wilsonLine,"Contraction 3pt",prop_a, prop_b, su3_a, +1, gammas, sources);

  /* Benchmark Wilson line update */
  PLEGMA_Su3field<float> su3_b;
  PLEGMA_Su3field<float> su3_c;
  PLEGMA_Su3field<float> su3_d;
  PLEGMA_benchmark(&su3_d,&PLEGMA_Su3field<float>::wilsonLineUpdate,"Update of the Wilson line",su3_b,su3_c,4+2);

  /* Benchmark shift routine */
  PLEGMA_Propagator<float> prop_c(BOTH);
  PLEGMA_benchmark<void,PLEGMA_Field<float>>(&prop_a,&PLEGMA_Field<float>::shift,"Shift routine",prop_c,4+2);

  /* Benchmark Meson contration */
  PLEGMA_Propagator<float> prop_d(BOTH);
  PLEGMA_Propagator<float> prop_e(BOTH);
  PLEGMA_benchmark(&corr,&PLEGMA_Correlator<float>::contractMesons,"Contraction mesons",prop_d, prop_e, sources);

  /* Benchmark Baryons contractions */
  PLEGMA_Propagator<float> prop_f(BOTH);
  PLEGMA_Propagator<float> prop_g(BOTH);
  PLEGMA_benchmark(&corr,&PLEGMA_Correlator<float>::contractBaryons,"Contraction Baryons",prop_f, prop_g, sources);
  free(sources);

  finalize();
  exit(0);
}



