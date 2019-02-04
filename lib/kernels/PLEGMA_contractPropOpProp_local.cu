#include <PLEGMA_contractPropOpProp.cuh>
template<typename FloatC,typename FloatA, typename FloatB>
void contractPropOpProp_local(PLEGMA_Correlator<FloatC> &corr, propTex<FloatA> prop1, propTex<FloatB> prop2, int signProps, int it, std::vector<GAMMAS> gammas){
  su3Tex<FloatC> su3;
  su3.tex=0;
  contractPropOpProp<FloatC,FloatA,FloatB,FloatC,false,-1,false>(corr,prop1,prop2,signProps,su3,it, gammas);
}

template void contractPropOpProp_local<float,float,float>(PLEGMA_Correlator<float> &corr, propTex<float> prop1, propTex<float> prop2, int signProps, int it, std::vector<GAMMAS> gammas);
template void contractPropOpProp_local<double,double,double>(PLEGMA_Correlator<double> &corr, propTex<double> prop1, propTex<double> prop2, int signProps, int it, std::vector<GAMMAS> gammas);
