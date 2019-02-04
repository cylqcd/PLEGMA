#include <PLEGMA_contractPropOpProp.cuh>

template<typename FloatC,typename FloatA, typename FloatB, typename FloatS>
void contractPropOpProp_wilsonLine(PLEGMA_Correlator<FloatC> &corr, propTex<FloatA> prop1, propTex<FloatB> prop2, int signProps,
			       su3Tex<FloatS> su3, int it, std::vector<GAMMAS> gammas){
  contractPropOpProp<FloatC,FloatA,FloatB,FloatC,true,-1,false>(corr,prop1,prop2,signProps,su3,it, gammas);
}

template void contractPropOpProp_wilsonLine<float,float,float,float>(PLEGMA_Correlator<float> &corr, propTex<float> prop1, propTex<float> prop2, int signProps, su3Tex<float> su3, int it, std::vector<GAMMAS> gammas);
template void contractPropOpProp_wilsonLine<double,double,double,double>(PLEGMA_Correlator<double> &corr, propTex<double> prop1, propTex<double> prop2, int signProps, su3Tex<double> su3, int it, std::vector<GAMMAS> gammas);
