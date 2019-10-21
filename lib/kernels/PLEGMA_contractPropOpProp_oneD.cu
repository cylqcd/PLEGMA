#include <PLEGMA_contractPropOpProp.cuh>

template<typename FloatC,typename FloatA, typename FloatB, typename FloatS>
void contractPropOpProp_oneD(PLEGMA_Correlator<FloatC> &corr, propTex<FloatA> prop1, propTex<FloatB> prop2, int signProps,
			     su3Tex<FloatS> su3, int dir, std::vector<GAMMAS> gammas){
  if(dir < 0 || dir > 3) PLEGMA_error("Allowed directions are 0,1,2,3\n");
  switch(dir){
  case(0): contractPropOpProp<FloatC,FloatA,FloatB,FloatC,true,0,false>(corr,prop1,prop2,signProps,su3,gammas); break;
  case(1): contractPropOpProp<FloatC,FloatA,FloatB,FloatC,true,1,false>(corr,prop1,prop2,signProps,su3,gammas);	break;
  case(2): contractPropOpProp<FloatC,FloatA,FloatB,FloatC,true,2,false>(corr,prop1,prop2,signProps,su3,gammas);	break;
  case(3): contractPropOpProp<FloatC,FloatA,FloatB,FloatC,true,3,false>(corr,prop1,prop2,signProps,su3,gammas);	break;
  }
}

template void contractPropOpProp_oneD<float,float,float,float>(PLEGMA_Correlator<float> &corr, propTex<float> prop1, propTex<float> prop2, int signProps, su3Tex<float> su3, int dir, std::vector<GAMMAS> gammas);
template void contractPropOpProp_oneD<double,double,double,double>(PLEGMA_Correlator<double> &corr, propTex<double> prop1, propTex<double> prop2, int signProps, su3Tex<double> su3, int dir, std::vector<GAMMAS> gammas);

