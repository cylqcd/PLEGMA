#include <PLEGMA_contractPropOpProp.cuh>

template<typename FloatC,typename FloatA, typename FloatB, typename FloatS>
void contractPropOpProp_noe(PLEGMA_Correlator<FloatC> &corr, propTex<FloatA> prop1, propTex<FloatB> prop2, int signProps,
			    su3Tex<FloatS> su3, int it, int dir, std::vector<GAMMAS> gammas){
  if(dir < 0 || dir > 3) errorQuda("Allowed directions are 0,1,2,3\n");
  if(gammas.size() != 0) errorQuda("For the conserved the gammas list should be passed as empty");
  switch(dir){
  case(0):
    gammas.push_back(G1);
    contractPropOpProp<FloatC,FloatA,FloatB,FloatC,true,0,true>(corr,prop1,prop2,signProps,su3,it, gammas);
    break;
  case(1):
    gammas.push_back(G2);
    contractPropOpProp<FloatC,FloatA,FloatB,FloatC,true,1,true>(corr,prop1,prop2,signProps,su3,it, gammas);
    break;
  case(2):
    gammas.push_back(G3);
    contractPropOpProp<FloatC,FloatA,FloatB,FloatC,true,2,true>(corr,prop1,prop2,signProps,su3,it, gammas);
    break;
  case(3):
    gammas.push_back(G4);
    contractPropOpProp<FloatC,FloatA,FloatB,FloatC,true,3,true>(corr,prop1,prop2,signProps,su3,it, gammas);
    break;
  }
}

template void contractPropOpProp_noe<float,float,float,float>(PLEGMA_Correlator<float> &corr, propTex<float> prop1, propTex<float> prop2, int signProps, su3Tex<float> su3, int it, int dir, std::vector<GAMMAS> gammas);
template void contractPropOpProp_noe<double,double,double,double>(PLEGMA_Correlator<double> &corr, propTex<double> prop1, propTex<double> prop2, int signProps, su3Tex<double> su3, int it, int dir, std::vector<GAMMAS> gammas);

