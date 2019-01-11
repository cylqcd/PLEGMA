#include <PLEGMA_contractPropOpProp.cuh>

template void contractPropOpProp_local<float,float,float>(PLEGMA_Correlator<float> &corr, propTex<float> prop1, propTex<float> prop2, int signProps, int it);
template void contractPropOpProp_local<double,double,double>(PLEGMA_Correlator<double> &corr, propTex<double> prop1, propTex<double> prop2, int signProps, int it);
