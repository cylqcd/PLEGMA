#include <PLEGMA_contractPropOpProp.cuh>

template void contractPropOpProp_oneD<float,float,float,float>(PLEGMA_Correlator<float> &corr, propTex<float> prop1, propTex<float> prop2, int signProps, su3Tex<float> su3, int it, int dir);
template void contractPropOpProp_oneD<double,double,double,double>(PLEGMA_Correlator<double> &corr, propTex<double> prop1, propTex<double> prop2, int signProps, su3Tex<double> su3, int it, int dir);

