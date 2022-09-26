#pragma once
#include <PLEGMA_Propagator.h>


template<typename FloatC,typename FloatA, typename FloatB>
void threep_local(PLEGMA_Correlator<FloatC> &corr, PLEGMA_Propagator<FloatA>& prop1, PLEGMA_Propagator<FloatB>& prop2, int signProps, std::vector<GAMMAS>& gammas, bool isZfac);


template<typename FloatC,typename FloatA, typename FloatB, typename FloatG>
void threep_noe(PLEGMA_Correlator<FloatC> &corr, PLEGMA_Propagator<FloatA>& prop1, PLEGMA_Propagator<FloatB>& prop2, int signProps, PLEGMA_Gauge<FloatG>& gauge);


template<typename FloatC,typename FloatA, typename FloatB, typename FloatG>
void threep_oneD(PLEGMA_Correlator<FloatC> &corr, PLEGMA_Propagator<FloatA>& prop1, PLEGMA_Propagator<FloatB>& prop2, int signProps, PLEGMA_Gauge<FloatG>& gauge, std::vector<GAMMAS>& gammas, bool isZfac);


template<typename FloatC,typename FloatA, typename FloatB, typename FloatG>
void threep_twoD(PLEGMA_Correlator<FloatC> &corr, PLEGMA_Propagator<FloatA>& prop1, PLEGMA_Propagator<FloatB>& prop2, int signProps, PLEGMA_Gauge<FloatG>& gauge, std::vector<GAMMAS>& gammas, bool isZfac);


template<typename FloatC,typename FloatA, typename FloatB, typename FloatG>
void threep_threeD_part1(PLEGMA_Correlator<FloatC> &corr, PLEGMA_Propagator<FloatA>& prop1, PLEGMA_Propagator<FloatB>& prop2, int signProps, PLEGMA_Gauge<FloatG>& gauge, std::vector<GAMMAS>& gammas, bool isZfac);

template<typename FloatC,typename FloatA, typename FloatB, typename FloatG>
void threep_threeD_part2(PLEGMA_Correlator<FloatC> &corr, PLEGMA_Propagator<FloatA>& prop1, PLEGMA_Propagator<FloatB>& prop2, int signProps, PLEGMA_Gauge<FloatG>& gauge, std::vector<GAMMAS>& gammas, bool isZfac);

template<typename FloatC,typename FloatA, typename FloatB, typename FloatG>
void threep_threeD_part3(PLEGMA_Correlator<FloatC> &corr, PLEGMA_Propagator<FloatA>& prop1, PLEGMA_Propagator<FloatB>& prop2, int signProps, PLEGMA_Gauge<FloatG>& gauge, std::vector<GAMMAS>& gammas, bool isZfac);

template<typename FloatC,typename FloatA, typename FloatB, typename FloatG>
void threep_threeD_part4(PLEGMA_Correlator<FloatC> &corr, PLEGMA_Propagator<FloatA>& prop1, PLEGMA_Propagator<FloatB>& prop2, int signProps, PLEGMA_Gauge<FloatG>& gauge, std::vector<GAMMAS>& gammas, bool isZfac);



template<typename FloatC,typename FloatA, typename FloatB, typename FloatG>
void threep_wilsonLine(PLEGMA_Correlator<FloatC> &corr, PLEGMA_Propagator<FloatA>& prop1, PLEGMA_Propagator<FloatB>& prop2, int signProps, PLEGMA_Su3field<FloatG>& gauge, std::vector<GAMMAS>& gammas);


