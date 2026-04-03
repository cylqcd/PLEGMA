#pragma once
#include <PLEGMA_Propagator.h>

template<bool b,typename FloatC,typename FloatA, typename FloatB>
void threep_local(PLEGMA_Correlator<FloatC> &corr,  typename std::conditional<b==true, PLEGMA_Propagator<FloatA>&, PLEGMA_Vector<FloatA>&>::type prop1,
                 typename std::conditional<b==true, PLEGMA_Propagator<FloatB>&, PLEGMA_Vector<FloatB>&>::type prop2,
 int signProps, std::vector<GAMMAS>& gammas, bool isZfac);


template<typename FloatC,typename FloatA, typename FloatB, typename FloatG>
void threep_noe(PLEGMA_Correlator<FloatC> &corr, PLEGMA_Propagator<FloatA>& prop1, PLEGMA_Propagator<FloatB>& prop2, int signProps, PLEGMA_Gauge<FloatG>& gauge);


template<typename FloatC,typename FloatA, typename FloatB, typename FloatG>
	void threep_oneD(PLEGMA_Correlator<FloatC> &corr, PLEGMA_Vector<FloatA>& vec1, PLEGMA_Vector<FloatB>& vec2, int signProps, PLEGMA_Gauge<FloatG>& gauge, std::vector<GAMMAS>& gammas);

template<bool b,typename FloatC,typename FloatA, typename FloatB, typename FloatG>
void threep_oneD(PLEGMA_Correlator<FloatC> &corr,  typename std::conditional<b==true, PLEGMA_Propagator<FloatA>&, PLEGMA_Vector<FloatA>&>::type prop1,
                 typename std::conditional<b==true, PLEGMA_Propagator<FloatB>&, PLEGMA_Vector<FloatB>&>::type prop2,
int signProps, PLEGMA_Gauge<FloatG>& gauge, std::vector<GAMMAS>& gammas, bool isZfac);


template<bool b,typename FloatC,typename FloatA, typename FloatB, typename FloatG>
void threep_twoD(PLEGMA_Correlator<FloatC> &corr,  typename std::conditional<b==true, PLEGMA_Propagator<FloatA>&, PLEGMA_Vector<FloatA>&>::type prop1,
                 typename std::conditional<b==true, PLEGMA_Propagator<FloatB>&, PLEGMA_Vector<FloatB>&>::type prop2,
int signProps, PLEGMA_Gauge<FloatG>& gauge, std::vector<GAMMAS>& gammas, bool isZfac);


template<bool b,typename FloatC,typename FloatA, typename FloatB, typename FloatG>
void threep_threeD_part1(PLEGMA_Correlator<FloatC> &corr, typename std::conditional<b==true, PLEGMA_Propagator<FloatA>&, PLEGMA_Vector<FloatA>&>::type prop1,
                 typename std::conditional<b==true, PLEGMA_Propagator<FloatB>&, PLEGMA_Vector<FloatB>&>::type prop2,
int signProps, PLEGMA_Gauge<FloatG>& gauge, std::vector<GAMMAS>& gammas, bool isZfac);

template<bool b,typename FloatC,typename FloatA, typename FloatB, typename FloatG>
void threep_threeD_part2(PLEGMA_Correlator<FloatC> &corr, typename std::conditional<b==true, PLEGMA_Propagator<FloatA>&, PLEGMA_Vector<FloatA>&>::type prop1,
                 typename std::conditional<b==true, PLEGMA_Propagator<FloatB>&, PLEGMA_Vector<FloatB>&>::type prop2,
int signProps, PLEGMA_Gauge<FloatG>& gauge, std::vector<GAMMAS>& gammas, bool isZfac);

template<bool b,typename FloatC,typename FloatA, typename FloatB, typename FloatG>
void threep_threeD_part3(PLEGMA_Correlator<FloatC> &corr, typename std::conditional<b==true, PLEGMA_Propagator<FloatA>&, PLEGMA_Vector<FloatA>&>::type prop1,
                 typename std::conditional<b==true, PLEGMA_Propagator<FloatB>&, PLEGMA_Vector<FloatB>&>::type prop2,
 int signProps, PLEGMA_Gauge<FloatG>& gauge, std::vector<GAMMAS>& gammas, bool isZfac);

template<bool b,typename FloatC,typename FloatA, typename FloatB, typename FloatG>
void threep_threeD_part4(PLEGMA_Correlator<FloatC> &corr, typename std::conditional<b==true, PLEGMA_Propagator<FloatA>&, PLEGMA_Vector<FloatA>&>::type prop1,
                 typename std::conditional<b==true, PLEGMA_Propagator<FloatB>&, PLEGMA_Vector<FloatB>&>::type prop2,
int signProps, PLEGMA_Gauge<FloatG>& gauge, std::vector<GAMMAS>& gammas, bool isZfac);

template<typename FloatC,typename FloatA, typename FloatB, typename FloatG>
void threep_wilsonLine(PLEGMA_Correlator<FloatC> &corr, PLEGMA_Propagator<FloatA>& prop1, PLEGMA_Propagator<FloatB>& prop2, int signProps, PLEGMA_Su3field<FloatG>& gauge, std::vector<GAMMAS>& gammas);

template<typename FloatC,typename FloatA, typename FloatB, typename FloatG>
void threep_staple(PLEGMA_Correlator<FloatC> &corr, PLEGMA_Propagator<FloatA>& prop1, PLEGMA_Propagator<FloatB>& prop2, int signProps, PLEGMA_Su3field<FloatG>& gauge, std::vector<GAMMAS>& gammas, bool isZfac);

template<typename Float>
void threep_qgq(PLEGMA_Correlator<Float> &corr,
		PLEGMA_Propagator<Float>& prop1, PLEGMA_Propagator<Float>& prop2,
		int signProps, PLEGMA_Su3field<Float>& su3_l,
		PLEGMA_Fmunu<Float> &Fmunu, std::pair<int,int> munu,
		PLEGMA_Su3field<Float>& su3_r,
		std::vector<GAMMAS>& gammas);
