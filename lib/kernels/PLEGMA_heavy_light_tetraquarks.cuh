#pragma once
#include <PLEGMA_Propagator.h>
#include <PLEGMA_Correlator.h>
#include "PLEGMA_heavy_light_tetraquarks_arrays.cuh"
#include "PLEGMA_heavy_light_tetraquarks_arrays_stoch.cuh"

template<typename FloatA, typename FloatC>
void contract_tetraquarks(PLEGMA_Propagator<FloatA>& propLT, PLEGMA_Propagator<FloatA>& propST, 
                          PLEGMA_Propagator<FloatA>& propCH, PLEGMA_Propagator<FloatA>& propBT,
                          PLEGMA_Correlator<FloatC> &corr, std::vector<int> &todo);
			   
template<typename FloatC,typename FloatA, typename FloatB>
void contract_tetraquark_scattering_open_index(PLEGMA_Correlator<FloatC> &corr, PLEGMA_Propagator<FloatA>& prop1, 
PLEGMA_Propagator<FloatB>& prop2, std::vector<GAMMAS>& gammas, int s1);


template<typename FloatA, typename FloatC>
void contract_tetraquarks_stochastic(PLEGMA_Propagator<FloatA>&propLT1, 
                                     PLEGMA_Propagator<FloatA>&propLT2, 
                                     PLEGMA_Propagator<FloatA>&propST1, 
                                     PLEGMA_Propagator<FloatA>&propST2,
                                     PLEGMA_Propagator<FloatA>&propBT1, 
                                     PLEGMA_Propagator<FloatA>&propBT2,
                                     PLEGMA_Correlator<FloatC> &corr, 
                                     std::vector<int> &todo);


