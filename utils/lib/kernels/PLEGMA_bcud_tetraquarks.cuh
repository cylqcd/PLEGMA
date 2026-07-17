#pragma once
#include <PLEGMA_Propagator.h>
#include <PLEGMA_Correlator.h>
#include <PLEGMA_bcud_tetraquarks_arrays.cuh>
#include <PLEGMA_bcud_tetraquarks_arrays_stoch.cuh>

template<typename FloatA, typename FloatC>
void contract_tetraquarks_bcud(PLEGMA_Propagator<FloatA>& propLT, PLEGMA_Propagator<FloatA>& propST, 
                          PLEGMA_Propagator<FloatA>& propCH, PLEGMA_Propagator<FloatA>& propBT,
                          PLEGMA_Correlator<FloatC> &corr, std::vector<int> &todo);


// The scattring part is independent of the array files, so here is no additional function required.
// The scattering computation is done via contract_tetraquark_scattering_open_index in PLEGMA_heavy_light_tetraquarks_open_contractions.cuh
/*template<typename FloatC,typename FloatA, typename FloatB>
void contract_tetraquark_scattering_open_index(PLEGMA_Correlator<FloatC> &corr, PLEGMA_Propagator<FloatA>& prop1, 
PLEGMA_Propagator<FloatB>& prop2, std::vector<GAMMAS>& gammas, int s1);*/

template<typename FloatA, typename FloatC>
void contract_tetraquarks_bcud_stochastic(  PLEGMA_Propagator<FloatA>&propLT1, PLEGMA_Propagator<FloatA>&propLT2, 
                                            PLEGMA_Propagator<FloatA>&propST1, PLEGMA_Propagator<FloatA>&propST2,
                                            PLEGMA_Propagator<FloatA>&propCH1, PLEGMA_Propagator<FloatA>&propCH2,
                                            PLEGMA_Propagator<FloatA>&propBT1, PLEGMA_Propagator<FloatA>&propBT2,
                                            PLEGMA_Correlator<FloatC> &corr, std::vector<int> &todo);

