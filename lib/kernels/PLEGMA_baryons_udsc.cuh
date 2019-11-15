#pragma once
#include <PLEGMA_Propagator.h>
#include <PLEGMA_Correlator.h>
#include <PLEGMA_baryons_udsc_arrays.cuh>

template<typename FloatA, typename FloatC>
void contract_baryons_udsc(PLEGMA_Propagator<FloatA>& propUP, PLEGMA_Propagator<FloatA>& propDN,
			   PLEGMA_Propagator<FloatA>& propST, PLEGMA_Propagator<FloatA>& propCH,
			   PLEGMA_Correlator<FloatC> &corr, std::vector<int> &todo);

