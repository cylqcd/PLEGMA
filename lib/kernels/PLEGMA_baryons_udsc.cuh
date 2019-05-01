#pragma once
#include <PLEGMA_Correlator.h>
#include <PLEGMA_baryons_udsc_arrays.cuh>

template<typename FloatA, typename FloatC>
void contract_baryons_udsc(propTex<FloatA> texPropUP, propTex<FloatA> texPropDN,
			   propTex<FloatA> texPropST, propTex<FloatA> texPropCH,
			   PLEGMA_Correlator<FloatC> &corr, int it, std::vector<int> &todo);

