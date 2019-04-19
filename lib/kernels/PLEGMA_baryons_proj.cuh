#pragma once

#include <PLEGMA_baryons_proj_arrays.cuh>

template<typename FloatA, typename FloatC>
void contract_baryons_proj(propTex<FloatA> texPropUP, propTex<FloatA> texPropDN,
			   propTex<FloatA> texPropST, propTex<FloatA> texPropCH,
			   PLEGMA_Correlator<FloatC> &corr, int it, std::vector<int> &todo);

