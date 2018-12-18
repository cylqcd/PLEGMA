#pragma once

template<typename FloatA, typename FloatC>
void contract_baryons_1o2(propTex<FloatA> texPropUP, propTex<FloatA> texPropDN,
			  propTex<FloatA> texPropST, propTex<FloatA> texPropCH,
			  PLEGMA_Correlator<FloatC> &corr, int it);

template<typename FloatA, typename FloatC>
void contract_baryons_3o2(propTex<FloatA> texPropUP, propTex<FloatA> texPropDN,
			  propTex<FloatA> texPropST, propTex<FloatA> texPropCH,
			  PLEGMA_Correlator<FloatC> &corr, int it);
