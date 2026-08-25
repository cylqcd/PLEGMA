#pragma once

template<typename FloatC, typename FloatA>
void contractNucleonDeltaSeqSource(vector2<FloatC> vec, propTex<FloatA>& prop1,  GAMMAS_SCATT gamma_source, GAMMAS_SCATT gamma_sink, WHICHPROJECTOR proj, WHICHPARTICLE particle, int c_nu, int c_c2, int c_sigma);

template<typename FloatC, typename FloatA, typename FloatB>
void contractNucleonDeltaSeqSource(vector2<FloatC> vec, propTex<FloatA>& prop1, propTex<FloatB>& prop2,  GAMMAS_SCATT gamma_source, GAMMAS_SCATT gamma_sink, WHICHPROJECTOR proj, WHICHPARTICLE particle, int c_nu, int c_c2, int c_sigma);
