#pragma once

template<typename FloatC, typename FloatA>
void contractNucleonDeltaSeqSource(vector2<FloatC> vec, propTex<FloatA>& prop1, WHICHPROJECTOR proj, WHICHPARTICLE particle, int c_nu, int c_c2, int c_sigma);

template<typename FloatC, typename FloatA, typename FloatB>
void contractNucleonDeltaSeqSource(vector2<FloatC> vec, propTex<FloatA>& prop1, propTex<FloatB>& prop2, WHICHPROJECTOR proj, WHICHPARTICLE particle, int c_nu, int c_c2, int c_sigma);
