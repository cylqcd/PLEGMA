#pragma once

template<typename FloatC, typename FloatA>
void contractNucleonSeqSource(vector2<FloatC> vec, propTex<FloatA>& prop1, WHICHPROJECTOR proj, WHICHPARTICLE particle, int c_nu, int c_c2);

template<typename FloatC, typename FloatA, typename FloatB>
void contractNucleonSeqSource(vector2<FloatC> vec, propTex<FloatA>& prop1, propTex<FloatB>& prop2, WHICHPROJECTOR proj, WHICHPARTICLE particle, int c_nu, int c_c2);
