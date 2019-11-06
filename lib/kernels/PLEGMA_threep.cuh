template<typename FloatC,typename FloatA, typename FloatB>
void threep_local(PLEGMA_Correlator<FloatC> &corr, propTex<FloatA> prop1, propTex<FloatB> prop2, int signProps, std::vector<GAMMAS> gammas);


template<typename FloatC,typename FloatA, typename FloatB, typename FloatG>
void threep_noe(PLEGMA_Correlator<FloatC> &corr, propTex<FloatA> prop1, propTex<FloatB> prop2, int signProps, gaugeTex<FloatG> gauge);


template<typename FloatC,typename FloatA, typename FloatB, typename FloatG>
void threep_oneD(PLEGMA_Correlator<FloatC> &corr, propTex<FloatA> prop1, propTex<FloatB> prop2, int signProps, gaugeTex<FloatG> gauge, std::vector<GAMMAS> gammas);


template<typename FloatC,typename FloatA, typename FloatB, typename FloatG>
void threep_wilsonLine(PLEGMA_Correlator<FloatC> &corr, propTex<FloatA> prop1, propTex<FloatB> prop2, int signProps, su3Tex<FloatG> gauge, std::vector<GAMMAS> gammas);


