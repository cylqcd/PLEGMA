#include <PLEGMA.h>
#include <PLEGMA_utils.h>
std::vector<double> runtime;
#define TIME(fnc)                                                              \
    runtime.push_back(MPI_Wtime());                                            \
    fnc;                                                                       \
    PLEGMA_printf("TIME for " #fnc " %f sec\n", MPI_Wtime() - runtime.back()); \
    runtime.pop_back()

extern int device;
static std::vector<std::string> listOpt = {"verbosity", "load-gauge", "nsrc", "src-filename", "momlisttwopt-filename", "momlistthreept-filename", "nsmear-APE", "alpha-APE", "nsmear-gauss", "alpha-gauss", "tSinks"};

int main(int argc, char **argv)
{
    /******************************************************
     *
     *   Initialiazation
     *
     ******************************************************/
    initializeOptions(argc, argv, true, listOpt);

    int confnumber_int;
    std::string outdiagramPrefix, flagFinish;
    HGC_options->set("flagFinish", "An empty file created indicating the completion of a run", verbosity, flagFinish);
    HGC_options->set("confnumber", "Integer determining the index of the gauge configuration", verbosity, confnumber_int);
    HGC_options->set("outdiagramPrefix", "Prefix of the resulting diagrams", verbosity, outdiagramPrefix);

    int seed_stoc;
    HGC_options->set("seed_stoc", "Seed for stoc", verbosity, seed_stoc);
    // HGC_options->set("num_stoc", "Num of stoc", verbosity, num_stoc);

    initializePLEGMA();

    {
        plegma::PLEGMA_Gauge<double> gauge, smearedGauge;
        gauge.readFile(latfile, LIME_FORMAT);
        gauge.load();
        gauge.calculatePlaq();
        initGaugeQuda(gauge, true);
        plaqQuda();

        smearedGauge.APEsmearing(gauge, nsmearAPE, alphaAPE, 3);
        smearedGauge.calculatePlaq();

        updateOptions(LIGHT);
        quda::QUDA_solver solver(mu);

#if 1 // Utilities

#if 1 // general
        bool smearQ = true;

        enum FlavorYan
        {
            u,
            d
        };
        enum SmearFlagYan
        {
            SS,
            SL,
            LS,
            LL,
            single2double,
            SB,
            LB
        };
#endif

#if 1 // solve
        auto solve0 = [&](plegma::PLEGMA_Vector<double> &out, plegma::PLEGMA_Vector<double> &in, FlavorYan f)
        {
            FlavorYan f2 = (mu > 0 ? u : d);
            double run_mu = (f == f2 ? mu : -mu);
            plegma::PLEGMA_Vector<double> auxVector;

            auxVector.rotateToPhysicalBasis(in, run_mu / std::abs(run_mu));
            double norm = auxVector.norm();
            auxVector.scale(1 / norm);
            if (f != f2)
            {
                mu = run_mu;
                solver.UpdateSolver();
            }
            solver.solve(auxVector, auxVector);
            auxVector.scale(norm);
            out.rotateToPhysicalBasis(auxVector, run_mu / std::abs(run_mu));
        };
        auto solve1 = [&](plegma::PLEGMA_Vector<double> &out, plegma::PLEGMA_Vector3D<double> &in, int inTime, FlavorYan f, SmearFlagYan s)
        {
            assert(s < single2double);

            if (!smearQ)
            {
                plegma::PLEGMA_Vector<double> auxVector;
                auxVector.absorb(in, inTime);
                solve0(out, auxVector, f);
                return;
            }

            plegma::PLEGMA_Vector<double> auxVector;
            plegma::PLEGMA_Vector3D<double> auxVector3D;

            if (s == SL || s == SS)
            {
                plegma::PLEGMA_Gauge3D<double> smearedGauge3D;
                smearedGauge3D.absorb(smearedGauge, inTime);
                auxVector3D.gaussianSmearing(in, smearedGauge3D, nsmearGauss, alphaGauss);
            }
            else
                auxVector3D.copy(in);

            auxVector.absorb(auxVector3D, inTime);
            solve0(auxVector, auxVector, f);

            if (s == LS || s == SS)
            {
                out.gaussianSmearing(auxVector, smearedGauge, nsmearGauss, alphaGauss);
            }
            else
                out.copy(auxVector);
        };
        auto solve2 = [&](plegma::PLEGMA_Vector<double> &outS, plegma::PLEGMA_Vector<double> &outL, plegma::PLEGMA_Vector3D<double> &in, int inTime, FlavorYan f, SmearFlagYan s)
        {
            assert(single2double < s);
            if (!smearQ)
            {
                plegma::PLEGMA_Vector<double> auxVector;
                auxVector.absorb(in, inTime);
                solve0(outL, auxVector, f);
                outS.copy(outL);
                return;
            }

            plegma::PLEGMA_Vector<double> auxVector;

            if (s == SB)
            {
                plegma::PLEGMA_Gauge3D<double> smearedGauge3D;
                plegma::PLEGMA_Vector3D<double> auxVector3D;
                smearedGauge3D.absorb(smearedGauge, inTime);
                auxVector3D.gaussianSmearing(in, smearedGauge3D, nsmearGauss, alphaGauss);
                auxVector.absorb(auxVector3D, inTime);
            }
            else
                auxVector.absorb(in, inTime);

            solve0(auxVector, auxVector, f);

            outS.gaussianSmearing(auxVector, smearedGauge, nsmearGauss, alphaGauss);
            outL.copy(auxVector);
        };
#endif

#if 1 // setupPointPropagator
        auto setupPointPropagator1 = [&](plegma::PLEGMA_Propagator<float> &out, plegma::site &source, FlavorYan f, SmearFlagYan s)
        {
            assert(s < single2double);

            plegma::PLEGMA_Vector<double> auxVector;
            plegma::PLEGMA_Vector<float> auxVectorF;
            plegma::PLEGMA_Vector3D<double> auxVector3D;
            for (int isc = 0; isc < 12; isc++)
            {
                auxVector3D.pointSource(source, isc / 3, isc % 3, DEVICE);
                solve1(auxVector, auxVector3D, source[DIM_T], f, s);
                auxVectorF.copy(auxVector);
                out.absorb(auxVectorF, isc / 3, isc % 3);
            }
        };
        auto setupPointPropagator2 = [&](plegma::PLEGMA_Propagator<float> &outS, plegma::PLEGMA_Propagator<float> &outL, plegma::site &source, FlavorYan f, SmearFlagYan s)
        {
            assert(single2double < s);
            plegma::PLEGMA_Vector<double> auxVectorS, auxVectorL;
            plegma::PLEGMA_Vector<float> auxVectorSF, auxVectorLF;
            plegma::PLEGMA_Vector3D<double> auxVector3D;

            for (int isc = 0; isc < 12; isc++)
            {
                auxVector3D.pointSource(source, isc / 3, isc % 3, DEVICE);
                solve2(auxVectorS, auxVectorL, auxVector3D, source[DIM_T], f, s);
                auxVectorSF.copy(auxVectorS);
                auxVectorLF.copy(auxVectorL);
                outS.absorb(auxVectorSF, isc / 3, isc % 3);
                outL.absorb(auxVectorLF, isc / 3, isc % 3);
            }
        };
#endif

#if 0 // setupSequentialPropagator
        auto setupSequentialPropagator1 = [&](plegma::PLEGMA_Propagator<float> &out, plegma::PLEGMA_Propagator<float> &in, int inTime, plegma::GAMMAS Gamma, std::vector<int> mom, int momSign, FlavorYan f, SmearFlagYan s)
        {
            assert(s < single2double);

            plegma::PLEGMA_Vector<double> auxVectorD;
            plegma::PLEGMA_Vector<float> auxVectorF;
            plegma::PLEGMA_Vector3D<double> auxVector3DD;
            plegma::PLEGMA_Vector3D<float> auxVector3DF;

            for (int isc = 0; isc < 12; isc++)
            {
                auxVector3DF.absorb(in, inTime, isc / 3, isc % 3);
                auxVector3DF.apply_gamma(Gamma);
                auxVector3DF.mulMomentumPhases(mom, momSign);
                auxVector3DD.copy(auxVector3DF);
                solve1(auxVectorD, auxVector3DD, inTime, f, s);
                auxVectorF.copy(auxVectorD);
                out.absorb(auxVectorF, isc / 3, isc % 3);
            }
        };
        auto setupSequentialPropagator2 = [&](plegma::PLEGMA_Propagator<float> &outS, plegma::PLEGMA_Propagator<float> &outL, plegma::PLEGMA_Propagator<float> &in, int inTime, plegma::GAMMAS Gamma, std::vector<int> mom, int momSign, FlavorYan f, SmearFlagYan s)
        {
            assert(s > single2double);

            plegma::PLEGMA_Vector<double> auxVectorDS, auxVectorDL;
            plegma::PLEGMA_Vector<float> auxVectorFS, auxVectorFL;
            plegma::PLEGMA_Vector3D<double> auxVector3DD;
            plegma::PLEGMA_Vector3D<float> auxVector3DF;

            for (int isc = 0; isc < 12; isc++)
            {
                auxVector3DF.absorb(in, inTime, isc / 3, isc % 3);
                // auxVector3DF.apply_gamma_scatt(G_5);
                auxVector3DF.apply_gamma(Gamma);
                auxVector3DF.mulMomentumPhases(mom, momSign);
                auxVector3DD.copy(auxVector3DF);
                solve2(auxVectorDS, auxVectorDL, auxVector3DD, inTime, f, s);
                auxVectorFS.copy(auxVectorDS);
                auxVectorFL.copy(auxVectorDL);
                outS.absorb(auxVectorFS, isc / 3, isc % 3);
                outL.absorb(auxVectorFL, isc / 3, isc % 3);
            }
        };
#endif

#if 1 // pack
        auto packVector = [&](plegma::PLEGMA_Vector<float> &out, plegma::PLEGMA_Vector<float> &in, int timeSlice)
        {
            PLEGMA_Vector3D<float> auxVector3D;
            auxVector3D.absorb(in, timeSlice, true);

            for (int dt = 0; dt < HGC_totalL[3]; dt++)
            {
                out.absorb(auxVector3D, dt, false);
            }
        };
        auto packVectorD = [&](plegma::PLEGMA_Vector<double> &out, plegma::PLEGMA_Vector<double> &in, int timeSlice)
        {
            PLEGMA_Vector3D<double> auxVector3D;
            auxVector3D.absorb(in, timeSlice, true);

            for (int dt = 0; dt < HGC_totalL[3]; dt++)
            {
                out.absorb(auxVector3D, dt, false);
            }
        };
        auto packPropagator = [&](plegma::PLEGMA_Propagator<float> &out, plegma::PLEGMA_Propagator<float> &in, int timeSlice)
        {
            PLEGMA_Vector<float> auxVector;
            for (int isc = 0; isc < 12; isc++)
            {
                auxVector.absorb(in, isc / 3, isc % 3);
                packVector(auxVector, auxVector, timeSlice);
                out.absorb(auxVector, isc / 3, isc % 3);
            }
        };
#endif

#if 1 // phiV3
        auto phiV3 = [&](plegma::PLEGMA_Propagator<float> &out, plegma::PLEGMA_Vector<float> &phi, plegma::PLEGMA_ScattCorrelator<float> &V3)
        {
            for (int isc = 0; isc < 12; isc++)
            {
                plegma::PLEGMA_Vector<float> aux;
                for (int dt = 0; dt < HGC_totalL[3]; dt++)
                {
                    plegma::PLEGMA_Vector3D<float> aux3D;
                    aux3D.absorb(phi,dt);
                    float *auxFloat = V3.Corr(dt, 0, 0);
                    std::complex<float> auxC(auxFloat[2 * isc + 0], auxFloat[2 * isc + 1]);
                    aux3D.cscale(auxC);
                    aux.absorb(aux3D,dt,false);
                }
                out.absorb(aux, isc / 3, isc % 3);
            }
        };
#endif

#endif

        /******************************************************
         *
         *   General input
         *
         ******************************************************/
        // Get the confnumber for latfile
        char *ssource;
        asprintf(&ssource, "%04d", confnumber_int);
        std::string confnumber = ssource;
        free(ssource);

        std::vector<plegma::GAMMAS_SCATT> gscatts_ID = {ID};
        std::vector<plegma::GAMMAS_SCATT> gscatts_c = {ID, G_1, G_2, G_3, G_4, G_5, G_5_G_1, G_5_G_2, G_5_G_3, G_5_G_4, S_23, S_13, S_12, S_41, S_42, S_43}; //  S_ij=-i*[gi,gj]/2; (+i) required to make [gi,gj]/2 from S_jk ; (-1) required to make S_31 from S_13;
        // std::vector<plegma::GAMMAS> gammas_c = {ONE, G1, G2, G3, G4, G5, G5G1, G5G2, G5G3, G5G4};

        std::vector<plegma::GAMMAS_SCATT> gscatts_i1 = {CG_5};
        std::vector<plegma::GAMMAS_SCATT> gscatts_f1 = {CG_5};
        std::vector<plegma::GAMMAS_SCATT> gscatts_i2 = {G_5};
        std::vector<plegma::GAMMAS_SCATT> gscatts_f2 = {G_5};

        PLEGMA_printf("###Momentum list read from : %s", pathListMomenta_twopt.c_str());
        plegma::momList momList2pt(3, pathListMomenta_twopt, {1, 2}); // pi2, pf1, pf2
        PLEGMA_printf("N momenta in sourcemomentumList: %d\n", momList2pt.size());
        if (momList2pt.empty())
            PLEGMA_error("threept momentumList empty");
        auto momVects2pt_pi2 = momList2pt.uniq_p(0);
        plegma::momList momList2pt_pi2(1, {momVects2pt_pi2}, {0});

        PLEGMA_printf("###Momentum list read from : %s", pathListMomenta_threept.c_str());
        plegma::momList momList3pt(4, pathListMomenta_threept, {1, 2, 3}); // pi2, pf1, pf2, pc
        PLEGMA_printf("N momenta in sourcemomentumList: %d\n", momList3pt.size());
        if (momList3pt.empty())
            PLEGMA_error("threept momentumList empty");
        auto momVects3pt_pf1 = momList3pt.uniq_p(1);
        auto momVects3pt_pi2 = momList3pt.uniq_p(0);
        auto momVects3pt_pf2 = momList3pt.uniq_p(2);
        auto momVects3pt_pc = momList3pt.uniq_p(3);
        plegma::momList momList3pt_pc(1, {momVects3pt_pc}, {0});
        plegma::momList momList_0(1, {{{0,0,0}}}, {0});
        std::vector<std::vector<int>> momVects_0 = {{0,0,0}};

        /******************************************************
         *
         *   Main
         *
         ******************************************************/
        for(int i_src = 0; i_src < numSourcePositions; i_src++)
        {
            auto src = sourcePositions[i_src];
            struct plegma::site src_zero({0, 0, 0, src[DIM_T]});

            asprintf(&ssource,"sx%02dsy%02dsz%02dst%03d", src[0], src[1], src[2], src[3]);
            std::string sourcepositiontext= (std::string)"_" + ssource;
            std::string outfilename = outdiagramPrefix + "_Bm_" + confnumber + sourcepositiontext;

            int ti,tf;
            ti=src[DIM_T];

            plegma::PLEGMA_Propagator<float> propUSS, propUSL;
            plegma::PLEGMA_Propagator<float> propDSS, propDSL;

            setupPointPropagator2(propUSS, propUSL, src, u, SB);
            setupPointPropagator2(propDSS, propDSL, src, d, SB);

            plegma::PLEGMA_Vector<double> stoc, stocXi, stocEta;
            stoc.randInit(seed_stoc); 
            stoc.stochastic_Z(4); stocXi.copy(stoc);
            stoc.stochastic_Z(4); stocEta.copy(stoc);

            

            for (int i_pi2 = 0; i_pi2 < momVects3pt_pi2.size(); ++i_pi2)
            {
                auto &mom_pi2 = momVects3pt_pi2[i_pi2];
                std::string str_pi2 = "pi2=" + std::to_string(mom_pi2[0]) + "_" + std::to_string(mom_pi2[1]) + "_" + std::to_string(mom_pi2[2]);

                plegma::PLEGMA_Vector<double> stoc_Uci_Gi2_Eta;
                {
                    plegma::PLEGMA_Vector3D<double> aux;
                    aux.absorb(stocEta, ti);
                    aux.apply_gamma_scatt(gscatts_i2[0]);
                    aux.mulMomentumPhases(mom_pi2, +1);
                    solve1(stoc_Uci_Gi2_Eta, aux, ti, u, SL);
                }

                for (int i_tfi=0; i_tfi< tSinks.size(); ++i_tfi)
                {
                    std::string str_dt = "dt=" + std::to_string(tSinks[i_tfi]);
                    tf=(ti+tSinks[i_tfi])% HGC_totalL[3];

                    for (int i_pf2 = 0; i_pf2 < momVects3pt_pf2.size(); ++i_pf2)
                    {
                        auto &mom_pf2 = momVects3pt_pf2[i_pf2];
                        std::string str_pf2 = "pf2=" + std::to_string(mom_pf2[0]) + "_" + std::to_string(mom_pf2[1]) + "_" + std::to_string(mom_pf2[2]);

                        plegma::PLEGMA_Vector<double> stoc_g5_Dcf_g5_Gf2dgr_Xi;
                        {
                            plegma::PLEGMA_Vector3D<double> aux;
                            aux.absorb(stocXi, tf);
                            aux.apply_gamma_scatt(gscatts_f2[0]); // Should be \dagger of G_f2
                            aux.mulMomentumPhases(mom_pf2, +1);
                            aux.apply_gamma5();
                            solve1(stoc_g5_Dcf_g5_Gf2dgr_Xi, aux, tf, d, SL);
                            stoc_g5_Dcf_g5_Gf2dgr_Xi.apply_gamma5();
                        }

                        plegma::PLEGMA_ScattCorrelator<float> auxPhiPhi(src_zero, momList3pt_pc);
                        auxPhiPhi.initialize_diagram(gscatts_c, "");
                        {
                            plegma::PLEGMA_Vector<float> aux1, aux2;
                            aux1.copy(stoc_Uci_Gi2_Eta);
                            aux2.copy(stoc_g5_Dcf_g5_Gf2dgr_Xi); 
                            auxPhiPhi.PhiPhi(aux1,gscatts_c,aux2,true);
                        }
                        auxPhiPhi.setGroups(std::vector<std::string>{"PhiPhi"});
                        auxPhiPhi.setDatasets(std::vector<std::string>{str_pi2 + "/" + str_pf2 + "/" + str_dt});
                        auxPhiPhi.writeHDF5(outfilename);

                    } 
                }
            }

            plegma::PLEGMA_ScattCorrelator<float> auxV3(src_zero, momList_0);
            {
                plegma::PLEGMA_Vector<float> aux, aux2;
                plegma::PLEGMA_Propagator<float> auxProp;
                aux2.copy(stocEta);
                packVector(aux, aux2, ti);
                packPropagator(auxProp, propUSS, ti);
                auxV3.V3(aux, gscatts_ID, auxProp);

                auxV3.setGroups(std::vector<std::string>{"V3"});
                auxV3.setDatasets(std::vector<std::string>{"data"});
                auxV3.writeHDF5(outfilename);
            }

            for (int i_tfi=0; i_tfi< tSinks.size(); ++i_tfi)
            {
                std::string str_dt = "dt=" + std::to_string(tSinks[i_tfi]);
                tf=(ti+tSinks[i_tfi])% HGC_totalL[3];

                plegma::PLEGMA_Propagator<float> propUSS_tf, propDSS_tf;
                packPropagator(propUSS_tf, propUSS, tf);
                packPropagator(propDSS_tf, propDSS, tf);

                plegma::PLEGMA_Vector<double> stoc_Uff_Xi;
                {
                    plegma::PLEGMA_Vector3D<double> aux;
                    aux.absorb(stocXi,tf);
                    solve1(stoc_Uff_Xi, aux, tf, u, SS);
                }
                plegma::PLEGMA_Propagator<float> prop_phiV3_tf;
                {
                    plegma::PLEGMA_Propagator<float> prop_phiV3;
                    plegma::PLEGMA_Vector<float> aux;
                    aux.copy(stoc_Uff_Xi);
                    phiV3(prop_phiV3, aux, auxV3);
                    packPropagator(prop_phiV3_tf, prop_phiV3, tf);
                }

                {
                    plegma::PLEGMA_ScattCorrelator<float> sc(src_zero, momVects3pt_pf1);
                    plegma::PLEGMA_Vector<float> aux;
                    aux.copy(stoc_Uff_Xi);
                    sc.V2(aux, gscatts_f1, propDSS_tf, propUSS_tf); 
                    sc.setGroups(std::vector<std::string>{"V24"});
                    sc.setDatasets(std::vector<std::string>{"V2/" + str_dt});
                    sc.writeHDF5(outfilename);
                }
                {
                    plegma::PLEGMA_ScattCorrelator<float> sc(src_zero, momVects3pt_pf1);
                    plegma::PLEGMA_Vector<float> aux;
                    aux.copy(stoc_Uff_Xi);
                    sc.V4(aux, gscatts_f1, propUSS_tf, propDSS_tf); 
                    sc.setGroups(std::vector<std::string>{"V24"});
                    sc.setDatasets(std::vector<std::string>{"V4/" + str_dt});
                    sc.writeHDF5(outfilename);
                }

                auto outputfile = [&](plegma::PLEGMA_ScattCorrelator<float> sc, std::string name, float factor)
                {
                    float temp_sgn[2] = {factor, 0.};
                    x_e_cx<float>(sc.H_elem(), temp_sgn, sc.getTotalSize());

                    if (tf < ti)
                    {
                        float overall_sign[2] = {-1., 0.};
                        x_e_cx<float>(sc.H_elem(), overall_sign, sc.getTotalSize());
                    }
                    
                    sc.setGroups(std::vector<std::string>{"Bm"});
                    sc.setDatasets(std::vector<std::string>{name + "/" + str_dt});
                    sc.writeHDF5(outfilename);
                };
                {
                    plegma::PLEGMA_ScattCorrelator<float> aux(src_zero, momVects3pt_pf1);
                    aux.T1(gscatts_i1, gscatts_f1, propUSS_tf, propDSS_tf, prop_phiV3_tf);
                    outputfile(aux, "1", 1);
                }
                {
                    plegma::PLEGMA_ScattCorrelator<float> aux(src_zero, momVects3pt_pf1);
                    aux.T2(gscatts_i1, gscatts_f1, propUSS_tf, propDSS_tf, prop_phiV3_tf);
                    outputfile(aux, "2", 1);
                }
                {
                    plegma::PLEGMA_ScattCorrelator<float> aux(src_zero, momVects3pt_pf1);
                    aux.T2(gscatts_i1, gscatts_f1, prop_phiV3_tf, propDSS_tf, propUSS_tf);
                    outputfile(aux, "3", 1);
                }
                {
                    plegma::PLEGMA_ScattCorrelator<float> aux(src_zero, momVects3pt_pf1);
                    aux.T1(gscatts_i1, gscatts_f1, prop_phiV3_tf, propDSS_tf, propUSS_tf);
                    outputfile(aux, "4", 1);
                }
            }
        }
    }
    finalize();
    std::ofstream output(flagFinish);
    return 0;
}
