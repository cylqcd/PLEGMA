/*
Here we compute PJP diagrams.
pi-J-pi = output
*/

#include <PLEGMA.h>
#include <PLEGMA_utils.h>
std::vector<double> runtime;
#define TIME(fnc)                                                              \
    runtime.push_back(MPI_Wtime());                                            \
    fnc;                                                                       \
    PLEGMA_printf("TIME for " #fnc " %f sec\n", MPI_Wtime() - runtime.back()); \
    runtime.pop_back()

extern int device;
static std::vector<std::string> listOpt = {"verbosity", "load-gauge", "seed1", "confnumber", "outdiagramPrefix", "momlisttwopt-filename", "momlistthreept-filename", "nsmear-APE", "alpha-APE", "nsmear-gauss", "alpha-gauss",
                                           "src-filename", "tSinks"};

int main(int argc, char **argv)
{
    /******************************************************
     *
     *   Initialiazation
     *
     ******************************************************/
    initializeOptions(argc, argv, true, listOpt);

    int seed_oet, confnumber_int;
    std::string outdiagramPrefix, flagfile;

    HGC_options->set("flagfile", "An empty file created indicating the completion of a run", verbosity, flagfile);
    HGC_options->set("seed_oet", "Seed for initialization of stochastic sources for the oet", verbosity, seed_oet);
    HGC_options->set("confnumber", "Integer determining the index of the gauge configuration", verbosity, confnumber_int);
    HGC_options->set("outdiagramPrefix", "Prefix of the resulting diagrams", verbosity, outdiagramPrefix);

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

#if 0 // setupPointPropagator
                    auto setupPointPropagator = [&](plegma::PLEGMA_Propagator<float> &outS, plegma::PLEGMA_Propagator<float> &outL, plegma::site &source, FlavorYan f, SmearFlagYan s)
                    {
                        if (s < single2double)
                        {
                            plegma::PLEGMA_Vector<double> auxVector;
                            plegma::PLEGMA_Vector<float> auxVectorF;
                            plegma::PLEGMA_Vector3D<double> auxVector3D;
                            for (int isc = 0; isc < 12; isc++)
                            {
                                auxVector3D.pointSource(source, isc / 3, isc % 3, DEVICE);
                                solve(auxVector, auxVector3D, source[DIM_T], f, s);
                                auxVectorF.copy(auxVector);
                                outS.absorb(auxVectorF, isc / 3, isc % 3);
                            }
                        }
                        else
                        {
                            plegma::PLEGMA_Vector<double> auxVectorS, auxVectorL;
                            plegma::PLEGMA_Vector<float> auxVectorSF, auxVectorLF;
                            plegma::PLEGMA_Vector3D<double> auxVector3D;

                            for (int isc = 0; isc < 12; isc++)
                            {
                                auxVector3D.pointSource(source, isc / 3, isc % 3, DEVICE);
                                solve(auxVectorS, auxVectorL, auxVector3D, source[DIM_T], f, s);
                                auxVectorSF.copy(auxVectorS);
                                auxVectorLF.copy(auxVectorL);
                                outS.absorb(auxVectorSF, isc / 3, isc % 3);
                                outL.absorb(auxVectorLF, isc / 3, isc % 3);
                            }
                        }
                    };
#endif

#if 0 // setupSequentialPropagator
                    auto ToolYan::setupSequentialPropagator = [&](plegma::PLEGMA_Propagator<float> &outS, plegma::PLEGMA_Propagator<float> &outL, plegma::PLEGMA_Propagator<float> &in, int inTime, plegma::GAMMAS Gamma, std::vector<int> mom, int momSign, FlavorYan f, SmearFlagYan s)
                    {
                        if (s < single2double)
                        {
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
                                solve(auxVectorD, auxVector3DD, inTime, f, s);
                                auxVectorF.copy(auxVectorD);
                                outS.absorb(auxVectorF, isc / 3, isc % 3);
                            }
                        }
                        else
                        {
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
                                solve(auxVectorDS, auxVectorDL, auxVector3DD, inTime, f, s);
                                auxVectorFS.copy(auxVectorDS);
                                auxVectorFL.copy(auxVectorDL);
                                outS.absorb(auxVectorFS, isc / 3, isc % 3);
                                outL.absorb(auxVectorFL, isc / 3, isc % 3);
                            }
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
        std::vector<plegma::GAMMAS_SCATT> gscatts_c = {ID, G_1, G_2, G_3, G_4, G_5, G_5_G_1, G_5_G_2, G_5_G_3, G_5_G_4, S_12, S_23, S_13, S_41, S_42, S_43};
        // std::vector<plegma::GAMMAS> gammas_c = {ONE, G1, G2, G3, G4, G5, G5G1, G5G2, G5G3, G5G4};

        std::vector<plegma::GAMMAS_SCATT> gscatts_pion = {G_5};
        std::string filenamePost = "PJP";

        // PLEGMA_printf("###Momentum list read from : %s", pathListMomenta_twopt.c_str());
        // plegma::momList momList2pt(3, pathListMomenta_twopt, {1, 2}); // pi2, pf1, pf2
        // PLEGMA_printf("N momenta in sourcemomentumList: %d\n", momList2pt.size());
        // if (momList2pt.empty())
        //     PLEGMA_error("threept momentumList empty");
        // auto momVects2pt_pi2 = momList2pt.uniq_p(0);
        // auto momVects2pt_pf2 = momList2pt.uniq_p(2);
        // plegma::momList momList2pt_pi2(1, {momVects2pt_pi2}, {0});
        // plegma::momList momList2pt_pf2(1, {momVects2pt_pf2}, {0});

        PLEGMA_printf("###Momentum list read from : %s", pathListMomenta_threept.c_str());
        plegma::momList momList_3pt(4, pathListMomenta_threept, {1, 2, 3}); // pi2, pf1, pf2, pc
        PLEGMA_printf("N momenta in sourcemomentumList: %d\n", momList_3pt.size());
        if (momList_3pt.empty())
            PLEGMA_error("threept momentumList empty");
        auto momVects_pi2 = momList_3pt.uniq_p(0);
        auto momVects_pf2 = momList_3pt.uniq_p(2);
        auto momVects_pc = momList_3pt.uniq_p(3);
        plegma::momList momList_pc(1, {momVects_pc}, {0});

        /******************************************************
         *
         *   Main
         *
         ******************************************************/
        int st = sourcePositions[0][3];
        struct plegma::site source({0, 0, 0, st});

        plegma::PLEGMA_Vector<double> stocXi;
        {
            stocXi.randInit(seed_oet);
            stocXi.stochastic_Z(4);
        }
        int time_i = source[3];

        plegma::PLEGMA_Vector<double> stocPhiU_ti_SS, stocPhiD_ti_SS;
        std::vector<PLEGMA_Vector<float> *> stocPhiUs_ti_mpi2_SL;
        std::vector<PLEGMA_Vector<float> *> stocPhiDs_ti_mpi2_SL;
        for (int i_pi2 = 0; i_pi2 < momVects_pi2.size(); ++i_pi2)
        {
            stocPhiUs_ti_mpi2_SL.push_back(new PLEGMA_Vector<float>(HOST));
            stocPhiDs_ti_mpi2_SL.push_back(new PLEGMA_Vector<float>(HOST));
        }

        // do U part
        {
            plegma::PLEGMA_Vector3D<double> auxVector3D;
            auxVector3D.absorb(stocXi, time_i);
            solve1(stocPhiU_ti_SS, auxVector3D, time_i, u, SS);
        }
        for (int i_pi2 = 0; i_pi2 < momVects_pi2.size(); ++i_pi2)
        {
            auto &mom_pi2 = momVects_pi2[i_pi2];
            plegma::PLEGMA_Vector<double> auxVector;
            plegma::PLEGMA_Vector<float> auxVectorF;
            plegma::PLEGMA_Vector3D<double> auxVector3D;
            {
                plegma::PLEGMA_Vector<double> aux;
                aux.copy(stocXi);
                auxVector3D.absorb(aux, time_i);
            }
            auxVector3D.mulMomentumPhases(mom_pi2, +1);
            solve1(auxVector, auxVector3D, time_i, u, SL);
            auxVectorF.copy(auxVector);
            auxVectorF.unload();
            stocPhiUs_ti_mpi2_SL[i_pi2]->copy(auxVectorF, HOST);
        }
        // do D part
        {
            plegma::PLEGMA_Vector3D<double> auxVector3D;
            auxVector3D.absorb(stocXi, time_i);
            solve1(stocPhiD_ti_SS, auxVector3D, time_i, d, SS);
        }
        for (int i_pi2 = 0; i_pi2 < momVects_pi2.size(); ++i_pi2)
        {
            auto &mom_pi2 = momVects_pi2[i_pi2];
            plegma::PLEGMA_Vector<double> auxVector;
            plegma::PLEGMA_Vector<float> auxVectorF;
            plegma::PLEGMA_Vector3D<double> auxVector3D;
            {
                plegma::PLEGMA_Vector<double> aux;
                aux.copy(stocXi);
                auxVector3D.absorb(aux, time_i);
            }
            auxVector3D.mulMomentumPhases(mom_pi2, +1);
            solve1(auxVector, auxVector3D, time_i, d, SL);
            auxVectorF.copy(auxVector);
            auxVectorF.unload();
            stocPhiDs_ti_mpi2_SL[i_pi2]->copy(auxVectorF, HOST);
        }

        auto signFlip = [&](plegma::PLEGMA_ScattCorrelator<float> sc)
        {
            float overall_sign[2] = {-1., 0.};
            x_e_cx<float>(sc.H_elem(), overall_sign, sc.getTotalSize());
        };


        // do D part
        for (int i_pf2 = 0; i_pf2 < momVects_pf2.size(); ++i_pf2)
        {
            auto &mom_pf2 = momVects_pf2[i_pf2];
            std::string str_pf2 = "pf2=" + std::to_string(mom_pf2[0]) + "_" + std::to_string(mom_pf2[1]) + "_" + std::to_string(mom_pf2[2]);
            std::string str_pf2_neg = "pf2=" + std::to_string(-mom_pf2[0]) + "_" + std::to_string(-mom_pf2[1]) + "_" + std::to_string(-mom_pf2[2]);

            for (int i_tSink = 0; i_tSink < tSinks.size(); i_tSink++)
            {
                std::string str_dt = "dt" + std::to_string(tSinks[i_tSink]);
                int time_f = (time_i + tSinks[i_tSink]) % HGC_totalL[3];

                plegma::PLEGMA_Vector<double> stocSeqPhiDU, stocSeqPhiDD;
                {
                    plegma::PLEGMA_Vector3D<double> auxVector3D;
                    auxVector3D.absorb(stocPhiU_ti_SS, time_f);
                    auxVector3D.apply_gamma5();
                    auxVector3D.mulMomentumPhases(mom_pf2, -1);
                    solve1(stocSeqPhiDU, auxVector3D, time_f, d, SL);
                }
                {
                    plegma::PLEGMA_Vector3D<double> auxVector3D;
                    auxVector3D.absorb(stocPhiD_ti_SS, time_f);
                    auxVector3D.apply_gamma5();
                    auxVector3D.mulMomentumPhases(mom_pf2, -1);
                    solve1(stocSeqPhiDD, auxVector3D, time_f, d, SL);
                }

                for (int i_pi2 = 0; i_pi2 < momVects_pi2.size(); ++i_pi2)
                {
                    auto &mom_pi2 = momVects_pi2[i_pi2];
                    std::string str_pi2 = "pi2=" + std::to_string(mom_pi2[0]) + "_" + std::to_string(mom_pi2[1]) + "_" + std::to_string(mom_pi2[2]);
                    std::string str_pi2_neg = "pi2=" + std::to_string(-mom_pi2[0]) + "_" + std::to_string(-mom_pi2[1]) + "_" + std::to_string(-mom_pi2[2]);

                    {
                        plegma::PLEGMA_ScattCorrelator<float> PJP(source, momList_pc);
                        PJP.initialize_diagram(gscatts_pion, gscatts_c, str_dt + "/" + str_pf2_neg + "/" + str_pi2 + "/pi+_ju_pi+");
                        {
                            plegma::PLEGMA_Vector<float> aux0, aux1;
                            aux0.copy(*stocPhiUs_ti_mpi2_SL[i_pi2], HOST);
                            aux0.load();
                            aux1.copy(stocSeqPhiDU);
                            PJP.P_diagrams(aux0, aux1, -1);
                            signFlip(PJP);
                        }
                        std::string outfilename = outdiagramPrefix + confnumber + "_" + filenamePost;
                        PJP.writeHDF5(outfilename);
                    }
                    {
                        plegma::PLEGMA_ScattCorrelator<float> PJP(source, momList_pc);
                        PJP.initialize_diagram(gscatts_pion, gscatts_c, str_dt + "/" + str_pf2 + "/" + str_pi2_neg + "/pi+_jd_pi+");
                        {
                            plegma::PLEGMA_Vector<float> aux0, aux1;
                            aux1.copy(*stocPhiUs_ti_mpi2_SL[i_pi2], HOST);
                            aux1.load();
                            aux0.copy(stocSeqPhiDU);
                            PJP.P_diagrams(aux0, aux1, -1);
                            signFlip(PJP);
                        }
                        std::string outfilename = outdiagramPrefix + confnumber + "_" + filenamePost;
                        PJP.writeHDF5(outfilename);
                    }
                    {
                        plegma::PLEGMA_ScattCorrelator<float> PJP(source, momList_pc);
                        PJP.initialize_diagram(gscatts_pion, gscatts_c, str_dt + "/" + str_pf2_neg + "/" + str_pi2 + "/pi0u_ju_pi0u1");
                        {
                            plegma::PLEGMA_Vector<float> aux0, aux1;
                            aux0.copy(*stocPhiUs_ti_mpi2_SL[i_pi2], HOST);
                            aux0.load();
                            aux1.copy(stocSeqPhiDD);
                            PJP.P_diagrams(aux0, aux1, -1);
                            signFlip(PJP);
                        }
                        std::string outfilename = outdiagramPrefix + confnumber + "_" + filenamePost;
                        PJP.writeHDF5(outfilename);
                    }
                    {
                        plegma::PLEGMA_ScattCorrelator<float> PJP(source, momList_pc);
                        PJP.initialize_diagram(gscatts_pion, gscatts_c, str_dt + "/" + str_pf2 + "/" + str_pi2_neg + "/pi0d_jd_pi0d2");
                        {
                            plegma::PLEGMA_Vector<float> aux0, aux1;
                            aux1.copy(*stocPhiUs_ti_mpi2_SL[i_pi2], HOST);
                            aux1.load();
                            aux0.copy(stocSeqPhiDD);
                            PJP.P_diagrams(aux0, aux1, -1);
                            signFlip(PJP);
                        }
                        std::string outfilename = outdiagramPrefix + confnumber + "_" + filenamePost;
                        PJP.writeHDF5(outfilename);
                    }
                }
            }
        }

        // do U part
        for (int i_pf2 = 0; i_pf2 < momVects_pf2.size(); ++i_pf2)
        {
            auto &mom_pf2 = momVects_pf2[i_pf2];
            std::string str_pf2 = "pf2=" + std::to_string(mom_pf2[0]) + "_" + std::to_string(mom_pf2[1]) + "_" + std::to_string(mom_pf2[2]);
            std::string str_pf2_neg = "pf2=" + std::to_string(-mom_pf2[0]) + "_" + std::to_string(-mom_pf2[1]) + "_" + std::to_string(-mom_pf2[2]);

            for (int i_tSink = 0; i_tSink < tSinks.size(); i_tSink++)
            {
                std::string str_dt = "dt" + std::to_string(tSinks[i_tSink]);
                int time_f = (time_i + tSinks[i_tSink]) % HGC_totalL[3];

                plegma::PLEGMA_Vector<double> stocSeqPhiUU, stocSeqPhiUD;
                {
                    plegma::PLEGMA_Vector3D<double> auxVector3D;
                    auxVector3D.absorb(stocPhiU_ti_SS, time_f);
                    auxVector3D.apply_gamma5();
                    auxVector3D.mulMomentumPhases(mom_pf2, -1);
                    solve1(stocSeqPhiUU, auxVector3D, time_f, u, SL);
                }
                {
                    plegma::PLEGMA_Vector3D<double> auxVector3D;
                    auxVector3D.absorb(stocPhiD_ti_SS, time_f);
                    auxVector3D.apply_gamma5();
                    auxVector3D.mulMomentumPhases(mom_pf2, -1);
                    solve1(stocSeqPhiUD, auxVector3D, time_f, u, SL);
                }

                for (int i_pi2 = 0; i_pi2 < momVects_pi2.size(); ++i_pi2)
                {
                    auto &mom_pi2 = momVects_pi2[i_pi2];
                    std::string str_pi2 = "pi2=" + std::to_string(mom_pi2[0]) + "_" + std::to_string(mom_pi2[1]) + "_" + std::to_string(mom_pi2[2]);
                    std::string str_pi2_neg = "pi2=" + std::to_string(-mom_pi2[0]) + "_" + std::to_string(-mom_pi2[1]) + "_" + std::to_string(-mom_pi2[2]);

                    {
                        plegma::PLEGMA_ScattCorrelator<float> PJP(source, momList_pc);
                        PJP.initialize_diagram(gscatts_pion, gscatts_c, str_dt + "/" + str_pf2 + "/" + str_pi2_neg + "/pi-_ju_pi-");
                        {
                            plegma::PLEGMA_Vector<float> aux0, aux1;
                            aux1.copy(*stocPhiDs_ti_mpi2_SL[i_pi2], HOST);
                            aux1.load();
                            aux0.copy(stocSeqPhiUD);
                            PJP.P_diagrams(aux0, aux1, -1);
                            signFlip(PJP);
                        }
                        std::string outfilename = outdiagramPrefix + confnumber + "_" + filenamePost;
                        PJP.writeHDF5(outfilename);
                    }
                    {
                        plegma::PLEGMA_ScattCorrelator<float> PJP(source, momList_pc);
                        PJP.initialize_diagram(gscatts_pion, gscatts_c, str_dt + "/" + str_pf2_neg + "/" + str_pi2 + "/pi-_jd_pi-");
                        {
                            plegma::PLEGMA_Vector<float> aux0, aux1;
                            aux0.copy(*stocPhiDs_ti_mpi2_SL[i_pi2], HOST);
                            aux0.load();
                            aux1.copy(stocSeqPhiUD);
                            PJP.P_diagrams(aux0, aux1, -1);
                            signFlip(PJP);
                        }
                        std::string outfilename = outdiagramPrefix + confnumber + "_" + filenamePost;
                        PJP.writeHDF5(outfilename);
                    }
                    {
                        plegma::PLEGMA_ScattCorrelator<float> PJP(source, momList_pc);
                        PJP.initialize_diagram(gscatts_pion, gscatts_c, str_dt + "/" + str_pf2 + "/" + str_pi2_neg + "/pi0u_ju_pi0u2");
                        {
                            plegma::PLEGMA_Vector<float> aux0, aux1;
                            aux1.copy(*stocPhiDs_ti_mpi2_SL[i_pi2], HOST);
                            aux1.load();
                            aux0.copy(stocSeqPhiUU);
                            PJP.P_diagrams(aux0, aux1, -1);
                            signFlip(PJP);
                        }
                        std::string outfilename = outdiagramPrefix + confnumber + "_" + filenamePost;
                        PJP.writeHDF5(outfilename);
                    }
                    {
                        plegma::PLEGMA_ScattCorrelator<float> PJP(source, momList_pc);
                        PJP.initialize_diagram(gscatts_pion, gscatts_c, str_dt + "/" + str_pf2_neg + "/" + str_pi2 + "/pi0d_jd_pi0d1");
                        {
                            plegma::PLEGMA_Vector<float> aux0, aux1;
                            aux0.copy(*stocPhiDs_ti_mpi2_SL[i_pi2], HOST);
                            aux0.load();
                            aux1.copy(stocSeqPhiUU);
                            PJP.P_diagrams(aux0, aux1, -1);
                            signFlip(PJP);
                        }
                        std::string outfilename = outdiagramPrefix + confnumber + "_" + filenamePost;
                        PJP.writeHDF5(outfilename);
                    }
                }
            }
        }
    }

    finalize();
    std::ofstream output(flagfile);
    return 0;
}
