#include <PLEGMA.h>
#include <PLEGMA_utils.h>
std::vector<double> runtime;
#define TIME(fnc)                                                              \
    runtime.push_back(MPI_Wtime());                                            \
    fnc;                                                                       \
    PLEGMA_printf("TIME for " #fnc " %f sec\n", MPI_Wtime() - runtime.back()); \
    runtime.pop_back()

extern int device;
static std::vector<std::string> listOpt = {"verbosity", "load-gauge", "nsrc", "src-filename", "seed1", "confnumber", "outdiagramPrefix", "momlistthreept-filename", "nsmear-APE", "alpha-APE", "nsmear-gauss", "alpha-gauss"};

int main(int argc, char **argv)
{
    /******************************************************
     *
     *   Initialiazation
     *
     ******************************************************/
    initializeOptions(argc, argv, true, listOpt);

    int seed_oet;
    int confnumber_int;
    std::string outdiagramPrefix = "";

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
        std::vector<plegma::GAMMAS_SCATT> gscatts_N = {CG_5};
        std::vector<plegma::GAMMAS_SCATT> gscatts_pi = {G_5};
        std::vector<plegma::GAMMAS_SCATT> gscatts_c = {ID, G_1, G_2, G_3, G_4, G_5, G_5_G_1, G_5_G_2, G_5_G_3, G_5_G_4};
        std::vector<plegma::GAMMAS> gammas_c = {ONE, G1, G2, G3, G4, G5, G5G1, G5G2, G5G3, G5G4};

        PLEGMA_printf("###Momentum list read from : %s", pathListMomenta_threept.c_str());
        plegma::momList momList_3pt(4, pathListMomenta_threept, {1, 2, 3}); // pi2, pf1, pf2, pc
        PLEGMA_printf("N momenta in sourcemomentumList: %d\n", momList_3pt.size());
        if (momList_3pt.empty())
            PLEGMA_error("threept momentumList empty");
        auto momVects_pi2 = momList_3pt.uniq_p(0);
        auto momVects_pc = momList_3pt.uniq_p(3);
        plegma::momList momList_pc(1, {momVects_pc}, {0});

        /******************************************************
         *
         *   Main
         *
         ******************************************************/
        for (int isource = 0; isource < numSourcePositions; ++isource)
        {
            plegma::site &source = sourcePositions[isource];
            PLEGMA_printf(("TestYan: source=" + std::to_string(source[0]) + "_" + std::to_string(source[1]) + "_" + std::to_string(source[2]) + "_" + std::to_string(source[3])).c_str());
            assert(source[0] == source[1] == source[2] == 0 && 0 <= source[3] < HGC_totalL[3]);

            plegma::PLEGMA_Vector<double> stocXi;
            {
                stocXi.randInit(seed_oet);
                stocXi.stochastic_Z(4);
            }
            int time_i = source[3];

            plegma::PLEGMA_Vector<double> stocPhiU_ti_SL, stocPhiD_ti_SL;
            plegma::PLEGMA_Vector3D<double> auxVector3D;
            auxVector3D.absorb(stocXi, time_i);
            solve1(stocPhiU_ti_SL, auxVector3D, time_i, u, SL);
            solve1(stocPhiD_ti_SL, auxVector3D, time_i, d, SL);

            // Here we do u and d separately for saving mu-changing time
            // This the u part
            for (int i_pi2 = 0; i_pi2 < momVects_pi2.size(); ++i_pi2)
            {
                auto &mom_pi2 = momVects_pi2[i_pi2];
                std::string str_pi2 = "pi2=" + std::to_string(mom_pi2[0]) + "_" + std::to_string(mom_pi2[1]) + "_" + std::to_string(mom_pi2[2]);
                PLEGMA_printf(("TestYan: pi2=" + str_pi2).c_str());

                plegma::PLEGMA_Vector<double> stocPhiD_ti_mpi2_SL;
                plegma::PLEGMA_Vector3D<double> auxVector3D;
                auxVector3D.absorb(stocXi, time_i);
                auxVector3D.mulMomentumPhases(mom_pi2, -1);
                solve1(stocPhiD_ti_mpi2_SL, auxVector3D, time_i, d, SL);

                plegma::PLEGMA_ScattCorrelator<float> pi0Insertion(source, momList_pc);
                pi0Insertion.initialize_diagram(gscatts_pi, gscatts_c, str_pi2 + "/up");
                {
                    plegma::PLEGMA_Vector<float> aux, aux_pi2;
                    aux.copy(stocPhiU_ti_SL);
                    aux_pi2.copy(stocPhiD_ti_mpi2_SL);
                    pi0Insertion.P_diagrams(aux, aux_pi2, -1);
                }

                asprintf(&ssource, "st%03d", time_i);
                std::string sourcepositiontext = (std::string) "_" + ssource;
                free(ssource);
                std::string outfilename = outdiagramPrefix + confnumber + sourcepositiontext + "_pi0Isert";

                pi0Insertion.writeHDF5(outfilename);
            }
            // This is the d part
            for (int i_pi2 = 0; i_pi2 < momVects_pi2.size(); ++i_pi2)
            {
                auto &mom_pi2 = momVects_pi2[i_pi2];
                std::string str_pi2 = "pi2=" + std::to_string(mom_pi2[0]) + "_" + std::to_string(mom_pi2[1]) + "_" + std::to_string(mom_pi2[2]);
                PLEGMA_printf(("TestYan: pi2=" + str_pi2).c_str());

                plegma::PLEGMA_Vector<double> stocPhiU_ti_mpi2_SL;
                plegma::PLEGMA_Vector3D<double> auxVector3D;
                auxVector3D.absorb(stocXi, time_i);
                auxVector3D.mulMomentumPhases(mom_pi2, -1);
                solve1(stocPhiU_ti_mpi2_SL, auxVector3D, time_i, u, SL);

                plegma::PLEGMA_ScattCorrelator<float> pi0Insertion(source, momList_pc);
                pi0Insertion.initialize_diagram(gscatts_pi, gscatts_c, str_pi2 + "/dn");
                {
                    plegma::PLEGMA_Vector<float> aux, aux_pi2;
                    aux.copy(stocPhiD_ti_SL);
                    aux_pi2.copy(stocPhiU_ti_mpi2_SL);
                    pi0Insertion.P_diagrams(aux, aux_pi2, -1);
                }

                asprintf(&ssource, "st%03d", time_i);
                std::string sourcepositiontext = (std::string) "_" + ssource;
                free(ssource);
                std::string outfilename = outdiagramPrefix + confnumber + sourcepositiontext + "_pi0Isert";

                pi0Insertion.writeHDF5(outfilename);
            }
        }
    }

    finalize();
    return 0;
}
