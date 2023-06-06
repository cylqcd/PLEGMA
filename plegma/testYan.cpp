#include <PLEGMA.h>
#include <PLEGMA_utils.h>
std::vector<double> runtime;
#define TIME(fnc)                                                              \
    runtime.push_back(MPI_Wtime());                                            \
    fnc;                                                                       \
    PLEGMA_printf("TIME for " #fnc " %f sec\n", MPI_Wtime() - runtime.back()); \
    runtime.pop_back()

extern int device;
static std::vector<std::string> listOpt = {"verbosity", "load-gauge", "momlisttwopt-filename", "momlistthreept-filename", "nsmear-APE", "alpha-APE", "nsmear-gauss", "alpha-gauss"};

int main(int argc, char **argv)
{
    /******************************************************
     *
     *   Initialiazation
     *
     ******************************************************/
    initializeOptions(argc, argv, true, listOpt);

    int confnumber_int;
    HGC_options->set("confnumber", "Integer determining the index of the gauge configuration", verbosity, confnumber_int);

    int seed_stoc;
    HGC_options->set("seed_stoc", "Seed for stoc", verbosity, seed_stoc);

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

        std::vector<plegma::GAMMAS_SCATT> gscatts_meson = {};
        std::string filenameEnd = "";

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
        auto momVects3pt_pi2 = momList3pt.uniq_p(0);
        auto momVects3pt_pc = momList3pt.uniq_p(3);
        plegma::momList momList3pt_pc(1, {momVects3pt_pc}, {0});

        /******************************************************
         *
         *   Main
         *
         ******************************************************/

        struct plegma::site source({1, 2, 3, 4});
        struct plegma::site source_reduction({0, 0, 0, 4});

        plegma::PLEGMA_Propagator<float> propU, propD;
        setupPointPropagator1(propU, source, u, LL);
        setupPointPropagator1(propD, source, d, LL);

        std::vector<GAMMAS_SCATT> glist_source_nucleon = {CG_5};
        std::vector<GAMMAS_SCATT> glist_sink_nucleon = {CG_5};

        std::vector<GAMMAS_SCATT> glist_source_meson = {G_5};
        std::vector<GAMMAS_SCATT> glist_sink_meson = {G_5};

        std::vector<GAMMAS_SCATT> glist_source_nucleon_unpaired = {ID};
        std::vector<GAMMAS_SCATT> glist_sink_nucleon_unpaired = {ID};

        momList sourcemomentumList_threept(4, pathListMomenta_threept, {1, 2, 3});

        momList sourcemomentumList_twopt(3, pathListMomenta_twopt, {1, 2});

        std::vector<std::vector<int>> mpi2_threept = sourcemomentumList_threept.uniq_p(0);
        momList list_mpi2_threept(1, {mpi2_threept}, {0});
        for (int i_mpi2 = 0; i_mpi2 < mpi2_threept.size(); ++i_mpi2)
        {

            auto &momentum_i2 = mpi2_threept[i_mpi2];
            // List of momenta corresponding to a fix value of p_i2
            momList filtered_sourcemomentumList = sourcemomentumList_threept.extract(momentum_i2, 0);

            momList filtered_sourcemomentumList_2pt = sourcemomentumList_twopt.extract(momentum_i2, 0);

            std::vector<std::vector<int>> mptot_filt = filtered_sourcemomentumList_2pt.uniq_p(3);

            std::vector<std::vector<int>> mpi2_filt;
            mpi2_filt.assign(mptot_filt.size(), momentum_i2);
            // PLEGMA_printf("mptot_filt.size() %d\n",mptot_filt.size());
            momList list_mpi2ptot(2, {mpi2_filt, mptot_filt}, {1});

            PLEGMA_ScattCorrelator<float> reductionsT1(source_reduction, mptot_filt);
            PLEGMA_ScattCorrelator<float> reductionsT2(source_reduction, mptot_filt);

            PLEGMA_ScattCorrelator<float> corrTproton_protonpizero1(source, list_mpi2ptot);
            PLEGMA_ScattCorrelator<float> corrTproton_protonpizero2(source, list_mpi2ptot);
            // PLEGMA_ScattCorrelator<float> corrTproton_protonpizero3(source, list_mpi2ptot);
            // PLEGMA_ScattCorrelator<float> corrTproton_protonpizero4(source, list_mpi2ptot);

            corrTproton_protonpizero1.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, "12", "Tseq21");

            corrTproton_protonpizero2.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, "12", "Tseq22");

            // corrTproton_protonpizero3.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, "12", "Tseq23");

            // corrTproton_protonpizero4.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, "12", "Tseq24");

            // PLEGMA_ScattCorrelator<float> corrD1ff1389(source, filtered_sourcemomentumList_2pt);
            PLEGMA_ScattCorrelator<float> corrD1ff24710(source, filtered_sourcemomentumList_2pt);

            corrD1ff24710.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "D1ff2-4-7-10");
            // corrD1ff1389.initialize_diagram(glist_source_nucleon_unpaired, glist_sink_nucleon_unpaired, glist_source_nucleon, glist_source_meson, glist_sink_nucleon, glist_sink_meson, "12", "D1ff1-3-8-9");

            TIME(reductionsT1.T1(glist_source_nucleon, glist_sink_nucleon, propU, propD, propU));
            TIME(corrTproton_protonpizero1.convertTreductiontoDiagram(reductionsT1, 0, false, true, true));

            TIME(reductionsT2.T2(glist_source_nucleon, glist_sink_nucleon, propU, propD, propU));
            TIME(corrTproton_protonpizero2.convertTreductiontoDiagram(reductionsT2, 0, false, true, true));

            TIME(corrD1ff24710.LT_diagrams(reductionsT1, reductionsT2, 1));
            corrD1ff24710.writeHDF5("test_D");
            // TIME(produceOutput_2pt_packed(corrD1ff24710, outfilename, "T", parallel_sources, attract_lookup_table));

            // TIME(reductionsT1.T1(glist_source_nucleon, glist_sink_nucleon, propUP_SS_packed, propDN_SS_packed, propTS_SS_packed));
            // TIME(corrTproton_protonpizero3.convertTreductiontoDiagram(reductionsT1, 0, false, true, true));

            // TIME(reductionsT1.T2(glist_source_nucleon, glist_sink_nucleon, propUP_SS_packed, propDN_SS_packed, propTS_SS_packed));
            // TIME(corrTproton_protonpizero4.convertTreductiontoDiagram(reductionsT1, 0, false, true, true));

            // TIME(corrD1ff1389.LT_diagrams(reductionsT1, reductionsT2, 1));
            // TIME(produceOutput_2pt_packed(corrD1ff1389, outfilename, "T", parallel_sources, attract_lookup_table));

            // TIME(produceOutput_2pt_packed(corrTproton_protonpizero1, outfilename, "T", parallel_sources, attract_lookup_table));
            // TIME(produceOutput_2pt_packed(corrTproton_protonpizero2, outfilename, "T", parallel_sources, attract_lookup_table));
            // TIME(produceOutput_2pt_packed(corrTproton_protonpizero3, outfilename, "T", parallel_sources, attract_lookup_table));
            // TIME(produceOutput_2pt_packed(corrTproton_protonpizero4, outfilename, "T", parallel_sources, attract_lookup_table));

            corrTproton_protonpizero1.writeHDF5("test_T");
            corrTproton_protonpizero2.writeHDF5("test_T");
        }
    }

    finalize();
    return 0;
}
