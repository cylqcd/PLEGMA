/*
Here we compute quark loops, including smeared (for pi0) and local (for insertion),
 with or without one-end-trick (oet not supported yet).

The output data structure is like:
sx00sy00sz00st00/stoc(seed_stoc)_(num_stoc))/[up or dn] dim=[N_time,N_mom,N_gamma,2 for real and imag]

The momenta are multiplied as sink momenta, i.e., with the Fourier phase exp(-i p x).

Only one of up and dn will be done for each run. up and dn are either conjugate or
 negative-conjugate to each other (with momentum flipped)

Each run should start with a random seed, and end up with either creating a new data file
 or appending an existed data file with a group name indicating the random seed and number
  of stochastic sources used.
It is recommended to backup data files before a new run that increases the statistics
 in case of the possibility that the new run destroys the existed data file.

Here the f_0(500)/sigma resonance (ubaru+dbard) is also supported.
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
    std::string outdiagramPrefix, flagfile;
    HGC_options->set("flagfile", "An empty file created indicating the completion of a run", verbosity, flagfile);
    HGC_options->set("confnumber", "Integer determining the index of the gauge configuration", verbosity, confnumber_int);
    HGC_options->set("outdiagramPrefix", "Prefix of the resulting diagrams", verbosity, outdiagramPrefix);

    std::string caseToDo, whichMeson, readStocPath;
    int readStoc, seed_stoc, num_stoc;
    HGC_options->set("caseToDo", "meson or insertion - saveSample or saveAverage", verbosity, caseToDo);
    HGC_options->set("whichMeson", "pi0 or sigma", verbosity, whichMeson);
    HGC_options->set("readStocPath", "Path for stoc", verbosity, readStocPath);
    HGC_options->set("readStoc", "Set 1 to read stoc from NJNpi_N0P+", verbosity, readStoc);
    HGC_options->set("seed_stoc", "Seed for stoc", verbosity, seed_stoc);
    HGC_options->set("num_stoc", "Num of stoc", verbosity, num_stoc);

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
        std::vector<plegma::GAMMAS_SCATT> gscatts_c = {ID, G_1, G_2, G_3, G_4, G_5, G_5_G_1, G_5_G_2, G_5_G_3, G_5_G_4};
        std::vector<plegma::GAMMAS> gammas_c = {ONE, G1, G2, G3, G4, G5, G5G1, G5G2, G5G3, G5G4};

        std::vector<plegma::GAMMAS_SCATT> gscatts_meson = {};
        std::string filenameEnd = "";
        if (caseToDo == "meson-saveSample" || caseToDo == "meson-saveAverage")
        {
            if (whichMeson == "pi0")
            {
                gscatts_meson.push_back(G_5);
                filenameEnd = "pi0Loop";
            }
            else if (whichMeson == "sigma")
            {
                gscatts_meson.push_back(ID);
                filenameEnd = "sigmaLoop";
            }
            else
            {
                PLEGMA_error("whichMeson = %s not supported", whichMeson);
            }
        }
        else
        {
            filenameEnd = "insertLoop";
        }

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
        if (caseToDo == "meson-saveSample")
        {
            if (readStoc == 0)
            {
                struct plegma::site src({0, 0, 0, 0});
                plegma::PLEGMA_Vector<double> stocSrcUnsmeared;
                stocSrcUnsmeared.randInit(seed_stoc);
                for (int i = 0; i < num_stoc; i++)
                {
                    plegma::PLEGMA_ScattCorrelator<float> pi0Loop(src, momList2pt_pi2);
                    pi0Loop.initialize_diagram(gscatts_meson, "stoc" + std::to_string(seed_stoc) + "id" + std::to_string(i) + "_1" + "/up");

                    plegma::PLEGMA_Vector<float> stocSrc, stocProp;
                    plegma::PLEGMA_Vector<double> stocPropSmeared;

                    stocSrcUnsmeared.stochastic_Z(4);
                    plegma::PLEGMA_Vector<double> stocSrcSmeared;
                    stocSrcSmeared.gaussianSmearing(stocSrcUnsmeared, smearedGauge, nsmearGauss, alphaGauss);
                    for (int tc = 0; tc < HGC_totalL[3]; tc++)
                    {
                        plegma::PLEGMA_Vector<double> aux, stocPropUnsmeared;
                        plegma::PLEGMA_Vector3D<double> aux3D, aux3D2;
                        aux.absorbTimeslice(stocSrcSmeared, tc, true);
                        solve0(stocPropUnsmeared, aux, u);
                        aux3D.absorb(stocPropUnsmeared, tc);

                        plegma::PLEGMA_Gauge3D<double> smearedGauge3D;
                        smearedGauge3D.absorb(smearedGauge, tc);
                        aux3D2.gaussianSmearing(aux3D, smearedGauge3D, nsmearGauss, alphaGauss);

                        stocPropSmeared.absorb(aux3D2, tc, tc == 0 ? true : false);
                    }
                    stocSrc.copy(stocSrcUnsmeared); // Note: smearing a prop Q gives SQS, with stoc, it's SQS\xi \xi^\dag, so no smearing is needed for the src.
                    stocProp.copy(stocPropSmeared);
                    pi0Loop.Loop_diagrams(stocProp, stocSrc, false);

                    std::string outfilename = outdiagramPrefix + confnumber + "_" + filenameEnd;
                    pi0Loop.writeHDF5(outfilename);
                }
            }
            else if (readStoc == 1)
            {
                struct plegma::site src({0, 0, 0, 0});
                for (int i = 0; i < num_stoc; i++)
                {
                    plegma::PLEGMA_ScattCorrelator<float> pi0Loop(src, momList2pt_pi2);
                    pi0Loop.initialize_diagram(gscatts_meson, "stocRead1id" + std::to_string(i) + "_1" + "/up");

                    plegma::PLEGMA_Vector<float> stocSrc, stocProp;
                    plegma::PLEGMA_Vector<double> stocSrcD, stocPropD;

                    std::string inputfilename = readStocPath + "globalTfulltimedilution_source_nstoch" + std::to_string(i) + "_" + confnumber;
                    PLEGMA_printf("Read stochastic source from: %s\n", inputfilename.c_str());
                    stocSrcD.readFile(inputfilename, LIME_FORMAT);
                    inputfilename = readStocPath + "globalTfulltimedilution_propagator_nstoch" + std::to_string(i) + "_" + confnumber;
                    PLEGMA_printf("Read stochastic propagator from: %s\n", inputfilename.c_str());
                    stocProp.readFile(inputfilename, LIME_FORMAT);

                    stocSrc.copy(stocSrcD);
                    pi0Loop.Loop_diagrams(stocProp, stocSrc, false);

                    std::string outfilename = outdiagramPrefix + confnumber + "_" + filenameEnd;
                    pi0Loop.writeHDF5(outfilename);
                }
            }
        }
        else if (caseToDo == "meson-saveAverage")
        {
            if (readStoc == 0)
            {
                struct plegma::site src({0, 0, 0, 0});
                plegma::PLEGMA_ScattCorrelator<float> pi0Loop(src, momList2pt_pi2);
                pi0Loop.initialize_diagram(gscatts_meson, "stoc" + std::to_string(seed_stoc) + "_" + std::to_string(num_stoc) + "/up");

                plegma::PLEGMA_Vector<double> stocSrcUnsmeared;
                stocSrcUnsmeared.randInit(seed_stoc);
                for (int i = 0; i < num_stoc; i++)
                {
                    plegma::PLEGMA_Vector<float> stocSrc, stocProp;
                    plegma::PLEGMA_Vector<double> stocPropSmeared;

                    stocSrcUnsmeared.stochastic_Z(4);
                    plegma::PLEGMA_Vector<double> stocSrcSmeared;
                    stocSrcSmeared.gaussianSmearing(stocSrcUnsmeared, smearedGauge, nsmearGauss, alphaGauss);
                    for (int tc = 0; tc < HGC_totalL[3]; tc++)
                    {
                        plegma::PLEGMA_Vector<double> aux, stocPropUnsmeared;
                        plegma::PLEGMA_Vector3D<double> aux3D, aux3D2;
                        aux.absorbTimeslice(stocSrcSmeared, tc, true);
                        solve0(stocPropUnsmeared, aux, u);
                        aux3D.absorb(stocPropUnsmeared, tc);

                        plegma::PLEGMA_Gauge3D<double> smearedGauge3D;
                        smearedGauge3D.absorb(smearedGauge, tc);
                        aux3D2.gaussianSmearing(aux3D, smearedGauge3D, nsmearGauss, alphaGauss);

                        stocPropSmeared.absorb(aux3D2, tc, tc == 0 ? true : false);
                    }
                    stocSrc.copy(stocSrcUnsmeared); // Note: smearing a prop Q gives SQS, with stoc, it's SQS\xi \xi^\dag, so no smearing is needed for the src.
                    stocProp.copy(stocPropSmeared);
                    pi0Loop.Loop_diagrams(stocProp, stocSrc, i == 0 ? false : true);
                }
                pi0Loop.normalize_nstoch(num_stoc);
                std::string outfilename = outdiagramPrefix + confnumber + "_" + filenameEnd;
                pi0Loop.writeHDF5(outfilename);
            }
            else if (readStoc == 1)
            {
                struct plegma::site src({0, 0, 0, 0});
                plegma::PLEGMA_ScattCorrelator<float> pi0Loop(src, momList2pt_pi2);
                pi0Loop.initialize_diagram(gscatts_meson, "stocRead1_" + std::to_string(num_stoc) + "/up");

                plegma::PLEGMA_Vector<float> stocSrc, stocProp;
                plegma::PLEGMA_Vector<double> stocSrcD, stocPropD;
                for (int i = 0; i < num_stoc; i++)
                {
                    std::string inputfilename = readStocPath + "globalTfulltimedilution_source_nstoch" + std::to_string(i) + "_" + confnumber;
                    PLEGMA_printf("Read stochastic source from: %s\n", inputfilename.c_str());
                    stocSrcD.readFile(inputfilename, LIME_FORMAT);
                    inputfilename = readStocPath + "globalTfulltimedilution_propagator_nstoch" + std::to_string(i) + "_" + confnumber;
                    PLEGMA_printf("Read stochastic propagator from: %s\n", inputfilename.c_str());
                    stocProp.readFile(inputfilename, LIME_FORMAT);

                    stocSrc.copy(stocSrcD);
                    pi0Loop.Loop_diagrams(stocProp, stocSrc, i == 0 ? false : true);
                }
                pi0Loop.normalize_nstoch(num_stoc);
                std::string outfilename = outdiagramPrefix + confnumber + "_" + filenameEnd;
                pi0Loop.writeHDF5(outfilename);
            }
        }
        else if (caseToDo == "insertion-saveSample")
        {
            struct plegma::site src({0, 0, 0, 0});
            plegma::PLEGMA_Vector<double> stocSrcUnsmeared, stocPropUnsmeared;
            stocSrcUnsmeared.randInit(seed_stoc);
            for (int i = 0; i < num_stoc; i++)
            {
                plegma::PLEGMA_ScattCorrelator<float> insertLoop(src, momList3pt_pc);
                insertLoop.initialize_diagram(gscatts_c, "stoc" + std::to_string(seed_stoc) + "id" + std::to_string(i) + "_1" + "/up");

                plegma::PLEGMA_Vector<float> stocSrc, stocProp;

                stocSrcUnsmeared.stochastic_Z(4);
                for (int tc = 0; tc < HGC_totalL[3]; tc++)
                {
                    plegma::PLEGMA_Vector<double> aux, aux2;
                    aux.absorbTimeslice(stocSrcUnsmeared, tc, true);
                    solve0(aux2, aux, u);
                    stocPropUnsmeared.absorbTimeslice(aux2, tc, tc == 0 ? true : false);
                }
                stocSrc.copy(stocSrcUnsmeared);
                stocProp.copy(stocPropUnsmeared);
                // stocProp.writeHDF5(confnumber+"stocProp");
                // stocSrc.writeHDF5(confnumber+"stocSrc");
                insertLoop.Loop_diagrams(stocProp, stocSrc, false);

                std::string outfilename = outdiagramPrefix + confnumber + "_" + filenameEnd;
                insertLoop.writeHDF5(outfilename);
            }
        }
        else if (caseToDo == "insertion-saveAverage")
        {
            struct plegma::site src({0, 0, 0, 0});
            plegma::PLEGMA_ScattCorrelator<float> insertLoop(src, momList3pt_pc);
            insertLoop.initialize_diagram(gscatts_c, "stoc" + std::to_string(seed_stoc) + "_" + std::to_string(num_stoc) + "/up");

            plegma::PLEGMA_Vector<double> stocSrcUnsmeared, stocPropUnsmeared;
            stocSrcUnsmeared.randInit(seed_stoc);
            for (int i = 0; i < num_stoc; i++)
            {
                plegma::PLEGMA_Vector<float> stocSrc, stocProp;

                stocSrcUnsmeared.stochastic_Z(4);
                for (int tc = 0; tc < HGC_totalL[3]; tc++)
                {
                    plegma::PLEGMA_Vector<double> aux, aux2;
                    aux.absorbTimeslice(stocSrcUnsmeared, tc, true);
                    solve0(aux2, aux, u);
                    stocPropUnsmeared.absorbTimeslice(aux2, tc, tc == 0 ? true : false);
                }
                stocSrc.copy(stocSrcUnsmeared);
                stocProp.copy(stocPropUnsmeared);
                insertLoop.Loop_diagrams(stocProp, stocSrc, i == 0 ? false : true);
            }
            insertLoop.normalize_nstoch(num_stoc);
            std::string outfilename = outdiagramPrefix + confnumber + "_" + filenameEnd;
            insertLoop.writeHDF5(outfilename);
        }
        else
        {
            PLEGMA_printf("%s not supported", caseToDo.c_str());
        }
    }

    finalize();
    std::ofstream output(flagfile);
    return 0;
}
