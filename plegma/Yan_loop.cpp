/*
Here we compute <(qbar.G.q)(x,t) e^(-i.pf.xf)> = - PhiPhi(phi0,Gf,phi1,true)
To get data for the above formula, the output file requires the following additional changes:
# (-1) for the (-1) in front of PhiPhi
# we compute q=u here, flavor d can be got from g5-Hermiticity
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
static std::vector<std::string> listOpt = {"verbosity", "load-gauge", "momlistthreept-filename", "nsmear-APE", "alpha-APE", "nsmear-gauss", "alpha-gauss"};

int main(int argc, char **argv)
{
    /******************************************************
     *
     *   Initialiazation
     *
     ******************************************************/
    initializeOptions(argc, argv, true, listOpt);

    int confnumber_int;
    std::string outdiagramPrefix;
    HGC_options->set("confnumber", "Integer determining the index of the gauge configuration", verbosity, confnumber_int);
    HGC_options->set("outdiagramPrefix", "Prefix of the resulting diagrams", verbosity, outdiagramPrefix);

    std::string SorL;
    int seed_stoc, num_stoc;
    HGC_options->set("SorL", "S: smear; L: local", verbosity, SorL);
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

        std::string outfilename = outdiagramPrefix + "_" + confnumber + "_loop";

        std::vector<plegma::GAMMAS_SCATT> gscatts = {ID, G_1, G_2, G_3, G_4, G_5, G_5_G_1, G_5_G_2, G_5_G_3, G_5_G_4, S_23, S_13, S_12, S_41, S_42, S_43};

        PLEGMA_printf("###Momentum list read from : %s", pathListMomenta_threept.c_str());
        plegma::momList momList3pt(4, pathListMomenta_threept, {1, 2, 3}); // pi2, pf1, pf2, pc
        PLEGMA_printf("N momenta in sourcemomentumList: %d\n", momList3pt.size());
        if (momList3pt.empty())
            PLEGMA_error("threept momentumList empty");
        auto momVects = momList3pt.uniq_p(3);
        plegma::momList momList(1, {momVects}, {0});

        /******************************************************
         *
         *   Main
         *
         ******************************************************/
        struct plegma::site src({0, 0, 0, 0});
        plegma::PLEGMA_Vector<double> stocSrcUnsmeared;
        stocSrcUnsmeared.randInit(seed_stoc);
        for (int i = 0; i < num_stoc; i++)
        {
            plegma::PLEGMA_ScattCorrelator<float> auxPhiPhi(src, momList);
            auxPhiPhi.initialize_diagram(gscatts, "");

            plegma::PLEGMA_Vector<float> phi0, phi1;
            plegma::PLEGMA_Vector<double> stocPropSmeared;

            stocSrcUnsmeared.stochastic_Z(4);
            plegma::PLEGMA_Vector<double> stocSrcSmeared;
            if (SorL == "S")
            {
                stocSrcSmeared.gaussianSmearing(stocSrcUnsmeared, smearedGauge, nsmearGauss, alphaGauss);
            }
            else if (SorL == "L")
            {
                stocSrcSmeared.copy(stocSrcUnsmeared);
            }
            else
            {
                PLEGMA_printf("invalid SorL");
                return -1;
            }
            
            for (int tc = 0; tc < HGC_totalL[3]; tc++)
            {
                plegma::PLEGMA_Vector<double> aux, stocPropUnsmeared;
                plegma::PLEGMA_Vector3D<double> aux3D, aux3D2;
                aux.absorbTimeslice(stocSrcSmeared, tc, true);
                solve0(stocPropUnsmeared, aux, u);
                aux3D.absorb(stocPropUnsmeared, tc);

                if (SorL == "true")
                {
                    plegma::PLEGMA_Gauge3D<double> smearedGauge3D;
                    smearedGauge3D.absorb(smearedGauge, tc);
                    aux3D2.gaussianSmearing(aux3D, smearedGauge3D, nsmearGauss, alphaGauss);
                }
                else if (SorL == "false")
                {
                    aux3D2.copy(aux3D);
                }

                stocPropSmeared.absorb(aux3D2, tc, tc == 0 ? true : false);
            }
            phi1.copy(stocSrcUnsmeared); // Note: smearing a prop Q gives SQS, with stoc, it's SQS\xi \xi^\dag, so no smearing is needed for the src.
            phi0.copy(stocPropSmeared);
            auxPhiPhi.PhiPhi(phi0, gscatts, phi1, true); // Additional (-1) needed

            auxPhiPhi.setGroups(std::vector<std::string>{"PhiPhi"}); 
            auxPhiPhi.setDatasets(std::vector<std::string>{"seed=" + std::to_string(seed_stoc) + "/id=" + std::to_string(i) + "/up"});
            auxPhiPhi.writeHDF5(outfilename);
        }
    }

    finalize();
    PLEGMA_printf("Yan_flagFinish");
    return 0;
}
