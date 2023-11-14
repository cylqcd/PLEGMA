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

auto packVector = [](plegma::PLEGMA_Vector<float> &out, plegma::PLEGMA_Vector<float> &in, int timeSlice)
{
    PLEGMA_Vector3D<float> auxVector3D;
    auxVector3D.absorb(in, timeSlice, true);

    for (int dt = 0; dt < HGC_totalL[3]; dt++)
    {
        out.absorb(auxVector3D, dt, false);
    }
};
auto packVectorD = [](plegma::PLEGMA_Vector<double> &out, plegma::PLEGMA_Vector<double> &in, int timeSlice)
{
    PLEGMA_Vector3D<double> auxVector3D;
    auxVector3D.absorb(in, timeSlice, true);

    for (int dt = 0; dt < HGC_totalL[3]; dt++)
    {
        out.absorb(auxVector3D, dt, false);
    }
};
auto packPropogator = [](plegma::PLEGMA_Propagator<float> &out, plegma::PLEGMA_Propagator<float> &in, int timeSlice)
{
    PLEGMA_Vector<float> auxVector;
    for (int isc = 0; isc < 12; isc++)
    {
        auxVector.absorb(in, isc / 3, isc % 3);
        packVector(auxVector, auxVector, timeSlice);
        out.absorb(auxVector, isc / 3, isc % 3);
    }
};

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

#if 1 // setupSequentialPropagator
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
        std::vector<plegma::GAMMAS_SCATT> gscatts_c_temp = {G_4};
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
        auto momVects3pt_pf1 = momList3pt.uniq_p(1);
        auto momVects3pt_pc = momList3pt.uniq_p(3);
        auto momVects3pt_pi1 = momList3pt.pi1();
        plegma::momList momList3pt_pc(1, {momVects3pt_pc}, {0});

        /******************************************************
         *
         *   Main
         *
         ******************************************************/

        std::string saveDir = "tempData/";

        struct plegma::site source({1, 2, 3, 4});
        struct plegma::site sourceZero({0, 0, 0, 4});

        int time_fi = 10;
        int time_i = source[DIM_T], time_f = (source[DIM_T] + time_fi) % HGC_totalL[3];

        plegma::PLEGMA_Propagator<float> propUSS, propUSL;
        plegma::PLEGMA_Propagator<float> propDSS, propDSL;

        setupPointPropagator2(propUSS, propUSL, source, u, SB);
        setupPointPropagator2(propDSS, propDSL, source, d, SB);

        plegma::PLEGMA_Propagator<float> propUSS_pack, propUSL_pack;
        plegma::PLEGMA_Propagator<float> propDSS_pack, propDSL_pack;
        packPropogator(propUSS_pack, propUSS, time_f);
        packPropogator(propUSL_pack, propUSL, time_f);
        packPropogator(propDSS_pack, propDSS, time_f);
        packPropogator(propDSL_pack, propDSL, time_f);

        PLEGMA_Propagator<float> seqUSSDSS, seqUSLDSS, seqUSSUSS, seqUSLUSS, seqDSSDSS, seqDSLDSS;
        PLEGMA_Propagator<float> seqUSSDSS_pack, seqUSLDSS_pack, seqUSSUSS_pack, seqUSLUSS_pack, seqDSSDSS_pack, seqDSLDSS_pack;
        setupSequentialPropagator2(seqUSSDSS, seqUSLDSS, propDSS, time_i, G5, momVects3pt_pi2[0], +1, u, SB);
        setupSequentialPropagator2(seqUSSUSS, seqUSLUSS, propUSS, time_i, G5, momVects3pt_pi2[0], +1, u, SB);
        setupSequentialPropagator2(seqDSSDSS, seqDSLDSS, propDSS, time_i, G5, momVects3pt_pi2[0], +1, d, SB);
        packPropogator(seqUSSDSS_pack, seqUSSDSS, time_f);
        packPropogator(seqUSLDSS_pack, seqUSLDSS, time_f);
        packPropogator(seqUSSUSS_pack, seqUSSUSS, time_f);
        packPropogator(seqUSLUSS_pack, seqUSLUSS, time_f);
        packPropogator(seqDSSDSS_pack, seqDSSDSS, time_f);
        packPropogator(seqDSLDSS_pack, seqDSLDSS, time_f);

        plegma::PLEGMA_Vector<double> stocXi, stocG5Xi;
        plegma::PLEGMA_Vector<double> stocXi_pack, stocG5Xi_pack;

        plegma::PLEGMA_Vector<double> stocPhiUTfSL, stocG5PhiUTfSL;
        plegma::PLEGMA_Vector<double> stocPhiDTfSL, stocG5PhiDTfSL;

        plegma::PLEGMA_Vector<double> stocPhiUTiSL, stocPhiDTiSL, stocG5PhiUTiSL, stocG5PhiDTiSL;
        plegma::PLEGMA_Vector<double> stocPhiUG5Gi2TiSS, stocPhiDG5Gi2TiSS; // this should acutally be Gi2G5 instead of G5Gi2, though Gi2=G5, they are the same.
        plegma::PLEGMA_Vector<double> stocPhiUG5Gi2TiSS_pack, stocPhiDG5Gi2TiSS_pack;
        {
            plegma::PLEGMA_Vector3D<double> auxVector3D;
            stocXi.randInit(0000);
            stocXi.stochastic_Z(4);
            stocG5Xi.copy(stocXi);
            stocG5Xi.apply_gamma5();
            //
            packVectorD(stocXi_pack, stocXi, time_f);
            packVectorD(stocG5Xi_pack, stocG5Xi, time_f);

            // stocXi.writeHDF5(saveDir + "stocXi");
            // stocXi_pack.writeHDF5(saveDir + "stocXi_pack");

            auxVector3D.absorb(stocXi, time_f);
            solve1(stocPhiUTfSL, auxVector3D, time_f, u, SL);
            stocG5PhiUTfSL.copy(stocPhiUTfSL);
            stocG5PhiUTfSL.apply_gamma5();
            //
            auxVector3D.absorb(stocXi, time_f);
            solve1(stocPhiDTfSL, auxVector3D, time_f, d, SL);
            stocG5PhiDTfSL.copy(stocPhiDTfSL);
            stocG5PhiDTfSL.apply_gamma5();

            auxVector3D.absorb(stocXi, time_i);
            auxVector3D.mulMomentumPhases(momVects3pt_pi2[0], -1);
            solve1(stocPhiUTiSL, auxVector3D, time_i, u, SL);
            stocG5PhiUTiSL.copy(stocPhiUTiSL);
            stocG5PhiUTiSL.apply_gamma5();
            solve1(stocPhiDTiSL, auxVector3D, time_i, d, SL);
            stocG5PhiDTiSL.copy(stocPhiDTiSL);
            stocG5PhiDTiSL.apply_gamma5();
            //
            auxVector3D.absorb(stocXi, time_i);
            // apply g5*g5=ID
            // auxVector3D.mulMomentumPhases(momVects3pt_pi2[0],+1);
            solve1(stocPhiUG5Gi2TiSS, auxVector3D, time_i, u, SS);
            solve1(stocPhiDG5Gi2TiSS, auxVector3D, time_i, d, SS);
            packVectorD(stocPhiUG5Gi2TiSS_pack, stocPhiUG5Gi2TiSS, time_f);
            packVectorD(stocPhiDG5Gi2TiSS_pack, stocPhiDG5Gi2TiSS, time_f);
        }

        PLEGMA_Propagator<float> phiV31;
        {
            plegma::PLEGMA_ScattCorrelator<float> auxV3(sourceZero, momList3pt_pc);
            plegma::PLEGMA_Vector<float> auxVector;
            auxVector.copy(stocG5PhiDTfSL);
            auxV3.V3(auxVector, gscatts_c_temp, seqUSLDSS);
            // auxV3.writeHDF5(saveDir + "temp1");

            for (int isc = 0; isc < 12; isc++)
            {
                plegma::PLEGMA_Vector<float> auxVector1;
                auxVector1.zero_where(BOTH);
                plegma::PLEGMA_Vector3D<double> auxVector3DD;
                plegma::PLEGMA_Vector3D<float> auxVector3D;
                auxVector3DD.absorb(stocG5Xi_pack, time_f, true);
                auxVector3D.copy(auxVector3DD);
                // if (isc == 5)
                //     auxVector3D.writeHDF5(saveDir + "temp2");
                for (int dt = 0; dt < HGC_totalL[3]; dt++)
                {
                    plegma::PLEGMA_Vector3D<float> auxVector3D1;
                    auxVector3D1.copy(auxVector3D);
                    // float *aux = auxV3.Corr((dt+time_i)%HGC_totalL[3], 0, 0);
                    float *aux = auxV3.Corr(dt, 0, 0);
                    std::complex<float> auxC(aux[2 * isc + 0], aux[2 * isc + 1]);
                    auxVector3D1.cscale(auxC);
                    // auxVector1.absorb(auxVector3D1, dt, false);
                    auxVector1.absorb(auxVector3D1, dt, false);
                }
                // if (isc == 5)
                //     auxVector1.writeHDF5(saveDir + "temp3");
                phiV31.absorb(auxVector1, isc / 3, isc % 3);
            }
            // phiV31.writeHDF5(saveDir + "temp4");
        }

        PLEGMA_Propagator<float> phiV31UU;
        {
            plegma::PLEGMA_ScattCorrelator<float> auxV3(sourceZero, momList3pt_pc);
            plegma::PLEGMA_Vector<float> auxVector;
            auxVector.copy(stocG5PhiDTfSL);
            auxV3.V3(auxVector, gscatts_c_temp, seqUSLUSS);
            // auxV3.writeHDF5(saveDir + "temp1");

            for (int isc = 0; isc < 12; isc++)
            {
                plegma::PLEGMA_Vector<float> auxVector1;
                auxVector1.zero_where(BOTH);
                plegma::PLEGMA_Vector3D<double> auxVector3DD;
                plegma::PLEGMA_Vector3D<float> auxVector3D;
                auxVector3DD.absorb(stocG5Xi_pack, time_f, true);
                auxVector3D.copy(auxVector3DD);
                // if (isc == 5)
                //     auxVector3D.writeHDF5(saveDir + "temp2");
                for (int dt = 0; dt < HGC_totalL[3]; dt++)
                {
                    plegma::PLEGMA_Vector3D<float> auxVector3D1;
                    auxVector3D1.copy(auxVector3D);
                    // float *aux = auxV3.Corr((dt+time_i)%HGC_totalL[3], 0, 0);
                    float *aux = auxV3.Corr(dt, 0, 0);
                    std::complex<float> auxC(aux[2 * isc + 0], aux[2 * isc + 1]);
                    auxVector3D1.cscale(auxC);
                    // auxVector1.absorb(auxVector3D1, dt, false);
                    auxVector1.absorb(auxVector3D1, dt, false);
                }
                // if (isc == 5)
                //     auxVector1.writeHDF5(saveDir + "temp3");
                phiV31UU.absorb(auxVector1, isc / 3, isc % 3);
            }
            // phiV31.writeHDF5(saveDir + "temp4");
        }
        PLEGMA_Propagator<float> phiV31DD;
        {
            plegma::PLEGMA_ScattCorrelator<float> auxV3(sourceZero, momList3pt_pc);
            plegma::PLEGMA_Vector<float> auxVector;
            auxVector.copy(stocG5PhiUTfSL);
            auxV3.V3(auxVector, gscatts_c_temp, seqDSLDSS);
            // auxV3.writeHDF5(saveDir + "temp1");

            for (int isc = 0; isc < 12; isc++)
            {
                plegma::PLEGMA_Vector<float> auxVector1;
                auxVector1.zero_where(BOTH);
                plegma::PLEGMA_Vector3D<double> auxVector3DD;
                plegma::PLEGMA_Vector3D<float> auxVector3D;
                auxVector3DD.absorb(stocG5Xi_pack, time_f, true);
                auxVector3D.copy(auxVector3DD);
                // if (isc == 5)
                //     auxVector3D.writeHDF5(saveDir + "temp2");
                for (int dt = 0; dt < HGC_totalL[3]; dt++)
                {
                    plegma::PLEGMA_Vector3D<float> auxVector3D1;
                    auxVector3D1.copy(auxVector3D);
                    // float *aux = auxV3.Corr((dt+time_i)%HGC_totalL[3], 0, 0);
                    float *aux = auxV3.Corr(dt, 0, 0);
                    std::complex<float> auxC(aux[2 * isc + 0], aux[2 * isc + 1]);
                    auxVector3D1.cscale(auxC);
                    // auxVector1.absorb(auxVector3D1, dt, false);
                    auxVector1.absorb(auxVector3D1, dt, false);
                }
                // if (isc == 5)
                //     auxVector1.writeHDF5(saveDir + "temp3");
                phiV31DD.absorb(auxVector1, isc / 3, isc % 3);
            }
            // phiV31.writeHDF5(saveDir + "temp4");
        }

        PLEGMA_Propagator<float> phiV32;
        {
            plegma::PLEGMA_ScattCorrelator<float> auxV3(sourceZero, momList3pt_pc);
            plegma::PLEGMA_Vector<float> auxVector;
            auxVector.copy(stocG5PhiDTfSL);
            auxV3.V3(auxVector, gscatts_c_temp, propUSL);

            for (int isc = 0; isc < 12; isc++)
            {
                plegma::PLEGMA_Vector<float> auxVector1;
                plegma::PLEGMA_Vector3D<double> auxVector3DD;
                plegma::PLEGMA_Vector3D<float> auxVector3D;
                auxVector3DD.absorb(stocG5Xi_pack, time_f, true);
                auxVector3D.copy(auxVector3DD);
                for (int dt = 0; dt < HGC_totalL[3]; dt++)
                {
                    plegma::PLEGMA_Vector3D<float> auxVector3D1;
                    auxVector3D1.copy(auxVector3D);
                    float *aux = auxV3.Corr(dt, 0, 0);
                    std::complex<float> auxC(aux[2 * isc + 0], aux[2 * isc + 1]);
                    auxVector3D1.cscale(auxC);
                    auxVector1.absorb(auxVector3D1, dt, false);
                }
                phiV32.absorb(auxVector1, isc / 3, isc % 3);
            }
        }
        PLEGMA_Propagator<float> phiV32_DD;
        {
            plegma::PLEGMA_ScattCorrelator<float> auxV3(sourceZero, momList3pt_pc);
            plegma::PLEGMA_Vector<float> auxVector;
            auxVector.copy(stocG5PhiUTfSL);
            auxV3.V3(auxVector, gscatts_c_temp, propDSL);

            for (int isc = 0; isc < 12; isc++)
            {
                plegma::PLEGMA_Vector<float> auxVector1;
                plegma::PLEGMA_Vector3D<double> auxVector3DD;
                plegma::PLEGMA_Vector3D<float> auxVector3D;
                auxVector3DD.absorb(stocG5Xi_pack, time_f, true);
                auxVector3D.copy(auxVector3DD);
                for (int dt = 0; dt < HGC_totalL[3]; dt++)
                {
                    plegma::PLEGMA_Vector3D<float> auxVector3D1;
                    auxVector3D1.copy(auxVector3D);
                    float *aux = auxV3.Corr(dt, 0, 0);
                    std::complex<float> auxC(aux[2 * isc + 0], aux[2 * isc + 1]);
                    auxVector3D1.cscale(auxC);
                    auxVector1.absorb(auxVector3D1, dt, false);
                }
                phiV32_DD.absorb(auxVector1, isc / 3, isc % 3);
            }
        }


        PLEGMA_Propagator<float> phiV33;
        {
            plegma::PLEGMA_ScattCorrelator<float> auxV3(sourceZero, momList3pt_pc);
            plegma::PLEGMA_Vector<float> auxVector;
            auxVector.copy(stocG5PhiUTfSL);
            auxV3.V3(auxVector, gscatts_c_temp, propDSL);

            for (int isc = 0; isc < 12; isc++)
            {
                plegma::PLEGMA_Vector<float> auxVector1;
                plegma::PLEGMA_Vector3D<double> auxVector3DD;
                plegma::PLEGMA_Vector3D<float> auxVector3D;
                auxVector3DD.absorb(stocG5Xi_pack, time_f, true);
                auxVector3D.copy(auxVector3DD);
                for (int dt = 0; dt < HGC_totalL[3]; dt++)
                {
                    plegma::PLEGMA_Vector3D<float> auxVector3D1;
                    auxVector3D1.copy(auxVector3D);
                    float *aux = auxV3.Corr(dt, 0, 0);
                    std::complex<float> auxC(aux[2 * isc + 0], aux[2 * isc + 1]);
                    auxVector3D1.cscale(auxC);
                    auxVector1.absorb(auxVector3D1, dt, false);
                }
                phiV33.absorb(auxVector1, isc / 3, isc % 3);
            }
        }

        PLEGMA_Propagator<float> phiV34;
        {
            plegma::PLEGMA_ScattCorrelator<float> auxV3(sourceZero, momList3pt_pc);
            plegma::PLEGMA_Vector<float> auxVector;
            auxVector.copy(stocG5PhiUTiSL);
            auxV3.V3(auxVector, gscatts_c_temp, propDSL);

            for (int isc = 0; isc < 12; isc++)
            {
                plegma::PLEGMA_Vector<float> auxVector1;
                plegma::PLEGMA_Vector3D<double> auxVector3DD;
                plegma::PLEGMA_Vector3D<float> auxVector3D;
                auxVector3DD.absorb(stocPhiUG5Gi2TiSS_pack, time_f, true);
                auxVector3D.copy(auxVector3DD);
                for (int dt = 0; dt < HGC_totalL[3]; dt++)
                {
                    plegma::PLEGMA_Vector3D<float> auxVector3D1;
                    auxVector3D1.copy(auxVector3D);
                    float *aux = auxV3.Corr(dt, 0, 0);
                    std::complex<float> auxC(aux[2 * isc + 0], aux[2 * isc + 1]);
                    auxVector3D1.cscale(auxC);
                    auxVector1.absorb(auxVector3D1, dt, false);
                }
                phiV34.absorb(auxVector1, isc / 3, isc % 3);
            }
        }
        PLEGMA_Propagator<float> phiV34_UUU;
        {
            plegma::PLEGMA_ScattCorrelator<float> auxV3(sourceZero, momList3pt_pc);
            plegma::PLEGMA_Vector<float> auxVector;
            auxVector.copy(stocG5PhiDTiSL);
            auxV3.V3(auxVector, gscatts_c_temp, propUSL);

            for (int isc = 0; isc < 12; isc++)
            {
                plegma::PLEGMA_Vector<float> auxVector1;
                plegma::PLEGMA_Vector3D<double> auxVector3DD;
                plegma::PLEGMA_Vector3D<float> auxVector3D;
                auxVector3DD.absorb(stocPhiUG5Gi2TiSS_pack, time_f, true);
                auxVector3D.copy(auxVector3DD);
                for (int dt = 0; dt < HGC_totalL[3]; dt++)
                {
                    plegma::PLEGMA_Vector3D<float> auxVector3D1;
                    auxVector3D1.copy(auxVector3D);
                    float *aux = auxV3.Corr(dt, 0, 0);
                    std::complex<float> auxC(aux[2 * isc + 0], aux[2 * isc + 1]);
                    auxVector3D1.cscale(auxC);
                    auxVector1.absorb(auxVector3D1, dt, false);
                }
                phiV34_UUU.absorb(auxVector1, isc / 3, isc % 3);
            }
        }
        PLEGMA_Propagator<float> phiV34_DDD;
        {
            plegma::PLEGMA_ScattCorrelator<float> auxV3(sourceZero, momList3pt_pc);
            plegma::PLEGMA_Vector<float> auxVector;
            auxVector.copy(stocG5PhiUTiSL);
            auxV3.V3(auxVector, gscatts_c_temp, propDSL);

            for (int isc = 0; isc < 12; isc++)
            {
                plegma::PLEGMA_Vector<float> auxVector1;
                plegma::PLEGMA_Vector3D<double> auxVector3DD;
                plegma::PLEGMA_Vector3D<float> auxVector3D;
                auxVector3DD.absorb(stocPhiDG5Gi2TiSS_pack, time_f, true);
                auxVector3D.copy(auxVector3DD);
                for (int dt = 0; dt < HGC_totalL[3]; dt++)
                {
                    plegma::PLEGMA_Vector3D<float> auxVector3D1;
                    auxVector3D1.copy(auxVector3D);
                    float *aux = auxV3.Corr(dt, 0, 0);
                    std::complex<float> auxC(aux[2 * isc + 0], aux[2 * isc + 1]);
                    auxVector3D1.cscale(auxC);
                    auxVector1.absorb(auxVector3D1, dt, false);
                }
                phiV34_DDD.absorb(auxVector1, isc / 3, isc % 3);
            }
        }

        auto outputfile = [&](plegma::PLEGMA_ScattCorrelator<float> sc, std::string filename, bool sgn)
        {
            // momentum phase for pi1
            float phase = 2 * M_PI / (float)HGC_totalL[0] * momVects3pt_pi1[0][0] * source[0] +
                          2 * M_PI / (float)HGC_totalL[1] * momVects3pt_pi1[0][1] * source[1] +
                          2 * M_PI / (float)HGC_totalL[2] * momVects3pt_pi1[0][2] * source[2];
            float expPhase[2] = {cos(phase), sin(phase)};
            x_e_cx<float>(sc.H_elem(), expPhase, sc.getTotalSize());

            // additional sign coming from my Mathematica codes
            float overall_sign[2] = {-1., 0.};
            if (sgn)
                x_e_cx<float>(sc.H_elem(), overall_sign, sc.getTotalSize());

            if (time_f < time_i)
                x_e_cx<float>(sc.H_elem(), overall_sign, sc.getTotalSize());
            // sc.applyBoundaryConditions(true);
            sc.writeHDF5(saveDir + filename);
        };

        // p_ju_p&pi0u
        // B
        {
            plegma::PLEGMA_ScattCorrelator<float> cont(sourceZero, momVects3pt_pf1);
            cont.T1(gscatts_N, gscatts_N, propUSS_pack, propDSS_pack, phiV31UU);
            outputfile(cont, "Diagram_uuB1", false);
        }
        {
            plegma::PLEGMA_ScattCorrelator<float> cont(sourceZero, momVects3pt_pf1);
            cont.T2(gscatts_N, gscatts_N, propUSS_pack, phiV31UU, propDSS_pack);
            outputfile(cont, "Diagram_uuB2", false);
        }
        {
            plegma::PLEGMA_ScattCorrelator<float> cont(sourceZero, momVects3pt_pf1);
            cont.T2(gscatts_N, gscatts_N, phiV31UU, propDSS_pack, propUSS_pack);
            outputfile(cont, "Diagram_uuB3", false);
        }
        {
            plegma::PLEGMA_ScattCorrelator<float> cont(sourceZero, momVects3pt_pf1);
            cont.T1(gscatts_N, gscatts_N, phiV31UU, propDSS_pack, propUSS_pack);
            outputfile(cont, "Diagram_uuB4", false);
        }
        // W
        {
            plegma::PLEGMA_ScattCorrelator<float> cont(sourceZero, momVects3pt_pf1);
            cont.T2(gscatts_N, gscatts_N, seqUSSUSS_pack, phiV32, propDSS_pack);
            outputfile(cont, "Diagram_uuW1", false);
        }
        {
            plegma::PLEGMA_ScattCorrelator<float> cont(sourceZero, momVects3pt_pf1);
            cont.T1(gscatts_N, gscatts_N, seqUSSUSS_pack, propDSS_pack, phiV32);
            outputfile(cont, "Diagram_uuW2", false);
        }
        {
            plegma::PLEGMA_ScattCorrelator<float> cont(sourceZero, momVects3pt_pf1);
            cont.T1(gscatts_N, gscatts_N, phiV32, propDSS_pack, seqUSSUSS_pack);
            outputfile(cont, "Diagram_uuW3", false);
        }
        {
            plegma::PLEGMA_ScattCorrelator<float> cont(sourceZero, momVects3pt_pf1);
            cont.T2(gscatts_N, gscatts_N, phiV32, seqUSSUSS_pack, propDSS_pack);
            outputfile(cont, "Diagram_uuW4", false);
        }
        // Z
        {
            plegma::PLEGMA_ScattCorrelator<float> cont(sourceZero, momVects3pt_pf1);
            cont.T1(gscatts_N, gscatts_N, propUSS_pack, propDSS_pack, phiV34_UUU);
            outputfile(cont, "Diagram_uuZ1", false);
        }
        {
            plegma::PLEGMA_ScattCorrelator<float> cont(sourceZero, momVects3pt_pf1);
            cont.T2(gscatts_N, gscatts_N, propUSS_pack, phiV34_UUU, propDSS_pack);
            outputfile(cont, "Diagram_uuZ2", false);
        }
        {
            plegma::PLEGMA_ScattCorrelator<float> cont(sourceZero, momVects3pt_pf1);
            cont.T2(gscatts_N, gscatts_N, phiV34_UUU, propDSS_pack, propUSS_pack);
            outputfile(cont, "Diagram_uuZ3", false);
        }
        {
            plegma::PLEGMA_ScattCorrelator<float> cont(sourceZero, momVects3pt_pf1);
            cont.T1(gscatts_N, gscatts_N, phiV34_UUU, propDSS_pack, propUSS_pack);
            outputfile(cont, "Diagram_uuZ4", false);
        }

        // p_jd_p&pi0u
        // W
        {
            plegma::PLEGMA_ScattCorrelator<float> cont(sourceZero, momVects3pt_pf1);
            cont.T1(gscatts_N, gscatts_N, propUSS_pack, phiV32_DD, seqUSSUSS_pack);
            outputfile(cont, "Diagram_duW1", false);
        }
        {
            plegma::PLEGMA_ScattCorrelator<float> cont(sourceZero, momVects3pt_pf1);
            cont.T2(gscatts_N, gscatts_N, propUSS_pack, phiV32_DD, seqUSSUSS_pack);
            outputfile(cont, "Diagram_duW2", false);
        }
        {
            plegma::PLEGMA_ScattCorrelator<float> cont(sourceZero, momVects3pt_pf1);
            cont.T2(gscatts_N, gscatts_N, seqUSSUSS_pack, phiV32_DD, propUSS_pack);
            outputfile(cont, "Diagram_duW3", false);
        }
        {
            plegma::PLEGMA_ScattCorrelator<float> cont(sourceZero, momVects3pt_pf1);
            cont.T1(gscatts_N, gscatts_N, seqUSSUSS_pack, phiV32_DD, propUSS_pack);
            outputfile(cont, "Diagram_duW4", false);
        }

        // p_ju_p&pi0d
        // W
        {
            plegma::PLEGMA_ScattCorrelator<float> cont(sourceZero, momVects3pt_pf1);
            cont.T1(gscatts_N, gscatts_N, propUSS_pack, seqDSSDSS_pack, phiV32);
            outputfile(cont, "Diagram_udW1", false);
        }
        {
            plegma::PLEGMA_ScattCorrelator<float> cont(sourceZero, momVects3pt_pf1);
            cont.T2(gscatts_N, gscatts_N, propUSS_pack, phiV32, seqDSSDSS_pack);
            outputfile(cont, "Diagram_udW2", false);
        }
        {
            plegma::PLEGMA_ScattCorrelator<float> cont(sourceZero, momVects3pt_pf1);
            cont.T2(gscatts_N, gscatts_N, phiV32, seqDSSDSS_pack, propUSS_pack);
            outputfile(cont, "Diagram_udW3", false);
        }
        {
            plegma::PLEGMA_ScattCorrelator<float> cont(sourceZero, momVects3pt_pf1);
            cont.T1(gscatts_N, gscatts_N, phiV32, seqDSSDSS_pack, propUSS_pack);
            outputfile(cont, "Diagram_udW4", false);
        }

        // p_jd_p&pi0d
        // B
        {
            plegma::PLEGMA_ScattCorrelator<float> cont(sourceZero, momVects3pt_pf1);
            cont.T2(gscatts_N, gscatts_N, propUSS_pack, phiV31DD, propUSS_pack);
            outputfile(cont, "Diagram_ddB1", false);
        }
        {
            plegma::PLEGMA_ScattCorrelator<float> cont(sourceZero, momVects3pt_pf1);
            cont.T1(gscatts_N, gscatts_N, propUSS_pack, phiV31DD, propUSS_pack);
            outputfile(cont, "Diagram_ddB2", false);
        }
        // V_reduction version
        {
            plegma::PLEGMA_Vector<float> auxVector;
            plegma::PLEGMA_ScattCorrelator<float> auxV24(sourceZero, momVects3pt_pf1);
            auxVector.copy(stocG5Xi_pack);
            // auxVector.writeHDF5(saveDir + "t1");
            // propUSS_pack.writeHDF5(saveDir + "t2");
            // propUSS.writeHDF5(saveDir + "t3");
            // auxV4.V4(auxVector, GN, propDSS_pack, propUSS_pack);

            auxV24.V2(auxVector, gscatts_N, propUSS_pack, propUSS_pack); // this and the above line won't produce the same V4, but the same cont3
            auxVector.writeHDF5(saveDir + "test_V2_1");
            propUSS_pack.writeHDF5(saveDir + "test_V2_2");
            auxV24.writeHDF5(saveDir + "test_V2");


            plegma::PLEGMA_ScattCorrelator<float> auxV3(sourceZero, momList3pt_pc);
            auxVector.copy(stocG5PhiUTfSL);
            auxV3.V3(auxVector, gscatts_c, seqDSLDSS);
            auxVector.writeHDF5(saveDir + "test_V3_1");
            seqUSLDSS.writeHDF5(saveDir + "test_V3_2");
            auxV3.writeHDF5(saveDir + "test_V3");

            plegma::PLEGMA_ScattCorrelator<float> cont(sourceZero, momList3pt);
            cont.initialize_diagram(gscatts_ID, gscatts_ID, gscatts_N, gscatts_pi, gscatts_N, gscatts_c, "12", "B12");
            cont.B_diagrams(auxV3, auxV24, 0, 7, true);
            outputfile(cont, "Diagram_ddB1_V", false);
        }
        {
            plegma::PLEGMA_Vector<float> auxVector;
            plegma::PLEGMA_ScattCorrelator<float> auxV24(sourceZero, momVects3pt_pf1);
            auxVector.copy(stocG5Xi_pack);
            // auxVector.writeHDF5(saveDir + "t1");
            // propUSS_pack.writeHDF5(saveDir + "t2");
            // propUSS.writeHDF5(saveDir + "t3");
            // auxV4.V4(auxVector, GN, propDSS_pack, propUSS_pack);
            auxV24.V2(auxVector, gscatts_N, propUSS_pack, propUSS_pack); // this and the above line won't produce the same V4, but the same cont3
            // auxV4.writeHDF5(saveDir + "auxV4");

            plegma::PLEGMA_ScattCorrelator<float> auxV3(sourceZero, momList3pt_pc);
            auxVector.copy(stocG5PhiUTfSL);
            auxV3.V3(auxVector, gscatts_c, seqDSLDSS);
            // auxVector.writeHDF5(saveDir + "auxVector");
            // seqUSLDSS.writeHDF5(saveDir + "seqUSLDSS");
            // auxV3.writeHDF5(saveDir + "auxV3");

            plegma::PLEGMA_ScattCorrelator<float> cont(sourceZero, momList3pt);
            cont.initialize_diagram(gscatts_ID, gscatts_ID, gscatts_N, gscatts_pi, gscatts_N, gscatts_c, "12", "B12");
            cont.B_diagrams(auxV3, auxV24, 0, 8, true);
            outputfile(cont, "Diagram_ddB2_V", false);
        }
        // Z
        {
            plegma::PLEGMA_ScattCorrelator<float> cont(sourceZero, momVects3pt_pf1);
            cont.T2(gscatts_N, gscatts_N, propUSS_pack, phiV34_DDD, propUSS_pack);
            outputfile(cont, "Diagram_ddZ1", false);
        }
        {
            plegma::PLEGMA_ScattCorrelator<float> cont(sourceZero, momVects3pt_pf1);
            cont.T1(gscatts_N, gscatts_N, propUSS_pack, phiV34_DDD, propUSS_pack);
            outputfile(cont, "Diagram_ddZ2", false);
        }

        // p_ju_n&pi+ B:
        plegma::PLEGMA_ScattCorrelator<float> cont1(sourceZero, momVects3pt_pf1);
        cont1.T1(gscatts_N, gscatts_N, propUSS_pack, propDSS_pack, phiV31);
        outputfile(cont1, "cont1", true); // signs: Gf^t

        plegma::PLEGMA_ScattCorrelator<float> cont2(sourceZero, momVects3pt_pf1);
        cont2.T1(gscatts_N, gscatts_N, propUSS_pack, phiV31, propDSS_pack);
        outputfile(cont2, "cont2", false);

        plegma::PLEGMA_ScattCorrelator<float> cont3(sourceZero, momVects3pt_pf1);
        cont3.T2(gscatts_N, gscatts_N, phiV31, propDSS_pack, propUSS_pack);
        outputfile(cont3, "cont3", true); // signs: Gf^t

        plegma::PLEGMA_ScattCorrelator<float> cont4(sourceZero, momVects3pt_pf1);
        cont4.T1(gscatts_N, gscatts_N, phiV31, propUSS_pack, propDSS_pack);
        outputfile(cont4, "cont4", true); // signs: Gfi1^t

        // p_ju_n&pi+ W:

        plegma::PLEGMA_ScattCorrelator<float> cont5(sourceZero, momVects3pt_pf1);
        cont5.T2(gscatts_N, gscatts_N, seqUSSDSS_pack, phiV32, propDSS_pack);
        outputfile(cont5, "cont5", true); // signs: Gfi1^t

        plegma::PLEGMA_ScattCorrelator<float> cont6(sourceZero, momVects3pt_pf1);
        cont6.T1(gscatts_N, gscatts_N, seqUSSDSS_pack, phiV32, propDSS_pack);
        outputfile(cont6, "cont6", true); // signs: Gfi1^t

        plegma::PLEGMA_ScattCorrelator<float> cont7(sourceZero, momVects3pt_pf1);
        cont7.T1(gscatts_N, gscatts_N, phiV32, propDSS_pack, seqUSSDSS_pack);
        outputfile(cont7, "cont7", true); // signs: Gfi1^t

        plegma::PLEGMA_ScattCorrelator<float> cont8(sourceZero, momVects3pt_pf1);
        cont8.T1(gscatts_N, gscatts_N, phiV32, seqUSSDSS_pack, propDSS_pack);
        outputfile(cont8, "cont8", false); // signs:

        // p_jd_n&pi+ W:

        plegma::PLEGMA_ScattCorrelator<float> cont9(sourceZero, momVects3pt_pf1);
        cont9.T1(gscatts_N, gscatts_N, propUSS_pack, phiV33, seqUSSDSS_pack);
        outputfile(cont9, "cont9", true); // signs: Gfi1^t

        plegma::PLEGMA_ScattCorrelator<float> cont10(sourceZero, momVects3pt_pf1);
        cont10.T1(gscatts_N, gscatts_N, propUSS_pack, seqUSSDSS_pack, phiV33);
        outputfile(cont10, "cont10", false); // signs:

        plegma::PLEGMA_ScattCorrelator<float> cont11(sourceZero, momVects3pt_pf1);
        cont11.T2(gscatts_N, gscatts_N, seqUSSDSS_pack, phiV33, propUSS_pack);
        outputfile(cont11, "cont11", true); // signs: Gfi1^t

        plegma::PLEGMA_ScattCorrelator<float> cont12(sourceZero, momVects3pt_pf1);
        cont12.T1(gscatts_N, gscatts_N, seqUSSDSS_pack, propUSS_pack, phiV33);
        outputfile(cont12, "cont12", true); // signs: Gfi1^t

        // p_jd_n&pi+ Z:

        plegma::PLEGMA_ScattCorrelator<float> cont13(sourceZero, momVects3pt_pf1);
        cont13.T1(gscatts_N, gscatts_N, propUSS_pack, propDSS_pack, phiV34);
        outputfile(cont13, "cont13", true); // signs: Gfi1^t

        plegma::PLEGMA_ScattCorrelator<float> cont14(sourceZero, momVects3pt_pf1);
        cont14.T1(gscatts_N, gscatts_N, propUSS_pack, phiV34, propDSS_pack);
        outputfile(cont14, "cont14", false); // signs:

        plegma::PLEGMA_ScattCorrelator<float> cont15(sourceZero, momVects3pt_pf1);
        cont15.T2(gscatts_N, gscatts_N, phiV34, propDSS_pack, propUSS_pack);
        outputfile(cont15, "cont15", true); // signs: Gfi1^t

        plegma::PLEGMA_ScattCorrelator<float> cont16(sourceZero, momVects3pt_pf1);
        cont16.T1(gscatts_N, gscatts_N, phiV34, propUSS_pack, propDSS_pack);
        outputfile(cont16, "cont16", true); // signs: Gfi1^t

        // V reductions version
        // { // B12
        //     plegma::PLEGMA_Vector<float> auxVector;
        //     plegma::PLEGMA_ScattCorrelator<float> auxV4(sourceZero, momVects3pt_pf1);
        //     auxVector.copy(stocG5Xi_pack);
        //     auxVector.writeHDF5(saveDir + "t1");
        //     propUSS_pack.writeHDF5(saveDir + "t2");
        //     propUSS.writeHDF5(saveDir + "t3");
        //     // auxV4.V4(auxVector, GN, propDSS_pack, propUSS_pack);
        //     auxV4.V4(auxVector, GN, propUSS_pack, propDSS_pack); // this and the above line won't produce the same V4, but the same cont3
        //     auxV4.writeHDF5(saveDir + "auxV4");

        //     plegma::PLEGMA_ScattCorrelator<float> auxV3(sourceZero, momList3pt_pc);
        //     auxVector.copy(stocG5PhiDTfSL);
        //     auxV3.V3(auxVector, Gc, seqUSLDSS);
        //     // auxVector.writeHDF5(saveDir + "auxVector");
        //     // seqUSLDSS.writeHDF5(saveDir + "seqUSLDSS");
        //     auxV3.writeHDF5(saveDir + "auxV3");

        //     plegma::PLEGMA_ScattCorrelator<float> cont3V(sourceZero, momList);
        //     cont3V.initialize_diagram(GID, GID, GN, Gpi, GN, Gc, "12", "B12");
        //     cont3V.B_diagrams(auxV3, auxV4, 0, 12, true);
        //     outputfile(cont3V, "cont3V", false);
        // }

        // { // Z11
        //     plegma::PLEGMA_Vector<float> auxVector;
        //     plegma::PLEGMA_ScattCorrelator<float> auxV24(sourceZero, momVects3pt_pf1);
        //     auxVector.copy(stocPhiUG5Gi2TiSS_pack);
        //     auxV24.V2(auxVector, GN, propDSS_pack, propUSS_pack);
        //     auxV24.writeHDF5(saveDir + "Z11auxV24");

        //     plegma::PLEGMA_ScattCorrelator<float> auxV3(sourceZero, momList3pt_pc);
        //     auxVector.copy(stocG5PhiUTiSL);
        //     auxV3.V3(auxVector, Gc, propDSL);
        //     auxV3.writeHDF5(saveDir + "Z11auxV3");

        //     plegma::PLEGMA_ScattCorrelator<float> contZ11(sourceZero, momList);
        //     contZ11.initialize_diagram(GID, GID, GN, Gpi, GN, Gc, "12", "B12");
        //     contZ11.Z_diagrams_without_dilution(auxV3, auxV24, 0, 11);
        //     outputfile(contZ11, "contZ11", false);
        // }
    }

    finalize();
    return 0;
}
