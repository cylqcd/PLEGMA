#include <ToolYan.h>

/*
    N=Nucleon; p=proton; n=neutron
    P=pion; [pp]=pion+; [pm]=pion-; [p0]=pion0; [pu]=pion0u; [pd]=pion0d
    J=current; [ju]; [jd]
    i=initial; f=finial; c=current
    1=Nucleon; 2=pion
*/

extern int device;
static std::vector<std::string> listOpt = {"verbosity", "nsmear-APE", "alpha-APE", "nsmear-gauss", "alpha-gauss", "load-gauge", "nsrc", "src-filename", "momlist-filename", "readStochSamples", "time-dilution", "nstochSamples", "confnumber", "contractionstoch", "contractionstd", "contractionoet"};

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
    // ==== initialization
    ToolYan tyan;

    initializeOptions(argc, argv, true, listOpt);
    initializePLEGMA();

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

    tyan.setupGauge(gauge);
    tyan.setupSmearing(smearedGauge, nsmearGauss, alphaGauss);
    tyan.setupSolver(solver);
    // tyan.testTimeCost();

    // initialization ====

    // ==== general definitions
    std::string savingPath = "tempData/";

    struct plegma::site source({12, 14, 17, 9 + 12 * 3});
    int tSource = source[DIM_T];
    struct plegma::site sourceZero({0, 0, 0, tSource});

    // old version
    plegma::momList momList(3, pathListMomenta, {2});
    auto pi2List = momList.uniq_p(0);
    auto pf1List = momList.uniq_p(1);
    // pf2 + pf1 = 0
    auto pcList = momList.uniq_p(2);
    plegma::momList pcMomList(1, {pcList}, {0});
    plegma::momList pf1MomList(1, {pf1List}, {0});

    std::vector<int> pi1List = {0, 0, 0};

    std::vector<plegma::GAMMAS_SCATT> GID = {ID};
    std::vector<plegma::GAMMAS_SCATT> GN = {CG_5};
    std::vector<plegma::GAMMAS_SCATT> Gpi = {G_5};
    std::vector<plegma::GAMMAS_SCATT> Gc = {ID};
    // std::vector<plegma::GAMMAS> Gc_GAMMAS = {ONE, G1, G2, G3, G4, G5, G5G1, G5G2, G5G3, G5G4};
    // std::vector<plegma::GAMMAS_SCATT> Gc = {ID, G_1, G_2, G_3, G_4, G_5, G_5_G_1, G_5_G_2, G_5_G_3, G_5_G_4};
    // std::vector<plegma::GAMMAS> Gc_GAMMAS = {ONE, G1, G2, G3, G4, G5, G5G1, G5G2, G5G3, G5G4};

    int time_fi = 6;
    int time_i = source[DIM_T], time_f = (source[DIM_T] + time_fi) % HGC_totalL[3];

    // general definitions ====

    tyan.printTime("setupPointPropagator");

    plegma::PLEGMA_Propagator<float> propUSS, propUSL;
    plegma::PLEGMA_Propagator<float> propDSS, propDSL;

    tyan.setupPointPropagator(propUSS, propUSL, source, u, SB);
    tyan.setupPointPropagator(propDSS, propDSL, source, d, SB);

    plegma::PLEGMA_Propagator<float> propUSS_pack, propUSL_pack;
    plegma::PLEGMA_Propagator<float> propDSS_pack, propDSL_pack;
    packPropogator(propUSS_pack, propUSS, time_f);
    packPropogator(propUSL_pack, propUSL, time_f);
    packPropogator(propDSS_pack, propDSS, time_f);
    packPropogator(propDSL_pack, propDSL, time_f);

    propUSS.writeHDF5(savingPath + "propUSS");
    propUSS_pack.writeHDF5(savingPath + "propUSS_pack");

    // propDSS_pack.writeHDF5(savingPath + "propDSS_pack");

    // {
    //     if(mu<0)
    //     {
    //         mu=-mu;
    //         solver.UpdateSolver();
    //     }
    //     plegma::PLEGMA_Vector<double> auxVector, auxVector1;
    //     plegma::PLEGMA_Vector3D<double> auxVector3D, auxVector3D1;
    //     plegma::PLEGMA_Gauge3D<double> smearedGauge3D;
    //     smearedGauge3D.absorb(smearedGauge, 0);

    //     auxVector3D.pointSource(source, 0, 0, DEVICE);
    //     auxVector3D1.gaussianSmearing(auxVector3D, smearedGauge3D, nsmearGauss, alphaGauss);
    //     auxVector.absorb(auxVector3D1, 0);
    //     auxVector.writeHDF5(savingPath+"pointSmear");

    //     auxVector1.rotateToPhysicalBasis(auxVector,+1);
    //     solver.solve(auxVector1,auxVector1);
    //     auxVector.rotateToPhysicalBasis(auxVector1,+1);
    //     auxVector1.gaussianSmearing(auxVector, smearedGauge, nsmearGauss, alphaGauss);
    //     auxVector1.writeHDF5(savingPath+"aux_propU-SS");
    // }

    // {
    //     plegma::PLEGMA_Propagator<float> propULS, propULL;
    //     tyan.setupPointPropagator(propULS, propULL, source, u, LB);

    //     plegma::PLEGMA_Vector<float> aux;
    //     aux.absorb(propUSS, 0, 0);
    //     aux.writeHDF5(savingPath + "propUSS-00");
    //     aux.absorb(propULL, 0, 0);
    //     aux.writeHDF5(savingPath + "propULL-00");
    //     aux.absorb(propUSL, 0, 0);
    //     aux.writeHDF5(savingPath + "propUSL-00");

    //     // N Diagram
    //     plegma::PLEGMA_ScattCorrelator<float> T1Contraction(sourceZero, pf1List);
    //     plegma::PLEGMA_ScattCorrelator<float> T2Contraction(sourceZero, pf1List);
    //     T1Contraction.T1(GN, GN, propDSS, propUSS, propDSS);
    //     // T1Contraction.writeHDF5(savingPath + "T1");
    //     T2Contraction.T2(GN, GN, propDSS, propUSS, propDSS);
    //     // T2Contraction.writeHDF5(savingPath + "T2");

    //     plegma::PLEGMA_ScattCorrelator<float> corr_NN(source, pf1MomList);
    //     corr_NN.initialize_diagram(GID, GID, GN, GN, "N0");
    //     corr_NN.N_diagrams(T1Contraction, T2Contraction);
    //     corr_NN.writeHDF5(savingPath + "NN");
    // }

    tyan.printTime("setupSequentialPropagator");

    PLEGMA_Propagator<float> seqUSSDSS, seqUSLDSS;
    PLEGMA_Propagator<float> seqUSSDSS_pack, seqUSLDSS_pack;
    tyan.setupSequentialPropagator(seqUSSDSS, seqUSLDSS, propDSS, time_i, G5, pi2List[0], +1, u, SB);
    packPropogator(seqUSSDSS_pack, seqUSSDSS, time_f);
    packPropogator(seqUSLDSS_pack, seqUSLDSS, time_f);

    tyan.printTime("stoc");

    plegma::PLEGMA_Vector<double> stocXi, stocG5Xi;
    plegma::PLEGMA_Vector<double> stocXi_pack, stocG5Xi_pack;

    plegma::PLEGMA_Vector<double> stocPhiUTfSL, stocG5PhiUTfSL;
    plegma::PLEGMA_Vector<double> stocPhiDTfSL, stocG5PhiDTfSL;

    plegma::PLEGMA_Vector<double> stocPhiUTiSL, stocG5PhiUTiSL;
    plegma::PLEGMA_Vector<double> stocPhiUG5Gi2TiSS;
    plegma::PLEGMA_Vector<double> stocPhiUG5Gi2TiSS_pack;
    {
        plegma::PLEGMA_Vector3D<double> auxVector3D;
        stocXi.randInit(1234);
        stocXi.stochastic_Z(4);
        stocG5Xi.copy(stocXi);
        stocG5Xi.apply_gamma5();
        //
        packVectorD(stocXi_pack, stocXi, time_f);
        packVectorD(stocG5Xi_pack, stocG5Xi, time_f);

        stocXi.writeHDF5(savingPath + "stocXi");
        stocXi_pack.writeHDF5(savingPath + "stocXi_pack");
        tyan.printTime(std::to_string(HGC_totalL[3]));

        auxVector3D.absorb(stocXi, time_f);
        tyan.solve(stocPhiUTfSL, auxVector3D, time_f, u, SL);
        stocG5PhiUTfSL.copy(stocPhiUTfSL);
        stocG5PhiUTfSL.apply_gamma5();
        //
        auxVector3D.absorb(stocXi, time_f);
        tyan.solve(stocPhiDTfSL, auxVector3D, time_f, d, SL);
        stocG5PhiDTfSL.copy(stocPhiDTfSL);
        stocG5PhiDTfSL.apply_gamma5();

        auxVector3D.absorb(stocXi, time_i);
        auxVector3D.mulMomentumPhases(pi2List[0], -1);
        tyan.solve(stocPhiUTiSL, auxVector3D, time_i, u, SL);
        stocG5PhiUTiSL.copy(stocPhiUTiSL);
        stocG5PhiUTiSL.apply_gamma5();
        //
        auxVector3D.absorb(stocXi, time_i);
        // apply g5*g5=ID
        // auxVector3D.mulMomentumPhases(pi2List[0],+1);
        tyan.solve(stocPhiUG5Gi2TiSS, auxVector3D, time_i, u, SS);
        packVectorD(stocPhiUG5Gi2TiSS_pack, stocPhiUG5Gi2TiSS, time_f);
    }

    tyan.printTime("phiV3");

    PLEGMA_Propagator<float> phiV31;
    {
        plegma::PLEGMA_ScattCorrelator<float> auxV3(sourceZero, pcMomList);
        plegma::PLEGMA_Vector<float> auxVector;
        auxVector.copy(stocG5PhiDTfSL);
        auxV3.V3(auxVector, Gc, seqUSLDSS);
        // auxV3.writeHDF5(savingPath + "temp1");

        for (int isc = 0; isc < 12; isc++)
        {
            plegma::PLEGMA_Vector<float> auxVector1;
            auxVector1.zero_where(BOTH);
            plegma::PLEGMA_Vector3D<double> auxVector3DD;
            plegma::PLEGMA_Vector3D<float> auxVector3D;
            auxVector3DD.absorb(stocG5Xi_pack, time_f, true);
            auxVector3D.copy(auxVector3DD);
            // if (isc == 5)
            //     auxVector3D.writeHDF5(savingPath + "temp2");
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
            //     auxVector1.writeHDF5(savingPath + "temp3");
            phiV31.absorb(auxVector1, isc / 3, isc % 3);
        }
        phiV31.writeHDF5(savingPath + "temp4");
    }

    PLEGMA_Propagator<float> phiV32;
    {
        plegma::PLEGMA_ScattCorrelator<float> auxV3(sourceZero, pcMomList);
        plegma::PLEGMA_Vector<float> auxVector;
        auxVector.copy(stocG5PhiDTfSL);
        auxV3.V3(auxVector, Gc, propUSL);

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

    PLEGMA_Propagator<float> phiV33;
    {
        plegma::PLEGMA_ScattCorrelator<float> auxV3(sourceZero, pcMomList);
        plegma::PLEGMA_Vector<float> auxVector;
        auxVector.copy(stocG5PhiUTfSL);
        auxV3.V3(auxVector, Gc, propDSL);

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
        plegma::PLEGMA_ScattCorrelator<float> auxV3(sourceZero, pcMomList);
        plegma::PLEGMA_Vector<float> auxVector;
        auxVector.copy(stocG5PhiUTiSL);
        auxV3.V3(auxVector, Gc, propDSL);

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

    tyan.printTime("cont");

    auto outputfile = [&](plegma::PLEGMA_ScattCorrelator<float> sc, std::string filename, bool sgn)
    {
        // momentum phase for pi1
        float phase = 2 * M_PI / (float)HGC_totalL[0] * pi1List[0] * source[0] +
                      2 * M_PI / (float)HGC_totalL[1] * pi1List[1] * source[1] +
                      2 * M_PI / (float)HGC_totalL[2] * pi1List[2] * source[2];
        float expPhase[2] = {cos(phase), sin(phase)};
        x_e_cx<float>(sc.H_elem(), expPhase, sc.getTotalSize());

        // additional sign coming from my Mathematica codes
        float overall_sign[2] = {-1., 0.};
        if (sgn)
            x_e_cx<float>(sc.H_elem(), overall_sign, sc.getTotalSize());

        if (time_f < time_i)
            x_e_cx<float>(sc.H_elem(), overall_sign, sc.getTotalSize());
        // sc.applyBoundaryConditions(true);
        sc.writeHDF5(savingPath + filename);
    };

    float overall_sign[2] = {-1., 0.};
    // x_e_cx<float>( cont.H_elem(), overall_sign, cont.getTotalSize());

    plegma::PLEGMA_ScattCorrelator<float> cont1(sourceZero, pf1List);
    cont1.T1(GN, GN, propUSS_pack, propDSS_pack, phiV31);
    outputfile(cont1, "cont1", true); // signs: Gf^t

    plegma::PLEGMA_ScattCorrelator<float> cont2(sourceZero, pf1List);
    cont2.T1(GN, GN, propUSS_pack, phiV31, propDSS_pack);
    outputfile(cont2, "cont2", false);

    plegma::PLEGMA_ScattCorrelator<float> cont3(sourceZero, pf1List);
    cont3.T2(GN, GN, phiV31, propDSS_pack, propUSS_pack);
    outputfile(cont3, "cont3", true); // signs: Gf^t

    plegma::PLEGMA_ScattCorrelator<float> cont4(sourceZero, pf1List);
    cont4.T1(GN, GN, phiV31, propUSS_pack, propDSS_pack);
    outputfile(cont4, "cont4", true); // signs: Gfi1^t

    //

    plegma::PLEGMA_ScattCorrelator<float> cont5(sourceZero, pf1List);
    cont5.T2(GN, GN, seqUSSDSS_pack, phiV32, propDSS_pack);
    outputfile(cont5, "cont5", true); // signs: Gfi1^t

    plegma::PLEGMA_ScattCorrelator<float> cont6(sourceZero, pf1List);
    cont6.T1(GN, GN, seqUSSDSS_pack, phiV32, propDSS_pack);
    outputfile(cont6, "cont6", true); // signs: Gfi1^t

    plegma::PLEGMA_ScattCorrelator<float> cont7(sourceZero, pf1List);
    cont7.T1(GN, GN, phiV32, propDSS_pack, seqUSSDSS_pack);
    outputfile(cont7, "cont7", true); // signs: Gfi1^t

    plegma::PLEGMA_ScattCorrelator<float> cont8(sourceZero, pf1List);
    cont8.T1(GN, GN, phiV32, seqUSSDSS_pack, propDSS_pack);
    outputfile(cont8, "cont8", false); // signs:

    //

    plegma::PLEGMA_ScattCorrelator<float> cont9(sourceZero, pf1List);
    cont9.T1(GN, GN, propUSS_pack, phiV33, seqUSSDSS_pack);
    outputfile(cont9, "cont9", true); // signs: Gfi1^t

    plegma::PLEGMA_ScattCorrelator<float> cont10(sourceZero, pf1List);
    cont10.T1(GN, GN, propUSS_pack, seqUSSDSS_pack, phiV33);
    outputfile(cont10, "cont10", false); // signs:

    plegma::PLEGMA_ScattCorrelator<float> cont11(sourceZero, pf1List);
    cont11.T2(GN, GN, seqUSSDSS_pack, phiV33, propUSS_pack);
    outputfile(cont11, "cont11", true); // signs: Gfi1^t

    plegma::PLEGMA_ScattCorrelator<float> cont12(sourceZero, pf1List);
    cont12.T1(GN, GN, seqUSSDSS_pack, propUSS_pack, phiV33);
    outputfile(cont12, "cont12", true); // signs: Gfi1^t

    //

    plegma::PLEGMA_ScattCorrelator<float> cont13(sourceZero, pf1List);
    cont13.T1(GN, GN, propUSS_pack, propDSS_pack, phiV34);
    outputfile(cont13, "cont13", true); // signs: Gfi1^t

    plegma::PLEGMA_ScattCorrelator<float> cont14(sourceZero, pf1List);
    cont14.T1(GN, GN, propUSS_pack, phiV34, propDSS_pack);
    outputfile(cont14, "cont14", false); // signs:

    plegma::PLEGMA_ScattCorrelator<float> cont15(sourceZero, pf1List);
    cont15.T2(GN, GN, phiV34, propDSS_pack, propUSS_pack);
    outputfile(cont15, "cont15", true); // signs: Gfi1^t

    plegma::PLEGMA_ScattCorrelator<float> cont16(sourceZero, pf1List);
    cont16.T1(GN, GN, phiV34, propUSS_pack, propDSS_pack);
    outputfile(cont16, "cont16", true); // signs: Gfi1^t

    // for (int spin = 0; spin < 4; spin++)
    // {

    //     plegma::PLEGMA_Vector<double> sd_stocXi,sd_aux;
    //     plegma::PLEGMA_Vector<double> sd_stocPhiUTfSL, sd_stocG5PhiUTfSL;
    //     plegma::PLEGMA_Vector<double> sd_stocPhiUG5Gi2TfSS;
    //     plegma::PLEGMA_Vector<double> sd_stocPhiUG5Gi2TfSS_pack;
    //     {
    //         plegma::PLEGMA_Vector3D<double> sd_auxVector3D;
    //         sd_aux.randInit(1234);
    //         sd_aux.stochastic_Z(4);
    //         sd_stocXi.diluteSpinDisplace(sd_aux,spin,0);

    //         sd_auxVector3D.absorb(sd_stocXi, time_f);
    //         tyan.solve(sd_stocPhiUTfSL, sd_auxVector3D, time_f, u, SL);
    //         sd_stocG5PhiUTfSL.copy(sd_stocPhiUTfSL);
    //         sd_stocG5PhiUTfSL.apply_gamma5();

    //         sd_auxVector3D.absorb(sd_stocXi, time_f);
    //         // apply g5*g5=ID
    //         tyan.solve(sd_stocPhiUG5Gi2TfSS, sd_auxVector3D, time_f, u, SS);
    //         packVectorD(sd_stocPhiUG5Gi2TfSS_pack, sd_stocPhiUG5Gi2TfSS, time_f);
    //     }

    //     PLEGMA_Propagator<float> sd_phiV34;
    //     {
    //         plegma::PLEGMA_ScattCorrelator<float> auxV3(sourceZero, pcMomList);
    //         plegma::PLEGMA_Vector<float> auxVector;
    //         auxVector.copy(sd_stocG5PhiUTfSL);
    //         auxV3.V3(auxVector, Gc, propDSL);

    //         for (int isc = 0; isc < 12; isc++)
    //         {
    //             plegma::PLEGMA_Vector<float> auxVector1;
    //             plegma::PLEGMA_Vector3D<double> auxVector3DD;
    //             plegma::PLEGMA_Vector3D<float> auxVector3D;
    //             auxVector3DD.absorb(sd_stocPhiUG5Gi2TfSS_pack, time_f);
    //             auxVector3D.copy(auxVector3DD);
    //             for (int dt = 0; dt < HGC_totalL[3]; dt++)
    //             {
    //                 {
    //                     plegma::PLEGMA_Vector3D<float> auxVector3D1;
    //                     auxVector3D1.copy(auxVector3D);
    //                     float *aux = auxV3.Corr(dt, 0, 0);
    //                     std::complex<float> auxC(aux[2 * isc + 0], aux[2 * isc + 1]);
    //                     auxVector3D1.cscale(auxC);
    //                     auxVector1.absorb(auxVector3D1, dt, false);
    //                 }
    //             }
    //             sd_phiV34.absorb(auxVector1, isc / 3, isc % 3);
    //         }
    //     }

    //     plegma::PLEGMA_ScattCorrelator<float> sd_cont13(sourceZero, pf1List);
    //     sd_cont13.T1(GN, GN, propUSS_pack, propDSS_pack, sd_phiV34); // signs: Gfi1^t
    //     x_e_cx<float>(sd_cont13.H_elem(), overall_sign, sd_cont13.getTotalSize());
    //     // sd_cont13.apply_phase();
    //     sd_cont13.applyBoundaryConditions(true);
    //     sd_cont13.writeHDF5(savingPath + "cont13_" + std::to_string(spin));

    //     plegma::PLEGMA_ScattCorrelator<float> sd_cont14(sourceZero, pf1List);
    //     sd_cont14.T1(GN, GN, propUSS_pack, sd_phiV34, propDSS_pack); // signs:
    //     // sd_cont14.apply_phase();
    //     sd_cont14.applyBoundaryConditions(true);
    //     sd_cont14.writeHDF5(savingPath + "cont14_" + std::to_string(spin));

    //     plegma::PLEGMA_ScattCorrelator<float> sd_cont15(sourceZero, pf1List);
    //     sd_cont15.T2(GN, GN, sd_phiV34, propDSS_pack, propUSS_pack); // signs: Gfi1^t
    //     x_e_cx<float>(sd_cont15.H_elem(), overall_sign, sd_cont15.getTotalSize());
    //     // sd_cont15.apply_phase();
    //     sd_cont15.applyBoundaryConditions(true);
    //     sd_cont15.writeHDF5(savingPath + "cont15_" + std::to_string(spin));

    //     plegma::PLEGMA_ScattCorrelator<float> sd_cont16(sourceZero, pf1List);
    //     sd_cont16.T1(GN, GN, sd_phiV34, propUSS_pack, propDSS_pack); // signs: Gfi1^t
    //     x_e_cx<float>(sd_cont16.H_elem(), overall_sign, sd_cont16.getTotalSize());
    //     // sd_cont16.apply_phase();
    //     sd_cont16.applyBoundaryConditions(true);
    //     sd_cont16.writeHDF5(savingPath + "cont16_" + std::to_string(spin));
    // }

    // V reductions version
    { // B12
        plegma::PLEGMA_Vector<float> auxVector;
        plegma::PLEGMA_ScattCorrelator<float> auxV4(sourceZero, pf1List);
        auxVector.copy(stocG5Xi_pack);
        auxVector.writeHDF5(savingPath + "t1");
        propUSS_pack.writeHDF5(savingPath + "t2");
        propUSS.writeHDF5(savingPath + "t3");
        // auxV4.V4(auxVector, GN, propDSS_pack, propUSS_pack);
        auxV4.V4(auxVector, GN, propUSS_pack, propDSS_pack); // this and the above line won't produce the same V4, but the same cont3
        auxV4.writeHDF5(savingPath + "auxV4");

        plegma::PLEGMA_ScattCorrelator<float> auxV3(sourceZero, pcMomList);
        auxVector.copy(stocG5PhiDTfSL);
        auxV3.V3(auxVector, Gc, seqUSLDSS);
        // auxVector.writeHDF5(savingPath + "auxVector");
        // seqUSLDSS.writeHDF5(savingPath + "seqUSLDSS");
        auxV3.writeHDF5(savingPath + "auxV3");

        plegma::PLEGMA_ScattCorrelator<float> cont3V(sourceZero, momList);
        cont3V.initialize_diagram(GID, GID, GN, Gpi, GN, Gc, "12", "B12");
        cont3V.B_diagrams(auxV3, auxV4, 0, 12, true);
        cont3V.apply_sign("4pt");
        cont3V.applyBoundaryConditions(true);
        cont3V.writeHDF5(savingPath + "cont3V");
    }

    { // Z11
        plegma::PLEGMA_Vector<float> auxVector;
        plegma::PLEGMA_ScattCorrelator<float> auxV24(sourceZero, pf1List);
        auxVector.copy(stocPhiUG5Gi2TiSS_pack);
        auxV24.V2(auxVector, GN, propDSS_pack, propUSS_pack);
        auxV24.writeHDF5(savingPath + "Z11auxV24");

        plegma::PLEGMA_ScattCorrelator<float> auxV3(sourceZero, pcMomList);
        auxVector.copy(stocG5PhiUTiSL);
        auxV3.V3(auxVector, Gc, propDSL);
        auxV3.writeHDF5(savingPath + "Z11auxV3");

        plegma::PLEGMA_ScattCorrelator<float> contZ11(sourceZero, momList);
        contZ11.initialize_diagram(GID, GID, GN, Gpi, GN, Gc, "12", "B12");
        contZ11.Z_diagrams_without_dilution(auxV3, auxV24, 0, 11);
        contZ11.apply_sign("4pt");
        contZ11.applyBoundaryConditions(true);
        contZ11.writeHDF5(savingPath + "contZ11");
    }

    tyan.printTime("beforeFinalize");
    finalize();

    return 0;
}
