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
    //==================================
    // Initialiazation
    //==================================

    initializeOptions(argc, argv, true, listOpt);

    int confnumber_int, boundaryType;
    int Nstoc, seed, Noet;
    int whatToDo;
    std::string outdiagramPrefix, flagFinish;
    HGC_options->set("flagFinish", "An empty file created indicating the completion of a run", verbosity, flagFinish);
    HGC_options->set("confnumber", "Integer determining the index of the gauge configuration", verbosity, confnumber_int);
    HGC_options->set("outdiagramPrefix", "Prefix of the resulting diagrams", verbosity, outdiagramPrefix);
    HGC_options->set("boundaryType", "1:antiperiodic, 2:p+a", verbosity, boundaryType);
    assert(boundaryType==1 || boundaryType==2);
    HGC_options->set("Nstoc", "Number of time-diluted stochastic sources", verbosity, Nstoc);
    HGC_options->set("seed", "seed for stochastic sources", verbosity, seed);
    HGC_options->set("Noet", "Noet<=Nstoc", verbosity, Noet);
    assert(Noet<=Nstoc);
    HGC_options->set("whatToDo", "1:NBWZ, 2:N, 3:P", verbosity, whatToDo);
    assert(whatToDo==1 || whatToDo==2 || whatToDo==3);

    int nroots=4; // Z_{nroots} stochastic source

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
        quda::QUDA_solver solver_a(mu);

        updateGaugeQuda(gauge, false);
        plaqQuda();
        updateOptions(LIGHT);
        quda::QUDA_solver solver_p(mu);

        int currentSolver = -1; // -1 for a, +1 for p; it should be changed only when updateSolver is called
        updateGaugeQuda(gauge, true);
        solver_a.UpdateSolver();

#if 1 // Utilities

#if 1 // general
        bool smearQ = true;

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
        auto solve0 = [&](plegma::PLEGMA_Vector<double> &out, plegma::PLEGMA_Vector<double> &in, int solverType)
        {
            assert( solverType==-1 || solverType==+1 );
            plegma::PLEGMA_Vector<double> auxVector;
            
            auxVector.copy(in);
            double norm = auxVector.norm();
            auxVector.scale(1 / norm);

            if(solverType==-1)
            {
                if(currentSolver!=solverType)
                {
                    currentSolver=solverType;
                    updateGaugeQuda(gauge,true);
                    solver_a.UpdateSolver();
                }
                solver_a.solve(out, auxVector);
                out.scale(norm);
            }
            else
            {
                if(currentSolver!=solverType)
                {
                    currentSolver=solverType;
                    updateGaugeQuda(gauge,false);
                    solver_p.UpdateSolver();
                }
                solver_p.solve(out, auxVector);
                out.scale(norm);
            }
        };

        auto solve1 = [&](plegma::PLEGMA_Vector<double> &out, plegma::PLEGMA_Vector3D<double> &in, int inTime, int solverType, SmearFlagYan s)
        {
            assert(s < single2double);
            if (!smearQ)
            {
                plegma::PLEGMA_Vector<double> auxVector;
                auxVector.absorb(in, inTime);
                solve0(out, auxVector, solverType);
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
            solve0(auxVector, auxVector, solverType);

            if (s == LS || s == SS)
            {
                out.gaussianSmearing(auxVector, smearedGauge, nsmearGauss, alphaGauss);
            }
            else
            {
                out.copy(auxVector);
            }
        };
        auto solve2 = [&](plegma::PLEGMA_Vector<double> &outS, plegma::PLEGMA_Vector<double> &outL, plegma::PLEGMA_Vector3D<double> &in, int inTime, int solverType, SmearFlagYan s)
        {
            assert(single2double < s);
            if (!smearQ)
            {
                plegma::PLEGMA_Vector<double> auxVector;
                auxVector.absorb(in, inTime);
                solve0(outL, auxVector, solverType);
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

            solve0(auxVector, auxVector, solverType);

            outL.copy(auxVector);
            outS.gaussianSmearing(auxVector, smearedGauge, nsmearGauss, alphaGauss);
        };

        auto applyBoundaryCondition = [&](plegma::PLEGMA_Vector<float> &vec, int inTime, int srcTime)
        {
            // Let p0 and a0 be the propagator before applying the boundary condition
            // Let p and a be the propagator after applying the boundary condition
            // p+a gives 0~T-1 while p-a gives T~2T-1
            // Then p = p0, but a = +/- a0
            // ( inTime>=srcTime?+1:-1 ) and ( t>=srcTime?+1:-1 ) convert a0 to a
            // The other sign is meant to convert a to -a for the backward part
            int inSign = ( inTime>=srcTime?+1:-1 ) * ( (inTime-srcTime+HGC_totalL[DIM_T])%HGC_totalL[DIM_T]<HGC_totalL[DIM_T]/2?+1:-1 );
            plegma::PLEGMA_Vector3D<float> aux3D;
            for (int t=0; t<HGC_totalL[DIM_T]; t++)
            {
                int sign = inSign *  ( t>=srcTime?+1:-1 ) * ( (t-srcTime+HGC_totalL[DIM_T])%HGC_totalL[DIM_T]<HGC_totalL[DIM_T]/2?+1:-1 );

                if(sign==-1)
                {
                    aux3D.absorb(vec,t);
                    aux3D.scale(-1.);
                    vec.absorb(aux3D,t,false);
                }
            }
        };

        auto applyBoundaryConditionDouble = [&](plegma::PLEGMA_Vector<double> &vec, int inTime, int srcTime)
        {
            // see applyBoundaryCondition
            int inSign = ( inTime>=srcTime?+1:-1 ) * ( (inTime-srcTime+HGC_totalL[DIM_T])%HGC_totalL[DIM_T]<HGC_totalL[DIM_T]/2?+1:-1 );
            plegma::PLEGMA_Vector3D<double> aux3D;
            for (int t=0; t<HGC_totalL[DIM_T]; t++)
            {
                int sign = inSign *  ( t>=srcTime?+1:-1 ) * ( (t-srcTime+HGC_totalL[DIM_T])%HGC_totalL[DIM_T]<HGC_totalL[DIM_T]/2?+1:-1 );

                if(sign==-1)
                {
                    aux3D.absorb(vec,t);
                    aux3D.scale(-1.);
                    vec.absorb(aux3D,t,false);
                }
            }
        };

        auto applyBoundaryConditionProp = [&](plegma::PLEGMA_Propagator<float> &prop, int inTime, int srcTime)
        {
            // see applyBoundaryCondition
            int inSign = ( inTime>=srcTime?+1:-1 ) * ( (inTime-srcTime+HGC_totalL[DIM_T])%HGC_totalL[DIM_T]<HGC_totalL[DIM_T]/2?+1:-1 );
            plegma::PLEGMA_Propagator3D<float> aux3D;
            for (int t=0; t<HGC_totalL[DIM_T]; t++)
            {
                int sign = inSign *  ( t>=srcTime?+1:-1 ) * ( (t-srcTime+HGC_totalL[DIM_T])%HGC_totalL[DIM_T]<HGC_totalL[DIM_T]/2?+1:-1 );
                if(sign==-1)
                {
                    aux3D.absorb(prop,t);
                    aux3D.scale(-1.);
                    prop.absorb(aux3D,t,false);
                }
            }
        };

        auto applyBoundaryConditionA = [&](plegma::PLEGMA_Vector<float> &vec, int inTime, int srcTime)
        {
            int inSign = ( inTime>=srcTime?+1:-1 );
            plegma::PLEGMA_Vector3D<float> aux3D;
            for (int t=0; t<HGC_totalL[DIM_T]; t++)
            {
                int sign = inSign *  ( t>=srcTime?+1:-1 );

                if(sign==-1)
                {
                    aux3D.absorb(vec,t);
                    aux3D.scale(-1.);
                    vec.absorb(aux3D,t,false);
                }
            }
        };
#endif

#if 1 // setupPointPropagator
        auto setupPointPropagator1 = [&](plegma::PLEGMA_Propagator<float> &out, plegma::site &source, int solverType, SmearFlagYan s)
        {
            assert(s < single2double);

            plegma::PLEGMA_Vector<double> auxVector;
            plegma::PLEGMA_Vector<float> auxVectorF;
            plegma::PLEGMA_Vector3D<double> auxVector3D;
            for (int isc = 0; isc < 12; isc++)
            {
                auxVector3D.pointSource(source, isc / 3, isc % 3, DEVICE);
                solve1(auxVector, auxVector3D, source[DIM_T], solverType, s);
                auxVectorF.copy(auxVector);
                out.absorb(auxVectorF, isc / 3, isc % 3);
            }
        };
        auto setupPointPropagator2 = [&](plegma::PLEGMA_Propagator<float> &outS, plegma::PLEGMA_Propagator<float> &outL, plegma::site &source, int solverType, SmearFlagYan s)
        {
            assert(single2double < s);
            plegma::PLEGMA_Vector<double> auxVectorS, auxVectorL;
            plegma::PLEGMA_Vector<float> auxVectorSF, auxVectorLF;
            plegma::PLEGMA_Vector3D<double> auxVector3D;

            for (int isc = 0; isc < 12; isc++)
            {
                auxVector3D.pointSource(source, isc / 3, isc % 3, DEVICE);
                solve2(auxVectorS, auxVectorL, auxVector3D, source[DIM_T], solverType, s);
                auxVectorSF.copy(auxVectorS);
                auxVectorLF.copy(auxVectorL);
                outS.absorb(auxVectorSF, isc / 3, isc % 3);
                outL.absorb(auxVectorLF, isc / 3, isc % 3);
            }
        };
#endif

#if 1 // setupSequentialPropagator
        auto setupSequentialPropagator1 = [&](plegma::PLEGMA_Propagator<float> &out, plegma::PLEGMA_Propagator<float> &in, int inTime, plegma::GAMMAS_SCATT Gamma, std::vector<int> mom, int momSign, int solverType, SmearFlagYan s)
        {
            assert(s < single2double);

            plegma::PLEGMA_Vector<double> auxVectorD;
            plegma::PLEGMA_Vector<float> auxVectorF;
            plegma::PLEGMA_Vector3D<double> auxVector3DD;
            plegma::PLEGMA_Vector3D<float> auxVector3DF;

            for (int isc = 0; isc < 12; isc++)
            {
                auxVector3DF.absorb(in, inTime, isc / 3, isc % 3);
                auxVector3DF.apply_gamma_scatt(Gamma);
                auxVector3DF.mulMomentumPhases(mom, momSign);
                auxVector3DD.copy(auxVector3DF);
                solve1(auxVectorD, auxVector3DD, inTime, solverType, s);
                auxVectorF.copy(auxVectorD);
                out.absorb(auxVectorF, isc / 3, isc % 3);
            }
        };
        auto setupSequentialPropagator2 = [&](plegma::PLEGMA_Propagator<float> &outS, plegma::PLEGMA_Propagator<float> &outL, plegma::PLEGMA_Propagator<float> &in, int inTime, plegma::GAMMAS_SCATT Gamma, std::vector<int> mom, int momSign, int solverType, SmearFlagYan s)
        {
            assert(s > single2double);

            plegma::PLEGMA_Vector<double> auxVectorDS, auxVectorDL;
            plegma::PLEGMA_Vector<float> auxVectorFS, auxVectorFL;
            plegma::PLEGMA_Vector3D<double> auxVector3DD;
            plegma::PLEGMA_Vector3D<float> auxVector3DF;

            for (int isc = 0; isc < 12; isc++)
            {
                auxVector3DF.absorb(in, inTime, isc / 3, isc % 3);
                auxVector3DF.apply_gamma_scatt(Gamma);
                auxVector3DF.mulMomentumPhases(mom, momSign);
                auxVector3DD.copy(auxVector3DF);
                solve2(auxVectorDS, auxVectorDL, auxVector3DD, inTime, solverType, s);
                auxVectorFS.copy(auxVectorDS);
                auxVectorFL.copy(auxVectorDL);
                outS.absorb(auxVectorFS, isc / 3, isc % 3);
                outL.absorb(auxVectorFL, isc / 3, isc % 3);
            }
        };
#endif

#if 0 // pack
        auto packVector = [&](plegma::PLEGMA_Vector<float> &out, plegma::PLEGMA_Vector<float> &in, int timeSlice)
        {
            PLEGMA_Vector3D<float> auxVector3D;
            auxVector3D.absorb(in, timeSlice, true);

            for (int dt = 0; dt < HGC_totalL[DIM_T]; dt++)
            {
                out.absorb(auxVector3D, dt, false);
            }
        };
        auto packVectorD = [&](plegma::PLEGMA_Vector<double> &out, plegma::PLEGMA_Vector<double> &in, int timeSlice)
        {
            PLEGMA_Vector3D<double> auxVector3D;
            auxVector3D.absorb(in, timeSlice, true);

            for (int dt = 0; dt < HGC_totalL[DIM_T]; dt++)
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

#if 0 // phiV3
        auto phiV3 = [&](plegma::PLEGMA_Propagator<float> &out, plegma::PLEGMA_Vector<float> &phi, plegma::PLEGMA_ScattCorrelator<float> &V3)
        {
            for (int isc = 0; isc < 12; isc++)
            {
                plegma::PLEGMA_Vector<float> aux;
                for (int dt = 0; dt < HGC_totalL[DIM_T]; dt++)
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

        //==================================
        // General inputs
        //==================================

        // Get the confnumber for latfile
        char *ssource;
        asprintf(&ssource, "%04d", confnumber_int);
        std::string confnumber = ssource;
        free(ssource);

        std::vector<plegma::GAMMAS_SCATT> gscatts_ID = {ID};
        // std::vector<plegma::GAMMAS_SCATT> gscatts_c = {ID, G_1, G_2, G_3, G_4, G_5, G_5_G_1, G_5_G_2, G_5_G_3, G_5_G_4, S_23, S_13, S_12, S_41, S_42, S_43}; //  S_ij=-i*[gi,gj]/2; (+i) required to make [gi,gj]/2 from S_jk ; (-1) required to make S_31 from S_13;
        // // std::vector<plegma::GAMMAS> gammas_c = {ONE, G1, G2, G3, G4, G5, G5G1, G5G2, G5G3, G5G4};
        
        //  Not applicable if any of the following contains more than one Gamma
        std::vector<plegma::GAMMAS_SCATT> gscatts_i1 = {CG_5};
        std::vector<plegma::GAMMAS_SCATT> gscatts_f1 = {CG_5};
        std::vector<plegma::GAMMAS_SCATT> gscatts_i2 = {G_5};
        std::vector<plegma::GAMMAS_SCATT> gscatts_f2 = {G_5};

        PLEGMA_printf("###Momentum list read from : %s", pathListMomenta_twopt.c_str());
        plegma::momList momList2pt(3, pathListMomenta_twopt, {1, 2}); // pi2, pf1, pf2
        PLEGMA_printf("N momenta in sourcemomentumList: %d\n", momList2pt.size());
        if (momList2pt.empty())
            PLEGMA_error("twopt momentumList empty");
        auto momVects2pt_pi2 = momList2pt.uniq_p(0);
        auto momVects2pt_pf1 = momList2pt.uniq_p(1);
        auto momVects2pt_pf2 = momList2pt.uniq_p(2);
        plegma::momList momList2pt_pi2(1, {momVects2pt_pi2}, {0});
        plegma::momList momList2pt_pf1(1, {momVects2pt_pf1}, {0});
        plegma::momList momList2pt_pf2(1, {momVects2pt_pf2}, {0});

        plegma::momList momList_0(1, {{{0,0,0}}}, {0});


        // Here pathListMomenta_threept is used only for N diagram
        PLEGMA_printf("###Momentum list read from : %s", pathListMomenta_threept.c_str());
        plegma::momList momList3pt(4, pathListMomenta_threept, {1, 2, 3}); // pi2, pf1, pf2, pc
        PLEGMA_printf("N momenta in sourcemomentumList: %d\n", momList3pt.size());
        if (momList3pt.empty())
            PLEGMA_error("threept momentumList empty");
        auto momVects3pt_pf1 = momList3pt.uniq_p(1);
        plegma::momList momList3pt_pf1(1, {momVects3pt_pf1}, {0});

        //==================================
        // P diagram
        //==================================
        if(whatToDo==3)
        {
            assert(boundaryType==2);
            plegma::PLEGMA_Vector<double> auxXi;
            auxXi.randInit(seed);
            auxXi.stochastic_Z(nroots);

            for(int ti=0; ti<HGC_totalL[DIM_T]; ti++)
            {
                struct plegma::site src_zero({0, 0, 0, ti});

                plegma::PLEGMA_Vector<float> auxQ_tfti_Xi_A, auxQ_tfti_Xi_P;
                {
                    plegma::PLEGMA_Vector<double> out_first, out_second;
                    plegma::PLEGMA_Vector3D<double> in;
                    in.absorb(auxXi,ti);
                    int firstSolver=currentSolver;
                    solve1(out_first, in, ti, firstSolver, SS);
                    solve1(out_second, in, ti, -firstSolver, SS);
                    if(firstSolver==-1)
                    {
                        auxQ_tfti_Xi_A.copy(out_first);
                        auxQ_tfti_Xi_P.copy(out_second);
                    }
                    else
                    {
                        auxQ_tfti_Xi_P.copy(out_first);
                        auxQ_tfti_Xi_A.copy(out_second);
                    }
                    applyBoundaryConditionA(auxQ_tfti_Xi_A,ti,ti);
                }

                for (int i_pi2 = 0; i_pi2 < momVects2pt_pi2.size(); i_pi2++)
                {
                    auto &mom_pi2 = momVects2pt_pi2[i_pi2];
                    std::string str_pi2 = "pi2=" + std::to_string(mom_pi2[0]) + "_" + std::to_string(mom_pi2[1]) + "_" + std::to_string(mom_pi2[2]);

                    plegma::PLEGMA_Vector<float> auxG5Q_tfti_G5Gmi2Xi_A, auxG5Q_tfti_G5Gmi2Xi_P;
                    if(mom_pi2[0]==0 && mom_pi2[1]==0 && mom_pi2[2]==0 && gscatts_i2[0]==G_5)
                    {
                        auxG5Q_tfti_G5Gmi2Xi_A.copy(auxQ_tfti_Xi_A);
                        auxG5Q_tfti_G5Gmi2Xi_A.apply_gamma5();
                        auxG5Q_tfti_G5Gmi2Xi_P.copy(auxQ_tfti_Xi_P);
                        auxG5Q_tfti_G5Gmi2Xi_P.apply_gamma5();
                    }
                    else
                    {
                        plegma::PLEGMA_Vector<double> out_first, out_second;
                        plegma::PLEGMA_Vector3D<double> in;
                        in.absorb(auxXi,ti);
                        in.apply_gamma_scatt(gscatts_i2[0]);
                        in.apply_gamma5();
                        in.mulMomentumPhases(mom_pi2, -1);

                        int firstSolver=currentSolver;
                        solve1(out_first, in, ti, firstSolver, SS);
                        solve1(out_second, in, ti, -firstSolver, SS);
                        if(firstSolver==-1)
                        {
                            auxG5Q_tfti_G5Gmi2Xi_A.copy(out_first);
                            auxG5Q_tfti_G5Gmi2Xi_P.copy(out_second);
                        }
                        else
                        {
                            auxG5Q_tfti_G5Gmi2Xi_P.copy(out_first);
                            auxG5Q_tfti_G5Gmi2Xi_A.copy(out_second);
                        }
                        applyBoundaryConditionA(auxG5Q_tfti_G5Gmi2Xi_A,ti,ti);
                        auxG5Q_tfti_G5Gmi2Xi_A.apply_gamma5();
                        auxG5Q_tfti_G5Gmi2Xi_P.apply_gamma5();
                    }

                    {
                        plegma::PLEGMA_ScattCorrelator<float> sc(src_zero, momList2pt_pf2);
                        sc.initialize_diagram(gscatts_f2, "");
                        plegma::PLEGMA_Vector<float> aux0,aux1;
                        aux0.copy(auxQ_tfti_Xi_A); 
                        aux1.copy(auxG5Q_tfti_G5Gmi2Xi_A);
                        sc.PhiPhi(aux0,gscatts_f2,aux1,true); // need additional -1 multiplier
                        sc.setGroups(std::vector<std::string>{"PhiPhi"});
                        sc.setDatasets(std::vector<std::string>{str_pi2 + "/a_a"});
                        std::string outfilename_temp = outdiagramPrefix + confnumber + "_P";
                        sc.writeHDF5(outfilename_temp);
                    }

                    {
                        plegma::PLEGMA_ScattCorrelator<float> sc(src_zero, momList2pt_pf2);
                        sc.initialize_diagram(gscatts_f2, "");
                        plegma::PLEGMA_Vector<float> aux0,aux1;
                        aux0.copy(auxQ_tfti_Xi_P); 
                        aux1.copy(auxG5Q_tfti_G5Gmi2Xi_P);
                        sc.PhiPhi(aux0,gscatts_f2,aux1,true); // need additional -1 multiplier
                        sc.setGroups(std::vector<std::string>{"PhiPhi"});
                        sc.setDatasets(std::vector<std::string>{str_pi2 + "/p_p"});
                        std::string outfilename_temp = outdiagramPrefix + confnumber + "_P";
                        sc.writeHDF5(outfilename_temp);
                    }

                    {
                        plegma::PLEGMA_ScattCorrelator<float> sc(src_zero, momList2pt_pf2);
                        sc.initialize_diagram(gscatts_f2, "");
                        plegma::PLEGMA_Vector<float> aux0,aux1;
                        aux0.copy(auxQ_tfti_Xi_A); 
                        aux1.copy(auxG5Q_tfti_G5Gmi2Xi_P);
                        sc.PhiPhi(aux0,gscatts_f2,aux1,true); // need additional -1 multiplier
                        sc.setGroups(std::vector<std::string>{"PhiPhi"});
                        sc.setDatasets(std::vector<std::string>{str_pi2 + "/a_p"});
                        std::string outfilename_temp = outdiagramPrefix + confnumber + "_P";
                        sc.writeHDF5(outfilename_temp);
                    }

                    {
                        plegma::PLEGMA_ScattCorrelator<float> sc(src_zero, momList2pt_pf2);
                        sc.initialize_diagram(gscatts_f2, "");
                        plegma::PLEGMA_Vector<float> aux0,aux1;
                        aux0.copy(auxQ_tfti_Xi_P); 
                        aux1.copy(auxG5Q_tfti_G5Gmi2Xi_A);
                        sc.PhiPhi(aux0,gscatts_f2,aux1,true); // need additional -1 multiplier
                        sc.setGroups(std::vector<std::string>{"PhiPhi"});
                        sc.setDatasets(std::vector<std::string>{str_pi2 + "/p_a"});
                        std::string outfilename_temp = outdiagramPrefix + confnumber + "_P";
                        sc.writeHDF5(outfilename_temp);
                    }


                    {
                        plegma::PLEGMA_ScattCorrelator<float> sc(src_zero, momList2pt_pf2);
                        sc.initialize_diagram(gscatts_f2, "");
                        plegma::PLEGMA_Vector<float> aux0,aux1;
                        aux0.copy(auxQ_tfti_Xi_A); aux0.add(auxQ_tfti_Xi_P); 
                        aux1.copy(auxG5Q_tfti_G5Gmi2Xi_A); aux1.add(auxG5Q_tfti_G5Gmi2Xi_P);
                        sc.PhiPhi(aux0,gscatts_f2,aux1,true); // need additional -1 multiplier
                        sc.setGroups(std::vector<std::string>{"PhiPhi"});
                        sc.setDatasets(std::vector<std::string>{str_pi2 + "/p+a_p+a"});
                        std::string outfilename_temp = outdiagramPrefix + confnumber + "_P";
                        sc.writeHDF5(outfilename_temp);
                    }

                    {
                        plegma::PLEGMA_ScattCorrelator<float> sc(src_zero, momList2pt_pf2);
                        sc.initialize_diagram(gscatts_f2, "");
                        plegma::PLEGMA_Vector<float> aux0,aux1;
                        aux0.copy(auxQ_tfti_Xi_A); aux0.scale(-1.); aux0.add(auxQ_tfti_Xi_P); 
                        aux1.copy(auxG5Q_tfti_G5Gmi2Xi_A); aux1.scale(-1.); aux1.add(auxG5Q_tfti_G5Gmi2Xi_P);
                        sc.PhiPhi(aux0,gscatts_f2,aux1,true); // need additional -1 multiplier
                        sc.setGroups(std::vector<std::string>{"PhiPhi"});
                        sc.setDatasets(std::vector<std::string>{str_pi2 + "/p-a_p-a"});
                        std::string outfilename_temp = outdiagramPrefix + confnumber + "_P";
                        sc.writeHDF5(outfilename_temp);
                    }


                }
            }
        }

        //==================================
        // Stochastic
        //==================================
        std::vector<PLEGMA_Vector<float>*> stocXi, stocQXi;
        if(whatToDo==1)
        {
            plegma::PLEGMA_Vector<double> auxVStoc;
            auxVStoc.randInit(seed);

            int currentSolverForStoc=currentSolver;

            for(int i=0; i<Nstoc; i++)
            {
                auxVStoc.stochastic_Z(nroots);
                auto filename_Xi = "globalTfulltimedilution_source_nstoch"+std::to_string(i)+"_"+confnumber;
                std::ifstream infile(filename_Xi);
                
                if(infile.good())
                {
                    PLEGMA_printf("[TestYan] i_stoc=%d exist\n",i);
                    continue;
                }
                else
                {
                    PLEGMA_printf("[TestYan] i_stoc=%d not exist\n",i);
                }

                plegma::PLEGMA_Vector<double> auxForWrite;
                plegma::PLEGMA_Vector<float> auxForWriteF;

                auxForWriteF.copy(auxVStoc);
                auxForWriteF.unload();
                auxForWriteF.writeLIME(filename_Xi);
                
                for(int t=0; t<HGC_totalL[DIM_T]; t++)
                {
                    plegma::PLEGMA_Vector<double> out;
                    plegma::PLEGMA_Vector3D<double> in;
                    in.absorb(auxVStoc,t);
                    solve1(out, in, t, currentSolverForStoc, SS);
                    auxForWrite.absorbTimeslice(out, t, false);
                }
                auxForWriteF.copy(auxForWrite);

                if(boundaryType==2)
                {
                    currentSolverForStoc*=-1;
                    for(int t=0; t<HGC_totalL[DIM_T]; t++)
                    {
                        plegma::PLEGMA_Vector<double> out;
                        plegma::PLEGMA_Vector3D<double> in;
                        in.absorb(auxVStoc,t);
                        solve1(out, in, t, currentSolverForStoc, SS);
                        auxForWrite.absorbTimeslice(out, t, false);
                    }
                    plegma::PLEGMA_Vector<float> aux;
                    aux.copy(auxForWrite);
                    auxForWriteF.add(aux);
                }

                auto filename_QXi = "globalTfulltimedilution_propagator_nstoch"+std::to_string(i)+"_"+confnumber;
                auxForWriteF.unload();
                auxForWriteF.writeLIME(filename_QXi);
            }

            for(int i=0; i<Nstoc; i++)
            {
                auto filename_Xi = "globalTfulltimedilution_source_nstoch"+std::to_string(i)+"_"+confnumber;
                auto filename_QXi = "globalTfulltimedilution_propagator_nstoch"+std::to_string(i)+"_"+confnumber;
    
                stocXi.push_back(new PLEGMA_Vector<float>(HOST));
                stocQXi.push_back(new PLEGMA_Vector<float>(HOST));
    
                plegma::PLEGMA_Vector<float> vectorRead(BOTH);
                vectorRead.readFile(filename_Xi,LIME_FORMAT);
                // vectorRead.writeHDF5(filename_Xi);
                stocXi[i]->copy(vectorRead,HOST);
                vectorRead.readFile(filename_QXi,LIME_FORMAT);
                // vectorRead.writeHDF5(filename_QXi);
                stocQXi[i]->copy(vectorRead,HOST);
            }
        }


        //==================================
        // N and BWZ
        //==================================
        for(int i_src = 0; i_src < numSourcePositions; i_src++)
        {
            if(whatToDo==3)
                break;

            auto src = sourcePositions[i_src];
            struct plegma::site src_zero({0, 0, 0, src[DIM_T]});

            asprintf(&ssource,"sx%03dsy%03dsz%03dst%03d", src[0], src[1], src[2], src[3]);
            std::string outfilename = outdiagramPrefix + confnumber + "_" + ssource + "_NpiScatteringWilson";

            int ti,tf;
            ti=src[DIM_T];

            plegma::PLEGMA_Propagator<float> prop;
            setupPointPropagator1(prop, src, -1, SS);
            applyBoundaryConditionProp(prop, ti, ti);

            {
                plegma::PLEGMA_ScattCorrelator<float> corrN(src, momList3pt_pf1);
                corrN.initialize_diagram(gscatts_ID, gscatts_ID, gscatts_i1, gscatts_f1, "N_a");

                plegma::PLEGMA_ScattCorrelator<float> reductionsT1N(src_zero, momVects3pt_pf1);
                plegma::PLEGMA_ScattCorrelator<float> reductionsT2N(src_zero, momVects3pt_pf1);

                reductionsT1N.T1(gscatts_i1, gscatts_f1, prop, prop, prop);
                reductionsT2N.T2(gscatts_i1, gscatts_f1, prop, prop, prop);

                corrN.N_diagrams(reductionsT1N, reductionsT2N);
                corrN.apply_phase();
                corrN.apply_sign("N");
                std::string outfilename_temp = outdiagramPrefix + confnumber + "_" + ssource + "_N";
                corrN.writeHDF5(outfilename_temp);
            }

            if(boundaryType==2)
            {
                plegma::PLEGMA_Propagator<float> prop_p;
                setupPointPropagator1(prop_p, src, +1, SS);
                prop.add(prop_p);
            }
            
            {
                plegma::PLEGMA_ScattCorrelator<float> corrN(src, momList3pt_pf1);
                corrN.initialize_diagram(gscatts_ID, gscatts_ID, gscatts_i1, gscatts_f1, "N_ppa_pma");

                plegma::PLEGMA_ScattCorrelator<float> reductionsT1N(src_zero, momVects3pt_pf1);
                plegma::PLEGMA_ScattCorrelator<float> reductionsT2N(src_zero, momVects3pt_pf1);

                reductionsT1N.T1(gscatts_i1, gscatts_f1, prop, prop, prop);
                reductionsT2N.T2(gscatts_i1, gscatts_f1, prop, prop, prop);

                corrN.N_diagrams(reductionsT1N, reductionsT2N);
                corrN.apply_phase();
                corrN.apply_sign("N");
                std::string outfilename_temp = outdiagramPrefix + confnumber + "_" + ssource + "_N";
                corrN.writeHDF5(outfilename_temp);
            }

            if(whatToDo!=1)
                continue;

            for(int i_stoc=0; i_stoc<Nstoc; i_stoc++)
            {
                plegma::PLEGMA_Vector<float> auxXi, auxQXi;
                plegma::PLEGMA_Vector<float> auxG5Xi, auxG5QXi;
                auxXi.copy(*stocXi[i_stoc],HOST); auxXi.load();
                auxQXi.copy(*stocQXi[i_stoc],HOST); auxQXi.load();

                auxG5Xi.copy(auxXi); auxG5Xi.apply_gamma5();
                auxG5QXi.copy(auxQXi); auxG5QXi.apply_gamma5();
                
                {
                    plegma::PLEGMA_ScattCorrelator<float> sc(src_zero, momList2pt_pf1);
                    sc.V2(auxG5Xi, gscatts_f1, prop, prop);
                    sc.setGroups(std::vector<std::string>{"V2B_1"});
                    sc.setDatasets(std::vector<std::string>{"i_stoc="+std::to_string(i_stoc)});
                    sc.writeHDF5(outfilename);
                }
                {
                    plegma::PLEGMA_ScattCorrelator<float> sc(src_zero, momList2pt_pf1);
                    sc.V4(auxG5Xi, gscatts_f1, prop, prop);
                    sc.setGroups(std::vector<std::string>{"V4B_1"});
                    sc.setDatasets(std::vector<std::string>{"i_stoc="+std::to_string(i_stoc)});
                    sc.writeHDF5(outfilename);
                }
                
                {
                    plegma::PLEGMA_ScattCorrelator<float> sc(src_zero, momList2pt_pf1);
                    sc.V2(auxQXi, gscatts_f1, prop, prop);
                    sc.setGroups(std::vector<std::string>{"V2B_2"});
                    sc.setDatasets(std::vector<std::string>{"i_stoc="+std::to_string(i_stoc)});
                    sc.writeHDF5(outfilename);
                }
                {
                    plegma::PLEGMA_ScattCorrelator<float> sc(src_zero, momList2pt_pf1);
                    sc.V4(auxQXi, gscatts_f1, prop, prop);
                    sc.setGroups(std::vector<std::string>{"V4B_2"});
                    sc.setDatasets(std::vector<std::string>{"i_stoc="+std::to_string(i_stoc)});
                    sc.writeHDF5(outfilename);
                }

                {
                    plegma::PLEGMA_ScattCorrelator<float> sc(src_zero, momList2pt_pf2);
                    sc.V3(auxG5QXi, gscatts_f2, prop);
                    sc.setGroups(std::vector<std::string>{"V3W_1"});
                    sc.setDatasets(std::vector<std::string>{"i_stoc="+std::to_string(i_stoc)});
                    sc.writeHDF5(outfilename);
                }

                {
                    plegma::PLEGMA_ScattCorrelator<float> sc(src_zero, momList2pt_pf2);
                    sc.V3(auxXi, gscatts_f2, prop);
                    sc.setGroups(std::vector<std::string>{"V3W_2"});
                    sc.setDatasets(std::vector<std::string>{"i_stoc="+std::to_string(i_stoc)});
                    sc.writeHDF5(outfilename);
                }

                if(i_stoc>=Noet)
                    continue;

                plegma::PLEGMA_Vector<float> auxQ_tfti_Xi;
                {
                    plegma::PLEGMA_Vector<double> out;
                    plegma::PLEGMA_Vector3D<double> in;
                    out.copy(auxXi);
                    in.absorb(out,ti);
                    if(boundaryType==1)
                    {
                        solve1(out, in, ti, -1, SS);
                        applyBoundaryConditionDouble(out,ti,ti);
                    }
                    else if(boundaryType==2)
                    {
                        plegma::PLEGMA_Vector<double> out_sec;
                        int firstSolver=currentSolver;
                        solve1(out, in, ti, firstSolver, SS);
                        solve1(out_sec, in, ti, -firstSolver, SS);
                        if(firstSolver==-1)
                            applyBoundaryConditionDouble(out,ti,ti);
                        else
                            applyBoundaryConditionDouble(out_sec,ti,ti);
                        out.add(out_sec);
                    }
                    auxQ_tfti_Xi.copy(out);
                }

                {
                    plegma::PLEGMA_ScattCorrelator<float> sc(src_zero, momList2pt_pf1);
                    sc.V2(auxQ_tfti_Xi, gscatts_f1, prop, prop);
                    sc.setGroups(std::vector<std::string>{"V2Z_1"});
                    sc.setDatasets(std::vector<std::string>{"i_stoc="+std::to_string(i_stoc)});
                    sc.writeHDF5(outfilename);
                }
                {
                    plegma::PLEGMA_ScattCorrelator<float> sc(src_zero, momList2pt_pf1);
                    sc.V4(auxQ_tfti_Xi, gscatts_f1, prop, prop);
                    sc.setGroups(std::vector<std::string>{"V4Z_1"});
                    sc.setDatasets(std::vector<std::string>{"i_stoc="+std::to_string(i_stoc)});
                    sc.writeHDF5(outfilename);
                }

                for (int i_pi2 = 0; i_pi2 < momVects2pt_pi2.size(); i_pi2++)
                {
                    auto &mom_pi2 = momVects2pt_pi2[i_pi2];
                    std::string str_pi2 = "pi2=" + std::to_string(mom_pi2[0]) + "_" + std::to_string(mom_pi2[1]) + "_" + std::to_string(mom_pi2[2]);

                    plegma::PLEGMA_Vector<float> auxG5Q_tfti_G5Gmi2Xi;
                    if(mom_pi2[0]==0 && mom_pi2[1]==0 && mom_pi2[2]==0 && gscatts_i2[0]==G_5)
                    {
                        auxG5Q_tfti_G5Gmi2Xi.copy(auxQ_tfti_Xi);
                        auxG5Q_tfti_G5Gmi2Xi.apply_gamma5();
                    }
                    else
                    {
                        plegma::PLEGMA_Vector<double> out;
                        plegma::PLEGMA_Vector3D<double> in;
                        out.copy(auxXi);
                        in.absorb(out,ti);
                        in.apply_gamma_scatt(gscatts_i2[0]);
                        in.apply_gamma5();
                        in.mulMomentumPhases(mom_pi2, -1);
                        if(boundaryType==1)
                        {
                            solve1(out, in, ti, -1, SS);
                            applyBoundaryConditionDouble(out,ti,ti);
                        }
                        else if(boundaryType==2)
                        {
                            plegma::PLEGMA_Vector<double> out_sec;
                            int firstSolver=currentSolver;
                            solve1(out, in, ti, firstSolver, SS);
                            solve1(out_sec, in, ti, -firstSolver, SS);
                            if(firstSolver==-1)
                                applyBoundaryConditionDouble(out,ti,ti);
                            else
                                applyBoundaryConditionDouble(out_sec,ti,ti);
                            out.add(out_sec);
                        }
                        auxG5Q_tfti_G5Gmi2Xi.copy(out);
                        auxG5Q_tfti_G5Gmi2Xi.apply_gamma5();
                    }

                    {
                        plegma::PLEGMA_ScattCorrelator<float> sc(src_zero, momList2pt_pf2);
                        sc.V3(auxG5Q_tfti_G5Gmi2Xi, gscatts_f2, prop);
                        sc.setGroups(std::vector<std::string>{"V3Z_1"});
                        sc.setDatasets(std::vector<std::string>{str_pi2 + "/i_stoc=" + std::to_string(i_stoc)});
                        sc.writeHDF5(outfilename);
                    }

                }

            }
            
            // // This part is used to verify an identify between B2 and B3, which should NOT be used for production.
            // for(int i_stoc=0; i_stoc<1; i_stoc++)
            // {
            //     plegma::PLEGMA_Vector<float> auxXi;
            //     auxXi.copy(*stocXi[i_stoc],HOST); 
            //     auxXi.load();

            //     plegma::PLEGMA_Propagator<float> prop_ti;
            //     plegma::PLEGMA_Propagator3D<float> aux;
            //     aux.absorb(prop,ti,true);

            //     plegma::PLEGMA_Vector<float> auxG5Q_titf_Xi;
            //     plegma::PLEGMA_Vector<double> aux2;

            //     for(int t=0; t<HGC_totalL[DIM_T]; t++)
            //     {
            //         prop_ti.absorb(aux,t,false);

            //         plegma::PLEGMA_Vector<double> out,out_sec;
            //         plegma::PLEGMA_Vector3D<double> in;
            //         out.copy(auxXi);
            //         in.absorb(out,t);
            //         solve1(out, in, t, -1, SS);
            //         applyBoundaryConditionDouble(out,t,ti);
            //         if(boundaryType==2)
            //         {
            //             solve1(out_sec, in, t, +1, SS);
            //             out.add(out_sec);
            //         }
            //         in.absorb(out,ti,true);
            //         aux2.absorb(in,t,false);
            //     }
            //     auxG5Q_titf_Xi.copy(aux2);
            //     auxG5Q_titf_Xi.apply_gamma5();

            //     {
            //         plegma::PLEGMA_ScattCorrelator<float> sc(src_zero, momList2pt_pi2);
            //         sc.V3(auxG5Q_titf_Xi, gscatts_i2, prop_ti);
            //         auxG5Q_titf_Xi.writeHDF5("auxG5Q_titf_Xi");
            //         prop_ti.writeHDF5("prop_ti");
            //         sc.setGroups(std::vector<std::string>{"V3B3"});
            //         sc.setDatasets(std::vector<std::string>{"i_stoc="+std::to_string(i_stoc)});
            //         sc.writeHDF5(outfilename);
            //     }
            // }

            for (int i_pi2 = 0; i_pi2 < momVects2pt_pi2.size(); i_pi2++)
            {
                auto &mom_pi2 = momVects2pt_pi2[i_pi2];
                std::string str_pi2 = "pi2=" + std::to_string(mom_pi2[0]) + "_" + std::to_string(mom_pi2[1]) + "_" + std::to_string(mom_pi2[2]);

                plegma::PLEGMA_Propagator<float> seq;
                if(boundaryType==1)
                {
                    setupSequentialPropagator1(seq, prop, ti, gscatts_i2[0], mom_pi2, +1, -1, SS);
                    applyBoundaryConditionProp(seq, ti, ti);
                }
                else if(boundaryType==2)
                {
                    int firstSolver=currentSolver;
                    plegma::PLEGMA_Propagator<float> seq_sec;
                    setupSequentialPropagator1(seq, prop, ti, gscatts_i2[0], mom_pi2, +1, firstSolver, SS);
                    setupSequentialPropagator1(seq_sec, prop, ti, gscatts_i2[0], mom_pi2, +1, -firstSolver, SS);
                    if(firstSolver==-1)
                        applyBoundaryConditionProp(seq, ti, ti);
                    else
                        applyBoundaryConditionProp(seq_sec, ti, ti);
                    seq.add(seq_sec);
                }

                for(int i_stoc=0; i_stoc<Nstoc; i_stoc++)
                {
                    plegma::PLEGMA_Vector<float> auxXi, auxQXi;
                    plegma::PLEGMA_Vector<float> auxG5Xi, auxG5QXi;
                    auxXi.copy(*stocXi[i_stoc],HOST); auxXi.load();
                    auxQXi.copy(*stocQXi[i_stoc],HOST); auxQXi.load();
    
                    auxG5Xi.copy(auxXi); auxG5Xi.apply_gamma5();
                    auxG5QXi.copy(auxQXi); auxG5QXi.apply_gamma5();

                    {
                        plegma::PLEGMA_ScattCorrelator<float> sc(src_zero, momList2pt_pf2);
                        sc.V3(auxG5QXi, gscatts_f2, seq);
                        sc.setGroups(std::vector<std::string>{"V3B_1"});
                        sc.setDatasets(std::vector<std::string>{str_pi2 + "/i_stoc="+std::to_string(i_stoc)});
                        sc.writeHDF5(outfilename);
                    }

                    {
                        plegma::PLEGMA_ScattCorrelator<float> sc(src_zero, momList2pt_pf2);
                        sc.V3(auxXi, gscatts_f2, seq);
                        sc.setGroups(std::vector<std::string>{"V3B_2"});
                        sc.setDatasets(std::vector<std::string>{str_pi2 + "/i_stoc="+std::to_string(i_stoc)});
                        sc.writeHDF5(outfilename);
                    }

                    {
                        plegma::PLEGMA_ScattCorrelator<float> sc(src_zero, momList2pt_pf1);
                        sc.V2(auxG5Xi, gscatts_f1, seq, prop);
                        sc.setGroups(std::vector<std::string>{"V2W1_1"});
                        sc.setDatasets(std::vector<std::string>{str_pi2 + "/i_stoc="+std::to_string(i_stoc)});
                        sc.writeHDF5(outfilename);
                    }
                    {
                        plegma::PLEGMA_ScattCorrelator<float> sc(src_zero, momList2pt_pf1);
                        sc.V4(auxG5Xi, gscatts_f1, seq, prop);
                        sc.setGroups(std::vector<std::string>{"V4W1_1"});
                        sc.setDatasets(std::vector<std::string>{str_pi2 + "/i_stoc="+std::to_string(i_stoc)});
                        sc.writeHDF5(outfilename);
                    }
                    {
                        plegma::PLEGMA_ScattCorrelator<float> sc(src_zero, momList2pt_pf1);
                        sc.V2(auxG5Xi, gscatts_f1, prop, seq);
                        sc.setGroups(std::vector<std::string>{"V2W2_1"});
                        sc.setDatasets(std::vector<std::string>{str_pi2 + "/i_stoc="+std::to_string(i_stoc)});
                        sc.writeHDF5(outfilename);
                    }
                    {
                        plegma::PLEGMA_ScattCorrelator<float> sc(src_zero, momList2pt_pf1);
                        sc.V4(auxG5Xi, gscatts_f1, prop, seq);
                        sc.setGroups(std::vector<std::string>{"V4W2_1"});
                        sc.setDatasets(std::vector<std::string>{str_pi2 + "/i_stoc="+std::to_string(i_stoc)});
                        sc.writeHDF5(outfilename);
                    }

                    {
                        plegma::PLEGMA_ScattCorrelator<float> sc(src_zero, momList2pt_pf1);
                        sc.V2(auxQXi, gscatts_f1, seq, prop);
                        sc.setGroups(std::vector<std::string>{"V2W1_2"});
                        sc.setDatasets(std::vector<std::string>{str_pi2 + "/i_stoc="+std::to_string(i_stoc)});
                        sc.writeHDF5(outfilename);
                    }
                    {
                        plegma::PLEGMA_ScattCorrelator<float> sc(src_zero, momList2pt_pf1);
                        sc.V4(auxQXi, gscatts_f1, seq, prop);
                        sc.setGroups(std::vector<std::string>{"V4W1_2"});
                        sc.setDatasets(std::vector<std::string>{str_pi2 + "/i_stoc="+std::to_string(i_stoc)});
                        sc.writeHDF5(outfilename);
                    }
                    {
                        plegma::PLEGMA_ScattCorrelator<float> sc(src_zero, momList2pt_pf1);
                        sc.V2(auxQXi, gscatts_f1, prop, seq);
                        sc.setGroups(std::vector<std::string>{"V2W2_2"});
                        sc.setDatasets(std::vector<std::string>{str_pi2 + "/i_stoc="+std::to_string(i_stoc)});
                        sc.writeHDF5(outfilename);
                    }
                    {
                        plegma::PLEGMA_ScattCorrelator<float> sc(src_zero, momList2pt_pf1);
                        sc.V4(auxQXi, gscatts_f1, prop, seq);
                        sc.setGroups(std::vector<std::string>{"V4W2_2"});
                        sc.setDatasets(std::vector<std::string>{str_pi2 + "/i_stoc="+std::to_string(i_stoc)});
                        sc.writeHDF5(outfilename);
                    }

                    // for(int j_stoc=0; j_stoc<Nstoc; j_stoc++)
                    // {
                    //     plegma::PLEGMA_Vector<float> auxEta, auxQEta;
                    //     plegma::PLEGMA_Vector<float> auxG5Eta, auxG5QEta;
                    //     auxEta.copy(*stocXi[j_stoc],HOST); auxEta.load();
                    //     auxQEta.copy(*stocQXi[j_stoc],HOST); auxQEta.load();
        
                    //     auxG5Eta.copy(auxEta); auxG5Eta.apply_gamma5();
                    //     auxG5QEta.copy(auxQEta); auxG5QEta.apply_gamma5();
                        
                    // }
                }
            }
        }
    }
    finalize();
    std::ofstream output(flagFinish);
    return 0;
}
