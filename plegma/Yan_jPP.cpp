/*
Here we compute <(q1bar.Gf.q2)(xf,tf) e^(-i.pf.xf) (q2bar.Gi.q1)^dgr(xi,ti) e^(+i.pi.xi)> = - PhiPhi(phi0,Gf,phi1,true)
To get data for the above formula, the output file requires the following additional changes:
# (-1) for the (-1) in front of PhiPhi
# (g4 Gi^dgr g4 / Gi) for Gi
# (+i) for tensor in Gi&Gf
# (-1) for S_13 in Gi&Gf
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
static std::vector<std::string> listOpt = {"verbosity", "load-gauge", "seed1", "confnumber", "outdiagramPrefix", "momlistthreept-filename", "nsmear-APE", "alpha-APE", "nsmear-gauss", "alpha-gauss"};

int main(int argc, char **argv)
{
    /******************************************************
     *
     *   Initialiazation
     *
     ******************************************************/
    initializeOptions(argc, argv, true, listOpt);

    int seed_stoc, confnumber_int;
    std::string outdiagramPrefix, flagFinish;
    HGC_options->set("flagFinish", "An empty file created indicating the completion of a run", verbosity, flagFinish);

    HGC_options->set("seed_stoc", "Seed for initialization of stochastic sources for the oet", verbosity, seed_stoc);
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
        std::vector<plegma::GAMMAS_SCATT> gscatts_f = {ID, G_1, G_2, G_3, G_4, G_5, G_5_G_1, G_5_G_2, G_5_G_3, G_5_G_4, S_23, S_13, S_12, S_41, S_42, S_43}; //  S_ij=-i*[gi,gj]/2; (+i) required to make [gi,gj]/2 from S_jk ; (-1) required to make S_31 from S_13;
        std::vector<plegma::GAMMAS_SCATT> gscatts_i = {ID, G_5, G_5_G_4}; //  S_ij=-i*[gi,gj]/2; (+i) required to make [gi,gj]/2 from S_jk ; (-1) required to make S_31 from S_13;
        
        std::string outfilename = outdiagramPrefix + "_" + confnumber + "_jPP";

        std::vector<SmearFlagYan> smearFlags = {LL, SL, SS};
        std::vector<std::string> str_smearFlags = {"LL", "SL", "SS"};

        PLEGMA_printf("###Momentum list read from : %s", pathListMomenta_threept.c_str());
        plegma::momList momList_3pt(4, pathListMomenta_threept, {1, 2, 3}); // pi2, pf1, pf2, pc
        PLEGMA_printf("N momenta in sourcemomentumList: %d\n", momList_3pt.size());
        if (momList_3pt.empty())
            PLEGMA_error("threept momentumList empty");
        auto momVects_pi = momList_3pt.uniq_p(0);
        auto momVects_pf = momList_3pt.uniq_p(2);
        plegma::momList momList_pf(1, {momVects_pf}, {0});


        /******************************************************
         *
         *   Main
         *
         ******************************************************/

        for (int st = 0; st < HGC_totalL[3]; st++)
        {
            struct plegma::site source({0, 0, 0, st});

            plegma::PLEGMA_Vector<double> stocXi;
            {
                stocXi.randInit(seed_stoc);
                stocXi.stochastic_Z(4);
            } // what if I move this part outside st loop?
            int time_i = source[3];
            
            for (int i_sf = 0; i_sf < smearFlags.size(); i_sf ++)
            {
                std::vector<plegma::PLEGMA_Vector<float> *> phi0us;
                std::vector<plegma::PLEGMA_Vector<float> *> phi0ds;
                auto smearFlag = smearFlags[i_sf];
                auto str_smearFlag = str_smearFlags[i_sf];

                // u part
                for (int i_gammai = 0; i_gammai < gscatts_i.size(); i_gammai++)
                {
                    auto gammai = gscatts_i[i_gammai];
                    
                    plegma::PLEGMA_Vector<double> aux;
                    plegma::PLEGMA_Vector3D<double> aux3D;

                    aux3D.absorb(stocXi,time_i);
                    aux3D.apply_gamma5();
                    aux3D.apply_gamma_scatt(gammai); // Need sign (g4 Gi^dg g4 / Gi) in postproduction

                    solve1(aux, aux3D, time_i, u, smearFlag);
                    phi0us.push_back(new plegma::PLEGMA_Vector<float>);
                    phi0us[i_gammai]->copy(aux);
                }

                // d part
                for (int i_gammai = 0; i_gammai < gscatts_i.size(); i_gammai++)
                {
                    auto gammai = gscatts_i[i_gammai];

                    plegma::PLEGMA_Vector<double> aux;
                    plegma::PLEGMA_Vector3D<double> aux3D;

                    aux3D.absorb(stocXi,time_i);
                    aux3D.apply_gamma5();
                    aux3D.apply_gamma_scatt(gammai); // Need sign (g4 Gi^dg g4 / Gi) in postproduction

                    solve1(aux, aux3D, time_i, d, smearFlag);
                    phi0ds.push_back(new plegma::PLEGMA_Vector<float>);
                    phi0ds[i_gammai]->copy(aux);
                }

                // u part (but g5-Hermiticity makes it d)
                for (int i_pi = 0; i_pi < momVects_pi.size(); ++i_pi)
                {
                    auto &mom_pi = momVects_pi[i_pi];
                    std::string str_pi = "pi=" + std::to_string(mom_pi[0]) + "_" + std::to_string(mom_pi[1]) + "_" + std::to_string(mom_pi[2]);
                    // PLEGMA_printf(("TestYan: " + str_pi2).c_str());

                    plegma::PLEGMA_Vector<double> phi1;
                    plegma::PLEGMA_Vector3D<double> aux3D;
                    aux3D.absorb(stocXi,time_i);
                    aux3D.mulMomentumPhases(mom_pi, -1);
                    solve1(phi1, aux3D, time_i, d, smearFlag);
                    phi1.apply_gamma5();


                    for (int i_gammai = 0; i_gammai < gscatts_i.size(); i_gammai++)
                    {
                        {
                            plegma::PLEGMA_ScattCorrelator<float> auxPhiPhi(source, momList_pf);
                            auxPhiPhi.initialize_diagram(gscatts_f, "");
                            
                            plegma::PLEGMA_Vector<float> aux0, aux1;
                            aux0.copy(*phi0us[i_gammai]);
                            aux1.copy(phi1); 
                            auxPhiPhi.PhiPhi(aux0,gscatts_f,aux1,true); // need additional -1 multiplier
                            auxPhiPhi.setGroups(std::vector<std::string>{"PhiPhi"});
                            auxPhiPhi.setDatasets(std::vector<std::string>{str_pi + "/q1q2=uu/sisf=" + str_smearFlag + "/i_gi=" + std::to_string(i_gammai)});
                            auxPhiPhi.writeHDF5(outfilename);
                        }
                        {
                            plegma::PLEGMA_ScattCorrelator<float> auxPhiPhi(source, momList_pf);
                            auxPhiPhi.initialize_diagram(gscatts_f, "");
                            
                            plegma::PLEGMA_Vector<float> aux0, aux1;
                            aux0.copy(*phi0ds[i_gammai]);
                            aux1.copy(phi1); 
                            auxPhiPhi.PhiPhi(aux0,gscatts_f,aux1,true); // need additional -1 multiplier
                            auxPhiPhi.setGroups(std::vector<std::string>{"PhiPhi"});
                            auxPhiPhi.setDatasets(std::vector<std::string>{str_pi + "/q1q2=ud/sisf=" + str_smearFlag + "/i_gi=" + std::to_string(i_gammai)});
                            auxPhiPhi.writeHDF5(outfilename);
                        }

                    }
                }
                
                // d part (but g5-Hermiticity makes it u)
                for (int i_pi = 0; i_pi < momVects_pi.size(); ++i_pi)
                {
                    auto &mom_pi = momVects_pi[i_pi];
                    std::string str_pi = "pi=" + std::to_string(mom_pi[0]) + "_" + std::to_string(mom_pi[1]) + "_" + std::to_string(mom_pi[2]);
                    // PLEGMA_printf(("TestYan: " + str_pi2).c_str());

                    plegma::PLEGMA_Vector<double> phi1;
                    plegma::PLEGMA_Vector3D<double> aux3D;
                    aux3D.absorb(stocXi,time_i);
                    aux3D.mulMomentumPhases(mom_pi, -1);
                    solve1(phi1, aux3D, time_i, u, smearFlag);
                    phi1.apply_gamma5();


                    for (int i_gammai = 0; i_gammai < gscatts_i.size(); i_gammai++)
                    {
                        {
                            plegma::PLEGMA_ScattCorrelator<float> auxPhiPhi(source, momList_pf);
                            auxPhiPhi.initialize_diagram(gscatts_f, "");
                            
                            plegma::PLEGMA_Vector<float> aux0, aux1;
                            aux0.copy(*phi0us[i_gammai]);
                            aux1.copy(phi1); 
                            auxPhiPhi.PhiPhi(aux0,gscatts_f,aux1,true); // need additional -1 multiplier
                            auxPhiPhi.setGroups(std::vector<std::string>{"PhiPhi"});
                            auxPhiPhi.setDatasets(std::vector<std::string>{str_pi + "/q1q2=du/sisf=" + str_smearFlag + "/i_gi=" + std::to_string(i_gammai)});
                            auxPhiPhi.writeHDF5(outfilename);
                        }
                        {
                            plegma::PLEGMA_ScattCorrelator<float> auxPhiPhi(source, momList_pf);
                            auxPhiPhi.initialize_diagram(gscatts_f, "");
                            
                            plegma::PLEGMA_Vector<float> aux0, aux1;
                            aux0.copy(*phi0ds[i_gammai]);
                            aux1.copy(phi1); 
                            auxPhiPhi.PhiPhi(aux0,gscatts_f,aux1,true); // need additional -1 multiplier
                            auxPhiPhi.setGroups(std::vector<std::string>{"PhiPhi"});
                            auxPhiPhi.setDatasets(std::vector<std::string>{str_pi + "/q1q2=dd/sisf=" + str_smearFlag + "/i_gi=" + std::to_string(i_gammai)});
                            auxPhiPhi.writeHDF5(outfilename);
                        }

                    }
                }

                for (int i_gammai = 0; i_gammai < gscatts_i.size(); i_gammai++)
                {
                    delete phi0us[i_gammai];
                    delete phi0ds[i_gammai];
                }
            }

        }
    }

    finalize();
    std::ofstream output(flagFinish);
    return 0;
}
