#include <ToolYan.h>

ToolYan::ToolYan()
{
    tStart = tLast = std::chrono::high_resolution_clock::now();
    printTime("start");
}
ToolYan::~ToolYan()
{
    printTime("end");
}

void ToolYan::printTime(std::string s, bool preciseQ)
{
    std::chrono::_V2::system_clock::time_point tNow = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> tTotal = tNow - tStart;
    std::chrono::duration<double, std::milli> tDura = tNow - tLast;
    if (preciseQ)
        PLEGMA_printf("[ToolYan: %s] Total time: %f; Duration: %f\n", s.c_str(), tTotal.count() / 1000, tDura.count() / 1000);
    else
        PLEGMA_printf("[ToolYan: %s] Total time: %d; Duration: %d\n", s.c_str(), int(tTotal.count() / 1000), int(tDura.count() / 1000));
    tLast = tNow;
}

void ToolYan::testTimeCost()
{
    printTime("testTimeCost start");
    for (int i = 0; i <= 3; i++)
    {
        plegma::PLEGMA_Vector<double> auxVector, auxVector1;
        plegma::PLEGMA_Vector3D<double> auxVector3D, auxVector3D1;
        struct plegma::site source({0, 0, 0, 0});

        auxVector.pointSource(source, 0, 0);
        auxVector3D.pointSource(source, 0, 0, DEVICE);

        printTime(("test " + std::to_string(i + 1)).c_str());

        solverP->solve(auxVector, auxVector);
        printTime("bare solve");

        auxVector1.rotateToPhysicalBasis(auxVector, +1);
        printTime("rotate to physical basis");

        if (smearQ)
        {
            auxVector1.gaussianSmearing(auxVector, *smearedGaugeP, this->nsmearGauss, this->alphaGauss);
            printTime("smear");
            plegma::PLEGMA_Gauge3D<double> smearedGauge3D;
            smearedGauge3D.absorb(*smearedGaugeP, 0);
            printTime("create gauge3D");
            auxVector3D1.gaussianSmearing(auxVector3D, smearedGauge3D, this->nsmearGauss, this->alphaGauss);
            printTime("smear3D");
        }
    }
    printTime("testTimeCost end");
}

void ToolYan::solve(plegma::PLEGMA_Vector<double> &out, plegma::PLEGMA_Vector<double> &in, FlavorYan f)
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
        solverP->UpdateSolver();
    }
    solverP->solve(auxVector, auxVector);
    auxVector.scale(norm);
    out.rotateToPhysicalBasis(auxVector, run_mu / std::abs(run_mu));
}

void ToolYan::solve(plegma::PLEGMA_Vector<double> &out, plegma::PLEGMA_Vector3D<double> &in, int inTime, FlavorYan f, SmearFlagYan s)
{
    assert(s < single2double);

    if (!smearQ)
    {
        plegma::PLEGMA_Vector<double> auxVector;
        auxVector.absorb(in, inTime);
        solve(out, auxVector, f);
        return;
    }

    plegma::PLEGMA_Vector<double> auxVector;
    plegma::PLEGMA_Vector3D<double> auxVector3D;

    if (s == SL || s == SS)
    {
        plegma::PLEGMA_Gauge3D<double> smearedGauge3D;
        smearedGauge3D.absorb(*smearedGaugeP, inTime);
        auxVector3D.gaussianSmearing(in, smearedGauge3D, this->nsmearGauss, this->alphaGauss);
    }
    else
        auxVector3D.copy(in);

    auxVector.absorb(auxVector3D, inTime);
    solve(auxVector, auxVector, f);

    if (s == LS || s == SS)
    {
        out.gaussianSmearing(auxVector, *smearedGaugeP, this->nsmearGauss, this->alphaGauss);
    }
    else
        out.copy(auxVector);
}

void ToolYan::solve(plegma::PLEGMA_Vector<double> &outS, plegma::PLEGMA_Vector<double> &outL, plegma::PLEGMA_Vector3D<double> &in, int inTime, FlavorYan f, SmearFlagYan s)
{
    assert(single2double < s);
    if (!smearQ)
    {
        plegma::PLEGMA_Vector<double> auxVector;
        auxVector.absorb(in, inTime);
        solve(outL, auxVector, f);
        outS.copy(outL);
        return;
    }

    plegma::PLEGMA_Vector<double> auxVector;

    if (s == SB)
    {
        plegma::PLEGMA_Gauge3D<double> smearedGauge3D;
        plegma::PLEGMA_Vector3D<double> auxVector3D;
        smearedGauge3D.absorb(*smearedGaugeP, inTime);
        auxVector3D.gaussianSmearing(in, smearedGauge3D, this->nsmearGauss, this->alphaGauss);
        auxVector.absorb(auxVector3D, inTime);
    }
    else
        auxVector.absorb(in, inTime);

    solve(auxVector, auxVector, f);

    outS.gaussianSmearing(auxVector, *smearedGaugeP, this->nsmearGauss, this->alphaGauss);
    outL.copy(auxVector);
}

void ToolYan::setupPointPropagator(plegma::PLEGMA_Propagator<float> &outS, plegma::PLEGMA_Propagator<float> &outL, plegma::site &source, FlavorYan f, SmearFlagYan s)
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
}

void ToolYan::setupSequentialPropagator(plegma::PLEGMA_Propagator<float> &outS, plegma::PLEGMA_Propagator<float> &outL, plegma::PLEGMA_Propagator<float> &in, int inTime, plegma::GAMMAS Gamma, std::vector<int> mom, int momSign, FlavorYan f, SmearFlagYan s)
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
}