#pragma once

#include <PLEGMA.h>
#include <PLEGMA_utils.h>

#include <chrono>
#include <string>
#include <vector>


enum FlavorYan {u,d};

// S=smear, L=local, B=both; left->in, right->out; Ex: SL => smear -> solve -> local (no smear);
enum SmearFlagYan {SS,SL,LS,LL,single2double,SB,LB};

class ToolYan
{
private:
protected:
    std::chrono::_V2::system_clock::time_point tStart;
    std::chrono::_V2::system_clock::time_point tLast;

    plegma::PLEGMA_Gauge<double> *gaugeP = NULL, *smearedGaugeP = NULL;
    quda::QUDA_solver *solverP = NULL;
    bool smearQ = false; // Global smearing flag
    int nsmearGauss;
    double alphaGauss;

public:
    ToolYan();
    ~ToolYan();

    void printTime(std::string s = "Regular", bool preciseQ = true);
    void testTimeCost();

    void setupGauge(plegma::PLEGMA_Gauge<double> &gauge) { assert(gaugeP == NULL); gaugeP = &gauge; };
    void setupSmearing(plegma::PLEGMA_Gauge<double> &smearedGauge, int nsmearGauss, double alphaGauss) { assert(smearedGaugeP == NULL); smearedGaugeP = &smearedGauge; this->nsmearGauss=nsmearGauss; this->alphaGauss=alphaGauss; smearQ = true; };
    void setupSolver(quda::QUDA_solver &solver) { assert(solverP == NULL); solverP = &solver; };

    // 1st: rotateToPhysicalBasis -> scale (for numerical purpose) -> solve
    void solve(plegma::PLEGMA_Vector<double> &out, plegma::PLEGMA_Vector<double> &in, FlavorYan f);
    // 2rd: smear -> 1st
    void solve(plegma::PLEGMA_Vector<double> &out, plegma::PLEGMA_Vector3D<double> &in, int inTime, FlavorYan f, SmearFlagYan s);
    // 3nd: smear -> 1st
    void solve(plegma::PLEGMA_Vector<double> &outS, plegma::PLEGMA_Vector<double> &outL, plegma::PLEGMA_Vector3D<double> &in, int inTime, FlavorYan f, SmearFlagYan s);

    // out = outS when s != SB/LB
    void setupPointPropagator(plegma::PLEGMA_Propagator<float> &outS, plegma::PLEGMA_Propagator<float> &outL, plegma::site &source, FlavorYan f, SmearFlagYan s);
    void setupSequentialPropagator(plegma::PLEGMA_Propagator<float> &outS, plegma::PLEGMA_Propagator<float> &outL, plegma::PLEGMA_Propagator<float> &in, int inTime, plegma::GAMMAS Gamma, std::vector<int> mom, int momSign, FlavorYan f, SmearFlagYan s);
};