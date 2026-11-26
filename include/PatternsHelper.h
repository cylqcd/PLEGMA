#pragma once

#include "PLEGMA_QLoops_fast.h"
#include "PLEGMA_fiveD.h"
#include <utils/QUDA_interface.h>
#include <memory>
#include <vector>

// ---------------------------------------------------------------------------
// PatternsHelper<PatFloat>
//
// Bundles the two persistent GPU accumulators (pat_std, pat_gen) and their
// shared FT object so that main() holds a single typed pointer regardless of
// precision.  contractAndFlush() gathers all Nsc diluted phi vectors, calls
// the multi-isc fast path, and writes the result to HDF5.
//
// PatFloat = float  (default, ~2-8x faster on GPU)
//          = double (higher precision, enabled by --pat-double)
// ---------------------------------------------------------------------------
template<typename PatFloat>
struct PatternsHelper {
    PLEGMA_QLoops_fast<PatFloat> pat_std;
    PLEGMA_QLoops_fast<PatFloat> pat_gen;
    PLEGMA_QLoops_fast<PatFloat> pat_direct;
    PLEGMA_FT<PatFloat>*         ft;

    PatternsHelper(int maxQsq) : pat_std(BOTH), pat_gen(BOTH), pat_direct(BOTH) {
        ft = new PLEGMA_FT<PatFloat>(maxQsq, 3);
    }
    ~PatternsHelper() { delete ft; }

    // Contract Nsc diluted components and flush all patterns to HDF5.
    // Uses contractG5_fiveD_patterns_multi: accumulates all Nsc terms on
    // the device with a single FT per pattern (vs. Nsc FTs per pattern if
    // called one-by-one).
    template<typename PhiContainer>
    void contractAndFlush(PhiContainer& phi_now, int i, int Nsc,
                          PLEGMA_Gauge<double>& gauge_flowed,
                          QUDA_dirac* D,
                          const QudaInvertParam& inv_params,
                          const std::string& fileprefix,
                          const std::string& confID,
                          FILE_FORMAT format, int isrc_global)
    {
        // Cast gauge to PatFloat once per checkpoint
        PLEGMA_Gauge<PatFloat> gauge_pf;
        gauge_pf.copy(gauge_flowed);
        gauge_pf.communicateGhost();

        std::vector<std::unique_ptr<PLEGMA_Vector<PatFloat>>> phi_pf_owned(Nsc);
        std::vector<PLEGMA_Vector<PatFloat>*>                 phi_pf(Nsc);
        std::vector<PatFloat> vals_std(Nsc, PatFloat(-1.0));

        for (int isc = 0; isc < Nsc; ++isc) {
            int idx = i * Nsc + isc;

            // Scale phi: 1/(2*kappa)
            PLEGMA_Vector<double> phi_scaled(BOTH);
            phi_scaled.copy(*phi_now[idx]);
            phi_scaled.scale(1.0 / (2.0 * inv_params.kappa));

            phi_pf_owned[isc].reset(new PLEGMA_Vector<PatFloat>(BOTH, FIRST_SIDE));
            phi_pf_owned[isc]->copy(phi_scaled);
            phi_pf_owned[isc]->communicateGhost();
            phi_pf[isc] = phi_pf_owned[isc].get();
        }

        // std part: x_l = phi, x_r = phi (accumulated over all Nsc)
        contractG5_fiveD_patterns_multi(phi_pf, phi_pf, gauge_pf,
                                        pat_std, *ft, vals_std,
                                        false, 5);

        pat_std.batchFlush(*ft, fileprefix + "pat_stoch_part_std",
                           confID, format, isrc_global);
    }

    // Direct contraction: x_l = flowed_src (unscaled), x_r = flowed_phi (scaled by 1/2kappa)
    // Computes Tr[ src_flowed^dag Gamma phi_flowed ] / (2*kappa)  for all higher-order patterns.
    template<typename PhiContainer>
    void contractAndFlush(PhiContainer& src_now, PhiContainer& phi_now, int i, int Nsc,
                          PLEGMA_Gauge<double>& gauge_flowed,
                          QUDA_dirac* D,
                          const QudaInvertParam& inv_params,
                          const std::string& fileprefix,
                          const std::string& confID,
                          FILE_FORMAT format, int isrc_global)
    {
        // Cast gauge to PatFloat once per checkpoint
        PLEGMA_Gauge<PatFloat> gauge_pf;
        gauge_pf.copy(gauge_flowed);
        gauge_pf.communicateGhost();

        std::vector<std::unique_ptr<PLEGMA_Vector<PatFloat>>> phi_pf_owned(Nsc);
        std::vector<PLEGMA_Vector<PatFloat>*>                 phi_pf(Nsc);
        std::vector<std::unique_ptr<PLEGMA_Vector<PatFloat>>> src_pf_owned(Nsc);
        std::vector<PLEGMA_Vector<PatFloat>*>                 src_pf(Nsc);
        std::vector<PatFloat> vals_direct(Nsc, PatFloat(-1.0));

        for (int isc = 0; isc < Nsc; ++isc) {
            int idx = i * Nsc + isc;

            // phi: scale by 1/(2*kappa)
            PLEGMA_Vector<double> phi_scaled(BOTH);
            phi_scaled.copy(*phi_now[idx]);
            phi_scaled.scale(1.0 / (2.0 * inv_params.kappa));

            phi_pf_owned[isc].reset(new PLEGMA_Vector<PatFloat>(BOTH, FIRST_SIDE));
            phi_pf_owned[isc]->copy(phi_scaled);
            phi_pf_owned[isc]->communicateGhost();
            phi_pf[isc] = phi_pf_owned[isc].get();

            // src: no scaling
            src_pf_owned[isc].reset(new PLEGMA_Vector<PatFloat>(BOTH, FIRST_SIDE));
            src_pf_owned[isc]->copy(*src_now[idx]);
            src_pf_owned[isc]->communicateGhost();
            src_pf[isc] = src_pf_owned[isc].get();
        }

        // direct part: x_l = src, x_r = phi
        contractG5_fiveD_patterns_multi(src_pf, phi_pf, gauge_pf,
                                        pat_direct, *ft, vals_direct,
                                        false, 5);

        pat_direct.batchFlush(*ft, fileprefix + "pat_stoch_part_direct",
                              confID, format, isrc_global);
    }
};
