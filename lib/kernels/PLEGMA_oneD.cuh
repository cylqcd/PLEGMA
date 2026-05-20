#include <PLEGMA_kernel_utils.cuh>
using namespace plegma;

enum DerivTimeMode {
  TIME_PLUS  =  1,
  TIME_MINUS = -1
};

template<typename FloatOut, typename FloatIn, typename FloatGauge, bool noGhost=false>
__global__ void covariant_derivative_kernel(vector2<FloatOut> out,
                                            vectorTex<FloatIn> vecInTex,
                                            gaugeTex<FloatGauge> gaugeTex,
                                            short dir,
                                            int time_mode) {
  int sid = blockIdx.x * blockDim.x + threadIdx.x;
  if (sid >= out.volume()) return;

  Float2<FloatGauge> G[N_COLS][N_COLS];
  Float2<FloatIn>    S[N_SPINS][N_COLS];
  Float2<FloatOut>   tmp[N_SPINS][N_COLS];

  #pragma unroll
  for (int mu = 0; mu < N_SPINS; mu++)
    #pragma unroll
    for (int c = 0; c < N_COLS; c++)
      tmp[mu][c] = 0.;

  // Spatial directions:
  // D_i psi(x) = U_i(x) psi(x + i) - U_i^\dagger(x - i) psi(x - i)
  if (dir < N_DIMS - 1) {
    vecInTex.get<noGhost ? PlusNoGhost : Plus>(S, sid, dir);
    if (isNotZeroV(S)) {
      gaugeTex.get(G, dir, sid);
      mul_G_V<FloatOut, FloatGauge, FloatIn, ACC_PLUS>(tmp, G, S);
    }

    vecInTex.get<noGhost ? MinusNoGhost : Minus>(S, sid, dir);
    if (isNotZeroV(S)) {
      gaugeTex.get<Minus>(G, dir, sid, dir);
      mul_Gdag_V<FloatOut, FloatGauge, FloatIn, ACC_MINUS>(tmp, G, S);
    }
  }
  // Temporal direction:
  // Use only the selected orientation.
  else {
    if (time_mode == TIME_PLUS) {
      vecInTex.get<noGhost ? PlusNoGhost : Plus>(S, sid, dir);
      if (isNotZeroV(S)) {
        gaugeTex.get(G, dir, sid);
        mul_G_V<FloatOut, FloatGauge, FloatIn, ACC_PLUS>(tmp, G, S);
      }
    } else {
      vecInTex.get<noGhost ? MinusNoGhost : Minus>(S, sid, dir);
      if (isNotZeroV(S)) {
        gaugeTex.get<Minus>(G, dir, sid, dir);
        mul_Gdag_V<FloatOut, FloatGauge, FloatIn, ACC_MINUS>(tmp, G, S);
      }
    }
  }

  out.setSid(sid);

  #pragma unroll
  for (int mu = 0; mu < N_SPINS; mu++)
    #pragma unroll
    for (int c = 0; c < N_COLS; c++)
      out[mu * N_COLS + c] = tmp[mu][c];
}

template<typename FloatOut, typename FloatIn, typename FloatGauge>
__global__ void covariant_derivative_only_ghost_kernel(vector2<FloatOut> out,
                                                       vectorTex<FloatIn> vecInTex,
                                                       gaugeTex<FloatGauge> gaugeTex,
                                                       short dir,
                                                       int time_mode,
                                                       bool tuning = 0) {
  size_t sid = blockIdx.x * blockDim.x + threadIdx.x;
  if (sid >= 2 * out.sideGhostL(dir)) return;

  ORIENTATION sign = sid >= out.sideGhostL(dir) ? DIR_MINUS : DIR_PLUS;

  size_t id[4];
  sid = sid % out.sideGhostL(dir);

  #pragma unroll
  for (int i = 0; i < N_DIMS; i++) {
    if (i == dir) {
      id[i] = (sign == DIR_PLUS) ? (DGC_localL[dir] - 1) : 0;
    } else {
      id[i] = sid % DGC_localL[i];
      sid /= DGC_localL[i];
    }
  }
  sid = LEXIC_ID(id);

  Float2<FloatGauge> G[N_COLS][N_COLS];
  Float2<FloatIn>    S[N_SPINS][N_COLS];
  Float2<FloatOut>   tmp[N_SPINS][N_COLS];

  #pragma unroll
  for (int mu = 0; mu < N_SPINS; mu++)
    #pragma unroll
    for (int c = 0; c < N_COLS; c++)
      tmp[mu][c] = 0.;

  if (dir < N_DIMS - 1) {
    // Spatial directions:
    // plus-face contributes with + sign
    // minus-face contributes with - sign
    if (sign == DIR_PLUS) {
      vecInTex.get<PlusOnlyGhost>(S, sid, dir);
      if (isNotZeroV(S)) {
        gaugeTex.get(G, dir, sid);
        mul_G_V<FloatOut, FloatGauge, FloatIn, ACC_PLUS>(tmp, G, S);
      }
    } else {
      vecInTex.get<MinusOnlyGhost>(S, sid, dir);
      if (isNotZeroV(S)) {
        gaugeTex.get<Minus>(G, dir, sid, dir);
        mul_Gdag_V<FloatOut, FloatGauge, FloatIn, ACC_MINUS>(tmp, G, S);
      }
    }
  } else {
    // Temporal direction:
    // apply only the selected orientation
    if (time_mode == TIME_PLUS && sign == DIR_PLUS) {
      vecInTex.get<PlusOnlyGhost>(S, sid, dir);
      if (isNotZeroV(S)) {
        gaugeTex.get(G, dir, sid);
        mul_G_V<FloatOut, FloatGauge, FloatIn, ACC_PLUS>(tmp, G, S);
      }
    } else if (time_mode == TIME_MINUS && sign == DIR_MINUS) {
      vecInTex.get<MinusOnlyGhost>(S, sid, dir);
      if (isNotZeroV(S)) {
        gaugeTex.get<Minus>(G, dir, sid, dir);
        mul_Gdag_V<FloatOut, FloatGauge, FloatIn, ACC_MINUS>(tmp, G, S);
      }
    }
  }

  out.setSid(sid);

  if (isNotZeroV(tmp)) {
    #pragma unroll
    for (int mu = 0; mu < N_SPINS; mu++)
      #pragma unroll
      for (int c = 0; c < N_COLS; c++)
        out[mu * N_COLS + c] += tuning ? 0 : tmp[mu][c];
  }
}

template<typename FloatOut, typename FloatIn, typename FloatGauge>
static void covariant_derivative_k(vector2<FloatOut> out,
                                   vectorTex<FloatIn>& vecInTex,
                                   gaugeTex<FloatGauge>& gaugeTex,
                                   short dir,
                                   int time_mode = TIME_PLUS) {
  ProfileStruct ps(out.volume());
  tuneAndRun(ps,
             "covariant_derivative_kernel",
             covariant_derivative_kernel<FloatOut, FloatIn, FloatGauge, false>,
             out, vecInTex, gaugeTex, dir, time_mode);
}

template<typename FloatOut, typename FloatIn, typename FloatGauge>
static void covariant_derivative_no_ghost_k(vector2<FloatOut> out,
                                            vectorTex<FloatIn>& vecInTex,
                                            gaugeTex<FloatGauge>& gaugeTex,
                                            short dir,
                                            int time_mode = TIME_PLUS) {
  ProfileStruct ps(out.volume());
  tuneAndRun(ps,
             "covariant_derivative_no_ghost_kernel",
             covariant_derivative_kernel<FloatOut, FloatIn, FloatGauge, true>,
             out, vecInTex, gaugeTex, dir, time_mode);
}

template<typename FloatOut, typename FloatIn, typename FloatGauge>
static void covariant_derivative_only_ghost_k(vector2<FloatOut> out,
                                              vectorTex<FloatIn>& vecInTex,
                                              gaugeTex<FloatGauge>& gaugeTex,
                                              short dir,
                                              int time_mode = TIME_PLUS) {
  if (HGC_dimBreak[dir]) {
    ProfileStruct ps(out.sideGhostL(dir) * 2);

    auto kernel = tuner(ps,
                        "covariant_derivative_only_ghost_kernel",
                        covariant_derivative_only_ghost_kernel<FloatOut, FloatIn, FloatGauge>,
                        out, vecInTex, gaugeTex, dir, time_mode, 0);

    if (!kernel->tuned()) {
      tune(ps,
           "covariant_derivative_only_ghost_kernel",
           covariant_derivative_only_ghost_kernel<FloatOut, FloatIn, FloatGauge>,
           out, vecInTex, gaugeTex, dir, time_mode, 1);
    }

    if (!kernel->tuned()) {
      PLEGMA_warning("ISSUE: Tuning again?!");
    }

    kernel->apply();
    delete kernel;
  }
}


template<typename FloatOut, typename FloatIn, typename FloatGauge>
static void covariant_derivative(PLEGMA_Vector<FloatOut>& out,
                                 PLEGMA_Vector<FloatIn>& vecIn,
                                 PLEGMA_Gauge<FloatGauge>& gauge,
                                 short dir,
                                 int time_mode = TIME_PLUS) {
  assert(out.checkVolume(vecIn, gauge));

  auto texVecIn   = toTexture<vectorTex>(vecIn);
  auto texGaugeIn = toTexture<gaugeTex>(gauge);

  covariant_derivative_k<FloatOut, FloatIn, FloatGauge>(
      toField2<vector2>(out), *texVecIn, *texGaugeIn, dir, time_mode);
}

template<typename FloatOut, typename FloatIn, typename FloatGauge, bool noGhost=false>
__global__ void covariant_derivative_prop_kernel(prop2<FloatOut> out,
                                                 propTex<FloatIn> propInTex,
                                                 gaugeTex<FloatGauge> gaugeTex,
                                                 short dir,
                                                 int time_mode) {
  int sid = blockIdx.x * blockDim.x + threadIdx.x;
  if (sid >= out.volume()) return;

  Float2<FloatGauge> G[N_COLS][N_COLS];
  Float2<FloatIn>    S[N_SPINS][N_SPINS][N_COLS][N_COLS];
  Float2<FloatOut>   tmp[N_SPINS][N_SPINS][N_COLS][N_COLS];

  #pragma unroll
  for (int mu = 0; mu < N_SPINS; mu++)
    #pragma unroll
    for (int nu = 0; nu < N_SPINS; nu++)
      #pragma unroll
      for (int c1 = 0; c1 < N_COLS; c1++)
        #pragma unroll
        for (int c2 = 0; c2 < N_COLS; c2++)
          tmp[mu][nu][c1][c2] = 0.;

  // Spatial directions:
  // D_i S(x) = U_i(x) S(x + i) - U_i^\dagger(x - i) S(x - i)
  if (dir < N_DIMS - 1) {
    propInTex.get<noGhost ? PlusNoGhost : Plus>(S, sid, dir);
    gaugeTex.get(G, dir, sid);
    mul_G<FloatOut, FloatGauge, FloatIn, ACC_PLUS>(tmp, G, S);

    propInTex.get<noGhost ? MinusNoGhost : Minus>(S, sid, dir);
    gaugeTex.get<Minus>(G, dir, sid, dir);
    mul_Gdag<FloatOut, FloatGauge, FloatIn, ACC_MINUS>(tmp, G, S);
  }
  // Temporal direction:
  // Use only the selected orientation.
  else {
    if (time_mode == TIME_PLUS) {
      propInTex.get<noGhost ? PlusNoGhost : Plus>(S, sid, dir);
      gaugeTex.get(G, dir, sid);
      mul_G<FloatOut, FloatGauge, FloatIn, ACC_PLUS>(tmp, G, S);
    } else {
      propInTex.get<noGhost ? MinusNoGhost : Minus>(S, sid, dir);
      gaugeTex.get<Minus>(G, dir, sid, dir);
      mul_Gdag<FloatOut, FloatGauge, FloatIn, ACC_MINUS>(tmp, G, S);
    }
  }

  out.setSid(sid);

  #pragma unroll
  for (int mu = 0; mu < N_SPINS; mu++)
    #pragma unroll
    for (int nu = 0; nu < N_SPINS; nu++)
      #pragma unroll
      for (int c1 = 0; c1 < N_COLS; c1++)
        #pragma unroll
        for (int c2 = 0; c2 < N_COLS; c2++)
          out[((mu * N_SPINS + nu) * N_COLS + c1) * N_COLS + c2] = tmp[mu][nu][c1][c2];
}

template<typename FloatOut, typename FloatIn, typename FloatGauge>
__global__ void covariant_derivative_prop_only_ghost_kernel(prop2<FloatOut> out,
                                                            propTex<FloatIn> propInTex,
                                                            gaugeTex<FloatGauge> gaugeTex,
                                                            short dir,
                                                            int time_mode,
                                                            bool tuning = 0) {
  size_t sid = blockIdx.x * blockDim.x + threadIdx.x;
  if (sid >= 2 * out.sideGhostL(dir)) return;

  ORIENTATION sign = sid >= out.sideGhostL(dir) ? DIR_MINUS : DIR_PLUS;

  size_t id[4];
  sid = sid % out.sideGhostL(dir);

  #pragma unroll
  for (int i = 0; i < N_DIMS; i++) {
    if (i == dir) {
      id[i] = (sign == DIR_PLUS) ? (DGC_localL[dir] - 1) : 0;
    } else {
      id[i] = sid % DGC_localL[i];
      sid /= DGC_localL[i];
    }
  }
  sid = LEXIC_ID(id);

  Float2<FloatGauge> G[N_COLS][N_COLS];
  Float2<FloatIn>    S[N_SPINS][N_SPINS][N_COLS][N_COLS];
  Float2<FloatOut>   tmp[N_SPINS][N_SPINS][N_COLS][N_COLS];

  #pragma unroll
  for (int mu = 0; mu < N_SPINS; mu++)
    #pragma unroll
    for (int nu = 0; nu < N_SPINS; nu++)
      #pragma unroll
      for (int c1 = 0; c1 < N_COLS; c1++)
        #pragma unroll
        for (int c2 = 0; c2 < N_COLS; c2++)
          tmp[mu][nu][c1][c2] = 0.;

  if (dir < N_DIMS - 1) {
    // Spatial directions:
    // plus-face contributes with + sign
    // minus-face contributes with - sign
    if (sign == DIR_PLUS) {
      propInTex.get<PlusOnlyGhost>(S, sid, dir);
      gaugeTex.get(G, dir, sid);
      mul_G<FloatOut, FloatGauge, FloatIn, ACC_PLUS>(tmp, G, S);
    } else {
      propInTex.get<MinusOnlyGhost>(S, sid, dir);
      gaugeTex.get<Minus>(G, dir, sid, dir);
      mul_Gdag<FloatOut, FloatGauge, FloatIn, ACC_MINUS>(tmp, G, S);
    }
  } else {
    // Temporal direction:
    // apply only the selected orientation
    if (time_mode == TIME_PLUS && sign == DIR_PLUS) {
      propInTex.get<PlusOnlyGhost>(S, sid, dir);
      gaugeTex.get(G, dir, sid);
      mul_G<FloatOut, FloatGauge, FloatIn, ACC_PLUS>(tmp, G, S);
    } else if (time_mode == TIME_MINUS && sign == DIR_MINUS) {
      propInTex.get<MinusOnlyGhost>(S, sid, dir);
      gaugeTex.get<Minus>(G, dir, sid, dir);
      mul_Gdag<FloatOut, FloatGauge, FloatIn, ACC_MINUS>(tmp, G, S);
    }
  }

  out.setSid(sid);

  #pragma unroll
  for (int mu = 0; mu < N_SPINS; mu++)
    #pragma unroll
    for (int nu = 0; nu < N_SPINS; nu++)
      #pragma unroll
      for (int c1 = 0; c1 < N_COLS; c1++)
        #pragma unroll
        for (int c2 = 0; c2 < N_COLS; c2++)
          out[((mu * N_SPINS + nu) * N_COLS + c1) * N_COLS + c2] += tuning ? 0 : tmp[mu][nu][c1][c2];
}

template<typename FloatOut, typename FloatIn, typename FloatGauge>
static void covariant_derivative_prop_k(prop2<FloatOut> out,
                                        propTex<FloatIn>& propInTex,
                                        gaugeTex<FloatGauge>& gaugeTex,
                                        short dir,
                                        int time_mode = TIME_PLUS) {
  ProfileStruct ps(out.volume());
  tuneAndRun(ps,
             "covariant_derivative_prop_kernel",
             covariant_derivative_prop_kernel<FloatOut, FloatIn, FloatGauge, false>,
             out, propInTex, gaugeTex, dir, time_mode);
}

template<typename FloatOut, typename FloatIn, typename FloatGauge>
static void covariant_derivative_prop_no_ghost_k(prop2<FloatOut> out,
                                                 propTex<FloatIn>& propInTex,
                                                 gaugeTex<FloatGauge>& gaugeTex,
                                                 short dir,
                                                 int time_mode = TIME_PLUS) {
  ProfileStruct ps(out.volume());
  tuneAndRun(ps,
             "covariant_derivative_prop_no_ghost_kernel",
             covariant_derivative_prop_kernel<FloatOut, FloatIn, FloatGauge, true>,
             out, propInTex, gaugeTex, dir, time_mode);
}

template<typename FloatOut, typename FloatIn, typename FloatGauge>
static void covariant_derivative_prop_only_ghost_k(prop2<FloatOut> out,
                                                   propTex<FloatIn>& propInTex,
                                                   gaugeTex<FloatGauge>& gaugeTex,
                                                   short dir,
                                                   int time_mode = TIME_PLUS) {
  if (HGC_dimBreak[dir]) {
    ProfileStruct ps(out.sideGhostL(dir) * 2);

    auto kernel = tuner(ps,
                        "covariant_derivative_prop_only_ghost_kernel",
                        covariant_derivative_prop_only_ghost_kernel<FloatOut, FloatIn, FloatGauge>,
                        out, propInTex, gaugeTex, dir, time_mode, 0);

    if (!kernel->tuned()) {
      tune(ps,
           "covariant_derivative_prop_only_ghost_kernel",
           covariant_derivative_prop_only_ghost_kernel<FloatOut, FloatIn, FloatGauge>,
           out, propInTex, gaugeTex, dir, time_mode, 1);
    }

    if (!kernel->tuned()) {
      PLEGMA_warning("ISSUE: Tuning again?!");
    }

    kernel->apply();
    delete kernel;
  }
}

template<typename FloatOut, typename FloatIn, typename FloatGauge>
static void covariant_derivative(PLEGMA_Propagator<FloatOut>& out,
                                 PLEGMA_Propagator<FloatIn>& propIn,
                                 PLEGMA_Gauge<FloatGauge>& gauge,
                                 short dir,
                                 int time_mode = TIME_PLUS) {
  assert(out.checkVolume(propIn, gauge));

  auto texPropIn   = toTexture<propTex>(propIn);
  auto texGaugeIn  = toTexture<gaugeTex>(gauge);

  covariant_derivative_prop_k<FloatOut, FloatIn, FloatGauge>(
      toField2<prop2>(out), *texPropIn, *texGaugeIn, dir, time_mode);
}