#include <PLEGMA_kernel_utils.cuh>
using namespace plegma;
template<typename Float>
static __global__ void su3Projection_kernel(Float* S){
  int sid = blockIdx.x*blockDim.x + threadIdx.x;
  if (sid >= c_threads) return;

  su3_2<Float> RS(S);
  Float2<Float> ThirdRootOne[2];
  Float ThirdRoot_18,ThirdRoot_12,ThirdRoot_2_3;
  Float2<Float> M[N_COLS][N_COLS] , H[N_COLS][N_COLS] , U[N_COLS][N_COLS];
  Float2<Float> detM;
  Float phase,cosPhase,sinPhase;
  Float sum;
  Float e[N_COLS];
  Float trace;
  Float a;
  Float2<Float> b,w;
  Float2<Float> v[N_COLS][N_COLS] , vr[N_COLS][N_COLS];
  // initialize some constants

  ThirdRootOne[0].x = 1.;
  ThirdRootOne[0].y = sqrt(3.);
  ThirdRootOne[1].x = 1.;
  ThirdRootOne[1].y = -sqrt(3.);
  ThirdRoot_12 =pow(12.,1./3.);
  ThirdRoot_18 =pow(18.,1./3.);
  ThirdRoot_2_3=pow((2./3.),1./3.);

  RS.get(M,sid);
  detM = det(M);
  phase = atan2(detM.y,detM.x)/3.;
  cosPhase=cos(phase);
  sinPhase=sin(phase);
  mul_Gdag_G(H,M,M);

  H[0][1].x = (H[0][1].x + H[1][0].x)/2.;                                    
  H[0][1].y = (H[0][1].y - H[1][0].y)/2.;
  H[1][0] =  conj(H[0][1]);
  H[0][2].x = (H[0][2].x + H[2][0].x)/2.;
  H[0][2].y = (H[0][2].y - H[2][0].y)/2.;
  H[2][0] =  conj(H[0][2]);
  H[1][2].x = (H[1][2].x + H[2][1].x)/2.;
  H[1][2].y = (H[1][2].y - H[2][1].y)/2.;
  H[2][1] =  conj(H[1][2]);

  sum = norm(H[0][1])+norm(H[0][2])+norm(H[1][2]);

  if(sum <= 1e-08){
#pragma unroll
    for(int c1=0; c1<N_COLS; c1++) e[c1]=1./sqrt(H[c1][c1].x);

#pragma unroll
    for(int c1=0; c1<N_COLS; c1++)
#pragma unroll
      for(int c2=0; c2<N_COLS; c2++)
	U[c1][c2] = e[c1] * M[c1][c2];

    RS.set(U,sid);
  }
  else{
    trace = real_trace<Float,Float>(H)/3.;
#pragma unroll
    for(int c1=0; c1<N_COLS; c1++) H[c1][c1].x -= trace;

    a = - ( norm2(H[0][1]) + norm2(H[0][2]) + norm2(H[1][2]) + H[2][2].x * H[2][2].x - H[0][0].x * H[1][1].x ); 
    b.x = - H[0][0].x * H[1][1].x * H[2][2].x + H[2][2].x * norm2(H[0][1])
      - (H[0][1] * H[1][2] * conj(H[0][2])).x + H[1][1].x * norm2(H[0][2]);
    b.y = H[2][2].x *(H[0][1] * conj(H[0][1])).y
      - (H[0][1] * H[1][2] * conj(H[0][2])).y + H[1][1].x * (H[0][2] * conj(H[0][2])).y;
    b.x +=   H[0][0].x * (H[1][2] * conj(H[1][2])).x - (H[0][2] * conj(H[0][1]) * conj(H[1][2]) ).x;               
    b.y +=   H[0][0].x * (H[1][2] * conj(H[1][2])).y - (H[0][2] * conj(H[0][1]) * conj(H[1][2]) ).y;
    
    Float2<Float> temp1;
    Float2<Float> D;

    temp1.x = 12. * a * a * a + 81. * (b * b).x ;
    temp1.y = 81. * (b * b).y;
    w = cpow<Float>(temp1,0.5);
    
    temp1.x = -9.0 * b.x + w.x;
    temp1.y = -9.0 * b.y + w.y;
    D = cpow<Float>(temp1,1./3.);

    temp1.x = a*ThirdRoot_2_3; temp1.y = 0.;
    e[0] = D.x / (ThirdRoot_18) - (temp1/D).x;
    temp1.x = D.x * ThirdRoot_12 ; temp1.y = D.y * ThirdRoot_12;
    e[1] = a * (ThirdRootOne[0] / temp1).x - (ThirdRootOne[1] * D).x / (ThirdRoot_18*2.);
    e[2] = -e[0]-e[1];

#pragma unroll
    for(int c1 = 0 ; c1 < N_COLS ; c1++){
      e[c1] += trace; H[c1][c1].x += trace;
    }

    // eigenvectors
    v[0][0].x = -(e[0]*H[2][0].x - H[2][0].x*H[1][1].x + (H[1][0]*H[2][1]).x);                         
    v[0][0].y = -(e[0]*H[2][0].y - H[2][0].y*H[1][1].x + (H[1][0]*H[2][1]).y);

    v[0][1].x = -((H[2][0]*H[0][1]).x + e[0]*H[2][1].x - H[0][0].x*H[2][1].x);
    v[0][1].y = -((H[2][0]*H[0][1]).y + e[0]*H[2][1].y - H[0][0].x*H[2][1].y);

    v[0][2].x =-e[0]*e[0] + e[0]*H[0][0].x + (H[0][1]*conj(H[0][1])).x + e[0]*H[1][1].x - H[0][0].x*H[1][1].x;
    v[0][2].y = 0.;

    v[1][0].x = -(e[1]*H[2][0].x - H[2][0].x*H[1][1].x + (H[1][0]*H[2][1]).x);
    v[1][0].y = -(e[1]*H[2][0].y - H[2][0].y*H[1][1].x + (H[1][0]*H[2][1]).y);

    v[1][1].x = -((H[2][0]*H[0][1]).x + e[1]*H[2][1].x - H[0][0].x*H[2][1].x);
    v[1][1].y = -((H[2][0]*H[0][1]).y + e[1]*H[2][1].y - H[0][0].x*H[2][1].y);

    v[1][2].x =-e[1]*e[1] + e[1]*H[0][0].x + (H[0][1]*conj(H[0][1])).x + e[1]*H[1][1].x - H[0][0].x*H[1][1].x;;
    v[1][2].y = 0.;

    Float norma;

    norma = norm2(v[0][0]) + norm2(v[0][1]) + norm2(v[0][2]);
    w = v[0][0] * conj(v[1][0]) + v[0][1] * conj(v[1][1]) + v[0][2] * conj(v[1][2]);
    w.x /= norma;
    w.y /= norma;

    v[1][0].x-= (w * v[0][0]).x;
    v[1][0].y-= (w * v[0][0]).y;

    v[1][1].x-= (w * v[0][1]).x;
    v[1][1].y-= (w * v[0][1]).y;

    v[1][2].x-= (w * v[0][2]).x;
    v[1][2].y-= (w * v[0][2]).y;

    norma=1./sqrt(norma);

    v[0][0].x *= norma;
    v[0][0].y *= norma;

    v[0][1].x *= norma;
    v[0][1].y *= norma;

    v[0][2].x *= norma;
    v[0][2].y *= norma;

    //////////////////////
    norma = norm2(v[1][0]) + norm2(v[1][1]) + norm2(v[1][2]);

    norma=1./sqrt(norma);

    v[1][0].x *= norma;
    v[1][0].y *= norma;

    v[1][1].x *= norma;
    v[1][1].y *= norma;

    v[1][2].x *= norma;
    v[1][2].y *= norma;

    /////////////////////////////

    v[2][0].x =  (v[0][1]*v[1][2]).x - (v[0][2]*v[1][1]).x;                                 
    v[2][0].y = -(v[0][1]*v[1][2]).y + (v[0][2]*v[1][1]).y;

    v[2][1].x = -(v[0][0]*v[1][2]).x + (v[0][2]*v[1][0]).x;
    v[2][1].y = +(v[0][0]*v[1][2]).y - (v[0][2]*v[1][0]).y;

    v[2][2].x =  (v[0][0]*v[1][1]).x - (v[0][1]*v[1][0]).x;
    v[2][2].y = -(v[0][0]*v[1][1]).y + (v[0][1]*v[1][0]).y;

    Float de;

#pragma unroll
    for(int c1 = 0 ; c1 < N_COLS ; c1++){
      de = 1./sqrt(e[c1]);
      b.x = de*cosPhase; b.y =-de*sinPhase;
#pragma unroll
      for(int c2 = 0 ; c2 < N_COLS ; c2++){
	vr[c1][c2] = b*v[c1][c2];
      }
    }
    
    mul_Gdag_G(H,M,v);
    mul_G_G(U,H,vr);

    norma = norm2(U[0][0]) + norm2(U[1][0]) + norm2(U[2][0]);
    w = U[0][0] * conj(U[0][1]) + U[1][0] * conj(U[1][1]) + U[2][0] * conj(U[2][1]);

    w.x /= norma;
    w.y /= norma;


    U[0][1].x -= (w*U[0][0]).x;
    U[0][1].y -= (w*U[0][0]).y;

    U[1][1].x -= (w*U[1][0]).x;
    U[1][1].y -= (w*U[1][0]).y;

    U[2][1].x -= (w*U[2][0]).x;
    U[2][1].y -= (w*U[2][0]).y;

    norma = 1./sqrt(norma);

    U[0][0].x*= norma;
    U[0][0].y*= norma;
    U[1][0].x*= norma;
    U[1][0].y*= norma;
    U[2][0].x*= norma;
    U[2][0].y*= norma;


    norma = norm2(U[0][1]) + norm2(U[1][1]) + norm2(U[2][1]);
    norma = 1./sqrt(norma);

    U[0][1].x *= norma;
    U[0][1].y *= norma;
    U[1][1].x *= norma;
    U[1][1].y *= norma;
    U[2][1].x *= norma;
    U[2][1].y *= norma;


    U[0][2].x =  (U[1][0]*U[2][1]).x - (U[2][0]*U[1][1]).x;                             
    U[0][2].y = -(U[1][0]*U[2][1]).y + (U[2][0]*U[1][1]).y;

    U[1][2].x = -(U[0][0]*U[2][1]).x + (U[2][0]*U[0][1]).x;
    U[1][2].y =  (U[0][0]*U[2][1]).y - (U[2][0]*U[0][1]).y;

    U[2][2].x =  (U[0][0]*U[1][1]).x - (U[1][0]*U[0][1]).x;
    U[2][2].y = -(U[0][0]*U[1][1]).y + (U[1][0]*U[0][1]).y;

    RS.set(U,sid);
  }
  
}

template<typename Float>
static void su3Projection_k(PLEGMA_Su3field<Float> &S){
  dim3 blockDim( THREADS_PER_BLOCK , 1, 1);
  dim3 gridDim( (GK_localVolume + blockDim.x -1)/blockDim.x , 1 , 1);
  su3Projection_kernel<Float><<<gridDim,blockDim>>>(S.D_elem());
  checkCudaError();
}
