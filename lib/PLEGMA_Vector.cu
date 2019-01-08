#include <PLEGMA_Vector.h>
#include <PLEGMA_Gauge.h>
#include <PLEGMA_Propagator.h>
#include <PLEGMA_lime.h>
#include <PLEGMA_vector_utils.cuh> 
#include <PLEGMA_gaussian_smearing.cuh> 
#include <PLEGMA_covD.cuh>
using namespace plegma;
using namespace quda;
//---------------------------//
// class PLEGMA_Vector //
//---------------------------//

template<typename Float>
PLEGMA_Vector<Float>::PLEGMA_Vector(ALLOCATION_FLAG alloc_flag, GHOST_FLAG ghost_flag): 
  PLEGMA_Field<Float>(alloc_flag, VECTOR, ghost_flag){ ; }


template<typename FloatOut, typename FloatIn>
static void copyVector(PLEGMA_Vector<FloatOut> &vecOut, PLEGMA_Vector<FloatIn> &vecIn){
  if(typeid(FloatIn) != typeid(FloatOut) )
    castVector(vecOut.D_elem(), vecIn.D_elem());
  else
    cudaMemcpy(vecOut.D_elem(), vecIn.D_elem(), vecIn.Bytes_total(), 
	       cudaMemcpyDeviceToDevice);
}

template<typename Float>
void PLEGMA_Vector<Float>::copy(PLEGMA_Vector<float> &vecIn) {
  copyVector(*this,vecIn);
}
template<typename Float>
void PLEGMA_Vector<Float>::copy(PLEGMA_Vector<double> &vecIn)  {
  copyVector(*this,vecIn);
}

template<typename Float>
void PLEGMA_Vector<Float>::gaussianSmearing(PLEGMA_Vector<Float> &vecIn,PLEGMA_Gauge<Float> &gaugeAPE){
  gaugeAPE.communicateSideGhost();
  vecIn.communicateSideGhost();

  gaugeTex<Float> texGauge;
  vectorTex<Float> texVecIn, texVecOut;

  texVecOut.tex = this->createTexObject();
  texVecIn.tex = vecIn.createTexObject();
  texGauge.tex = gaugeAPE.createTexObject();
  
  for(int i = 0 ; i < GK_nsmearGauss ; i++){
    if( (i%2) == 0){
      gaussian_smearing(this->D_elem(),texVecIn,texGauge);
      this->communicateSideGhost();
    }
    else{
      gaussian_smearing(vecIn.D_elem(),texVecOut,texGauge);
      vecIn.communicateSideGhost();
    }
  }

  if( (GK_nsmearGauss%2) == 0) cudaMemcpy(this->D_elem(),vecIn.D_elem(),PLEGMA_Field<Float>::bytes_total_length,cudaMemcpyDeviceToDevice);
  
  this->destroyTexObject(texVecOut.tex);
  vecIn.destroyTexObject(texVecIn.tex);
  gaugeAPE.destroyTexObject(texGauge.tex);
  checkCudaError();
}

template<typename Float>
void PLEGMA_Vector<Float>::copyToQUDA(ColorSpinorField *qudaVector, bool isEv){
  copy_to_QUDA(PLEGMA_Field<Float>::d_elem, *qudaVector, isEv);
}

template<typename Float>
void PLEGMA_Vector<Float>::copyFromQUDA(ColorSpinorField *qudaVector, bool isEv){
  copy_from_QUDA(PLEGMA_Field<Float>::d_elem, *qudaVector, isEv);
}

template<typename Float>
void  PLEGMA_Vector<Float>::scaleVector(Float a){
  scale_vector(a,PLEGMA_Field<Float>::d_elem);
}

template<typename Float>
void  PLEGMA_Vector<Float>::conjugate(){
  conjugate_vector(PLEGMA_Field<Float>::d_elem);
}

template<typename Float>
void  PLEGMA_Vector<Float>::apply_gamma5(){
  apply_gamma5_vector(PLEGMA_Field<Float>::d_elem);
}

template<typename Float>
void PLEGMA_Vector<Float>::norm2Host(){
  Float res = 0.;
  Float globalRes;

  for(int i = 0 ; i < N_SPINS*N_COLS*GK_localVolume ; i++){
    res += PLEGMA_Field<Float>::h_elem[i*2 + 0]*PLEGMA_Field<Float>::h_elem[i*2 + 0] + PLEGMA_Field<Float>::h_elem[i*2 + 1]*PLEGMA_Field<Float>::h_elem[i*2 + 1];
  }

  int rc = MPI_Allreduce(&res, &globalRes , 1, sizeof(Float)==4 ? MPI_FLOAT : MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  if( rc != MPI_SUCCESS ) errorQuda("Error in MPI reduction for plaquette");
  printfQuda("Vector norm2 is %e\n",globalRes);
}

template<typename Float>
void PLEGMA_Vector<Float>::copyPropagator3D(PLEGMA_Propagator3D<Float> &prop, int global_it, int nu , int c2){
  if(global_it >= GK_totalL[3]) errorQuda("The global time slice you provided exceed the temporal extent\n");
  int my_it = global_it - comm_coords(default_topo)[3] * GK_localL[3];
  bool is_myIt = (my_it >= 0) && ( my_it < GK_localL[3] );

  int V3 = GK_localVolume/GK_localL[3];
  Float *pointer_src = NULL;
  Float *pointer_dst = NULL;
  
  for(int mu = 0 ; mu < 4 ; mu++)
    for(int c1 = 0 ; c1 < 3 ; c1++){
      pointer_dst = (PLEGMA_Field<Float>::d_elem + mu*3*GK_localVolume*2 + c1*GK_localVolume*2 + global_it*V3*2);
      if(is_myIt){
	pointer_src = (prop.D_elem() + mu*4*3*3*V3*2 + nu*3*3*V3*2 + c1*3*V3*2 + c2*V3*2);
	cudaMemcpy(pointer_dst, pointer_src, V3*2 * sizeof(Float), cudaMemcpyDeviceToDevice);
      }
      else
	cudaMemset(pointer_dst, 0, V3*2 * sizeof(Float));
    }
  checkCudaError();
}

template<typename Float>
void PLEGMA_Vector<Float>::copyPropagator(PLEGMA_Propagator<Float> &prop, int nu , int c2){
  Float *pointer_src = NULL;
  Float *pointer_dst = NULL;
  
  for(int mu = 0 ; mu < 4 ; mu++)
    for(int c1 = 0 ; c1 < 3 ; c1++){
      pointer_dst = (PLEGMA_Field<Float>::d_elem + 
		     mu*3*GK_localVolume*2 + 
		     c1*GK_localVolume*2);
      pointer_src = (prop.D_elem() + 
		     mu*4*3*3*GK_localVolume*2 + 
		     nu*3*3*GK_localVolume*2 + 
		     c1*3*GK_localVolume*2 + 
		     c2*GK_localVolume*2);
      cudaMemcpy(pointer_dst, pointer_src, GK_localVolume*2 * sizeof(Float), cudaMemcpyDeviceToDevice);
    }
  
  pointer_src = NULL;
  pointer_dst = NULL;
  checkCudaError();

}

template<typename Float>
void PLEGMA_Vector<Float>::pointSource(int *sourceposition, int spin, int color, ALLOCATION_FLAG where){
  
  this->zero_where(where);
  int my_src[N_DIMS];
  size_t id=0;
  Float temp[1];
  temp[0] = 1.0;

  for(int i = N_DIMS-1; i >= 0; i--) {
    my_src[i] = (sourceposition[i] - comm_coords(default_topo)[i] * GK_localL[i]);

    // if out of the local lattice we break
    if((my_src[i]<0) || (my_src[i]>=GK_localL[i]))
      return;

    id = id * GK_localL[i] + my_src[i];
  }

  if( where == BOTH ){
    this->h_elem[((spin*N_COLS+color)*GK_localVolume + id)*2] = 1.0; 
    cudaMemcpy((this->d_elem + ((spin*N_COLS+color)*GK_localVolume + id)*2), temp,sizeof(Float),
                cudaMemcpyHostToDevice ); 
  }
  else if (where == HOST){
    this->h_elem[((spin*N_COLS+color)*GK_localVolume + id)*2] = 1.0; 
  }
  else if (where == DEVICE){
    cudaMemcpy((this->d_elem + ((spin*N_COLS+color)*GK_localVolume + id)*2), temp,sizeof(Float),
                cudaMemcpyHostToDevice ); 
  }
  else{
    errorQuda("Not supported %d\n",where);
  }
}

template<typename Float>
void PLEGMA_Vector<Float>::pointSource(int *sourceposition, int spin, int color){
  pointSource(sourceposition,spin,color,this->allocation);
}

template<typename Float>
void PLEGMA_Vector<Float>::write(char *filename){
  FILE *fid;
  int error_in_header=0;
  LimeWriter *limewriter;
  LimeRecordHeader *limeheader = NULL;
  int ME_flag=0, MB_flag=0, limeStatus;
  u_int64_t message_length;
  MPI_Offset offset;
  MPI_Datatype subblock;  //MPI-type, 5d subarray  
  MPI_File mpifid;
  MPI_Status status;
  int sizes[5], lsizes[5], starts[5];
  long int i;
  int chunksize,mu,c1;
  char *buffer;
  int x,y,z,t;
  char tmp_string[2048];

  if(comm_rank() == 0){ // master will write the lime header
    fid = fopen(filename,"w");
    if(fid == NULL){
      fprintf(stderr,"Error open file to write propagator in %s \n",__func__);
      comm_abort(-1);
    }
    else{
      limewriter = limeCreateWriter(fid);
      if(limewriter == (LimeWriter*)NULL) {
	fprintf(stderr, "Error in %s. LIME error in file for writing!\n", __func__);
	error_in_header=1;
	comm_abort(-1);
      }
      else
	{
	  sprintf(tmp_string, "DiracFermion_Sink");
	  message_length=(long int) strlen(tmp_string);
	  MB_flag=1; ME_flag=1;
	  limeheader = limeCreateHeader(MB_flag, ME_flag, "propagator-type", message_length);
	  if(limeheader == (LimeRecordHeader*)NULL)
	    {
	      fprintf(stderr, "Error in %s. LIME create header error.\n", __func__);
	      error_in_header=1;
	      comm_abort(-1);
	    }
	  limeStatus = limeWriteRecordHeader(limeheader, limewriter);
	  if(limeStatus < 0 )
	    {
	      fprintf(stderr, "Error in %s. LIME write header %d\n", __func__, limeStatus);
	      error_in_header=1;
	      comm_abort(-1);
	    }
	  limeDestroyHeader(limeheader);
	  limeStatus = limeWriteRecordData(tmp_string, &message_length, limewriter);
	  if(limeStatus < 0 )
	    {
	      fprintf(stderr, "Error in %s. LIME write header error %d\n", __func__, limeStatus);
	      error_in_header=1;
	      comm_abort(-1);
	    }

	  if( typeid(Float) == typeid(double) )
	    sprintf(tmp_string, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<etmcFormat>\n\t<field>diracFermion</field>\n\t<precision>64</precision>\n\t<flavours>1</flavours>\n\t<lx>%d</lx>\n\t<ly>%d</ly>\n\t<lz>%d</lz>\n\t<lt>%d</lt>\n\t<spin>4</spin>\n\t<colour>3</colour>\n</etmcFormat>", GK_totalL[0], GK_totalL[1], GK_totalL[2], GK_totalL[3]);
	  else
	    sprintf(tmp_string, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<etmcFormat>\n\t<field>diracFermion</field>\n\t<precision>32</precision>\n\t<flavours>1</flavours>\n\t<lx>%d</lx>\n\t<ly>%d</ly>\n\t<lz>%d</lz>\n\t<lt>%d</lt>\n\t<spin>4</spin>\n\t<colour>3</colour>\n</etmcFormat>", GK_totalL[0], GK_totalL[1], GK_totalL[2], GK_totalL[3]);

	  message_length=(long int) strlen(tmp_string); 
	  MB_flag=1; ME_flag=1;

	  limeheader = limeCreateHeader(MB_flag, ME_flag, "quda-propagator-format", message_length);
	  if(limeheader == (LimeRecordHeader*)NULL)
	    {
	      fprintf(stderr, "Error in %s. LIME create header error.\n", __func__);
	      error_in_header=1;
	      comm_abort(-1);
	    }
	  limeStatus = limeWriteRecordHeader(limeheader, limewriter);
	  if(limeStatus < 0 )
	    {
	      fprintf(stderr, "Error in %s. LIME write header %d\n", __func__, limeStatus);
	      error_in_header=1;
	      comm_abort(-1);
	    }
	  limeDestroyHeader(limeheader);
	  limeStatus = limeWriteRecordData(tmp_string, &message_length, limewriter);
	  if(limeStatus < 0 )
	    {
	      fprintf(stderr, "Error in %s. LIME write header error %d\n", __func__, limeStatus);
	      error_in_header=1;
	      comm_abort(-1);
	    }
	  
	  message_length = GK_totalVolume*4*3*2*sizeof(Float);
	  MB_flag=1; ME_flag=1;
	  limeheader = limeCreateHeader(MB_flag, ME_flag, "scidac-binary-data", message_length);
	  limeStatus = limeWriteRecordHeader( limeheader, limewriter);
	  if(limeStatus < 0 )
	    {
	      fprintf(stderr, "Error in %s. LIME write header error %d\n", __func__, limeStatus);
	      error_in_header=1;
	    }
	  limeDestroyHeader( limeheader );
	}
      message_length=1;
      limeWriteRecordData(tmp_string, &message_length, limewriter);
      limeDestroyWriter(limewriter);
      offset = ftell(fid)-1;
      fclose(fid);
    }
  }

  MPI_Bcast(&offset,sizeof(MPI_Offset),MPI_BYTE,0,MPI_COMM_WORLD);
  
  sizes[0]=GK_totalL[3];
  sizes[1]=GK_totalL[2];
  sizes[2]=GK_totalL[1];
  sizes[3]=GK_totalL[0];
  sizes[4]=4*3*2;
  lsizes[0]=GK_localL[3];
  lsizes[1]=GK_localL[2];
  lsizes[2]=GK_localL[1];
  lsizes[3]=GK_localL[0];
  lsizes[4]=sizes[4];
  starts[0]=comm_coords(default_topo)[3]*GK_localL[3];
  starts[1]=comm_coords(default_topo)[2]*GK_localL[2];
  starts[2]=comm_coords(default_topo)[1]*GK_localL[1];
  starts[3]=comm_coords(default_topo)[0]*GK_localL[0];
  starts[4]=0;  

  if( typeid(Float) == typeid(double) )
    MPI_Type_create_subarray(5,sizes,lsizes,starts,
			     MPI_ORDER_C,MPI_DOUBLE,&subblock);
  else
    MPI_Type_create_subarray(5,sizes,lsizes,starts,
			     MPI_ORDER_C,MPI_FLOAT,&subblock);

  MPI_Type_commit(&subblock);
  MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY, 
		MPI_INFO_NULL, &mpifid);
  MPI_File_set_view(mpifid, offset, MPI_FLOAT, subblock, 
		    "native", MPI_INFO_NULL);

  chunksize=4*3*2*sizeof(Float);
  buffer = (char*) malloc(chunksize*GK_localVolume);

  if(buffer==NULL)  
    {
      fprintf(stderr,"Error in %s! Out of memory\n", __func__);
      comm_abort(-1);
    }

  i=0;
                        
  for(t=0; t<GK_localL[3];t++)
  for(z=0; z<GK_localL[2];z++)
  for(y=0; y<GK_localL[1];y++)
  for(x=0; x<GK_localL[0];x++)
  for(mu=0; mu<4; mu++)
  for(c1=0; c1<3; c1++) 
    // works only for QUDA_DIRAC_ORDER (color inside spin)
    {
      ((Float *)buffer)[i] = 
	(PLEGMA_Field<Float>::h_elem[t*GK_localL[2]*GK_localL[1]*GK_localL[0]*4*3*2 + 
		    z*GK_localL[1]*GK_localL[0]*4*3*2 + 
		    y*GK_localL[0]*4*3*2 + 
		    x*4*3*2 + mu*3*2 + c1*2 + 0]);
      
      ((Float *)buffer)[i+1] = 
	(PLEGMA_Field<Float>::h_elem[t*GK_localL[2]*GK_localL[1]*GK_localL[0]*4*3*2 + 
		    z*GK_localL[1]*GK_localL[0]*4*3*2 + 
		    y*GK_localL[0]*4*3*2 + 
		    x*4*3*2 + mu*3*2 + c1*2 + 1]);
      i+=2;
    }
  if(!qcd_isBigEndian()){
    if( typeid(Float) == typeid(double) ) 
      qcd_swap_8((double*) buffer,2*4*3*GK_localVolume);
    else qcd_swap_4((float*) buffer,2*4*3*GK_localVolume);
  }
  if( typeid(Float) == typeid(double) )
    MPI_File_write_all(mpifid, buffer, 4*3*2*GK_localVolume, 
		       MPI_DOUBLE, &status);
  else
    MPI_File_write_all(mpifid, buffer, 4*3*2*GK_localVolume, 
		       MPI_FLOAT, &status);

  free(buffer);
  MPI_File_close(&mpifid);
  MPI_Type_free(&subblock);
}


template<typename Float>
void PLEGMA_Vector<Float>::covD(PLEGMA_Vector<Float> &vecIn, PLEGMA_Gauge<Float> &gauge, int dirOr){
  // to increase efficiency the communication of the ghost for the the vector should happen before calling this function
  if(dirOr < 0 || dirOr > 7) errorQuda("Wrong direction is given");
  vectorTex<Float> texVecIn;
  texVecIn.tex= vecIn.createTexObject();
  gaugeTex<Float> texGaugeIn;
  texGaugeIn.tex = gauge.createTexObject();
  covD_k<Float,Float,Float>(this->D_elem(), texVecIn, texGaugeIn, dirOr);
  vecIn.destroyTexObject(texVecIn.tex);
  gauge.destroyTexObject(texGaugeIn.tex);
}

template<typename FloatC, typename FloatA>
void contractNucleonSeqSource(PLEGMA_Vector<FloatC> &vec, genericTex<FloatA> prop1, WHICHPROJECTOR proj, WHICHPARTICLE particle, int timeslice, int c_nu, int c_c2);
template<typename Float>
void PLEGMA_Vector<Float>::seqSourceNucleon(PLEGMA_Propagator3D<Float> &prop, WHICHPROJECTOR proj, WHICHPARTICLE particle, int global_it, int c_nu, int c_c2){
  if(global_it >= GK_totalL[3]) errorQuda("The global time slice you provided exceed the temporal extent\n");
  int my_it = global_it - comm_coords(default_topo)[3] * GK_localL[3];
  bool is_myIt = (my_it >= 0) && ( my_it < GK_localL[3] );
  this->zero_device();
  if(is_myIt){
    genericTex<Float> texProp;
    texProp.tex = prop.createTexObject();
    contractNucleonSeqSource<Float,Float>(*this, texProp, proj, particle, my_it, c_nu, c_c2);
    prop.destroyTexObject(texProp.tex);
  }
  comm_barrier();
}

template<typename FloatC, typename FloatA, typename FloatB>
void contractNucleonSeqSource(PLEGMA_Vector<FloatC> &vec, genericTex<FloatA> prop1, genericTex<FloatB> prop2, WHICHPROJECTOR proj, WHICHPARTICLE particle, int timeslice, int c_nu, int c_c2);
template<typename Float>
void PLEGMA_Vector<Float>::seqSourceNucleon(PLEGMA_Propagator3D<Float> &prop1, PLEGMA_Propagator3D<Float> &prop2, WHICHPROJECTOR proj, WHICHPARTICLE particle, int global_it, int c_nu, int c_c2){
  if(global_it >= GK_totalL[3]) errorQuda("The global time slice you provided exceed the temporal extent\n");
  int my_it = global_it - comm_coords(default_topo)[3] * GK_localL[3];
  bool is_myIt = (my_it >= 0) && ( my_it < GK_localL[3] );
  this->zero_device();
  if(is_myIt){
    genericTex<Float> texProp1;
    texProp1.tex = prop1.createTexObject();
    genericTex<Float> texProp2;
    texProp2.tex = prop2.createTexObject();
    contractNucleonSeqSource<Float,Float,Float>(*this, texProp1,texProp2, proj, particle, my_it, c_nu, c_c2);
    prop1.destroyTexObject(texProp1.tex);
    prop2.destroyTexObject(texProp2.tex);
  }
  comm_barrier();
}

template class PLEGMA_Vector<float>;
template class PLEGMA_Vector<double>;
