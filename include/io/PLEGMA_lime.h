#pragma once
extern "C" {
#include <lime.h>
}

#include <PLEGMA_global.h>
#include <unistd.h>
#include <stdio.h>

inline bool exists_file (const char* name) {
  return ( access( name, F_OK ) != -1 );
}

static void swap_8(double *Rd, int N)
{
   register char *i,*j,*k;
   char swap;
   char *max;
   char *R = (char*) Rd;

   max = R+(N<<3);
   for(i=R;i<max;i+=8)
   {
      j=i; k=j+7;
      swap = *j; *j = *k;  *k = swap;
      j++; k--;
      swap = *j; *j = *k;  *k = swap;
      j++; k--;
      swap = *j; *j = *k;  *k = swap;
      j++; k--;
      swap = *j; *j = *k;  *k = swap;
   }
}

static void swap_4(float *Rd, int N)
{
  register char *i,*j,*k;
  char swap;
  char *max;
  char *R =(char*) Rd;

  max = R+(N<<2);
  for(i=R;i<max;i+=4)
  {
    j=i; k=j+3;
    swap = *j; *j = *k;  *k = swap;
    j++; k--;
    swap = *j; *j = *k;  *k = swap;
  }
}

static int isBigEndian()
{
   union{
     char C[4];
     int  R   ;
        }word;
   word.R=1;
   if(word.C[3]==1) return 1;
   if(word.C[0]==1) return 0;

   return -1;
}

static void print_xlf_info(LimeReader *limereader) {
  n_uint64_t lime_data_size = limeReaderBytes(limereader);
  char * lime_data;
  hostMalloc(lime_data, lime_data_size+1);
  limeReaderReadData((void *)lime_data, &lime_data_size, limereader);
  lime_data[lime_data_size]='\0';
  std::string css = lime_data;
  if(css.empty()) return;
  std::stringstream ss(css);
  if(HGC_verbosity>1) PLEGMA_printf("Begin LIME header,...\n");
  while(ss.good()){
    std::string substr;
    getline( ss, substr, ',' );
    if(HGC_verbosity>1) PLEGMA_printf("%s\n",substr.c_str());
  }
  if(HGC_verbosity>1) PLEGMA_printf("End LIME header.\n");
  
  hostFree(lime_data, lime_data_size+1);
}

template<typename T>
bool getValueFromXML(std::string str, std::string toMatch, T &value){
  std::stringstream ss(str);
  std::string prototype = "<" + toMatch + "></" + toMatch + ">";
  while(ss.good()){
    std::string line;
    getline(ss,line,'\n');
    if(HGC_verbosity > 2) PLEGMA_printf("Extract line: %s\n",line.c_str());
    if(line.empty()) PLEGMA_error("Problem with extracting from LIME XML\n Message:\n %s",str.c_str());
    line = line.substr(line.find_first_not_of(" \t"),line.find_last_not_of(" \t")-line.find_first_not_of(" \t")+1);
    if(line.find(toMatch) != std::string::npos){
      int left = line.find_first_of(">");
      int right = line.find_last_of("<");
      if(right <= left) PLEGMA_error("Cannot understand xml format in string %s\n",line.c_str());
      std::string sval = line.substr(left+1,right-left-1);
      if(sval.empty()) return false;
      line.erase(left+1,right-left-1);
      if(line != prototype) PLEGMA_error("Format %s does not match expected format %s\n",line.c_str(),prototype.c_str());
      std::stringstream ss(sval);
      ss >> value;
      if(ss.fail()) PLEGMA_warning("Cannot extract value from [%s], return with failure\n",toMatch.c_str());
      return true;
    }
  }
  return false;
}

static void get_ildg_info(LimeReader *limereader, int &prec, int &dof){
  n_uint64_t lime_data_size = limeReaderBytes(limereader);
  char * lime_data;
  hostMalloc(lime_data, lime_data_size+1);
  int status = limeReaderReadData((void *)lime_data, &lime_data_size, limereader);
  if( status < 0 && status != LIME_EOR ) PLEGMA_error("Lime read error occured: status =%d",status);
  lime_data[lime_data_size]='\0';
  std::string string_lime_data = lime_data;
  bool passCheck;
  
  passCheck = getValueFromXML(string_lime_data,"precision",prec);
  if(!passCheck) PLEGMA_error("LIME: Cannot extract precision (32/64) from the ildg format");
  prec /= 8;

  int ll=0;
  std::vector<std::string> xyzt = {"x","y","z","t"};
  for(int i = 0; i < N_DIMS; i++){
    passCheck = getValueFromXML(string_lime_data, "l"+xyzt[i], ll);
    if(!passCheck) PLEGMA_error("LIME: Cannot extract Lattice extend for %s-direction from the ildg format",xyzt[i].c_str());
    if(ll != HGC_totalL[i]) PLEGMA_error("LIME: Read l%s different from HGC_totalL[%d], (%d != %d)\n",xyzt[i].c_str(),i,ll,HGC_totalL[i]);
  }
  passCheck = getValueFromXML(string_lime_data,"dof",dof);
  if(!passCheck){
    dof=-1;
    PLEGMA_warning("LIME: Cannot read d.o.f of field from ildg header. Switch to use the one provided");
  }
  hostFree(lime_data, lime_data_size+1);
}

static void read_lime_header(LimeReader *limereader, int &prec, int &dof){
  while(limeReaderNextRecord(limereader) != LIME_EOF ) {
    char* lime_type = limeReaderType(limereader); 
    if(strcmp(lime_type,"ildg-binary-data") == 0)
      break;
    if(strcmp(lime_type,"xlf-info")==0)
      print_xlf_info(limereader);
    if(strcmp(lime_type,"ildg-format")==0)
      get_ildg_info(limereader,prec,dof);
  }
  if((prec != 4) && (prec != 8)) PLEGMA_error("Precision %d is not supported",prec);
}

static void write_lime_header_type(LimeWriter *limewriter,std::string &headerString, size_t messageLength
				   , int MB_flag = 1, int ME_flag = 1){
  LimeRecordHeader *limeheader = NULL;
  int limeStatus;
  if(headerString.length() == 0) PLEGMA_error("Header is empty, check");
  limeheader = limeCreateHeader(MB_flag, ME_flag,(char*)headerString.c_str(), messageLength);
  if(limeheader == (LimeRecordHeader *) NULL) PLEGMA_error("LIME: Cannot create header");
  limeStatus = limeWriteRecordHeader(limeheader,limewriter);
  if(limeStatus < 0) PLEGMA_error("LIME: Cannot write header record error is = %d\n",limeStatus);
  limeDestroyHeader(limeheader);
}

static void write_lime_header_message(LimeWriter *limewriter, std::string message){
  int limeStatus;
  size_t messageL = message.length();
  limeStatus = limeWriteRecordData((char*)message.c_str(), &(messageL), limewriter);
  if(limeStatus < 0) PLEGMA_error("LIME: Cannot write message record");
}

static void write_lime_header(LimeWriter *limewriter,std::string headerString, std::string message, int MB_flag = 1, int ME_flag = 1){
  write_lime_header_type(limewriter,headerString,message.length(),MB_flag,ME_flag);
  write_lime_header_message(limewriter,message);
}

static std::string lime_version_header(){
  return "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<ildgFormat xmlns=\"http://www.lqcd.org/ildg\"\n            xmlns:xsi=\"http\
://www.w3.org/2001/XMLSchema-instance\"\n            xsi:schemaLocation=\"http://www.lqcd.org/ildg filefmt.xsd\">\n  <version> 1.0 </version>\n";
}

template<typename Float>
static void write_binary_to_lime(std::string filename, FILE *fid, LimeWriter *limewriter, Float *data, int dof){
  std::string header = "ildg-binary-data";
  std::string msg_tmp = "X";
  MPI_Offset offset;
  if(comm_rank() == 0){
    write_lime_header_type(limewriter,header,HGC_totalVolume*dof*2*sizeof(Float),0,0); // make one fake record write to set the offset
    write_lime_header_message(limewriter,msg_tmp);
    offset = ftell(fid)-1;
    fclose(fid);
  }
  comm_barrier();
  int mpiErr = MPI_Bcast(&offset,sizeof(MPI_Offset),MPI_BYTE,0,MPI_COMM_WORLD);
  if(mpiErr != MPI_SUCCESS) PLEGMA_error("MPI_Bcast failed with error %d\n", mpiErr);

  Float *ftmp;
  long int sizeVec=((long int) dof)*HGC_localVolume;
  hostMalloc(ftmp, sizeVec*2*sizeof(Float));

  MPI_Datatype subblock;  //MPI-type, (N_DIMS+1)d subarray
  MPI_File mpifid;
  MPI_Status status;
  int sizes[N_DIMS+1], lsizes[N_DIMS+1], starts[N_DIMS+1];
  for(int i=0; i<N_DIMS; i++) {
    sizes[i] = HGC_totalL[N_DIMS-1-i];
    lsizes[i] = HGC_localL[N_DIMS-1-i];
    starts[i] = HGC_procPosition[N_DIMS-1-i]*HGC_localL[N_DIMS-1-i];
  }
  lsizes[N_DIMS] = sizes[N_DIMS] = dof*2;
  starts[N_DIMS] = 0;

  MPI_Type_create_subarray(N_DIMS+1,sizes,lsizes,starts,MPI_ORDER_C,MPI_Type(data),&subblock);
  MPI_Type_commit(&subblock);
	
  MPI_File_open(MPI_COMM_WORLD, filename.c_str(), MPI_MODE_WRONLY, MPI_INFO_NULL, &mpifid);
  MPI_File_set_view(mpifid, offset, MPI_Type(data), subblock, "native", MPI_INFO_NULL);

  for(size_t i = 0; i < HGC_localVolume; i++) {
    for(int s = 0; s < dof; s++) {
      ftmp[((long int) i)*dof*2 +s*2+0] = data[(s*HGC_localVolume + i)*2 + 0];
      ftmp[((long int) i)*dof*2 +s*2+1] = data[(s*HGC_localVolume + i)*2 + 1];
    }
  }

  if(!isBigEndian()){
    if(sizeof(Float) == 8)swap_8((double*)ftmp,sizeVec*2);
    else if(sizeof(Float) == 4) swap_4((float*)ftmp,sizeVec*2);
    else PLEGMA_error("Cannot byte swap with this precision");
  }
  mpiErr = MPI_File_write_all(mpifid,ftmp,sizeVec*2,MPI_Type(data),&status);
  if(mpiErr != MPI_SUCCESS) PLEGMA_error("MPI_File_write_all failed with error %d\n", mpiErr);
  
  hostFree(ftmp, sizeVec*2*sizeof(Float));
  MPI_File_close(&mpifid);
  MPI_Type_free(&subblock);
}

template<typename Float>
static void read_binary_from_lime(std::string filename, FILE *fid, LimeReader *limereader, Float *data, int dof){
#ifdef	MULTI_GPU
  MPI_Offset offset;
  // Read 1 byte to set file-pointer to start of binary data
  if(comm_rank() == 0){
    n_uint64_t one=1;
    char dummy;
    limeReaderReadData(&dummy,&one,limereader);
    offset = ftell(fid)-1;
  }
  comm_broadcast(&offset,sizeof(MPI_Offset));
#endif

  Float *ftmp;
  long int sizeVec=((long int) dof)*HGC_localVolume;
  hostMalloc(ftmp, sizeVec*2*sizeof(Float));

#ifdef	MULTI_GPU
  MPI_Datatype subblock;  //MPI-type, (N_DIMS+1)d subarray
  MPI_File mpifid;
  MPI_Status status;
  int sizes[N_DIMS+1], lsizes[N_DIMS+1], starts[N_DIMS+1];
  for(int i=0; i<N_DIMS; i++) {
    sizes[i] = HGC_totalL[N_DIMS-1-i];
    lsizes[i] = HGC_localL[N_DIMS-1-i];
    starts[i] = HGC_procPosition[N_DIMS-1-i]*HGC_localL[N_DIMS-1-i];
  }
  lsizes[N_DIMS] = sizes[N_DIMS] = dof*2;
  starts[N_DIMS] = 0;

  MPI_Type_create_subarray(N_DIMS+1,sizes,lsizes,starts,MPI_ORDER_C,MPI_Type(data),&subblock);
  MPI_Type_commit(&subblock);
	
  MPI_File_open(MPI_COMM_WORLD, filename.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &mpifid);
  MPI_File_set_view(mpifid, offset, MPI_Type(data), subblock, "native", MPI_INFO_NULL);

  if(sizeVec*2 > 2147483648) PLEGMA_warning("Be careful for possible integer overflow in MPI_File_read_all function");
      
  if(MPI_File_read_all(mpifid, ftmp, sizeVec*2, MPI_Type(data), &status) == 1)
    PLEGMA_error("Error in MPI_File_read_all\n");
#else
  if(fread(ftmp, sizeof(Float), sizeVec*2, fid) != sizeVec*2) {
    PLEGMA_error("Error, could not read proper amount of data");
  }
#endif
  if(!isBigEndian()){
    if(sizeof(Float) == 8)swap_8((double*)ftmp,sizeVec*2);
    else if(sizeof(Float) == 4) swap_4((float*)ftmp,sizeVec*2);
    else PLEGMA_error("Cannot byte swap with this precision");
  }

  for(size_t i = 0; i < HGC_localVolume; i++) {
    for(int s = 0; s < dof; s++) {
      data[(s*HGC_localVolume + i)*2 + 0] = ftmp[((long int) i)*dof*2 +s*2+0];
      data[(s*HGC_localVolume + i)*2 + 1] = ftmp[((long int) i)*dof*2 +s*2+1];
    }
  }
  hostFree(ftmp, sizeVec*2*sizeof(Float));
  MPI_File_close(&mpifid);
  MPI_Type_free(&subblock);
}
