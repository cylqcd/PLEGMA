/*
 * PLEGMA_printf, PLEGMA_error, PLEGMA_warning
 * inspired by QUDA.
 */
#pragma once

extern bool HGC_hold_exit; // used to hold exit until all the errors have been printed
extern bool HGC_init_PLEGMA_flag;
extern struct global_vars HGC_global_vars;
extern class Options * HGC_options;

#define PLEGMA_exit(value) do {			\
    if(! HGC_hold_exit) {			\
      if (HGC_init_PLEGMA_flag) {		\
	comm_abort(value);			\
      } else {					\
	int init = 0;				\
	MPI_Initialized(&init);			\
	if(init) {				\
	  comm_abort(value);			\
	}					\
      }						\
      exit(value);				\
    }						\
  } while (0)

#define PLEGMA_printf(...) {						\
    if (HGC_init_PLEGMA_flag) {						\
      sprintf(getPrintBuffer(), __VA_ARGS__);				\
      if (getRankVerbosity()) {						\
	fprintf(getOutputFile(), "%s", getOutputPrefix());		\
	fprintf(getOutputFile(), "%s", getPrintBuffer());		\
	fflush(getOutputFile());					\
      }									\
    } else {								\
      int rank = 0;							\
      MPI_Initialized(&rank);						\
      if(rank) {							\
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);				\
	if(rank==0) {							\
	  printf(__VA_ARGS__);						\
	}								\
      } else {								\
	printf(__VA_ARGS__);						\
      }									\
    }									\
  }

#define PLEGMA_error(...) do {						\
    HGC_hold_exit = true;						\
    HGC_options->checkErrors();						\
    if (HGC_init_PLEGMA_flag) {						\
      fprintf(getOutputFile(), "%sERROR: ", getOutputPrefix());		\
      fprintf(getOutputFile(), __VA_ARGS__);				\
      fprintf(getOutputFile(), " (rank %d, host %s, " __FILE__ ":%d in %s())\n", \
	      comm_rank(), comm_hostname(), __LINE__, __func__);	\
      fprintf(getOutputFile(), "%s       last kernel called was (name=%s,volume=%s,aux=%s)\n", \
	      getOutputPrefix(), getLastTuneKey().name,			\
	      getLastTuneKey().volume, getLastTuneKey().aux);		\
      HGC_global_vars.print();						\
      fflush(getOutputFile());						\
      quda::saveTuneCache(true);					\
    } else {								\
      int rank = 0;							\
      MPI_Initialized(&rank);						\
      if(rank) {							\
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);				\
	printf( "ERROR: ", rank);					\
	printf(__VA_ARGS__);						\
	printf(" (rank %d, " __FILE__ ":%d in %s())\n",			\
	       rank, __LINE__, __func__);				\
      } else {								\
	printf( "ERROR: ");						\
	printf(__VA_ARGS__);						\
	printf(" (" __FILE__ ":%d in %s())\n",				\
	       __LINE__, __func__);					\
      }									\
    }									\
    HGC_hold_exit = false;						\
    PLEGMA_exit(1);							\
  } while(0)

#define PLEGMA_warning(...) do {					\
    if (HGC_init_PLEGMA_flag) {						\
      if (getVerbosity() > QUDA_SILENT) {				\
	sprintf(getPrintBuffer(), __VA_ARGS__);				\
	if (getRankVerbosity()) {					\
	  fprintf(getOutputFile(), "%sWARNING: ", getOutputPrefix());	\
	  fprintf(getOutputFile(), "%s", getPrintBuffer());		\
	  fprintf(getOutputFile(), "\n");				\
	  fflush(getOutputFile());					\
	}								\
      }									\
    } else {								\
      int rank = 0;							\
      MPI_Initialized(&rank);						\
      if(rank) {							\
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);				\
	if(rank==0) {							\
	  printf("WARNING: ");						\
	  printf(__VA_ARGS__);						\
	}								\
      } else {								\
	printf("WARNING: ");						\
	printf(__VA_ARGS__);						\
      }									\
    }									\
  } while (0)


