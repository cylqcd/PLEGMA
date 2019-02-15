/*
 * PLEGMA_printf, PLEGMA_error, PLEGMA_warning
 * inspired by QUDA.
 */
#pragma once

extern bool HGC_init_PLEGMA_flag;

#define PLEGMA_printf(...) do {					\
    if (HGC_init_PLEGMA_flag) {					\
      sprintf(getPrintBuffer(), __VA_ARGS__);			\
      if (getRankVerbosity()) {					\
	fprintf(getOutputFile(), "%s", getOutputPrefix());	\
	fprintf(getOutputFile(), "%s", getPrintBuffer());	\
	fflush(getOutputFile());				\
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
  } while (0)

#define PLEGMA_error(...) do {						\
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
      comm_abort(1);							\
    } else {								\
      int rank = 0;							\
      MPI_Initialized(&rank);						\
      if(rank) {							\
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);				\
	printf( "ERROR: ", rank);					\
	printf(__VA_ARGS__);						\
	printf(" (rank %d, " __FILE__ ":%d in %s())\n",			\
	       rank, __LINE__, __func__);				\
	comm_abort(1);							\
      } else {								\
	printf( "ERROR: ");						\
	printf(__VA_ARGS__);						\
	printf(" (" __FILE__ ":%d in %s())\n",				\
	       __LINE__, __func__);					\
	exit(1);							\
      }									\
    }									\
  } while (0)

#define PLEGMA_warning(...) do {					\
    if (HGC_init_PLEGMA_flag) {						\
    if (getVerbosity() > QUDA_SILENT) {					\
      sprintf(getPrintBuffer(), __VA_ARGS__);				\
      if (getRankVerbosity()) {						\
	fprintf(getOutputFile(), "%sWARNING: ", getOutputPrefix());	\
	fprintf(getOutputFile(), "%s", getPrintBuffer());		\
	fprintf(getOutputFile(), "\n");					\
	fflush(getOutputFile());					\
      }									\
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


