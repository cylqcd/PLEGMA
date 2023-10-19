/*
 * PLEGMA_printf, PLEGMA_error, PLEGMA_warning
 * inspired by QUDA.
 */
#pragma once
//extern struct global_vars_both HGC;
//extern bool HGC.hold_exit; // used to hold exit until all the errors have been printed
//extern bool HGC.init_PLEGMA_flag;
//extern class Options * HGC.options;
#define PLEGMA_exit(value) do {			\
      exit(value);				\
  } while (0)

#define PLEGMA_printf(...) do {						\
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
  } while(0)

#define PLEGMA_error(...) do {						\
      fprintf(getOutputFile(), "%sERROR: ", getOutputPrefix());		\
      fprintf(getOutputFile(), __VA_ARGS__);				\
      fprintf(getOutputFile(), " (rank %d, host %s, " __FILE__ ":%d in %s())\n", \
             comm_rank(), comm_hostname(), __LINE__, __func__);         \
      fprintf(getOutputFile(), "%s       last kernel called was (name=%s,volume=%s,aux=%s)\n", \
	      getOutputPrefix(), getLastTuneKey().name,			\
	      getLastTuneKey().volume, getLastTuneKey().aux);		\
      fflush(getOutputFile());						\
      quda::saveTuneCache(true);					\
      PLEGMA_exit(1);							\
  } while(0)

#define PLEGMA_warning(...) do {					\
      if (getVerbosity() > QUDA_SILENT) {				\
	sprintf(getPrintBuffer(), __VA_ARGS__);				\
	if (getRankVerbosity()) {					\
	  fprintf(getOutputFile(), "%sWARNING: ", getOutputPrefix());	\
	  fprintf(getOutputFile(), "%s", getPrintBuffer());		\
	  fprintf(getOutputFile(), "\n");				\
	  fflush(getOutputFile());					\
	}								\
      }									\
  } while (0)


