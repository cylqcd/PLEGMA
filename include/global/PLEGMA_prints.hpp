/*
 * PLEGMA_printf, PLEGMA_error, PLEGMA_warning
 * inspired by QUDA.
 */
#pragma once

#ifdef MULTI_GPU

#define PLEGMA_printf(...) do {					\
    sprintf(getPrintBuffer(), __VA_ARGS__);			\
    if (getRankVerbosity()) {					\
      fprintf(getOutputFile(), "%s", getOutputPrefix());	\
      fprintf(getOutputFile(), "%s", getPrintBuffer());		\
      fflush(getOutputFile());					\
    }								\
  } while (0)

#define PLEGMA_error(...) do {						\
    fprintf(getOutputFile(), "%sERROR: ", getOutputPrefix());		\
    fprintf(getOutputFile(), __VA_ARGS__);				\
    fprintf(getOutputFile(), " (rank %d, host %s, " __FILE__ ":%d in %s())\n", \
	    comm_rank(), comm_hostname(), __LINE__, __func__);		\
    fprintf(getOutputFile(), "%s       last kernel called was (name=%s,volume=%s,aux=%s)\n", \
	    getOutputPrefix(), getLastTuneKey().name,			\
	    getLastTuneKey().volume, getLastTuneKey().aux);		\
    HGC_global_vars.print();						\
    fflush(getOutputFile());						\
    quda::saveTuneCache(true);						\
    comm_abort(1);							\
  } while (0)

#define PLEGMA_warning(...) do {					\
    if (getVerbosity() > QUDA_SILENT) {					\
      sprintf(getPrintBuffer(), __VA_ARGS__);				\
      if (getRankVerbosity()) {						\
	fprintf(getOutputFile(), "%sWARNING: ", getOutputPrefix());	\
	fprintf(getOutputFile(), "%s", getPrintBuffer());		\
	fprintf(getOutputFile(), "\n");					\
	fflush(getOutputFile());					\
      }									\
    }									\
  } while (0)

#else

#define PLEGMA_printf(...) do {				\
    fprintf(getOutputFile(), "%s", getOutputPrefix());	\
    fprintf(getOutputFile(), __VA_ARGS__);		\
    fflush(getOutputFile());				\
  } while (0)

#define PLEGMA_error(...) do {						\
    fprintf(getOutputFile(), "%sERROR: ", getOutputPrefix());		\
    fprintf(getOutputFile(), __VA_ARGS__);				\
    fprintf(getOutputFile(), " (" __FILE__ ":%d in %s())\n",		\
	    __LINE__, __func__);					\
    fprintf(getOutputFile(), "%s       last kernel called was (name=%s,volume=%s,aux=%s)\n", \
	    getOutputPrefix(), getLastTuneKey().name,			\
	    getLastTuneKey().volume, getLastTuneKey().aux);		\
    HGC_global_vars.print();						\
    quda::saveTuneCache(true);						\
    comm_abort(1);							\
  } while (0)

#define PLEGMA_warning(...) do {					\
    if (getVerbosity() > QUDA_SILENT) {					\
      fprintf(getOutputFile(), "%sWARNING: ", getOutputPrefix());	\
      fprintf(getOutputFile(), __VA_ARGS__);				\
      fprintf(getOutputFile(), "\n");					\
      fflush(getOutputFile());						\
    }									\
  } while (0)

#endif // MULTI_GPU
