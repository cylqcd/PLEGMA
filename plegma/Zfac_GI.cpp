#include <PLEGMA.h>
#include <PLEGMA_utils.h>
#include <cmath>

using namespace plegma;
using namespace quda;
static std::vector<std::string> listOpt = {"verbosity", "load-gauge", "load-gauge-list-filename", "nsrc", "rng-seed"};

int main(int argc, char **argv)
{
  initializeOptions(argc, argv, true, listOpt);//true => includes ops for solver
  //================ Add your options in this between initializeOptions and initializePLEGMA ================//
  std::string Nconf;
  HGC_options->set("Nconf","The configuration number of the lattice",verbosity,Nconf);  
  std::string pathOut = "./";
  HGC_options->set("output-path","Path to the directory to dump G_FF",verbosity,pathOut);
  bool doG_FF = true;
  HGC_options->set("doG_FF", "If we want to compute 2pt correlator for the fermionic traceless off-diagonal EMT, default true", verbosity,doG_FF);
  bool doG_GG = true;
  HGC_options->set("doG_GG", "If we want to compute 2pt correlator for the gluonic traceless off-diagonal EMT, default true", verbosity,doG_GG);
  int nsmearStout = 10;
  int nsmearStep = 5;
  double alphaStout = 0.129;
  HGC_options->set("nsmear-stout-Gprop", "Number of stout smearing step for Gprop",verbosity,nsmearStout);
  HGC_options->set("nsmear-step-Gprop", "Skipping umber of stout smearing step for Gprop",verbosity,nsmearStep);
  HGC_options->set("alpha-stout-Gprop", "Coefficient for the stout smearing for Gprop",verbosity,alphaStout);
  bool isS4D = false;
  HGC_options->set("Spatial4D", "True if we apply 4D Stout Smearing only to spatial links", verbosity, isS4D);
  bool is3D = false;
  HGC_options->set("add3D", "True if we need G_GG with 3D Stout Smearing as well", verbosity, is3D);
  //==========================================================================================================//
  initializePLEGMA();

  if ( doG_GG ) {
    // Assume: for the moment that temporal direction is not divided
    if ( nsmearStout%nsmearStep != 0 )
      PLEGMA_error("nsmear-stout-Gprop must be divisible by nsmear-step-Gprop\n");

    // setup output
    int n_s_dim = (is3D)?2:1;
    std::string sd = (isS4D)?"S4D":"4D";
    std::string outName3 = pathOut + "G_GG_3DStoutSmearing" + std::to_string(nsmearStout) +"by"+std::to_string(nsmearStep)+"with"+std::to_string(alphaStout);
    std::string outName4 = pathOut + "G_GG_"+sd+"StoutSmearing" + std::to_string(nsmearStout) +"by"+std::to_string(nsmearStep)+"with"+std::to_string(alphaStout);
    int rank = 0;
    MPI_Initialized(&rank);
    if(rank) {
       MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    }
    std::ofstream fpt3, fpt4;
    double factor = (double) 6*HGC_totalVolume/2;
    if(rank==0) {
      if(is3D) {
	fpt3.open(outName3, std::ios::trunc);
	fpt3.precision(8);
      }
      fpt4.open(outName4, std::ios::trunc);
      fpt4.precision(8);
    }

    std::vector<std::pair<int,int>> pairs;
    for ( int i=0; i<N_DIMS-1; i++)
      for ( int j=0; j<N_DIMS-1; j++ )
	if ( i<j )
	  pairs.push_back(std::make_pair(i,j));

    PLEGMA_Gauge<double> gauge, gauge1, gauge2;
    PLEGMA_Field<double> trace1(BOTH,SCALAR), trace2(BOTH,SCALAR); // could this be DEVICE?
    PLEGMA_Field3D<double> tr3D(BOTH,SCALAR);
    PLEGMA_FT<double> ft3D(0,3,false,dims[3]); // this performs 3D FT on each time slice
    PLEGMA_Fmunu<double> fmunu;
    PLEGMA_Su3field<double> one3x3; 
    one3x3.setUnit((std::vector<int>) {0,4,8});
    //std::complex<double> T[listGaugeConfs.size()][2][nsmearStout/nsmearStep+1][dims[3]][3];
    std::complex<double> T[n_s_dim][nsmearStout/nsmearStep+1][pairs.size()][HGC_totalL[DIM_T]];

    if(HGC_verbosity > 1) PLEGMA_printf("Will work on %d confs",listGaugeConfs.size());
    for(int iconf=0; iconf < listGaugeConfs.size(); iconf++){
      double t1=MPI_Wtime();
      std::string confStr=splitStrFwd(listGaugeConfs[iconf],'.');
      gauge.readFile(listGaugeConfs[iconf], LIME_FORMAT);
      PLEGMA_printf("Unsmeared Plaquette is: ");
      gauge.calculatePlaq();

      for ( int i_s = 0; i_s < n_s_dim; i_s++ ){
	int s_dim = (is3D && i_s == 0)?3:4;// smearing dimension: assume n_s_dim = 1 or 2
	for(int n=0; n<=nsmearStout;n+=nsmearStep){
	  if(n%(2*nsmearStep) == 0){
	    if(n==0) gauge2.copy(gauge);
	    else gauge2.stoutSmearing(gauge1,nsmearStep,alphaStout,s_dim,isS4D);
	  }
	  else{
	    gauge1.stoutSmearing(gauge2,nsmearStep,alphaStout,s_dim,isS4D);
	  }
	  fmunu.compute_leaves((n%(2*nsmearStep)==0)?gauge2:gauge1);
	  for(int p =0 ; p<pairs.size(); p++){ // for each pair (i,j), compute T_ij
	    int i = pairs[p].first, j = pairs[p].second;
	    double sign;
	    trace2.zero_device();
	    for(int mu =0; mu< N_DIMS;mu++){
	      if((i!=mu) && (j!=mu)){
		if(i<mu &&j<mu){ trace1.TrFmunuSu3FmunuSu3(fmunu, std::make_pair(i,mu), one3x3,fmunu,std::make_pair(j,mu), one3x3); sign=+1.;}
		else if(i<mu &&j>mu) { trace1.TrFmunuSu3FmunuSu3(fmunu, std::make_pair(i,mu), one3x3,fmunu,std::make_pair(mu,j), one3x3); sign=-1.;}
		else if(i>mu &&j<mu) { trace1.TrFmunuSu3FmunuSu3(fmunu, std::make_pair(mu,i), one3x3,fmunu,std::make_pair(j,mu), one3x3); sign=-1.;}
		else{trace1.TrFmunuSu3FmunuSu3(fmunu, std::make_pair(mu,i), one3x3,fmunu,std::make_pair(mu,j), one3x3); sign=+1.;}
		trace2.add(trace1,(std::complex<double>) {sign,0.}); // trace2 += sign*trace1
	      }
	    }
	    ft3D.apply(trace2,FT_GEMV);
	    ft3D.store3DFTs(T[i_s][n/nsmearStep][p], 0 );
	  }
	}
      }

      if ( rank == 0 ){
	int gT = HGC_totalL[DIM_T];
	std::complex<double> G_GG = 0;
	for(int i_s = 0; i_s < n_s_dim; i_s++)
	  for (int n=0; n<=nsmearStout;n+=nsmearStep)
	    for(int t0=0; t0 < gT; t0++)
	      for(int ts=0; ts < gT; ts++) {
		G_GG = 0;
		for ( int p=0; p<pairs.size(); p++ ) G_GG += T[i_s][n/nsmearStep][p][(t0+ts)%gT]*T[i_s][n/nsmearStep][p][t0];
		if(is3D && i_s==0)
		  fpt3 << iconf << " " << 3 << " " << n << " " << t0 << " " << ts << " " << std::scientific << G_GG.real()/factor << " " << G_GG.imag()/factor << std::endl;
		else
		  fpt4 << iconf << " " << 4 << " " << n << " " << t0 << " " << ts << " " << std::scientific << G_GG.real()/factor << " " << G_GG.imag()/factor << std::endl;
	      }
      }
	
      double t2=MPI_Wtime();
      PLEGMA_printf("conf.%s completed in %f secs\n",confStr.c_str(),t2-t1);
    }
    /*
    //write to a file
    std::string outName2 = pathOut + "G_GG_3D4DStoutSmearing" + std::to_string(nsmearStout)+"by"+std::to_string(nsmearStep)+"with"+std::to_string(alphaStout)+"Nconfs"+std::to_string(listGaugeConfs.size())+"ASCII";
   
    int rank = 0;
    MPI_Initialized(&rank);
    if(rank) {
      // sum up
      MPI_Comm_rank(MPI_COMM_WORLD,&rank);
      }
    if(rank==0) {
      std::ofstream fpt2(outName2);
      //double factor = (double) 2*6*HGC_totalVolume/2;
      fpt2.precision(8);
      for(int s_dim=0; s_dim<2; s_dim++ )
	for (int n=0; n<=nsmearStout;n+=nsmearStep)
	  for(int iconf=0; iconf < listGaugeConfs.size(); iconf++){
	    std::complex<double> T0[dims[3]][pairs.size()];
	    std::string confStr=splitStrFwd(listGaugeConfs[iconf],'.');
	    for ( int p=0; p<3; p++ ) {
	      std::ifstream infile(pathOut+"tmp/"+"G_GG_conf"+confStr+"dim"+std::to_string(3+s_dim)+"stout"+std::to_string(n)+"pair"+std::to_string(p));
	      std::string line;
	      for(int t=0; t < dims[3]; t++){
		double tmp, a, b;
		std::getline(infile, line);
		std::istringstream iss(line);
		for(int k=0;k<5;k++) iss >> tmp;
		iss >> a >> b;
		T0[t][p] = std::complex<double>(a,b);
	      }
	      infile.close();
	    }
	    for(int t0=0; t0 < dims[3]; t0++)
	      for(int ts=0; ts < dims[3]; ts++) {
		std::complex<double> G_GG = 0;
		//for ( int p=0; p<3; p++ ) G_GG += T[iconf][s_dim][n/nsmearStep][t0+ts][p]*T[iconf][s_dim][n/nsmearStep][t0][p];
		for ( int p=0; p<pairs.size(); p++ ) G_GG += T0[(t0+ts)%dims[3]][p]*T0[t0][p];
		fpt2 << s_dim+3 << " " << n << " " << iconf << " " << t0 << " " << ts << " " << std::scientific << G_GG.real()/factor << " " << G_GG.imag()/factor << std::endl;
	      }
	  }
      fpt2.close();
    }*/
    if(rank==0) {
      fpt4.close();
      if (is3D) fpt3.close();
    }
  }

  if ( doG_FF ) {
    // TODO: may not work when time dim of lattice is divided over processes
    PLEGMA_Gauge<double> gauge;
    gauge.readFile(latfile, LIME_FORMAT);
    gauge.calculatePlaq();
    initGaugeQuda(gauge, true);
    plaqQuda();
    
    QUDA_solver solver(mu);
    // We use 3D vector for time-dilution
    PLEGMA_Vector3D<double> Vstc, psi, phi;//alloc flags. etc...?
    PLEGMA_Vector<double> VtIn, VtOut[5], Vtmp, Vtmp1, Vtmp2;
    PLEGMA_Su3field<double> Umu(DEVICE), Umu_d(DEVICE), Unu_d(DEVICE);
    Vstc.randInit(rng_seed);

    //std::complex<double> G_FF[dims[3]];
    //for(int t=0; t < dims[3]; t++) G_FF[t]=0;
    double G_FF[dims[3]][dims[3]][numSourcePositions];
    for(int isc=0; isc < numSourcePositions; isc++) for(int t=0; t < dims[3]*dims[3]; t++) G_FF[t/dims[3]][t%dims[3]][isc]=0;
    
    std::string outName = pathOut + "G_FF" + Nconf + "_" + std::to_string(numSourcePositions);
    for(int isc=0; isc < numSourcePositions; isc++){
      Vstc.stochastic_Z(4);
      for(int t=0; t < dims[3]; t++){
	// inversion with xi w/&w/t gamma_5
	VtIn.absorb(Vstc,t);
	solver.solve(VtOut[0],VtIn);
	VtIn.apply_gamma5();
	solver.solve(VtOut[1],VtIn);
	for(int mu=0; mu<3; mu++ ){
	  // inversion with shifted xi w/&w/t gamma_5
	  Vtmp.shift(VtIn,mu);// check the convention!
	  solver.solve(VtOut[3],Vtmp);
	  Vtmp.apply_gamma5();
	  solver.solve(VtOut[2],Vtmp);
	    
	  Umu.absorbDir_device(gauge,mu);
	  Umu_d.copy(Umu,DEVICE);
	  Umu_d.Udag();
	  for(int nu=0; nu<3; nu++){
	    if ( mu != nu ){
	      // We only need U_\nu^\dagger
	      Unu_d.absorbDir_device(gauge,nu);
	      Unu_d.Udag();
	      
	      // inversion with mu&nu dependent rhs
	      Vtmp.zero_device();
	      Vtmp.absorb(Vstc,t);
	      VtIn.mulGV(Vtmp,Umu);//use sep. field!!!
	      VtIn.apply_gamma(static_cast<GAMMAS>(nu+1),LEFT);
	      VtIn.apply_gamma5();
	      solver.solve(VtOut[4],VtIn);

	      // (1) 
	      // compute psi
	      Vtmp1.shift(VtOut[4],N_DIMS+nu);
	      // compute phi
	      Vtmp2.mulGV(VtOut[2],Unu_d);
	      Vtmp2.apply_gamma(static_cast<GAMMAS>(mu+1),LEFT);
	      Vtmp2.apply_gamma5();
	      for(int ts=0; ts < dims[3]; ts++){//time separation
		psi.absorb(Vtmp1,(t+ts)%dims[3]);
		phi.absorb(Vtmp2,(t+ts)%dims[3]);
		G_FF[t][ts][isc] -= 2*(psi.dot(phi)).real();//complex conjugated?//do not partition time; could brak donw.
		//PLEGMA_printf("%g %g\n",G_FF[t][ts].real(),G_FF[t][ts].imag());
	      }
	      
	      // 2nd term
	      // compute psi
	      Vtmp1.shift(VtOut[4],N_DIMS+mu);
	      // compute phi
	      Umu.Udag();
	      Vtmp2.mulGV(VtOut[2],Umu);
	      Vtmp2.apply_gamma(static_cast<GAMMAS>(nu+1),LEFT);
	      Vtmp2.apply_gamma5();
	      Umu.Udag();
	      for(int ts=0; ts < dims[3]; ts++){//time separation
		psi.absorb(Vtmp1,(t+ts)%dims[3]);
		phi.absorb(Vtmp2,(t+ts)%dims[3]);
		G_FF[t][ts][isc] -= 4*(psi.dot(phi)).real();
	      }
		
	      // (3)
	      // compute psi
	      Vtmp.mulGV(VtOut[4],Unu_d);
	      Vtmp.apply_gamma5();
	      Vtmp.apply_gamma(static_cast<GAMMAS>(mu+1),LEFT);
	      Vtmp1.shift(Vtmp,nu);
	      for(int ts=0; ts < dims[3]; ts++){//time separation
		psi.absorb(Vtmp1,(t+ts)%dims[3]);
		phi.absorb(VtOut[2],(t+ts)%dims[3]);
		G_FF[t][ts][isc] += 2*(psi.dot(phi)).real();
	      }

	      // 2nd term
	      // compute psi
	      Umu.Udag();
	      Vtmp.mulGV(VtOut[4],Umu);
	      Vtmp.apply_gamma5();
	      Vtmp.apply_gamma(static_cast<GAMMAS>(nu+1),LEFT);
	      Vtmp1.shift(Vtmp,mu);
	      Umu.Udag();
	      for(int ts=0; ts < dims[3]; ts++){//time separation 
		psi.absorb(Vtmp1,(t+ts)%dims[3]);
                phi.absorb(VtOut[2],(t+ts)%dims[3]);
                G_FF[t][ts][isc] += 2*(psi.dot(phi)).real();
              }
		
	      // inversion with mu&nu dependent rhs
	      VtIn.zero_device();
	      VtIn.absorb(Vstc,t);
	      Vtmp.shift(VtIn,N_DIMS+mu);
	      VtIn.mulGV(Vtmp,Umu);
	      VtIn.apply_gamma(static_cast<GAMMAS>(nu+1),LEFT);
	      solver.solve(VtOut[4],VtIn);

	      // (2)
	      // compute psi
	      Vtmp1.shift(VtOut[1],N_DIMS+nu);
	      // compute phi
	      Vtmp2.mulGV(VtOut[4],Unu_d);
	      Vtmp2.apply_gamma(static_cast<GAMMAS>(mu+1),LEFT);
	      Vtmp2.apply_gamma5();
	      for(int ts=0; ts < dims[3]; ts++){//time separation
		psi.absorb(Vtmp1,(t+ts)%dims[3]);
		phi.absorb(Vtmp2,(t+ts)%dims[3]);
                G_FF[t][ts][isc] += 2*(psi.dot(phi)).real();
              }
	      
	      // 2nd term
	      // compute psi
	      Vtmp1.shift(VtOut[1],N_DIMS+mu);
	      // compute phi
	      Umu.Udag();
	      Vtmp2.mulGV(VtOut[4],Umu);
	      Vtmp2.apply_gamma(static_cast<GAMMAS>(nu+1),LEFT);
	      Vtmp2.apply_gamma5();
	      Umu.Udag();
	      for(int ts=0; ts < dims[3]; ts++){//time separation 
		psi.absorb(Vtmp1,(t+ts)%dims[3]);
                phi.absorb(Vtmp2,(t+ts)%dims[3]);
                G_FF[t][ts][isc] += 2*(psi.dot(phi)).real();
              }
		
	      // (4)
	      // compute psi
	      Vtmp.mulGV(VtOut[1],Unu_d);
	      Vtmp.apply_gamma5();
              Vtmp.apply_gamma(static_cast<GAMMAS>(mu+1),LEFT);
	      Vtmp1.shift(Vtmp,nu);
	      for(int ts=0; ts < dims[3]; ts++){//time separation  
		psi.absorb(Vtmp1,(t+ts)%dims[3]);
                phi.absorb(VtOut[4],(t+ts)%dims[3]);
		G_FF[t][ts][isc] -= 2*(psi.dot(phi)).real();
	      }
	      
	      // inversion with mu&nu dependent rhs
	      Vtmp.zero_device();
	      Vtmp.absorb(Vstc,t);
	      VtIn.mulGV(Vtmp,Umu);
	      VtIn.apply_gamma(static_cast<GAMMAS>(nu+1),LEFT);
	      solver.solve(VtOut[4],VtIn);
		
	      // (5)
	      // compute psi
	      Vtmp1.shift(VtOut[3],N_DIMS+nu);
	      // compute phi 
	      Vtmp2.mulGV(VtOut[4],Unu_d);
	      Vtmp2.apply_gamma(static_cast<GAMMAS>(mu+1),LEFT);
	      Vtmp2.apply_gamma5();
	      for(int ts=0; ts < dims[3]; ts++){//time separation
		psi.absorb(Vtmp1,(t+ts)%dims[3]);
                phi.absorb(Vtmp2,(t+ts)%dims[3]);
                G_FF[t][ts][isc] -= 2*(psi.dot(phi)).real();
              }

	      // 2nd term
	      // compute psi
	      Vtmp1.shift(VtOut[3],N_DIMS+mu);
	      // compute phi
	      Umu.Udag();
	      Vtmp2.mulGV(VtOut[4],Umu);
	      Vtmp2.apply_gamma(static_cast<GAMMAS>(nu+1),LEFT);
	      Vtmp.apply_gamma5();
	      Umu.Udag();
	      for(int ts=0; ts < dims[3]; ts++){//time separation
		psi.absorb(Vtmp1,(t+ts)%dims[3]);
                phi.absorb(Vtmp2,(t+ts)%dims[3]);
		G_FF[t][ts][isc] -= 4*(psi.dot(phi)).real();
              }
	      
	      // (7)
	      // compute psi
	      Vtmp.mulGV(VtOut[3],Unu_d);
	      Vtmp.apply_gamma5();
	      Vtmp.apply_gamma(static_cast<GAMMAS>(mu+1),LEFT);
	      Vtmp1.shift(Vtmp,nu);
	      for(int ts=0; ts < dims[3]; ts++){//time separation
		psi.absorb(Vtmp1,(t+ts)%dims[3]);
		phi.absorb(VtOut[4],(t+ts)%dims[3]);
		G_FF[t][ts][isc] += 2*(psi.dot(phi)).real();
              }

	      // 2nd term
	      // compute psi
	      Umu.Udag();
	      Vtmp.mulGV(VtOut[3],Umu);
	      Vtmp.apply_gamma5();
	      Vtmp.apply_gamma(static_cast<GAMMAS>(nu+1),LEFT);
	      Vtmp1.shift(Vtmp,mu);
	      Umu.Udag();
	      // compute phi
	      for(int ts=0; ts < dims[3]; ts++){//time separation
		psi.absorb(Vtmp1,(t+ts)%dims[3]);
                phi.absorb(VtOut[4],(t+ts)%dims[3]);
                G_FF[t][ts][isc] += 2*(psi.dot(phi)).real();
              }
	      
	      // inversion with mu&nu dependent rhs
	      VtIn.zero_device();
	      VtIn.absorb(Vstc,t);
	      Vtmp.shift(VtIn,N_DIMS+mu);
	      VtIn.mulGV(Vtmp,Umu);
	      VtIn.apply_gamma(static_cast<GAMMAS>(nu+1),LEFT);
	      VtIn.apply_gamma5();
	      solver.solve(VtOut[4],VtIn);
		
	      // (6)
	      // compute psi
	      Vtmp1.shift(VtOut[4],N_DIMS+nu);
	      // compute phi
	      Vtmp2.mulGV(VtOut[0],Unu_d);
	      Vtmp2.apply_gamma(static_cast<GAMMAS>(mu+1),LEFT);
	      Vtmp2.apply_gamma5();
	      for(int ts=0; ts < dims[3]; ts++){//time separation
		psi.absorb(Vtmp1,(t+ts)%dims[3]);
		phi.absorb(Vtmp2,(t+ts)%dims[3]);
		G_FF[t][ts][isc] += 2*(psi.dot(phi)).real();
	      }
	      
	      // 2nd term
	      // compute psi
	      Vtmp1.shift(VtOut[4],N_DIMS+mu);
	      // compute phi
	      Umu.Udag();
	      Vtmp2.mulGV(VtOut[0],Umu);
	      Vtmp2.apply_gamma(static_cast<GAMMAS>(nu+1),LEFT);
	      Vtmp2.apply_gamma5();
	      Umu.Udag();
	      for(int ts=0; ts < dims[3]; ts++){//time separation
		psi.absorb(Vtmp1,(t+ts)%dims[3]);
		phi.absorb(Vtmp2,(t+ts)%dims[3]);
                G_FF[t][ts][isc] += 2*(psi.dot(phi)).real();
	      }
	      
	      // (8)
	      // compute psi
	      Vtmp.mulGV(VtOut[4],Unu_d);
	      Vtmp.apply_gamma5();
	      Vtmp.apply_gamma(static_cast<GAMMAS>(mu+1),LEFT);
	      Vtmp1.shift(Vtmp,nu);
	      // compute phi
	      for(int ts=0; ts < dims[3]; ts++){//time separation
		psi.absorb(Vtmp1,(t+ts)%dims[3]);
                phi.absorb(VtOut[0],(t+ts)%dims[3]);
                G_FF[t][ts][isc] -= 2*(psi.dot(phi)).real();
              }
	    }
	  }
	}
      }
    }
    //write to a file
    int rank = 0;
    MPI_Initialized(&rank);
    if(rank) {
      // sum up
      //MPI_Allreduce(MPI_IN_PLACE, G_FF, dims[3], MPI_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);
      MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    }
    if(rank==0) { 
      std::ofstream fpt(outName);
      double factor = (double) 2*6*HGC_totalVolume*numSourcePositions;
      fpt.precision(8);
      for(int isc=0; isc < numSourcePositions; isc++)
	for(int t=0; t < dims[3]; t++)
	  for(int ts=0; ts < dims[3]; ts++) {
	    //std::complex<double> G_FF_t = std::pow((double) ts,5)*G_FF[t][ts]/((double)2*6*HGC_totalVolume*numSourcePositions);
	    fpt << isc<< " " << t << " " << ts << " " << std::scientific << G_FF[t][ts][isc]/factor <<  std::endl;
	  }
    }
  }
  finalize();
  return 0;
}
