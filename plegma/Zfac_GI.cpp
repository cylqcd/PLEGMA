
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
  int startT = 0;
  int endT   = 0;
  HGC_options->set("startT", "The start time for iteration over source timeslices", verbosity, startT);
  HGC_options->set("endT", "The end time for iteration over source timeslices", verbosity, endT);
  bool doG_GG = true;
  HGC_options->set("doG_GG", "If we want to compute 2pt correlator for the gluonic traceless off-diagonal EMT, default true", verbosity,doG_GG);
  int nsmearStoutStart = 0;
  int nsmearStout = 10;
  int nsmearStep = 5;
  double alphaStout = 0.129;
  HGC_options->set("nsmear-stout-start-Gprop", "The starting number of stout smearing step for Gprop",verbosity,nsmearStoutStart);
  HGC_options->set("nsmear-stout-Gprop", "Number of stout smearing step for Gprop",verbosity,nsmearStout);
  HGC_options->set("nsmear-step-Gprop", "Skipping umber of stout smearing step for Gprop",verbosity,nsmearStep);
  HGC_options->set("alpha-stout-Gprop", "Coefficient for the stout smearing for Gprop",verbosity,alphaStout);
  bool isS4D = false;
  HGC_options->set("Spatial4D", "True if we apply 4D Stout Smearing only to spatial links", verbosity, isS4D);
  bool add3D = false;
  HGC_options->set("add3D", "True if we need G_GG with 3D Stout Smearing as well", verbosity, add3D);
  //==========================================================================================================//
  initializePLEGMA();

  if ( doG_GG ) {
    /***  Here we compute the off-dialgonal elements of gluon loops, or trace of squares of Field Strength Tensor (FST)  ***/
    /* Clover definition of gluon loops off diagonals
     * Definition \mathcal{O}_i = unknown * \Tr[\sum_\mu F_{i,\mu} * F_{3,\mu}]
     * FST indices cannot be same
     * unknown is a factor which will be figured out later
     ************************************************************************************************************************/

    if ( (nsmearStout-nsmearStoutStart)%nsmearStep != 0 )
      PLEGMA_error("nsmear-stout-Gprop must be divisible by nsmear-step-Gprop\n");

    /***  Smearing Options  **********************************************************
     *   If add3D == true, we compute gluon loops with 3D smearing as well as 4D
     *   If isS4D == true, 4D smearing is performed but only for the spatial indicies
     *********************************************************************************/
    int n_s_dim = (add3D)?2:1;
    std::string sd = (isS4D)?"S4D":"4D";
    std::string outName3 = pathOut + "T_G_3DStoutSmearing" + std::to_string(nsmearStout) +"by"+std::to_string(nsmearStep)+"with"+std::to_string(alphaStout);
    std::string outName4 = pathOut + "T_G_"+sd+"StoutSmearing" + std::to_string(nsmearStout) +"by"+std::to_string(nsmearStep)+"with"+std::to_string(alphaStout);

    // create empty files: I append data later
    FILE *fp = NULL;
    if(comm_rank() == 0){
      if (add3D) {
	fp = fopen(outName3.c_str(),"w");
	if(fp == NULL) PLEGMA_error("Cannot open file:%s for writting\n",outName3.c_str());
	fclose(fp);
      }
      fp = fopen(outName4.c_str(),"w");
      if(fp == NULL) PLEGMA_error("Cannot open file:%s for writting\n",outName4.c_str());
      fclose(fp);
    }
    
    std::vector<std::pair<int,int>> pairs;
    for ( int i=0; i<N_DIMS-1; i++)
      for ( int j=0; j<N_DIMS-1; j++ )
	if ( i<j )
	  pairs.push_back(std::make_pair(i,j));

    PLEGMA_Gauge<double> gauge, gauge1, gauge2;
    PLEGMA_Field<double> trace1(BOTH,SCALAR), trace2(BOTH,SCALAR); // could this be DEVICE?
    PLEGMA_FT<double> ft3D(0,3,false,dims[3]); // this performs 3D FT on each time slice
    PLEGMA_Fmunu<double> fmunu;
    PLEGMA_Su3field<double> one3x3; 
    one3x3.setUnit((std::vector<int>) {0,4,8});

    if(HGC_verbosity > 1) PLEGMA_printf("Will work on %d confs",listGaugeConfs.size());
    for(int iconf=0; iconf < listGaugeConfs.size(); iconf++){
      double t1=MPI_Wtime();
      std::string confStr=splitStrFwd(listGaugeConfs[iconf],'.');
      gauge.readFile(listGaugeConfs[iconf], LIME_FORMAT);
      PLEGMA_printf("Unsmeared Plaquette is: ");
      gauge.calculatePlaq();

      for ( int i_s = 0; i_s < n_s_dim; i_s++ ){ // loop over smearing options
	int s_dim = (add3D && i_s == 0)?3:4;// smearing dimension: assume n_s_dim = 1 or 2
	// Smear the gauge and compute FST
	for(int n=nsmearStoutStart; n<=nsmearStout;n+=nsmearStep){
	  if((n-nsmearStoutStart)%(2*nsmearStep) == 0){
	    if((n-nsmearStoutStart)==0) {
	      if(n==0) gauge2.copy(gauge);
	      else gauge2.stoutSmearing(gauge,nsmearStoutStart,alphaStout,s_dim,isS4D);
	    }
	    else gauge2.stoutSmearing(gauge1,nsmearStep,alphaStout,s_dim,isS4D);
	  }
	  else{
	    gauge1.stoutSmearing(gauge2,nsmearStep,alphaStout,s_dim,isS4D);
	  }
	  fmunu.compute_leaves(((n-nsmearStoutStart)%(2*nsmearStep)==0)?gauge2:gauge1);
	  
	  // for each pair (i,j), compute T_ij
	  for(int p =0 ; p<pairs.size(); p++){ 
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
	    
	    //std::string tmp = "_"+std::to_string(iconf)+"_"+std::to_string(n)+"_"+std::to_string(p);
	    if(s_dim == 3)
	      ft3D.writeASCII(outName3, 0, true);
	    else if(s_dim == 4)
	      ft3D.writeASCII(outName4, 0, true);
	      //ft3D.store3DFTs(T[i_s][n/nsmearStep][p], 0 );
	    //PLEGMA_printf("%d %d %d\n",iconf,n,p);}
	  }
	}
      }
      double t2=MPI_Wtime();
      PLEGMA_printf("conf.%s completed in %f secs\n",confStr.c_str(),t2-t1);
    }
  }

  if ( doG_FF ) {
    if ( startT >= endT ) endT = HGC_totalL[3];
    // TODO: may not work when time dim of lattice is divided over processes
    PLEGMA_Gauge<double> gauge;
    if ( Nconf == "unit" ) {
      gauge.setUnit((std::vector<int>) {0,4,8, 9,13,17, 18,22,26, 27,31,35});
      gauge.unload();//I think this is not necessary
    }
    else {
      gauge.readFile(latfile, LIME_FORMAT);
      gauge.load();
    }
    gauge.calculatePlaq();
    initGaugeQuda(gauge, true);
    plaqQuda();

    //TODO
    // 1) reduce application of Udag op.
    // 2) split the loop into two for Diag (1,2,3,4) and Diag (5,6,7,8) to reduce #UpdateSolver() and storage from VtOut[3] to VtOut[2]
    // 3) in sacrifice to readability, we can remove VtIn to use only Vtmp1 and Vtmp2
    
    int T = HGC_totalL[3]; 
    int T0 = endT - startT; 
    // Prepare two solvers for D_u & D_d as updating the solver for MG takes some time
    QUDA_solver u_solver(mu);
    QUDA_solver d_solver(-mu);
    // TODO: we can use 3D vectors once dot prodct for 3D vectors is correctly implemented
    PLEGMA_Vector<double> psi, phi;//alloc flags. etc...?
    PLEGMA_Vector<double> VtIn, VtOut[2], Vtmp1, Vtmp2, Vstc;
    PLEGMA_Su3field<double> Umu(DEVICE), Unu(DEVICE);
    Vstc.randInit(rng_seed);

    double G_FF[T0][T][numSourcePositions];
    for(int isc=0; isc < numSourcePositions; isc++) for(int t=0; t < T0*T; t++) G_FF[t/T][t%T][isc]=0;

    std::string ext = "";
    if ( T0 != T ) ext = "from" + std::to_string(startT) + "to" + std::to_string(endT);
    std::string outName = pathOut + "G_FF" + Nconf + "_" + std::to_string(numSourcePositions) + ext;
    for(int isc=0; isc < numSourcePositions; isc++){
      Vstc.stochastic_Z(4);
      for(int t=startT; t < endT; t++){ // source time slice
	// TODO
	//  To enable dvision over time slice, we need to
	//   absorbTimeslice(Vstc,t); if t is in the range of time slices managed by the given rank
	//   set it to zero otherwise

	//mu = -in_mu;
	//solver.UpdateSolver();
	VtIn.absorbTimeslice(Vstc,t);
	//VtIn.zero_where(BOTH);//this is unncessary if we use 4D vectors
	VtIn.apply_gamma5();
	d_solver.solve(VtOut[0],VtIn);
	

	for(int mu_d=0; mu_d<3; mu_d++ ){
	  
	  Umu.absorbDir_device(gauge,mu_d);
	  Umu.Udag(); // to reduce #Udag() ops
	  for(int nu=0; nu<3; nu++){
	    if ( mu_d != nu ){
	      
	      //-- Notation: take <psi,phi>
	      /******  (<-, <-) Group  ******/

	      //---- inversion with mu&nu dependent rhs
	      // for Diag (1) & (3) // need Umu^dagger
	      
	      //mu = in_mu;
	      //solver.UpdateSolver();

	      Unu.shift(Umu, nu); // Here, Unu = Umu(x - nu)
	      //Unu.Udag(); //if Umu not daggered
	      Vtmp1.absorbTimeslice(Vstc,t);
	      VtIn.mulGV(Vtmp1,Unu);//use sep. field!!!
	      VtIn.apply_gamma(static_cast<GAMMAS>(nu+1),LEFT);
	      VtIn.apply_gamma5();
	      u_solver.solve(VtOut[1],VtIn);

	      // After this point, only Unu^dagger are used for Diag. (1,2,3,4) so that Unu is daggrered
	      Unu.absorbDir_device(gauge,nu);
	      Unu.Udag();

	      // Diag (1)

	      // (mu,nu;nu,mu)
	      // compute psi
	      Vtmp1.shift(VtOut[0],N_DIMS+nu);
	      // compute phi //need Unu^dagger
	      Vtmp2.mulGV(VtOut[1],Unu);
	      Vtmp2.apply_gamma(static_cast<GAMMAS>(mu_d+1),LEFT);
	      Vtmp2.apply_gamma5();
	      for(int ts=0; ts < T; ts++){//time separation
		psi.absorbTimeslice(Vtmp1,(t+ts)%T);
		phi.absorbTimeslice(Vtmp2,(t+ts)%T);
		G_FF[t-startT][ts][isc] -= (psi.dot(phi)).real();//does not partition time; could brak donw.
	      }
	      
	      // (nu,mu;nu,mu)
	      // compute psi
	      Vtmp1.shift(VtOut[0],N_DIMS+mu_d);
	      // compute phi //need Umu^dagger
	      //Umu.Udag(); //necessary if Umu not daggered 
	      Vtmp2.mulGV(VtOut[1],Umu);
	      Vtmp2.apply_gamma(static_cast<GAMMAS>(nu+1),LEFT);
	      Vtmp2.apply_gamma5();
	      //Umu.Udag(); //necessary if Umu not daggered 
	      for(int ts=0; ts < T; ts++){//time separation
		psi.absorbTimeslice(Vtmp1,(t+ts)%T);
		phi.absorbTimeslice(Vtmp2,(t+ts)%T);
		G_FF[t-startT][ts][isc] -= (psi.dot(phi)).real();
	      }
		
	      // Diag (3)
	      
	      // (mu,nu;nu,mu)
	      // compute psi // need Unu^dagger
	      Vtmp2.mulGV(VtOut[0],Unu);
	      Vtmp2.apply_gamma5();
	      Vtmp2.apply_gamma(static_cast<GAMMAS>(nu+1),LEFT);
	      Vtmp1.shift(Vtmp2,nu);
	      for(int ts=0; ts < T; ts++){//time separation
		psi.absorbTimeslice(Vtmp1,(t+ts)%T);
		phi.absorbTimeslice(VtOut[1],(t+ts)%T);
		G_FF[t-startT][ts][isc] += (psi.dot(phi)).real();
	      }

	      // (nu,mu;nu,mu)
	      // compute psi // need Umu^dagger
	      //Umu.Udag(); //necessary if Umu not daggered 
	      Vtmp2.mulGV(VtOut[0],Umu);
	      Vtmp2.apply_gamma5();
	      Vtmp2.apply_gamma(static_cast<GAMMAS>(nu+1),LEFT);
	      Vtmp1.shift(Vtmp2,mu_d);
	      //Umu.Udag(); //necessary if Umu not daggered 
	      for(int ts=0; ts < T; ts++){//time separation 
		psi.absorbTimeslice(Vtmp1,(t+ts)%T);
		phi.absorbTimeslice(VtOut[1],(t+ts)%T);
                G_FF[t-startT][ts][isc] += (psi.dot(phi)).real();
              }

	      //---- inversion with mu&nu dependent rhs
	      // for Diag (2) & (4) // need Umu

	      Umu.Udag(); // since Umu is daggered
	      VtIn.absorbTimeslice(Vstc,t);
	      Vtmp1.shift(VtIn,N_DIMS+mu_d);
	      VtIn.mulGV(Vtmp1,Umu);
	      VtIn.apply_gamma(static_cast<GAMMAS>(nu+1),LEFT);
	      u_solver.solve(VtOut[1],VtIn);
	      Umu.Udag(); // since Umu is daggered
	      
	      // Diag (2)

	      // (mu,nu;nu,mu) 
	      // compute psi
	      Vtmp1.shift(VtOut[0],N_DIMS+nu);
	      // compute phi // need U_nu^dagger
	      Vtmp2.mulGV(VtOut[1],Unu);
	      Vtmp2.apply_gamma(static_cast<GAMMAS>(mu_d+1),LEFT);
	      Vtmp2.apply_gamma5();
	      for(int ts=0; ts < T; ts++){//time separation
		psi.absorbTimeslice(Vtmp1,(t+ts)%T);
		phi.absorbTimeslice(Vtmp2,(t+ts)%T);
                G_FF[t-startT][ts][isc] += (psi.dot(phi)).real();
              }
	      
	      // (nu,mu;nu,mu) 
	      // compute psi
	      Vtmp1.shift(VtOut[0],N_DIMS+mu_d);
	      // compute phi // need Umu^dagger
	      //Umu.Udag(); //necessary if Umu not daggered 
	      Vtmp2.mulGV(VtOut[1],Umu);
	      Vtmp2.apply_gamma(static_cast<GAMMAS>(nu+1),LEFT);
	      Vtmp2.apply_gamma5();
	      //Umu.Udag(); //necessary if Umu not daggered 
	      for(int ts=0; ts < T; ts++){//time separation 
		psi.absorbTimeslice(Vtmp1,(t+ts)%T);
		phi.absorbTimeslice(Vtmp2,(t+ts)%T);
                G_FF[t-startT][ts][isc] += (psi.dot(phi)).real();
              }
		
	      // Diag (4)

	      // (mu,nu;nu,mu)   
	      // compute psi // need Unu^dagger
	      Vtmp2.mulGV(VtOut[0],Unu);
	      Vtmp2.apply_gamma5();
              Vtmp2.apply_gamma(static_cast<GAMMAS>(mu_d+1),LEFT);
	      Vtmp1.shift(Vtmp2,nu);
	      for(int ts=0; ts < T; ts++){//time separation  
		psi.absorbTimeslice(Vtmp1,(t+ts)%T);
		phi.absorbTimeslice(VtOut[1],(t+ts)%T);
		G_FF[t-startT][ts][isc] -= (psi.dot(phi)).real();
	      }

	      // (mu,nu;mu,nu)
	      // compute psi // need Umu^dagger
	      //Umu.Udag(); //necessary if Umu not daggered
	      Vtmp2.mulGV(VtOut[0],Umu);
              Vtmp2.apply_gamma5();
              Vtmp2.apply_gamma(static_cast<GAMMAS>(nu+1),LEFT);
              Vtmp1.shift(Vtmp2,mu_d);
	      //Umu.Udag(); //necessary if Umu not daggered
              for(int ts=0; ts < T; ts++){//time separation
                psi.absorbTimeslice(Vtmp1,(t+ts)%T);
                phi.absorbTimeslice(VtOut[1],(t+ts)%T);
                G_FF[t-startT][ts][isc] -= (psi.dot(phi)).real();
              }
	    }
	  }
	}

	//mu = in_mu;
        //solver.UpdateSolver();

	VtIn.absorbTimeslice(Vstc,t);
	u_solver.solve(VtOut[0],VtIn);	  
	
        for(int mu_d=0; mu_d<3; mu_d++ ){

          Umu.absorbDir_device(gauge,mu_d);
	  Umu.Udag(); // to reduce #Udag() ops 
          for(int nu=0; nu<3; nu++){
            if ( mu_d != nu ){
	      
	      /******  (<-, ->) Group  ******/
	      
	      //---- inversion with mu&nu dependent rhs
	      // for Diag (5) & (7)
	      //mu = -in_mu;
	      //solver.UpdateSolver();

	      Unu.shift(Umu, nu); // Here, Unu = Umu(x - nu) 
              //Unu.Udag(); // since Umu is daggered
	      Vtmp1.absorbTimeslice(Vstc,t);
	      VtIn.mulGV(Vtmp1,Unu);
	      VtIn.apply_gamma(static_cast<GAMMAS>(nu+1),LEFT);
	      VtIn.apply_gamma5();
	      d_solver.solve(VtOut[1],VtIn);

	      Unu.absorbDir_device(gauge,nu);
              Unu.Udag();

	      // Diag (5)
	      
	      // (mu,nu;nu,mu)
	      // compute psi
	      Vtmp1.shift(VtOut[1],N_DIMS+nu);
	      // compute phi // need Unu^dagger
	      Vtmp2.mulGV(VtOut[0],Unu);
	      Vtmp2.apply_gamma(static_cast<GAMMAS>(mu_d+1),LEFT);
	      Vtmp2.apply_gamma5();
	      for(int ts=0; ts < T; ts++){//time separation
		psi.absorbTimeslice(Vtmp1,(t+ts)%T);
		phi.absorbTimeslice(Vtmp2,(t+ts)%T);
                G_FF[t-startT][ts][isc] -= -(psi.dot(phi)).real();
              }

	      // (nu,mu;nu,mu)
	      // compute psi
	      Vtmp1.shift(VtOut[1],N_DIMS+mu_d);
	      // compute phi //need Umu^dagger
	      //Umu.Udag(); //necessary if Umu not daggered
	      Vtmp2.mulGV(VtOut[0],Umu);
	      Vtmp2.apply_gamma(static_cast<GAMMAS>(nu+1),LEFT);
	      Vtmp2.apply_gamma5();
	      //Umu.Udag(); //necessary if Umu not daggered
	      for(int ts=0; ts < T; ts++){//time separation
		psi.absorbTimeslice(Vtmp1,(t+ts)%T);
                phi.absorbTimeslice(Vtmp2,(t+ts)%T);
		G_FF[t-startT][ts][isc] -= -(psi.dot(phi)).real();
              }
	      
	      // Diag (7)
	      
	      // (mu,nu;nu,mu)  
	      // compute psi // need Unu^dagger
	      Vtmp2.mulGV(VtOut[1],Unu);
	      Vtmp2.apply_gamma5();
	      Vtmp2.apply_gamma(static_cast<GAMMAS>(mu_d+1),LEFT);
	      Vtmp1.shift(Vtmp2,nu);
	      for(int ts=0; ts < T; ts++){//time separation
		psi.absorbTimeslice(Vtmp1,(t+ts)%T);
		phi.absorbTimeslice(VtOut[0],(t+ts)%T);
		G_FF[t-startT][ts][isc] += -(psi.dot(phi)).real();
              }

	      // (nu,mu;nu,mu)
	      // compute psi // need Umu^dagger
	      //Umu.Udag(); //necessary if Umu not daggered
	      Vtmp2.mulGV(VtOut[1],Umu);
	      Vtmp2.apply_gamma5();
	      Vtmp2.apply_gamma(static_cast<GAMMAS>(nu+1),LEFT);
	      Vtmp1.shift(Vtmp2,mu_d);
	      //Umu.Udag(); //necessary if Umu not daggered
	      for(int ts=0; ts < T; ts++){//time separation
		psi.absorbTimeslice(Vtmp1,(t+ts)%T);
                phi.absorbTimeslice(VtOut[0],(t+ts)%T);
                G_FF[t-startT][ts][isc] += -(psi.dot(phi)).real();
              }
	      
	      //---- inversion with mu&nu dependent rhs
	      // for Diag (6) & (8) // need Umu   

	      Umu.Udag(); // since Umu is daggered
	      VtIn.absorbTimeslice(Vstc,t);
	      Vtmp1.shift(VtIn,N_DIMS+mu_d);
	      VtIn.mulGV(Vtmp1,Umu);
	      VtIn.apply_gamma(static_cast<GAMMAS>(nu+1),LEFT);
	      VtIn.apply_gamma5();
	      d_solver.solve(VtOut[1],VtIn);
	      Umu.Udag();
	      
	      // Diag (6)
	      
	      // (mu,nu;nu,mu) 
	      // compute psi
	      Vtmp1.shift(VtOut[1],N_DIMS+nu);
	      // compute phi //need Unu^dagger
	      Vtmp2.mulGV(VtOut[0],Unu);
	      Vtmp2.apply_gamma(static_cast<GAMMAS>(mu_d+1),LEFT);
	      Vtmp2.apply_gamma5();
	      for(int ts=0; ts < T; ts++){//time separation
		psi.absorbTimeslice(Vtmp1,(t+ts)%T);
		phi.absorbTimeslice(Vtmp2,(t+ts)%T);
		G_FF[t-startT][ts][isc] += -(psi.dot(phi)).real();
	      }

	      // (nu,mu;nu,mu)    
	      // compute psi
	      Vtmp1.shift(VtOut[1],N_DIMS+mu_d);
	      // compute phi // meed Umu^dagger
	      //Umu.Udag(); //necessary if Umu not daggered
	      Vtmp2.mulGV(VtOut[0],Umu);
	      Vtmp2.apply_gamma(static_cast<GAMMAS>(nu+1),LEFT);
	      Vtmp2.apply_gamma5();
	      //Umu.Udag(); //necessary if Umu not daggered
	      for(int ts=0; ts < T; ts++){//time separation
		psi.absorbTimeslice(Vtmp1,(t+ts)%T);
		phi.absorbTimeslice(Vtmp2,(t+ts)%T);
                G_FF[t-startT][ts][isc] += -(psi.dot(phi)).real();
	      }
	      
	      // Diag (8)

	      // (mu,nu;nu,mu) 
	      // compute psi // need Unu^dagger
	      Vtmp2.mulGV(VtOut[1],Unu);
	      Vtmp2.apply_gamma5();
	      Vtmp2.apply_gamma(static_cast<GAMMAS>(mu_d+1),LEFT);
	      Vtmp1.shift(Vtmp2,nu);
	      for(int ts=0; ts < T; ts++){//time separation
		psi.absorbTimeslice(Vtmp1,(t+ts)%T);
                phi.absorbTimeslice(VtOut[0],(t+ts)%T);
                G_FF[t-startT][ts][isc] -= -(psi.dot(phi)).real();
              }

	      // (nu,mu;nu,mu)
	      // compute psi // need Umu^dagger
	      //Umu.Udag(); //necessary if Umu not daggered
	      Vtmp2.mulGV(VtOut[1],Umu);
              Vtmp2.apply_gamma5();
              Vtmp2.apply_gamma(static_cast<GAMMAS>(nu+1),LEFT);
              Vtmp1.shift(Vtmp2,mu_d);
	      //Umu.Udag(); //necessary if Umu not daggered
              for(int ts=0; ts < T; ts++){//time separation
                psi.absorbTimeslice(Vtmp1,(t+ts)%T);
                phi.absorbTimeslice(VtOut[0],(t+ts)%T);
                G_FF[t-startT][ts][isc] -= -(psi.dot(phi)).real();
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
      fpt.precision(8);
      for(int isc=0; isc < numSourcePositions; isc++)
	for(int t=startT; t < endT; t++)
	  for(int ts=0; ts < T; ts++) {
	    fpt << isc<< " " << t << " " << ts << " " << std::scientific << G_FF[t][ts][isc]/16.0 <<  std::endl;
	  }
    }
  }
  finalize();
  return 0;
}
