#include <vector>
#include <iostream>
#include <armadillo>
#include <cmath>

#define WITH_MPI

#include "base/mpi_wrapper.h"
#include "base/quadrature.h"
#include "base/stopwatch.h"
#include "disc/dg.h"
#include "disc/dg_eqp.h"
#include "fe/fe.h"
#include "fe/fe_definition.h"
#include "mesh/fe_cache.h"
#include "mesh/mesh_generator.h"

#ifdef WITH_MPI
#include "mesh/parallel_mesh.h"
#include "mesh/parallel_fe_field.h"
#include "mesh/parallel_mesh_geometry.h"
#else
#include "mesh/mesh.h"
#include "mesh/fe_field.h"
#include "mesh/mesh_geometry.h"
#endif

#include "solver/dense_solver.h"
#include "solver/ptc_solver.h"
#include "solver/newton_solver.h"
#include "solver/gmres_solver.h"
#include "solver/block_ilu.h"
#include "solver/additive_schwarz.h"
#include "solver/block_sparse_matrix.h"
#include "io/vtk_io.h"
#include "io/gmsh.h"


#include "eqn/euler.h"
#include "eqn/rans_sa.h"
#include "eqn/navier_stokes_utils.h"

#define ONERA

//#define EULER
#define RANS
/*
 * how1 was using a coarse mesh
 * how2 works well; alpha \in [0,3] and M \in [0.2,0.4]
 */

#ifdef ONERA
#ifdef RANS
class ParametrizedEuler : public RANSSA<3> {
#endif
#ifdef EULER
class ParametrizedEuler : public Euler<3> {
#endif
#else
class ParametrizedEuler : public Euler<2> {
#endif
  //  class ParametrizedNS : public NavierStokes {
public:

  // mode
  // 0 : alpha only
  // 1 : M only
  // 2 : alpha and M
  const unsigned int param_mode = 2;
  std::vector<unsigned int> far_bnds;
  unsigned int foil_bnd;
  NSUtils* nsutils_;
  arma::Col<double> mu_;

public:
  ParametrizedEuler(unsigned int dim, const NSVariableType var_type_in, NSUtils* nsutils)
#ifdef EULER
    : Euler(var_type_in),
#endif
#ifdef RANS
    : RANSSA(var_type_in),
#endif
      nsutils_(nsutils)
      //    : NavierStokes(dim,var_type_in)
  {}

  // unsigned int quad_degree(const unsigned int poly_degree) const
  // {
  //   const unsigned int quad_min = 7;
  //   return std::max(quad_min, 3*poly_degree+1);
  // }


  unsigned int n_params() const {
    switch (param_mode) {
    case 0:
    case 1:
      return 1;
    case 2:
      return 2;
    default:
      Error("unsupported mode");
    }
  }

  void flow_vars(double& alpha, double& M)
  {
    alpha = 1.5*M_PI/180;
    M = 0.3;
    switch (param_mode) {
    case 0:
      alpha = mu_(0);
      break;
    case 1:
      M = mu_(0);
      break;
    case 2:
      alpha = mu_(0);
      M = mu_(1);
      break;
    default:
      Error("unknown mode");
    }
  }  
  
  virtual void set_parameters(const arma::Col<double> mu) {
    double M;
    double alpha;
    mu_ = mu;
    flow_vars(alpha,M);    

    // set initial condition
    nsutils_->set_mach_number(M);
#ifdef ONERA
    nsutils_->set_sideslip(alpha);
    nsutils_->set_angle_of_attack(0.0);
#else
    nsutils_->set_angle_of_attack(alpha);
#endif
#ifdef RANS
    double Re = 1e6;
    double fs_chi = 3.0;
    double sa_scale = sqrt(Re);
    nsutils_->set_reynolds_number(Re);
    nsutils_->set_freestream_chi(fs_chi);
    nsutils_->set_sa_scale(sa_scale);
#endif
    arma::Col<double> u_init = nsutils_->state();
    set_initial_state(u_init);
#ifdef RANS
    set_dynamic_viscosity(1.0/Re);
    set_constant_viscosity(true);
    set_freestream_chi(fs_chi);
#endif

    // set boundary conditions
    std::vector<double> ub = arma::conv_to<std::vector<double>>::from(u_init);
    for (unsigned int i = 0; i < far_bnds.size(); ++i) {
      set_boundary_type(far_bnds[i],NSBoundaryType::full_state);
      set_boundary_parameters(far_bnds[i],ub);
    }
#ifdef EULER
    set_boundary_type(foil_bnd,NSBoundaryType::flow_tangency);
#endif
#ifdef RANS
    set_boundary_type(foil_bnd,NSBoundaryType::adiabatic_wall);
#endif

    // lift output
    set_output_type(0,NSOutputType::directional_force);
    std::vector<unsigned int> bids = {foil_bnd};
    set_output_boundary_ids(0,bids);
    const double qinf = nsutils_->dynamic_pressure();
#ifdef ONERA
    const double sca = 2.0/1.15315084119231;
#ifdef EULER
    std::vector<double> bparams = {-sca*sin(alpha)/qinf,0.0,sca*cos(alpha)/qinf};
#endif
#ifdef RANS
    std::vector<double> bparams = {sca*cos(alpha)/qinf,0.0,sca*sin(alpha)/qinf};
#endif
    set_output_parameters(0,bparams);
#else
    std::vector<double> bparams = {-sin(alpha)/qinf,cos(alpha)/qinf,0.0};
    set_output_parameters(0,bparams);
#endif
  }
  
  void parameter_bound(arma::Mat<double>& mu_bnd)
  {
    //arma::Row<double> alpha_bnd = {{0.0*M_PI/180.0, 3.5*M_PI/180.0}};
    arma::Row<double> alpha_bnd = {{3.0*M_PI/180.0, 3.0*M_PI/180.0}};
    arma::Row<double> M_bnd = {{0.5, 0.5}};
    mu_bnd.set_size(n_params(),2);
    switch (param_mode) {
    case 0:
      mu_bnd.row(0) = alpha_bnd;
      break;
    case 1:
      mu_bnd.row(0) = M_bnd;
      break;
    case 2:
      mu_bnd.row(0) = alpha_bnd;
      mu_bnd.row(1) = M_bnd;
      break;
    default:
      Error("unknown mode");      
    }
  }

  void generate_structured_Xi(const unsigned int n_1d, arma::Mat<double>& Xi_train) {
    arma::Mat<double> mu_bnd;
    parameter_bound(mu_bnd);    
    const unsigned int P = mu_bnd.n_rows;
    Xi_train.set_size(P, pow(n_1d,P));
        
    unsigned int cnt = 0;
    switch (n_params()) {
    case 1:
      for (unsigned int i0 = 0; i0 < n_1d; ++i0) {
        Xi_train(0, cnt) = mu_bnd(0,0) + (mu_bnd(0,1)-mu_bnd(0,0))*((1.0*i0)/(n_1d-1.0));
        ++cnt;
      }
      break;
    case 2:
      for (unsigned int i1 = 0; i1 < n_1d; ++i1) {
        for (unsigned int i0 = 0; i0 < n_1d; ++i0) {
          Xi_train(0, cnt) = mu_bnd(0,0) + (mu_bnd(0,1)-mu_bnd(0,0))*((1.0*i0)/(n_1d-1.0));
          Xi_train(1, cnt) = mu_bnd(1,0) + (mu_bnd(1,1)-mu_bnd(1,0))*((1.0*i1)/(n_1d-1.0));
          ++cnt;
        }
      }
      break;
    case 3:
      for (unsigned int i2 = 0; i2 < n_1d; ++i2) {
        for (unsigned int i1 = 0; i1 < n_1d; ++i1) {
          for (unsigned int i0 = 0; i0 < n_1d; ++i0) {
            Xi_train(0, cnt) = mu_bnd(0,0) + (mu_bnd(0,1)-mu_bnd(0,0))*((1.0*i0)/(n_1d-1.0));
            Xi_train(1, cnt) = mu_bnd(1,0) + (mu_bnd(1,1)-mu_bnd(1,0))*((1.0*i1)/(n_1d-1.0));
            Xi_train(2, cnt) = mu_bnd(2,0) + (mu_bnd(2,1)-mu_bnd(2,0))*((1.0*i2)/(n_1d-1.0));
            ++cnt;
          }
        }
      }
      break;
    case 4:
      for (unsigned int i3 = 0; i3 < n_1d; ++i3) {
        for (unsigned int i2 = 0; i2 < n_1d; ++i2) {
          for (unsigned int i1 = 0; i1 < n_1d; ++i1) {
            for (unsigned int i0 = 0; i0 < n_1d; ++i0) {
              Xi_train(0, cnt) = mu_bnd(0,0) + (mu_bnd(0,1)-mu_bnd(0,0))*((1.0*i0)/(n_1d-1.0));
              Xi_train(1, cnt) = mu_bnd(1,0) + (mu_bnd(1,1)-mu_bnd(1,0))*((1.0*i1)/(n_1d-1.0));
              Xi_train(2, cnt) = mu_bnd(2,0) + (mu_bnd(2,1)-mu_bnd(2,0))*((1.0*i2)/(n_1d-1.0));
              Xi_train(3, cnt) = mu_bnd(3,0) + (mu_bnd(3,1)-mu_bnd(3,0))*((1.0*i3)/(n_1d-1.0));
              ++cnt;
            }
          }
        }
      }
      break;
    default:       
      Error("unknown number of parameters");
    }
    Assert(cnt == pow(n_1d,P), "count mismatch");
  }

  void generate_random_Xi(const unsigned int n_train, arma::Mat<double>& Xi_train) {
    arma::Mat<double> mu_bnd;
    parameter_bound(mu_bnd);    
    const unsigned int P = mu_bnd.n_rows;
    Xi_train = arma::randu(P,n_train);
    for (unsigned int i = 0; i < P; ++i) {
      Xi_train.row(i) = mu_bnd(i,0) + (mu_bnd(i,1)-mu_bnd(i,0))*Xi_train.row(i);
    }
  }
  
};

class EQPDriver {
public:
  EQPDriver()
#ifdef ONERA
    : dim(3),
      geom_degree(3),
#else
    : dim(2),
      geom_degree(4),
#endif
      nstype(NSVariableType::entropy),
#ifdef EULER
      nsutils(dim,nstype,NSUtils::Equation::euler,NSUtils::Normalization::unit_rho_a),
#endif
#ifdef RANS
      nsutils(dim,nstype,NSUtils::Equation::rans_sa,NSUtils::Normalization::unit_rho_a),
#endif
      eqn(dim, nstype, &nsutils),
      mesh(dim),
      mesh_geom(&mesh,&fe_set,geom_degree),
      dg_eqp_c(&eqn, &mesh, &mesh_geom, &fe_set, &quad_set)
  {}

  
  void load_onera_mesh() {
    // set boundary conditions
    eqn.set_n_boundaries(4);

    // output (drag)
    eqn.set_n_outputs(1);

    // set far-field boundaries
    eqn.far_bnds.resize(1);
    eqn.far_bnds[0] = 2;
    eqn.foil_bnd = 1;
    eqn.set_boundary_type(3,NSBoundaryType::symmetry_plane);
    
    // load the mesh
    std::string gmsh_file = "../oneram6_very_coarse.msh";
    unsigned int comm_rank = Utilities::MPI::mpi_comm_rank();

    // extract surface mesh
    std::vector<bool> wall_marker = {false,true,false,false};
    Mesh surf_mesh(dim-1,dim);
    MeshGeometry surf_geom(&surf_mesh,&fe_set,geom_degree);
 #ifdef RANS
    {
      Gmsh gmsh;
      Mesh mesh(dim);
      MeshGeometry mesh_geom(&mesh,&fe_set,geom_degree);
      if (comm_rank == 0) {
	gmsh.read_msh_file(gmsh_file);
      }
      gmsh.mpi_bcast();
      gmsh.load_mesh(mesh,mesh_geom);

      // extract surface mesh
      mesh_geom.set_wall_marker(wall_marker);
      mesh_geom.extract_surface_geometry(wall_marker, surf_mesh, surf_geom);
    }
#endif

    Gmsh gmsh;
    mesh.set_enforce_one_regular_hanging_node(true);
    if (Utilities::MPI::mpi_comm_rank() == 0) {
      gmsh.read_msh_file(gmsh_file);
      printf("onera: mode = %d, n_elem = %d\n", eqn.param_mode, mesh.n_elements());
    }
    gmsh.load_mesh(mesh, mesh_geom);

    mesh.prepare_parallelization();
    mesh.execute_repartition();
    mesh_geom.execute_repartition();
    mesh.finalize_repartition();

#ifdef RANS
    if (comm_rank == 0) {
      printf("compuing wall distance\n");
    }
    mesh_geom.set_wall_geometry(surf_mesh, surf_geom);
    mesh_geom.init_wall_distance();
#endif
    
  }

  void load_naca_how_mesh() {

    // set boundary conditions
    eqn.set_n_boundaries(4);

    // output (drag)
    eqn.set_n_outputs(1);
    
    // set far-field boundaries
    eqn.far_bnds.resize(2);
    eqn.far_bnds[0] = 2;
    eqn.far_bnds[1] = 3;
    eqn.foil_bnd = 1;
    
    // load the mesh
    std::string gmsh_file = "../naca_ref1.gmsh";
    Gmsh gmsh;
    mesh.set_enforce_one_regular_hanging_node(true);
    if (Utilities::MPI::mpi_comm_rank() == 0) {
      gmsh.read_msh_file(gmsh_file);
    }
    gmsh.load_mesh(mesh, mesh_geom);
    
    printf("naca: mode = %d, n_elem = %d\n", eqn.param_mode, mesh.n_elements());
  }

  void load_naca_orig() {
    
    // set boundary conditions
    eqn.set_n_boundaries(3);

    // output (drag)
    eqn.set_n_outputs(1);
    
    // set far-field boundaries
    eqn.far_bnds.resize(2);
    eqn.far_bnds[0] = 1;
    eqn.far_bnds[1] = 2;
    eqn.foil_bnd = 0;
    
    // mesh generator
    unsigned int sca = 1;
    NacaMeshGenerator naca_gen; // the mesh generator will construct a C mesh
    naca_gen.set_naca_digits("0012");
    naca_gen.set_n_foil_elements(12*sca); // # elements on foil along flow direction
    naca_gen.set_n_tail_elements(3*sca); // # elements after foil along flow direction
    naca_gen.set_n_radial_elements(6*sca); // # elements in radial direction
    naca_gen.set_n_span_elements(2*sca); // # spanwise elements
    naca_gen.set_first_element_radial_spacing(0.1);
    naca_gen.set_farfield_radius(30.0);
    naca_gen.set_sqrt_fitting(true); // use sqrt relation for wall-normal spacing
    naca_gen.set_mesh_entity_type(entity_type);
  
    // load the mesh
    mesh.set_enforce_one_regular_hanging_node(true);
    naca_gen.generate_mesh(mesh, mesh_geom);

    if (Utilities::MPI::mpi_comm_rank() == 0) {
      printf("naca: mode = %d, n_elem = %d\n", eqn.param_mode, mesh.n_elements());
    }
  }
  
  void setup_eqp() {
    // DG-EQP evaluator and contructor
    dg_eqp_c.set_spatio_parameter_adaptivity(true);
    //dg_eqp_c.set_spatial_adapt_on_first_iteration_only(true);
    dg_eqp_c.adapt()->set_adaptation_type(Adaptation<double>::type::h);

    // FP 2e-4 for 1%
    // Naca 5e-4 for 1%
    dg_eqp_c.adapt()->set_target_error(1e-4);
    dg_eqp_c.adapt()->set_adaptation_target_type(Adaptation<double>::target_type::output);
    dg_eqp_c.adapt()->set_max_iterations(20);
    //dg_eqp_c.adapt()->set_refinement_fraction(1.0);
  
    dg_eqp_c.set_initial_polynomial_degree(poly_degree);
    dg_eqp_c.set_eqp_tolerance(5e-5);
    
    dg_eqp_c.set_n_max_reduced_basis(30);
    dg_eqp_c.set_weak_greedy_tolerance(1e-4);
    dg_eqp_c.set_pod_tolerance(1e-10);
    
    dg_eqp_c.set_greedy_target_type(DGEQPConstructor<double>::GreedyTargetType::output);
    dg_eqp_c.set_eqp_form(EQPForm::elem_stable);
    //dg_eqp_c.set_eqp_form(EQPForm::original);
    dg_eqp_c.set_eqp_norm(EQPNorm::l2);
    dg_eqp_c.set_include_constant_rb(false);
    dg_eqp_c.set_include_stab_rb(false);
    dg_eqp_c.set_n_eqp_smoothing_iterations(3);
    dg_eqp_c.set_eqp_verbosity(-1); 
    dg_eqp_c.set_eqp_target_type(DGEQPConstructor<double>::EQPTargetType::output);
    //dg_eqp_c.set_eqp_unity_weights(true);  
    dg_eqp_c.set_eqp_element_constant_constraint(true);
    dg_eqp_c.set_eqp_facet_constant_constraint(true);
    dg_eqp_c.set_eqp_min_nnz_weights(0);

    //dg_eqp_c.set_write_reduced_mesh(true);
    //dg_eqp_c.set_write_reduced_basis(true);

    dg_eqp_c.set_p_sequencing_flag(true);
    dg_eqp_c.set_primal_fe_solver(&ptc);
    dg_eqp_c.set_primal_rb_solver(&ptc_rb);
    dg_eqp_c.adapt()->set_adjoint_solver(&adjoint_solver);
    dg_eqp_c.set_adjoint_fe_solver(&adjoint_solver);
    dg_eqp_c.set_adjoint_rb_solver(&adjoint_rb_solver);
    
    
    // set training parameters
    arma::arma_rng::set_seed(20);
    eqn.generate_structured_Xi(5, Xi_train);
    //arma::Mat<double> Xi_train_2;
    //eqn.generate_random_Xi(30, Xi_train_2);
    //Xi_train = arma::join_rows(Xi_train, Xi_train_2);
    dg_eqp_c.set_training_parameters(Xi_train);

    // set test parameters
    arma::arma_rng::set_seed(100);
    unsigned int n_test = 20;  
    eqn.generate_random_Xi(n_test, Xi_test);

    //Xi_test = Xi_train;
    
    //dg_eqp_c.set_test_parameters(Xi_test);
    
    dg_eqp_c.set_test_rb_truth_error(true); 
    dg_eqp_c.set_test_res(true);

    if (Utilities::MPI::mpi_comm_rank() == 0) {
      printf("n train = %d\n", static_cast<int>(Xi_train.n_cols));
      printf("n test = %d\n", static_cast<int>(Xi_test.n_cols));
    }
  };

  void set_rb_solver() {
    ptc_rb.set_linear_solver(&dense_linsolver);
    ptc_rb.set_max_iter(50);
    ptc_rb.set_verbosity(-1); 
    ptc_rb.set_abs_tol(1e-10);
    ptc_rb.set_initial_timestep(1e1); 
    ptc_rb.set_timestep_increase_multiplier(3.0);
    ptc_rb.set_line_search_maximum_physical_change(0.9); 
    //ptc_rb.set_line_search_type(PTCSolver<double>::LineSearchType::unsteady);
    //ptc_rb.set_line_search_maximum_residual_increase(2.0);
    //ptc_rb.set_maximum_timestep(1e2);
    //ptc_rb.set_ping_jacobian(true);

    // set adjoint solver
    adjoint_rb_solver.set_linear_solver(&dense_linsolver);
    adjoint_rb_solver.set_residual_tolerance(1e-9);
    adjoint_rb_solver.set_verbosity(-1);
  };

  void set_fe_solver() {
    // set primal fe solver
    ilu.set_ordering(BlockOrdering<double>::Type::MDF);
#ifdef WITH_MPI
    addsch.set_type(AdditiveSchwarz<double>::Type::restricted);
    addsch.set_local_preconditioner(&ilu);
    gmres.set_preconditioner(&addsch);
#else
    gmres.set_preconditioner(&ilu);
#endif
    gmres.set_max_inner_iterations(200);
    gmres.set_max_outer_iterations(2);
    gmres.set_verbosity(-1);

    ptc.set_verbosity(1);
    ptc.set_linear_solver(&gmres);
    ptc.set_max_iter(250);
    ptc.set_initial_timestep(1e0);
    ptc.set_timestep_increase_multiplier(3.0);
    ptc.set_abs_tol(1e-9);
    ptc.set_line_search_type(PTCSolver<double>::LineSearchType::unsteady);
    ptc.set_line_search_maximum_physical_change(0.2); 
    ptc.set_line_search_maximum_residual_increase(1.05);
    ptc.set_linear_solver_decrease_multiplier(1e-5);

    // set adjoint solver
    adjoint_solver.set_linear_solver(&gmres);
    adjoint_solver.set_residual_tolerance(1e-9);
  };

  void run_adaptive_fe_solve() {
    dg_eqp_c.adapt()->solve();
  }

  void run_weak_greedy() {
    dg_eqp_c.init();
    dg_eqp_c.weak_greedy(&dg_eqp_e);  
  };

  void save_mesh(const std::string& file_name) {
    std::ofstream ofs(file_name);
    boost::archive::binary_oarchive oa(ofs);
    oa << fe_set << mesh << mesh_geom;    
    ofs.close();
  };

  void load_mesh(const std::string& file_name) {
    std::ifstream ifs(file_name);
    boost::archive::binary_iarchive ia(ifs);
    ia >> fe_set >> mesh >> mesh_geom;
  };

  void save_training_data(const std::string& file_name) {
    std::ofstream ofs(file_name);
    boost::archive::binary_oarchive oa(ofs);
    std::vector<double> Xi_train_vec = arma::conv_to<std::vector<double>>::from(arma::vectorise(dg_eqp_c.training_parameters()));
    oa << Xi_train_vec;
    oa << dg_eqp_c.training_states();
    ofs.close();
  }

  void load_training_data(const std::string& file_name) {
    const unsigned int n_param = eqn.n_params();
    
    std::ifstream ifs(file_name);
    boost::archive::binary_iarchive ia(ifs);
    std::vector<double> Xi_train_vec;
    ia >> Xi_train_vec;
    Xi_train = arma::Mat<double>(Xi_train_vec.data(), n_param, Xi_train_vec.size()/n_param);
    ia >> train_vecs;
    dg_eqp_c.set_training_parameters(Xi_train);
    dg_eqp_c.set_training_states(train_vecs);
  }

  void save_test_data(const std::string& file_name) {
    std::ofstream ofs(file_name);
    boost::archive::binary_oarchive oa(ofs);
    std::vector<double> Xi_test_vec = arma::conv_to<std::vector<double>>::from(arma::vectorise(dg_eqp_c.test_parameters()));
    oa << Xi_test_vec;
    oa << dg_eqp_c.test_states();
    ofs.close();
  }

  void load_test_data(const std::string& file_name) {
    const unsigned int n_param = eqn.n_params();
    
    std::ifstream ifs(file_name);
    boost::archive::binary_iarchive ia(ifs);
    std::vector<double> Xi_test_vec;
    ia >> Xi_test_vec;
    Xi_test = arma::Mat<double>(Xi_test_vec.data(), n_param, Xi_test_vec.size()/n_param);
    ia >> test_vecs;
    dg_eqp_c.set_test_parameters(Xi_test);
    dg_eqp_c.set_test_states(test_vecs);
  }
  
  const MeshEntityType entity_type = MeshEntityType::quad;
  const unsigned int poly_degree = 2;
  const FiniteElementType fe_type = FiniteElementType::Legendre;
  
  unsigned int dim;
  unsigned int geom_degree;
  NSVariableType nstype;
  NSUtils nsutils;

  FESet fe_set;
  QuadratureSet quad_set;

  ParametrizedEuler eqn;

#ifdef WITH_MPI
  ParallelMesh mesh;
  ParallelMeshGeometry mesh_geom;
#else
  Mesh mesh;
  MeshGeometry mesh_geom;
#endif

  Mesh surf_mesh;
  MeshGeometry surf_geom;

  DGEQPEvaluator<double> dg_eqp_e;
  DGEQPConstructor<double> dg_eqp_c;

  // RB solvers
  DenseSolver<double> dense_linsolver;
  PTCSolver<double> ptc_rb;
  AdjointSolver<double> adjoint_rb_solver;
  //NewtonSolver<double> ptc_rb;

  // full fidelity solvers
  GMRESSolver<double> gmres;
  BlockILU<double> ilu;
  PTCSolver<double> ptc;
  AdjointSolver<double> adjoint_solver;
#ifdef WITH_MPI
  AdditiveSchwarz<double> addsch;
#endif

  arma::Mat<double> Xi_train;
  arma::Mat<double> Xi_test;

  std::vector<BlockVector<double>*> train_vecs;
  std::vector<BlockVector<double>*> test_vecs;
};

  
int main(int argc, char *argv[])
{
#ifdef WITH_MPI
  Utilities::MPI::mpi_init(argc, argv);
#endif
  unsigned int comm_rank = Utilities::MPI::mpi_comm_rank();
  
  std::string mesh_file = "mesh.dat";
  std::string train_file = "train.dat";
  std::string test_file = "test.dat";
  
  EQPDriver eqpd;
  DGEQPConstructor<double>& dg_eqp_c = eqpd.dg_eqp_c;

  //eqpd.load_naca_orig();
  //eqpd.eqn.s_shift = 0;
  

#ifdef ONERA
  eqpd.load_onera_mesh();
#else
  eqpd.load_naca_how_mesh();
#endif

#ifdef WITH_MPI
    eqpd.mesh.prepare_parallelization();
    eqpd.mesh.execute_repartition();
    eqpd.mesh_geom.execute_repartition();
    eqpd.mesh.finalize_repartition();
#endif  
  
  eqpd.set_fe_solver();
  eqpd.set_rb_solver();
  eqpd.setup_eqp();

  if ((eqpd.Xi_train.n_elem == eqpd.Xi_test.n_elem) && 
      (arma::norm(arma::vectorise(eqpd.Xi_train - eqpd.Xi_test)) == 0)) {
    if (Utilities::MPI::mpi_comm_rank() == 0) {
      printf("setting test = train\n");
    }
    test_file = train_file;
  }

  bool pod_test = false;
  if (pod_test) {
  bool restart = false;
  
  
  if (restart) {
    eqpd.load_mesh(mesh_file);
    eqpd.load_training_data(train_file);
    eqpd.load_test_data(test_file);    
    dg_eqp_c.init();

    std::cout << "n_dofs = " <<  dg_eqp_c.fe_field()->n_dofs() << std::endl;
    std::cout << "n_train = " << eqpd.Xi_train.n_cols << std::endl;
    std::cout << "n_test = " << eqpd.Xi_test.n_cols << std::endl;

  } else {
    arma::Col<double> mu = eqpd.Xi_train.col(0);
    mu(0) = 3.0*M_PI/180.0;
    mu(1) = 0.5;
    eqpd.eqn.set_parameters(mu);    
    eqpd.run_adaptive_fe_solve();
    
    return 0;
    
    const BlockVector<double>* fe_init_vec = dg_eqp_c.adapt()->state_vector();
    dg_eqp_c.set_test_init_fe_state(fe_init_vec);

    std::cout << "solving for training states" << std::endl;
    dg_eqp_c.solve_training_states();

    std::cout << "solving for test states" << std::endl;
    dg_eqp_c.solve_test_states();
    
    eqpd.save_mesh(mesh_file);
    eqpd.save_training_data(train_file);
    eqpd.save_test_data(test_file);
  }


  dg_eqp_c.pod_training_states();

  std::cout << "n_rb = " << dg_eqp_c.n_reduced_basis() << std::endl;
  
  dg_eqp_c.compute_eqp_weights();

  DGEQPEvaluator<double> dg_eqp_e;
  dg_eqp_c.set_evaluator(&dg_eqp_e);
  dg_eqp_c.construct_eqp_evaluator(&dg_eqp_e);
  
  // running test
  arma::Col<double> rb_init_vec;
  BlockVector<double>* fe_init_vec = dg_eqp_c.training_state(0);
  dg_eqp_c.project_fe_vector_onto_rb(fe_init_vec, rb_init_vec);  
  dg_eqp_c.set_test_init_rb_state(rb_init_vec);
  dg_eqp_c.set_test_init_fe_state(fe_init_vec);

  // BlockVector<double>* alpha_rb = dg_eqp_e.allocate_vector();
  // BlockVector<double>* res_rb = dg_eqp_e.allocate_vector();
  // for (unsigned int i = 0; i < alpha_rb->n(); ++i) {
  //   alpha_rb->value(i) = rb_init_vec(i);
  // }
  // dg_eqp_e.compute_residual(alpha_rb, res_rb, nullptr);


  dg_eqp_c.run_test();
  dg_eqp_c.print_greedy_test_history();

  } else {

    // //prepare the initial mesh
    // arma::Col<double> mu = eqpd.Xi_train.col(0);
    // mu(0) = 3.5*M_PI/180.0;
    // mu(1) = 0.5;
    // eqpd.eqn.set_parameters(mu);
    // dg_eqp_c.adapt()->solve();
    // return 0; 

    // mu(0) = 1.5*M_PI/180.0;
    // mu(1) = 0.3;
    // eqpd.eqn.set_parameters(mu);
    // dg_eqp_c.adapt()->solve();

    eqpd.run_weak_greedy();
    
  }
  // BlockVector<double>* alpha_rb = dg_eqp_e.allocate_vector();
  // BlockVector<double>* res_rb = dg_eqp_e.allocate_vector();
  // for (unsigned int i = 0; i < alpha_rb->n(); ++i) {
  //   alpha_rb->value(i) = rb_init_vec(i);
  // }
  //dg_eqp_e.compute_residual(alpha_rb, res_rb, nullptr);




  


#ifdef WITH_MPI
  Utilities::MPI::mpi_finalize();
#endif

  return 0;

  // if (1) {
  //   FEField* fe_field = dg_eqp_c.fe_field();
  //   for (unsigned int irb = 0; irb < dg_eqp_c.n_reduced_basis(); ++irb) {
  //     VtkIO vtkio(fe_field, &mesh_geom, dg_eqp_c.reduced_basis(irb));
  //     std::string rb_file_name = "rb" + std::to_string(irb);
  //     vtkio.set_n_refine(1);
  //     vtkio.write_volume_data(rb_file_name);
  //   }
  // }
 

  // //Xi_test = Xi_train;
  // //n_test = Xi_test.n_cols;
  
  // // loop over test points
  // DG<double>* dg = dg_eqp_c.dg();
  // BlockVector<double>* alpha_rb = dg_eqp_e.allocate_vector();
  // BlockVector<double>* u_rb = dg->allocate_vector();
  // BlockVector<double>* u_fe = dg->allocate_vector();
  // BlockVector<double>* u_diff = dg->allocate_vector();
  // BlockVector<double>* res = dg->allocate_vector();
  // BlockSparseMatrix<double>* mass_mat = dg->allocate_matrix();
  // BlockVector<double>* temp = dg->allocate_vector();

  // ptc.set_verbosity(-1);

  // dg_eqp_e.fill_vector(eqn.initial_state(), alpha_rb);
  // u_fe->equal(1.0, dg_eqp_c.adapt()->state_vector());
  // //dg->fill_vector(eqn.initial_state(), u_rb);
  // //dg->fill_vector(eqn.initial_state(), u_fe);

  // dg->add_temporal_flux(u_fe, u_diff, mass_mat); // u_diff is a dummy

  // std::vector<Stopwatch> sws(2);

  // // initialization
  // {
  //   printf("initialization");
  //   ptc.solve(u_fe);
  //   dg_eqp_c.project_fe_vector_onto_rb(u_fe, alpha_rb->values_vector());
  //   printf("...done\n");
  // }
  
  // for (unsigned int itest = 0; itest < n_test; ++itest) {
  //   eqn.set_parameters(Xi_test.col(itest));

  //   // rb solve
  //   sws[0].restart();
  //   ptc_rb.reset_iteration_count();
  //   ptc_rb.solve(alpha_rb);
  //   sws[0].stop();
    
  //   dg_eqp_c.rb_coeff_to_fe_vector(alpha_rb, u_rb);
  //   double j_rb;
  //   dg->compute_output(0, u_rb, j_rb);
    
  //   // fe solve
  //   sws[1].restart();
  //   //u_fe->equal(1.0, u_rb);
  //   ptc.solve(u_fe);
  //   sws[1].stop();

  //   double j_fe;
  //   dg->compute_output(0, u_fe, j_fe);
    
  //   // compute the L2 norm of difference
  //   mass_mat->mat_vec_mult(u_fe, temp);
  //   double u_norm = sqrt(u_fe->dot(temp));
  //   u_diff->equal(1.0, u_fe);
  //   u_diff->add(-1.0, u_rb);
  //   mass_mat->mat_vec_mult(u_diff, temp);
  //   double err = sqrt(u_diff->dot(temp));

  //   dg->compute_residual(u_rb, res, nullptr); // we will use u_diff
  //   dg->multiply_inv_broken_h1_matrix(res, temp);
  //   double res_err = sqrt(res->dot(temp));
    
    
  //   printf("%d %.6e %.6e %.6e %.6e\n", itest, err, u_norm, err/u_norm, res_err);
  //   printf("%d %.6e %.6e %.6e\n", itest, j_fe, j_rb, (j_fe-j_rb)/j_fe);
  //   printf("%.6e %.6e\n", sws[0].last_split_time(), sws[1].last_split_time());
    
  // } // end of test loop

  
  
  // // MC sampling
  // if (1) {
  // const unsigned int n_mc = 1000;  
  // arma::Mat<double> Xi_mc;
  // arma::arma_rng::set_seed(0);
  // eqn.generate_random_Xi(n_mc, Xi_mc);
  // arma::Col<double> out_mc(n_mc);

  // std::ofstream mf;
  // mf.open("out_mc.txt");
  
  // for (unsigned int i_mc = 0; i_mc < n_mc; ++i_mc) {
  //   eqn.set_parameters(Xi_mc.col(i_mc));

  //   // rb solve
  //   ptc_rb.reset_iteration_count();
  //   ptc_rb.solve(alpha_rb);    
  //   dg_eqp_c.rb_coeff_to_fe_vector(alpha_rb, u_rb);
  //   double j_rb;
  //   dg->compute_output(0, u_rb, j_rb);

  //   out_mc(i_mc) = j_rb;

  //   mf << std::setprecision(15) << out_mc[i_mc] << '\n';
  // }

  // mf.close();
  // }
  
  // printf("time: %.6e %.6e\n", sws[0].elapsed_time(), sws[1].elapsed_time());

  // if (1) {
  //   FEField* fe_field = dg_eqp_c.fe_field();
    
  // VtkIO vtkio(fe_field, &mesh_geom, u_rb);
  // vtkio.set_equation(&eqn);
  // vtkio.set_n_refine(1);
  // vtkio.write_volume_data("state_rb");

  // }

  // delete alpha_rb;
  // delete u_rb;
  // delete u_fe;
  // delete u_diff;
  // delete res;
  // delete temp;
  // delete mass_mat;
  
  // return 0;
}

  
