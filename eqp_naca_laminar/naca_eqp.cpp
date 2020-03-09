#include <vector>

#include <cmath>
#include <armadillo>

#define WITH_MPI
#define WITH_FFD

#include "base/mpi_wrapper.h"
#include "base/quadrature.h"
#include "disc/dg.h"
#include "disc/dg_eqp.h"
#include "fe/fe.h"
#include "fe/fe_definition.h"
#include "mesh/mesh.h"
#include "mesh/fe_cache.h"
#include "mesh/fe_field.h"
#include "mesh/mesh_generator.h"
#include "solver/mumps_solver.h"
#include "solver/ptc_solver.h"
#include "solver/block_sparse_matrix.h"
#include "io/vtk_io.h"
#include "io/gmsh.h"
#include "eqn/navier_stokes.h"
#include "eqn/navier_stokes_utils.h"

#include "solver/dense_solver.h"
#include "solver/gmres_solver.h"
#include "solver/block_ilu.h"
#include "solver/additive_schwarz.h"

#ifdef WITH_MPI
#include "mesh/parallel_mesh.h"
#include "mesh/parallel_fe_field.h"
#include "mesh/parallel_mesh_geometry.h"
#else
#include "mesh/mesh.h"
#include "mesh/fe_field.h"
#include "mesh/mesh_geometry.h"
#endif
#include "mesh/ffd.h"

#include "disc/dg_adaptation.h"

// subsonic laminar flow over an NACA0012 airfoil
//
// This case is setup for a parallel run.  The cmake configuration line should look something like the following
//
// $ cmake -DAPSINC=~/aps/include -DAPSLIB=~/aps/build_mpi_opt/lib/libaps.so -DBLAS=Intel ..
//
// where ~/aps/build_mpi_opt is the build directory for the MPI version of the APS library with optimization (i.e. a RELEASE build).  The code should then be run with a command
//
// $ mpirun -np 4 --bind-to core ./rae
//
// where the parameter for "-np" should be adjusted based on the number of available cores.

// Set (i.e. define) the ADAPT flag to perform adaptive analysis.  Otherwise comment out hte line to run the single-solve branch.
#define ADAPT

class ParametrizedNS : public NavierStokes<2> {

public:
  // mode
  // 0 : alpha only
  // 1 : M only
  // 2 : alpha and M
  // 3 : alpha, Re, and M
  // 4 : alpha, Re, and M, with FFD

  const unsigned int param_mode_ = 4;

  // internal parameters
  std::vector<unsigned int> far_bnds_;
  unsigned int foil_bnd_;
private:
  NSUtils* nsutils_;
  arma::Col<double> mu_;
  arma::Mat<double> mu_bnd_;
public:
  FFD* ffd_;
  
public:
  ParametrizedNS(unsigned int dim, const NSVariableType var_type_in, NSUtils* nsutils, FFD* ffd = nullptr)
    : NavierStokes(var_type_in),
      nsutils_(nsutils),
      ffd_(ffd)
  {}

  /** initialize with default parameter bounds */
  void init_default_parameter_bound()
  {
    // this section can be modified to change the parameter range
    arma::Row<double> alpha_bnd = {{0*M_PI/180.0, 5.0*M_PI/180.0}};
    arma::Row<double> M_bnd = {{0.3,0.5}};
    arma::Row<double> Re_bnd = {{5e3, 8e3}};
    arma::Row<double> geom_delta_bnd = {{-0.2, 0.2}};
    mu_bnd_.set_size(n_parameters(),2);
    switch (param_mode_) {
    case 0:
      mu_bnd_.row(0) = alpha_bnd;
      break;
    case 1:
      mu_bnd_.row(0) = M_bnd;
      break;
    case 2:
      mu_bnd_.row(0) = alpha_bnd;
      mu_bnd_.row(1) = M_bnd;
      break;
    case 3:
      mu_bnd_.row(0) = alpha_bnd;
      mu_bnd_.row(1) = M_bnd;
      mu_bnd_.row(2) = Re_bnd;
      break;
    case 4:
      mu_bnd_.row(0) = alpha_bnd;
      mu_bnd_.row(1) = M_bnd;
      mu_bnd_.row(2) = Re_bnd;
      for (unsigned int i = 3; i < n_parameters(); ++i) {
        mu_bnd_.row(i) = geom_delta_bnd;
      }
      break;
    default:
      Error("unknown mode");
    }
  }


  virtual unsigned int n_parameters() const
  {
    switch (param_mode_) {
    case 0:
    case 1:
      return 1;
    case 2:
      return 2;
    case 3:
      return 3;
    case 4:
      return 3 + ffd_->get_n_dof();
    default:
      Error("unsupported mode");
    }
  }

  void flow_vars(double& alpha, double& M, double& Re)
  {
    // Map the parameter to the flow variables.  The actual mapping depends on the parameter mode.
    alpha = 1.0*M_PI/180;
    M = 0.3;
    Re = 5e3;
    switch (param_mode_) {
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
    case 3:
    case 4:
      alpha = mu_(0);
      M = mu_(1);
      Re = mu_(2);
      break;
    default:
      Error("unknown mode");
    }
  }

  virtual void set_parameters(const arma::Col<double> mu) {
    // set the boundary conditions based on the parameters specified.
    double M;
    double alpha;
    double Re;
    mu_ = mu;
    flow_vars(alpha,M,Re);

    // set initial condition
    nsutils_->set_mach_number(M);
    nsutils_->set_angle_of_attack(alpha);
    nsutils_->set_reynolds_number(Re);
    nsutils_->set_reference_temperature_kelvin(273.15); // only used for nonconst visc
    nsutils_->set_sutherland_constant_kelvin(110.4); // only used for nonconst visc
      
    arma::Col<double> u_init = nsutils_->state();
    set_initial_state(u_init);

    // set all boundary conditions to full_state
    std::vector<double> ub = arma::conv_to<std::vector<double>>::from(u_init);
    for (unsigned int i = 0; i < far_bnds_.size(); ++i) {
      set_boundary_type(far_bnds_[i],NSBoundaryType::full_state);
      set_boundary_parameters(far_bnds_[i],ub);
    }
    // re-set foil boundary to adiabatic_wall
    set_boundary_type(foil_bnd_,NSBoundaryType::adiabatic_wall);

    // output is the drag on the foil.  Note that this is the directional force in the direction of the flow.
    set_output_type(0,NSOutputType::directional_force);
    std::vector<unsigned int> bids = {foil_bnd_};
    set_output_boundary_ids(0,bids);
    const double qinf = nsutils_->dynamic_pressure();
    std::vector<double> bparams = {cos(alpha)/qinf,sin(alpha)/qinf,0.0};
    set_output_parameters(0,bparams);

    // set ffd parameters
    if (param_mode_ == 4) {
      ffd_->set_transformed_control_point(mu.subvec(3,3 + ffd_->get_n_dof() - 1),1);
    }
  }

  virtual const arma::Mat<double>& parameter_domain() const
  {
    return mu_bnd_;
  }

};

// Class that drives the actual computation
class EQPDriver {
public:
  EQPDriver()
    : dim(2),
      geom_degree(4),
      nstype(NSVariableType::entropy),
      // note these settings are specific to navier_stokes eqns
      nsutils(dim,nstype,NSUtils::Equation::navier_stokes,NSUtils::Normalization::unit_rho_a),
      eqn(dim, nstype, &nsutils),
      mesh(dim),
      mesh_geom(&mesh,&fe_set,geom_degree),
      dg_eqp_c(&eqn, &mesh, &mesh_geom, &fe_set, &quad_set)
  {}

  void init_naca_c_mesh() {
    // generate mesh
    NacaMeshGenerator naca_gen; // the mesh generator will construct a C mesh
    naca_gen.set_naca_digits("0012");
    naca_gen.set_n_foil_elements(7); // # elements on foil along flow direction
    naca_gen.set_n_tail_elements(3); // # elements after foil along flow direction
    naca_gen.set_n_radial_elements(5); // # elements in radial direction
    naca_gen.set_first_element_radial_spacing(0.1/4); 
    naca_gen.set_sqrt_fitting(true); // use sqrt relation for wall-normal spacing
    naca_gen.generate_mesh(mesh, mesh_geom);

#ifdef WITH_MPI
    // paralellize the mesh and geometry
    mesh.prepare_parallelization();
    mesh.execute_repartition();
    mesh_geom.execute_repartition();
    mesh.finalize_repartition();
#endif

    // set mesh info to eqn
    eqn.set_n_boundaries(3);
    // output (drag)
    eqn.set_n_outputs(1);

    // set far-field boundary indices
    eqn.far_bnds_.resize(2);
    eqn.far_bnds_[0] = 1;
    eqn.far_bnds_[1] = 2;
    eqn.foil_bnd_ = 0;
  }
  
  void setup_eqp() {
    // DG-EQP evaluator and contructor
    dg_eqp_c.set_spatio_parameter_adaptivity(true);
    // dg_eqp_c.set_spatial_adapt_on_first_iteration_only(true);
    dg_eqp_c.adapt()->set_adaptation_type(Adaptation<double>::type::anisotropic_h);
    
    // FE tolerance
    dg_eqp_c.adapt()->set_target_error(5e-4);
    dg_eqp_c.adapt()->set_adaptation_target_type(Adaptation<double>::target_type::output);
    dg_eqp_c.adapt()->set_max_iterations(25);
    dg_eqp_c.adapt()->set_refinement_fraction(0.15);
    dg_eqp_c.set_initial_polynomial_degree(poly_degree);

    // RB-EQP Greedy settings
    dg_eqp_c.set_n_max_reduced_basis(13);
    dg_eqp_c.set_weak_greedy_tolerance(1e-3);
    // dg_eqp_c.set_pod_tolerance(1e-10);
    dg_eqp_c.set_eqp_tolerance(1e-4);
    dg_eqp_c.set_greedy_target_type(DGEQPConstructor<double>::GreedyTargetType::output);
    dg_eqp_c.set_eqp_form(EQPForm::elem_stable);
    dg_eqp_c.set_eqp_norm(EQPNorm::l2);
    dg_eqp_c.set_include_constant_rb(false);
    dg_eqp_c.set_include_stab_rb(false);
    dg_eqp_c.set_n_eqp_smoothing_iterations(3);
    dg_eqp_c.set_eqp_verbosity(-1);
    dg_eqp_c.set_eqp_target_type(DGEQPConstructor<double>::EQPTargetType::output);
    // dg_eqp_c.set_eqp_unity_weights(true);
    dg_eqp_c.set_eqp_element_constant_constraint(true);
    dg_eqp_c.set_eqp_facet_constant_constraint(true);
    dg_eqp_c.set_eqp_min_nnz_weights(0);

    dg_eqp_c.set_write_reduced_mesh(false);
    dg_eqp_c.set_write_reduced_basis(false);

    // set solvers
    dg_eqp_c.set_p_sequencing_flag(true);
    dg_eqp_c.set_primal_fe_solver(&ptc);
    dg_eqp_c.set_primal_rb_solver(&ptc_rb);
    dg_eqp_c.adapt()->set_adjoint_solver(&adjoint_solver);
    dg_eqp_c.set_adjoint_fe_solver(&adjoint_solver);
    dg_eqp_c.set_adjoint_rb_solver(&adjoint_rb_solver);

    // set training parameters
    arma::arma_rng::set_seed(0);
    dg_eqp_c.generate_structured_parameter_set(10,Xi_train);
    dg_eqp_c.set_training_parameters(Xi_train);

    // set test parameters
    arma::arma_rng::set_seed(100);
    unsigned int n_test = 10;
    dg_eqp_c.generate_random_parameter_set(n_test, Xi_test);
   
    dg_eqp_c.set_test_parameters(Xi_test);
    dg_eqp_c.set_test_rb_truth_error(true);
    dg_eqp_c.set_test_res(true);

    if (Utilities::MPI::mpi_comm_rank() == 0) {
      printf("n train = %d\n", static_cast<int>(Xi_train.n_cols));
      printf("n test = %d\n", static_cast<int>(Xi_test.n_cols));
    }
  };

  void set_rb_solver() {
    // set primal RB solver
    ptc_rb.set_linear_solver(&dense_linsolver);
    ptc_rb.set_max_iter(50);
    ptc_rb.set_verbosity(-1);
    ptc_rb.set_abs_tol(1e-10);
    ptc_rb.set_initial_timestep(1e2);
    ptc_rb.set_timestep_increase_multiplier(4.0);
    ptc_rb.set_line_search_maximum_physical_change(0.9);

    // the below should not be needed for the NACA problem
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
    // set primal FE solver
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
    ptc.set_max_iter(500);
    ptc.set_initial_timestep(1e-1);
    ptc.set_timestep_increase_multiplier(4.0);
    ptc.set_abs_tol(1e-9);
    ptc.set_line_search_type(PTCSolver<double>::LineSearchType::unsteady);
    ptc.set_line_search_maximum_physical_change(0.2); 
    ptc.set_line_search_maximum_residual_increase(1.02);
    ptc.set_linear_solver_decrease_multiplier(1e-6);

    // set adjoint solver
    adjoint_solver.set_linear_solver(&gmres);
    adjoint_solver.set_residual_tolerance(1e-9);
    adjoint_solver.set_verbosity(-1);
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

  // void save_training_data(const std::string& file_name) {
  //   std::ofstream ofs(file_name);
  //   boost::archive::binary_oarchive oa(ofs);
  //   std::vector<double> Xi_train_vec = arma::conv_to<std::vector<double>>::from(arma::vectorise(dg_eqp_c.training_parameters()));
  //   oa << Xi_train_vec;
  //   oa << dg_eqp_c.training_states();
  //   ofs.close();
  // }

  // void load_training_data(const std::string& file_name) {
  //   const unsigned int n_param = eqn.n_parameters();

  //   std::ifstream ifs(file_name);
  //   boost::archive::binary_iarchive ia(ifs);
  //   std::vector<double> Xi_train_vec;
  //   ia >> Xi_train_vec;
  //   Xi_train = arma::Mat<double>(Xi_train_vec.data(), n_param, Xi_train_vec.size()/n_param);
  //   ia >> train_vecs;
  //   dg_eqp_c.set_training_parameters(Xi_train);
  //   dg_eqp_c.set_training_states(train_vecs);
  // }

  void save_test_data(const std::string& file_name) {
    std::ofstream ofs(file_name);
    boost::archive::binary_oarchive oa(ofs);
    std::vector<double> Xi_test_vec = arma::conv_to<std::vector<double>>::from(arma::vectorise(dg_eqp_c.test_parameters()));
    oa << Xi_test_vec;
    oa << dg_eqp_c.test_states();
    ofs.close();
  }

  void load_test_data(const std::string& file_name) {
    const unsigned int n_param = eqn.n_parameters();

    std::ifstream ifs(file_name);
    boost::archive::binary_iarchive ia(ifs);
    std::vector<double> Xi_test_vec;
    ia >> Xi_test_vec;
    Xi_test = arma::Mat<double>(Xi_test_vec.data(), n_param, Xi_test_vec.size()/n_param);
    ia >> test_vecs;
    dg_eqp_c.set_test_parameters(Xi_test);
    dg_eqp_c.set_test_states(test_vecs);
  }

  const MeshEntityType etype = MeshEntityType::quad;
  const unsigned int poly_degree = 2;
  const MeshEntityType entity_type = MeshEntityType::quad;
  const FiniteElementType fe_type = FiniteElementType::Legendre;

  unsigned int dim;
  unsigned int geom_degree;
  NSVariableType nstype;
  NSUtils nsutils;

  FESet fe_set;
  QuadratureSet quad_set;

  ParametrizedNS eqn;

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
  EQPDriver eqpd;
  DGEQPConstructor<double>& dg_eqp_c = eqpd.dg_eqp_c;

#ifdef WITH_FFD
  // setup ffd here.
  // set mesh transformation
  arma::Mat<double> bounds = {{-0.5,1.5},{-1.0,1.0}};
  std::vector<unsigned int> n_points = {3,3};
  unsigned int continuity = 0;
  // initialize ffd
  FFD ffd(eqpd.dim, bounds, n_points, continuity);

  // pass the ffd and mesh transform information
  eqpd.eqn.ffd_ = &ffd;
  eqpd.mesh_geom.set_ffd(&ffd);
#endif
  
  // call EQPDriver intialization routines
  eqpd.eqn.init_default_parameter_bound();
  eqpd.init_naca_c_mesh();
  eqpd.set_fe_solver();
  eqpd.set_rb_solver();
  eqpd.setup_eqp();

  // run the weak greedy algorithm
  dg_eqp_c.set_adaptive_eqp_training(true);
  eqpd.run_weak_greedy();

#ifdef WITH_MPI
  Utilities::MPI::mpi_finalize();
#endif

  return 0;
}

