#include <vector>

#include <cmath>
#include <armadillo>

#define WITH_MPI

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

#include "mesh/parallel_mesh.h"
#include "mesh/parallel_mesh_geometry.h"
#include "mesh/parallel_fe_field.h"

#include "disc/dg_adaptation.h"
a
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
  const unsigned int param_mode = 2;

  // internal parameters
  std::vector<unsigned int> far_bnds;
  unsigned int foil_bnd;
  NSUtils* nsutils_;
  arma::Col<double> mu_;

public:
  ParametrizedNS(unsigned int dim, const NSVariableType var_type_in, NSUtils* nsutils)
    : NavierStokes(var_type_in),
      nsutils_(nsutils)
  {
    // set the parameter bound
    // this function can be modified to change the parameter range
    arma::Row<double> alpha_bnd = {{0*M_PI/180.0, 5.0*M_PI/180.0}};
    arma::Row<double> M_bnd = {{0.3,0.5}};
    mu_bnd_.set_size(n_parameters(),2);
    switch (param_mode) {
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
    default:
      Error("unknown mode");
    }
  }

  virtual unsigned int n_parameters() const
  {
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
    // Map the parameter to the flow variables.  The actual mapping depends on the parameter mode.
    alpha = 1.0*M_PI/180;
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
    // set the boundary conditions based on the parameters specified.
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
    arma::Col<double> u_init = nsutils_->state();
    set_initial_state(u_init);

    // set boundary conditions
    std::vector<double> ub = arma::conv_to<std::vector<double>>::from(u_init);
    for (unsigned int i = 0; i < far_bnds.size(); ++i) {
      set_boundary_type(far_bnds[i],NSBoundaryType::full_state);
      set_boundary_parameters(far_bnds[i],ub);
    }
    set_boundary_type(foil_bnd,NSBoundaryType::flow_tangency);

    // lift output
    set_output_type(0,NSOutputType::directional_force);
    std::vector<unsigned int> bids = {foil_bnd};
    set_output_boundary_ids(0,bids);
    const double qinf = nsutils_->dynamic_pressure();
#ifdef ONERA
    std::vector<double> bparams = {-sin(alpha)/qinf,0.0,cos(alpha)/qinf};
    set_output_parameters(0,bparams);
#else
    std::vector<double> bparams = {-sin(alpha)/qinf,cos(alpha)/qinf,0.0};
    set_output_parameters(0,bparams);
#endif
  }

  virtual const arma::Mat<double>& parameter_domain() const
  {
    return mu_bnd_;
  }

private:
  arma::Mat<double> mu_bnd_;

};

// Class that drives the actual computation
class EQPDriver {
public:
  EQPDriver()
    : dim(2),
      geom_degree(4),
      nstype(NSVariableType::entropy),
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
    //dg_eqp_c.set_pod_tolerance(1e-10);
    dg_eqp_c.set_eqp_tolerance(1e-4);
    dg_eqp_c.set_greedy_target_type(DGEQPConstructor<double>::GreedyTargetType::output);
    dg_eqp_c.set_eqp_form(EQPForm::elem_stable);
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
    //Xi_test = Xi_train;

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
    ptc.set_initial_timestep(1e-2);
    ptc.set_timestep_increase_multiplier(4.0);
    ptc.set_abs_tol(1e-9);
    ptc.set_linear_solver_decrease_multiplier(1e-3);
    ptc.set_line_search_maximum_physical_change(0.3);

    // the below should not be needed for the NACA problem
    //ptc.set_line_search_type(PTCSolver<double>::LineSearchType::unsteady);
    //ptc.set_line_search_maximum_residual_increase(1.02);

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

  eqpd.set_fe_solver();
  eqpd.set_rb_solver();
  eqpd.setup_eqp();

  if ((eqpd.Xi_train.n_elem == eqpd.Xi_test.n_elem) &&
      (arma::norm(arma::vectorise(eqpd.Xi_train - eqpd.Xi_test)) == 0)) {
    if (Utilities::MPI::mpi_comm_rank() == 0) {
      printf("setting test = train\n");
    }
  }

  // run the weak greedy algorithm
  dg_eqp_c.set_adaptive_eqp_training(true);
  eqpd.run_weak_greedy();

#ifdef WITH_MPI
  Utilities::MPI::mpi_finalize();
#endif

  return 0;
}

/*
int main(int argc, char *argv[])
{
  // initialize MPI and define useful variables
  Utilities::MPI::mpi_init(argc, argv);
  unsigned int comm_rank = Utilities::MPI::mpi_comm_rank();

  // Set basic parameters.  Parameters are set for p=2 anisotropic-h adaptation. Change poly_degree and adapt_type parameters to consider other polynomial degrees or adaptation types.
  const int dim = 2;
  const MeshEntityType etype = MeshEntityType::quad;  
  const int geom_degree = 4;
  FiniteElementType fe_type = FiniteElementType::Legendre;
  const unsigned int poly_degree = 2;
  const auto adapt_type = Adaptation<double>::type::anisotropic_h;
  const unsigned int n_adapt_iter = 15;
  
  // flow parameters
  double M = 0.2;  
  double alpha = 0.5*M_PI/180;
  double Re = 5e3;

  // initialize the Navier-Stokes utility class.  The class will help us set parameters and boundary conditions.
  NSVariableType nstype = NSVariableType::entropy;
  NSUtils nsutils(dim, nstype, NSUtils::Equation::navier_stokes, NSUtils::Normalization::unit_rho_a);
  nsutils.set_mach_number(M);
  nsutils.set_angle_of_attack(alpha);
  nsutils.set_reynolds_number(Re);
  nsutils.set_reference_temperature_kelvin(273.15); // only used for nonconst visc
  nsutils.set_sutherland_constant_kelvin(110.4); // only used for nonconst visc

  // initialize Navier-Stokes equations
  NavierStokes<dim> eqn(nstype);  
  eqn.set_dynamic_viscosity(nsutils.dynamic_viscosity());
  eqn.set_constant_viscosity(true);
  eqn.set_reference_temperature(nsutils.reference_temperature()); // only used for nonconst visc
  eqn.set_sutherland_constant(nsutils.sutherland_constant()); // only used for nonconst visc
  arma::Col<double> u_init = nsutils.state();
  eqn.set_initial_state(u_init);

  // set shock capturing.  We do not need shock capturing for this case.
  eqn.set_shock_capturing(false);
  eqn.set_shock_viscosity_type(Euler<dim>::shock_viscosity_type::physical);

  // set boundary conditions
  std::vector<double> ub = arma::conv_to<std::vector<double>>::from(u_init);  
  eqn.set_n_boundaries(3);
  // inflow
  eqn.set_boundary_type(1,NSBoundaryType::full_state);
  eqn.set_boundary_parameters(1,ub);
  // outflow
  eqn.set_boundary_type(2,NSBoundaryType::full_state);
  eqn.set_boundary_parameters(2,ub);
  // foil
  eqn.set_boundary_type(0,NSBoundaryType::adiabatic_wall);
  
  // output is the drag on the foil.  Note that this is the directional force in the direction of the flow.
  eqn.set_n_outputs(1);
  eqn.set_output_type(0,NSOutputType::directional_force);
  std::vector<unsigned int> bids = {0}; 
  eqn.set_output_boundary_ids(0,bids);
  const double qinf = nsutils.dynamic_pressure();
  std::vector<double> bparams = {cos(alpha)/qinf,sin(alpha)/qinf,0.0};
  eqn.set_output_parameters(0,bparams);

  // initialize FE and quadrature set
  FESet fe_set;
  QuadratureSet quad_set;

  // generate mesh
  ParallelMesh mesh(dim);
  ParallelMeshGeometry mesh_geom(&mesh,&fe_set,geom_degree);
  NacaMeshGenerator naca_gen; // the mesh generator will construct a C mesh
  naca_gen.set_naca_digits("0012");
  naca_gen.set_n_foil_elements(7); // # elements on foil along flow direction
  naca_gen.set_n_tail_elements(3); // # elements after foil along flow direction
  naca_gen.set_n_radial_elements(5); // # elements in radial direction
  naca_gen.set_first_element_radial_spacing(0.1/4); 
  naca_gen.set_sqrt_fitting(true); // use sqrt relation for wall-normal spacing
  naca_gen.generate_mesh(mesh, mesh_geom);

  // paralellize the mesh and geometry
  mesh.prepare_parallelization();
  mesh.execute_repartition();
  mesh_geom.execute_repartition();
  mesh.finalize_repartition();

  // set linear solver
  // linear solver is GMRES
  // local preconditioner is block ILU0 with minimum
  GMRESSolver<double> gmres;
  AdditiveSchwarz<double> addsch;
  addsch.set_type(AdditiveSchwarz<double>::Type::restricted);
  BlockILU<double> ilu;
  ilu.set_ordering(BlockOrdering<double>::Type::MDF);
  addsch.set_local_preconditioner(&ilu);
  gmres.set_preconditioner(&addsch);
  gmres.set_max_inner_iterations(200);
  gmres.set_max_outer_iterations(2);
  gmres.set_verbosity(0);

  // set nonlinear solver
  // nonlinear solver is pseudo-time continuation solver with unsteady line search
  PTCSolver<double> ptc;
  ptc.set_linear_solver(&gmres);
  ptc.set_max_iter(150);
  ptc.set_initial_timestep(1e-1);
  ptc.set_timestep_increase_multiplier(4.0);
  ptc.set_abs_tol(1e-9);
  ptc.set_line_search_type(PTCSolver<double>::LineSearchType::unsteady);
  ptc.set_line_search_maximum_physical_change(0.2); 
  ptc.set_line_search_maximum_residual_increase(1.02);
  ptc.set_linear_solver_decrease_multiplier(1e-6);

#ifdef ADAPT
  // adaptive solve
  
  // set adjoint solver
  AdjointSolver<double> adjoint_solver;
  adjoint_solver.set_linear_solver(&gmres);
  adjoint_solver.set_residual_tolerance(1e-9);

  // set adaptation parameters
  Adaptation<double> adapt(&eqn, &mesh, &mesh_geom, &fe_set, &quad_set);
  adapt.set_primal_solver(&ptc);
  adapt.set_adjoint_solver(&adjoint_solver);
  adapt.set_max_iterations(n_adapt_iter);
  adapt.set_fe_type(fe_type);
  adapt.set_initial_polynomial_degree(poly_degree);
  adapt.set_output_file_name("naca");
  adapt.set_adaptation_type(adapt_type);
  adapt.solve();

#else
  // single-solve branch

  // initialize FE field
  ParallelFEField fe_field(&mesh,&fe_set,dim,eqn.n_components(),fe_type,poly_degree);

  // initialize DG discretization
  DG<double> dg(&eqn,&fe_set,&quad_set,&fe_field,&mesh_geom);

  // initialize state vector
  BlockVector<double>* state_vec = dg.allocate_vector();
  dg.fill_vector(u_init, state_vec);

  // set the discretization
  ptc.set_discretization(&dg);

  // solve the nonlinear problem
  ptc.solve(state_vec);

  // compute output
  double output;
  dg.compute_output(0,state_vec,output);
  if (comm_rank == 0) {
    printf("output = %22.15e\n", output);
  }

  // output the solution
  VtkIO vtkio(&fe_field, &mesh_geom, state_vec);
  vtkio.set_n_refine(1);
  vtkio.set_equation(&eqn);
  vtkio.write_volume_data("naca");   
  
  // free state vector
  delete state_vec;  

#endif
    
  // finalize MPI
  Utilities::MPI::mpi_finalize();
  return 0;
}
*/
