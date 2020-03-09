#include <vector>

#include <armadillo>
#include <cmath>

#define WITH_MPI

#include "base/mpi_wrapper.h"
#include "base/serialization.h"
#include "base/quadrature.h"
#include "disc/dg.h"
#include "disc/dg_adaptation.h"
#include "fe/fe.h"
#include "mesh/mesh.h"
#include "mesh/fe_cache.h"
#include "mesh/mesh_generator.h"
#include "mesh/mesh_iterator.h"
#include "mesh/mesh_entity.h"
#include "mesh/parallel_mesh.h"
#include "mesh/parallel_mesh_geometry.h"
#include "mesh/parallel_fe_field.h"
#include "solver/additive_schwarz.h"
#include "solver/adjoint_solver.h"
#include "solver/gmres_solver.h"
#include "solver/ptc_solver.h"
#include "solver/block_ilu.h"
#include "solver/block_sparse_matrix.h"
#include "io/vtk_io.h"
#include "io/gmsh.h"
#include "eqn/rans_sa.h"
#include "eqn/navier_stokes_utils.h"


// transonic RANS flow over an RAE2822 airfoil
//
// This case is setup for a parallel run. The cmake configuration line should look something like the following
//
// $ cmake -DAPSINC=~/aps/include -DAPSLIB=~/aps/build_mpi_opt/lib/libaps.so -DBLAS=Intel ..
//
// where ~/aps/build_mpi_opt is the build directory for the MPI version of the APS library with optimization (i.e. a RELEASE build).  The code should then be run with a command
//
// $ mpirun -np 4 --bind-to core ./rae
//
// where the parameter for "-np" should be adjusted based on the number of available cores.

int main(int argc, char *argv[])
{
  // initialize MPI and define useful variables
  Utilities::MPI::mpi_init(argc, argv);
  unsigned int comm_rank = Utilities::MPI::mpi_comm_rank();

  // Set basic parameters.  Parameters are set for p=2 anisotropic-h adaptation. Change poly_degree and adapt_type parameters to consider other polynomial degrees or adaptation types.
  const unsigned int dim = 2;
  MeshEntityType entity_type = MeshEntityType::quad;
  const unsigned int poly_degree = 2; // polynomial degree
  const unsigned int geom_degree = 4; // geom degree is compatible with the high-order mesh
  const unsigned int n_adapt_iter = 20; // number of adaptation iterations
  const auto adapt_type = Adaptation<double>::type::anisotropic_h;   
  const FiniteElementType fe_type = FiniteElementType::Legendre;  

  // initial mesh
  std::string gmsh_file = "../rae2822_level5.m4.msh";

  // flow parameters
  double M = 0.734;  // Mach number
  double alpha = 2.79*M_PI/180.0;  // angle of attack (in radians)
  double Re = 6.5e6;  // Reynolds number
  double fs_chi = 3.0;  // freestream chi for SA equations
  double sa_scale = sqrt(Re);  // SA scale

  // initialize the Navier-Stokes utility class.  The class will help us set parameters and boundary conditions.  
  NSVariableType nstype = NSVariableType::entropy;
  NSUtils nsutils(dim, nstype, NSUtils::Equation::rans_sa, NSUtils::Normalization::unit_rho_a);
  nsutils.set_mach_number(M);
  nsutils.set_angle_of_attack(alpha);
  nsutils.set_reynolds_number(Re);
  nsutils.set_reference_temperature_kelvin(273.15); // only used for nonconst visc
  nsutils.set_sutherland_constant_kelvin(110.4); // only used for nonconst visc
  nsutils.set_freestream_chi(fs_chi);
  nsutils.set_sa_scale(sa_scale);

  // initialize the RANS equations
  RANSSA<dim> eqn(nstype);
  arma::Col<double> u_init = nsutils.state();
  eqn.set_freestream_chi(fs_chi);
  eqn.set_sa_scale(sa_scale);
  eqn.set_initial_state(u_init);
  eqn.set_dynamic_viscosity(nsutils.dynamic_viscosity());
  eqn.set_constant_viscosity(true);
  eqn.set_reference_temperature(nsutils.reference_temperature()); // only used for nonconst visc
  eqn.set_sutherland_constant(nsutils.sutherland_constant()); // only used for nonconst visc

  // set shock capturing.  For this transonic case, shock capturing is actually not required if the entropy variables are used.
  eqn.set_shock_capturing(false);
  eqn.set_shock_viscosity_type(Euler<dim>::shock_viscosity_type::physical);

  // set boundary conditions
  std::vector<double> ub = arma::conv_to<std::vector<double>>::from(u_init);
  eqn.set_n_boundaries(4);
  // inflow
  eqn.set_boundary_type(2,NSBoundaryType::full_state);
  eqn.set_boundary_parameters(2,ub);
  // outflow
  eqn.set_boundary_type(3,NSBoundaryType::full_state);
  eqn.set_boundary_parameters(3,ub);
  // plane
  eqn.set_boundary_type(1,NSBoundaryType::adiabatic_wall);
  std::vector<bool> wall_marker = {false, true, false, false}; // boundary ID 1 is the wall

  // output is drag on the foil.  Note that this is the directional force in the direction of the flow.
  eqn.set_n_outputs(1);
  eqn.set_output_type(0,NSOutputType::directional_force);
  std::vector<unsigned int> bids = {1};
  eqn.set_output_boundary_ids(0,bids);
  const double qinf = nsutils.dynamic_pressure();
  std::vector<double> bparams = {cos(alpha)/qinf,sin(alpha)/qinf,0.0};
  eqn.set_output_parameters(0,bparams);

  // initilize FE and quadrature set
  FESet fe_set;
  QuadratureSet quad_set;

  // extract surface mesh; the surface mesh is needed for RANS wall-distance calculation
  Mesh surf_mesh(dim-1,dim);
  MeshGeometry surf_geom(&surf_mesh,&fe_set,geom_degree);
  {
    Gmsh gmsh;    
    Mesh mesh(dim);
    MeshGeometry mesh_geom(&mesh,&fe_set,geom_degree);

    // read the gmesh on the root process and then broadcast
    if (comm_rank == 0) {
      gmsh.read_msh_file(gmsh_file);
    }
    gmsh.mpi_bcast();

    // load the mesh and extract the surface mesh 
    gmsh.load_mesh(mesh,mesh_geom);  
    mesh_geom.set_wall_marker(wall_marker);
    mesh_geom.extract_surface_geometry(wall_marker, surf_mesh, surf_geom);
  }

  // load the mesh
  Gmsh gmsh;
  ParallelMesh mesh(dim);
  ParallelMeshGeometry mesh_geom(&mesh,&fe_set,geom_degree);
  if (comm_rank == 0) {
    gmsh.read_msh_file(gmsh_file);
    printf("nelements = %d\n", gmsh.n_elems);
  }
  gmsh.load_mesh(mesh, mesh_geom);

  // parallelize the mesh and mesh geometry
  mesh.prepare_parallelization();
  mesh.execute_repartition();
  mesh_geom.execute_repartition();
  mesh.finalize_repartition();

  // set wall geometry and initialize wall distance
  mesh_geom.set_wall_geometry(surf_mesh, surf_geom);
  mesh_geom.init_wall_distance();

  // set linear solver
  // linear solver is GMRES
  // local preconditioner is block ILU0 with minimum-discarded fill reordering
  // global preconditioner is additive Schwarz
  GMRESSolver<double> gmres;
  AdditiveSchwarz<double> addsch;
  addsch.set_type(AdditiveSchwarz<double>::Type::restricted);  
  BlockILU<double> ilu;
  addsch.set_local_preconditioner(&ilu);
  ilu.set_ordering(BlockOrdering<double>::Type::MDF);
  gmres.set_preconditioner(&addsch);
  gmres.set_max_inner_iterations(500);
  gmres.set_max_outer_iterations(2);
  gmres.set_verbosity(0);

  // set nonlinear solver
  // nonlinear solver is pseudo-time continuation solver with unsteady line search
  PTCSolver<double> ptc;
  ptc.set_linear_solver(&gmres);
  ptc.set_max_iter(500);
  ptc.set_abs_tol(1e-8);  
  ptc.set_linear_solver_decrease_multiplier(1e-6);
  ptc.set_timestep_increase_multiplier(1.5);
  ptc.set_initial_timestep(1e-4);
  ptc.set_maximum_timestep(1e8);
  ptc.set_minimum_timestep(1e-8);
  ptc.set_line_search_type(PTCSolver<double>::LineSearchType::unsteady);
  ptc.set_line_search_maximum_physical_change(0.2); 
  ptc.set_line_search_maximum_residual_increase(1.02);

  // set adjoint solver
  // we will use the same linear solver for the adjoint problem
  AdjointSolver<double> adjoint_solver;
  adjoint_solver.set_linear_solver(&gmres);
  adjoint_solver.set_residual_tolerance(1e-8);
  adjoint_solver.set_max_iter(10);

  // set adaptation parameters
  Adaptation<double> adapt(&eqn, &mesh, &mesh_geom, &fe_set, &quad_set);
  adapt.set_primal_solver(&ptc);
  adapt.set_adjoint_solver(&adjoint_solver);
  adapt.set_adaptation_type(adapt_type);
  adapt.set_max_iterations(n_adapt_iter);
  adapt.set_fe_type(fe_type);
  adapt.set_refinement_fraction(0.1);
  adapt.set_initial_polynomial_degree(poly_degree);
  adapt.set_output_file_name("rae");

  // solve the problem adaptively
  adapt.solve();

  // finalize MPI  
  Utilities::MPI::mpi_finalize();
  
  return 0;

}
