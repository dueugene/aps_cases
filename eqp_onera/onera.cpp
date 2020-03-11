#include <vector>
#include <iostream>
#include <cmath>
#include <armadillo>

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
#include "solver/dense_solver.h"
#include "solver/ptc_solver.h"
#include "solver/newton_solver.h"
#include "solver/gmres_solver.h"
#include "solver/block_ilu.h"
#include "solver/additive_schwarz.h"
#include "solver/block_sparse_matrix.h"
#include "io/vtk_io.h"
#include "io/gmsh.h"

#ifdef WITH_MPI
#include "mesh/parallel_mesh.h"
#include "mesh/parallel_fe_field.h"
#include "mesh/parallel_mesh_geometry.h"
#else
#include "mesh/mesh.h"
#include "mesh/fe_field.h"
#include "mesh/mesh_geometry.h"
#endif

#include "eqn/rans_sa.h"
#include "eqn/navier_stokes_utils.h"

//#define FP
//#define RAE
#define ONERA
/**
 * Parametrized RANS-SA equation for UQ
 * The three parameters are as follows: 1) sigma, 2) kappa, 3) cw_3
 * param_mode 0: sigma, kappa, cw_3
 * param_mode 1: sigma, kappa, cw_3, cw_2
 * param_mode 2: kappa
 * param_mode 3: kappa, chi
 * param_mode 4: alpha
 * param_mode 5: alpha, mach
 */
#ifdef ONERA
class ParametrizedRANSSA : public RANSSA<3> {
#else
class ParametrizedRANSSA : public RANSSA<2> {
#endif
public:
  const unsigned int param_mode = 5;
  NSUtils* nsutils_;
  std::vector<unsigned int> far_bnds;
  std::vector<unsigned int> foil_bnds;
  double fs_chi_0;

  double alpha_0;
  double M_0;
  
public:
  ParametrizedRANSSA(const NSVariableType var_type_in, NSUtils* nsutils)
    : RANSSA(var_type_in),
      nsutils_(nsutils)
  {
    //arma::Row<double> alpha_bnd = {{0.0*M_PI/180.0, 3.5*M_PI/180.0}};
    arma::Row<double> sig_bnd = {{0.6,1.0}};
    arma::Row<double> kap_bnd = {{0.38,0.42}};
    arma::Row<double> cw3_bnd = {{1.75, 2.5}};
    arma::Row<double> cw2_bnd = {{0.055, 0.3525}};
    arma::Row<double> fs_chi_log_bnd = {{0.0,1.0}};
    arma::Row<double> alpha_delta_bnd = {{-1.0*M_PI/180.0, 1.0*M_PI/180.0}};
    arma::Row<double> M_delta_bnd = {{-0.1,0.1}};
    //arma::Row<double> M_delta_bnd = {{-0.15,0.15}};
    mu_bnd_.set_size(n_parameters(),2);
    switch (param_mode) {
    case 0:
      mu_bnd_.row(0) = sig_bnd;
      mu_bnd_.row(1) = kap_bnd;
      mu_bnd_.row(2) = cw3_bnd;
      break;
    case 1:
      mu_bnd_.row(0) = sig_bnd;
      mu_bnd_.row(1) = kap_bnd;
      mu_bnd_.row(2) = cw3_bnd;
      mu_bnd_.row(3) = cw2_bnd;
      break;
    case 2:
      mu_bnd_.row(0) = kap_bnd;
      break;
    case 3:
      mu_bnd_.row(0) = kap_bnd;
      mu_bnd_.row(1) = fs_chi_log_bnd;
      break;
    case 4:
      mu_bnd_.row(0) = alpha_delta_bnd;
      break;
    case 5:
      mu_bnd_.row(0) = alpha_delta_bnd;
      mu_bnd_.row(1) = M_delta_bnd;
      break;    
    default:
      Error("unknown mode");      
    }
  }

  unsigned int n_parameters() const {
    switch (param_mode) {
    case 0:
      return 3;
    case 1:
      return 4;
    case 2:
      return 1;
    case 3:
      return 2;
    case 4:
      return 1;
    case 5:
      return 2;
    default:
      Error("unsupported mode");
    }
  }
  
  virtual void set_parameters(const arma::Col<double> mu) {
    double sig = 2.0/3.0;
    double kap = 0.41;
    double cw3 = 2.0;
    double fs_chi_log = 0.0;
    double alpha = alpha_0;
    double M = M_0;
    switch (param_mode) {
    case 0:
      sig = mu(0);
      kap = mu(1);
      cw3 = mu(2);
      break;
    case 1:
      sig = mu(0);
      kap = mu(1);
      cw3 = mu(2);
      break;
    case 2:
      kap = mu(0);
      break;
    case 3:
      kap = mu(0);
      fs_chi_log = mu(1);
      break;
    case 4:
      alpha = alpha_0 + mu(0);
      break;
    case 5:
      alpha = alpha_0 + mu(0);
      M = M_0 + mu(1);
      break;
    default:
      Error("unsupported mode");
    }

    // useful variable
    double sig2 = sig*sig;
    double sig3 = sig*sig*sig;

    // variation considered by Schaefer et al AIAA2017-1710
    if (0) {
    this->cb1_ = -0.0291*sig3 + 0.0909*sig2 - 0.1095*sig + 0.1768;
    this->cb2_ = -0.1828*sig3 + 0.5309*sig2 - 0.2943*sig + 0.6357;
    this->sigma_ = sig;
    this->cv1_ = 7.1 + 37.5*(kap - 0.41);
    this->kappa_sa_ = kap;
    this->cw1_ = this->cb1_/(kap*kap) + (1.0+this->cb2_)/sig;
    if (param_mode != 1) {
      this->cw2_ = -1.5672*sig3 + 4.1858*sig2 - 4.4125*sig + 1.8478;
    } else {
      this->cw2_ = mu(3);
    }
    this->cw3_ = cw3;
    };
    
    double fs_chi = fs_chi_0*pow(10.0,fs_chi_log);      
    nsutils_->set_freestream_chi(fs_chi);
#ifdef ONERA
    nsutils_->set_angle_of_attack(0.0);
    nsutils_->set_sideslip(alpha);
#else
    nsutils_->set_angle_of_attack(alpha);
#endif
    nsutils_->set_mach_number(M);
    
    // initialize the RANS equations
    arma::Col<double> u_init = nsutils_->state();
    set_freestream_chi(fs_chi);
    set_initial_state(u_init);
    
    // set boundary conditions
    std::vector<double> ub = arma::conv_to<std::vector<double>>::from(u_init);
    for (unsigned int i = 0; i < far_bnds.size(); ++i) {
      set_boundary_parameters(far_bnds[i],ub);
    }

    // drag output
    set_output_type(0,NSOutputType::directional_force);
    set_output_boundary_ids(0,foil_bnds);
    const double qinf = nsutils_->dynamic_pressure();
#ifdef ONERA
    const double sca = 2.0/1.15315084119231;
    std::vector<double> bparams = {sca*cos(alpha)/qinf,0.0,sca*sin(alpha)/qinf};
#else
    std::vector<double> bparams = {cos(alpha)/qinf,sin(alpha)/qinf,0.0};
#endif
    set_output_parameters(0,bparams);
  }

  virtual const arma::Mat<double>& parameter_domain() const
  {
    return mu_bnd_;
  }

private:
  arma::Mat<double> mu_bnd_;
  
};

class EQPDriver {
public:
  EQPDriver()
    : comm_rank(Utilities::MPI::mpi_comm_rank()),
#ifdef ONERA
      dim(3),
      geom_degree(3),
#else
      dim(2),
      geom_degree(4),
#endif
      nstype(NSVariableType::entropy),
      nsutils(dim,nstype,NSUtils::Equation::rans_sa,NSUtils::Normalization::unit_rho_a),
      eqn(nstype, &nsutils),
      mesh(dim),
      mesh_geom(&mesh,&fe_set,geom_degree),
      surf_mesh(dim-1,dim),
      surf_geom(&surf_mesh,&fe_set,geom_degree),
      dg_eqp_c(&eqn, &mesh, &mesh_geom, &fe_set, &quad_set)
  {}

  void load_fp() {
      
    // flow parameters
    double M = 0.5;  
    double alpha = 0.0*M_PI/180;
    double Re = 1e5;
    double fs_chi = 3.0;
    double sa_scale = 10;

    nsutils.set_mach_number(M);
    nsutils.set_reynolds_number(Re);
    nsutils.set_angle_of_attack(alpha);
    nsutils.set_freestream_chi(fs_chi);
    nsutils.set_sa_scale(sa_scale);

    // initialize RANS-SA equations
    eqn.set_dynamic_viscosity(nsutils.dynamic_viscosity());
    eqn.set_prandtl_number(0.72);
    eqn.set_constant_viscosity(true);
    eqn.set_freestream_chi(fs_chi);
    eqn.set_sa_scale(sa_scale);
    
    arma::Col<double> u_init = nsutils.state();
    eqn.set_initial_state(u_init);

    // set boundary conditions  
    eqn.set_n_boundaries(2*dim+1);
    // inflow
    eqn.set_boundary_type(0,NSBoundaryType::Tt_Pt_alpha);
    std::vector<double> TtPta = nsutils.TtPta();
    eqn.set_boundary_parameters(0,TtPta);
    // outflow
    eqn.set_boundary_type(1,NSBoundaryType::static_p);
    std::vector<double> p_static = {nsutils.pressure()};
    eqn.set_boundary_parameters(1,p_static);
    // before and after bump
    eqn.set_boundary_type(2,NSBoundaryType::symmetry_plane);
    // top
    eqn.set_boundary_type(3,NSBoundaryType::symmetry_plane);
    // flatplate
    eqn.set_boundary_type(2*dim,NSBoundaryType::adiabatic_wall);  
    if (dim == 3) {
      // side
      eqn.set_boundary_type(4,NSBoundaryType::symmetry_plane);
      eqn.set_boundary_type(5,NSBoundaryType::symmetry_plane);
    }  
      
    // output is the drag on the foil.  Note that this is the directional force in the direction of the flow.
    eqn.set_n_outputs(1);
    eqn.set_output_type(0,NSOutputType::directional_force);
    std::vector<unsigned int> bids = {2*dim}; 
    eqn.set_output_boundary_ids(0,bids);
    const double qinf = nsutils.dynamic_pressure();
    std::vector<double> bparams = {1.0/(2.0*qinf),0.0,0.0}; // length of the plate is 2.0
    eqn.set_output_parameters(0,bparams);   

    // load surface mesh
    ExtendedFlatPlateMeshGenerator fp_gen;
    fp_gen.set_mesh_entity_type(entity_type);
    fp_gen.set_n_leading_elements(2);
    fp_gen.set_n_trailing_elements(2);
    fp_gen.set_n_plate_elements(4);
    fp_gen.set_n_wall_normal_elements(4);
    fp_gen.set_n_cross_flow_elements(1);
    fp_gen.set_first_element_wall_normal_spacing(0.02);
    //fp_gen.set_first_element_wall_normal_spacing(0.1);
    fp_gen.set_leading_length(0.5);
    fp_gen.set_trailing_length(0.5);
    fp_gen.set_plate_length(2.0);
    fp_gen.set_height(1.0);
    fp_gen.set_square_edge_elements(true);
    std::vector<bool> wall_marker(2*dim+1,false);
    wall_marker[2*dim] = true;
    {
      Mesh mesh(dim);
      MeshGeometry mesh_geom(&mesh,&fe_set,geom_degree);
      fp_gen.generate_mesh(mesh, mesh_geom);
      
      mesh_geom.set_wall_marker(wall_marker);
      mesh_geom.extract_surface_geometry(wall_marker, surf_mesh, surf_geom);
    }
    
    // load the mesh
    mesh.set_enforce_one_regular_hanging_node(true);
    fp_gen.generate_mesh(mesh, mesh_geom);

#ifdef WITH_MPI
    mesh.prepare_parallelization();
    mesh.execute_repartition();
    mesh_geom.execute_repartition();
    mesh.finalize_repartition();
#endif 

    // set wall geometry and initialize wall distance
    mesh_geom.set_wall_geometry(surf_mesh, surf_geom);
    mesh_geom.init_wall_distance();
  };

  void load_rae() {
    std::string gmsh_file = "../rae2822_level5.m4.msh";
  
    // flow parameters
    double M = 0.3;  // Mach number
    //double M = 0.734;
    double alpha = 2.0*M_PI/180.0;  // angle of attack (in radians)
    double Re = 6.5e6;  // Reynolds number
    double fs_chi = 3.0;  // freestream chi for SA equations
    double sa_scale = sqrt(Re);  // SA scale

    eqn.alpha_0 = alpha;
    eqn.M_0 = M;
    
    nsutils.set_mach_number(M);
    nsutils.set_reynolds_number(Re);
    nsutils.set_angle_of_attack(alpha);
    nsutils.set_freestream_chi(fs_chi);
    nsutils.set_sa_scale(sa_scale);

    // initialize the RANS equations
    arma::Col<double> u_init = nsutils.state();
    eqn.set_freestream_chi(fs_chi);
    eqn.set_sa_scale(sa_scale);
    eqn.set_dynamic_viscosity(nsutils.dynamic_viscosity());
    eqn.set_constant_viscosity(true);
    eqn.set_initial_state(u_init);

    // set shock capturing.  For this transonic case, shock capturing is actually not required if the entropy variables are used.
    eqn.set_shock_capturing(false);
    //eqn.set_shock_viscosity_type(Euler::shock_viscosity_type::physical);

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

    // far field boundaries
    eqn.fs_chi_0 = 3.0;
    eqn.far_bnds.resize(2);
    eqn.far_bnds[0] = 2;
    eqn.far_bnds[1] = 3;
    eqn.foil_bnds.resize(1);
    eqn.foil_bnds[0] = 1;

    // output is drag on the foil.  Note that this is the directional force in the direction of the flow.
    eqn.set_n_outputs(1);
    eqn.set_output_type(0,NSOutputType::directional_force);
    std::vector<unsigned int> bids = {1};
    eqn.set_output_boundary_ids(0,bids);
    const double qinf = nsutils.dynamic_pressure();
    std::vector<double> bparams = {cos(alpha)/qinf,sin(alpha)/qinf,0.0};
    eqn.set_output_parameters(0,bparams);

    // extract surface mesh; the surface mesh is needed for RANS wall-distance calculation
    std::vector<bool> wall_marker = {false, true, false, false}; // boundary ID 2 is the wall

    // extract surface mesh
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
    
    Gmsh gmsh;
    mesh.set_enforce_one_regular_hanging_node(true);
    if (comm_rank == 0) {
      gmsh.read_msh_file(gmsh_file);
    }
    gmsh.load_mesh(mesh, mesh_geom);

  };

  void load_mda() {
    std::string gmsh_file = "../c31_mesh_quad_q4.msh";
  
    // flow parameters
    double M = 0.2;  // Mach number
    double alpha = 16.0*M_PI/180.0;  // angle of attack (in radians)
    double Re = 9.0e6;  // Reynolds number
    double fs_chi = 5.0;  // freestream chi for SA equations
    double sa_scale = 10000;  // SA scale

    eqn.alpha_0 = alpha;
    eqn.M_0 = M;

    nsutils.set_mach_number(M);
    nsutils.set_reynolds_number(Re);
    nsutils.set_angle_of_attack(alpha);
    nsutils.set_freestream_chi(fs_chi);
    nsutils.set_sa_scale(sa_scale);

    // initialize the RANS equations
    arma::Col<double> u_init = nsutils.state();
    eqn.set_freestream_chi(fs_chi);
    eqn.set_sa_scale(sa_scale);
    eqn.set_dynamic_viscosity(nsutils.dynamic_viscosity());
    eqn.set_constant_viscosity(true);
    eqn.set_initial_state(u_init);

    // set shock capturing.  For this transonic case, shock capturing is actually not required if the entropy variables are used.
    eqn.set_shock_capturing(false);
    //eqn.set_shock_viscosity_type(Euler::shock_viscosity_type::physical);

    // set boundary conditions
    std::vector<double> ub = arma::conv_to<std::vector<double>>::from(u_init);
    eqn.set_n_boundaries(5);

    eqn.set_boundary_type(1,NSBoundaryType::full_state);
    eqn.set_boundary_parameters(1,ub);
    // wing
    eqn.set_boundary_type(2,NSBoundaryType::adiabatic_wall);
    // slat
    eqn.set_boundary_type(3,NSBoundaryType::adiabatic_wall);
    // flap
    eqn.set_boundary_type(4,NSBoundaryType::adiabatic_wall);
    
    // far field boundaries
    eqn.fs_chi_0 = 5.0;
    eqn.far_bnds.resize(1);
    eqn.far_bnds[0] = 1;

    eqn.foil_bnds.resize(3);
    eqn.foil_bnds[0] = 2;
    eqn.foil_bnds[1] = 3;
    eqn.foil_bnds[2] = 4;


    // output is drag on the foil.  Note that this is the directional force in the direction of the flow.
    eqn.set_n_outputs(1);
    eqn.set_output_type(0,NSOutputType::directional_force);
    std::vector<unsigned int> bids = {2,3,4};
    eqn.set_output_boundary_ids(0,bids);
    const double qinf = nsutils.dynamic_pressure();
    std::vector<double> bparams = {cos(alpha)/qinf,sin(alpha)/qinf,0.0};
    eqn.set_output_parameters(0,bparams);

    // extract surface mesh; the surface mesh is needed for RANS wall-distance calculation
    std::vector<bool> wall_marker = {false,false,true,true,true};

    // extract surface mesh
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
    
    Gmsh gmsh;
    mesh.set_enforce_one_regular_hanging_node(true);
    if (comm_rank == 0) {
      gmsh.read_msh_file(gmsh_file);
    }
    gmsh.load_mesh(mesh, mesh_geom);

  };

  void load_onera() {
    std::string gmsh_file = "../oneram6_very_coarse.msh";
  
    // flow parameters
    double M = 0.4;  // Mach number
    double alpha = 1.0*M_PI/180.0;  // angle of attack (in radians)
    double Re = 1.0e6;  // Reynolds number
    double fs_chi = 3.0;  // freestream chi for SA equations
    double sa_scale = sqrt(Re);  // SA scale

    eqn.alpha_0 = alpha;
    eqn.M_0 = M;

    nsutils.set_mach_number(M);
    nsutils.set_reynolds_number(Re);
    nsutils.set_angle_of_attack(0.0);
    nsutils.set_sideslip(alpha);
    nsutils.set_freestream_chi(fs_chi);
    nsutils.set_sa_scale(sa_scale);

    // initialize the RANS equations
    arma::Col<double> u_init = nsutils.state();
    eqn.set_freestream_chi(fs_chi);
    eqn.set_sa_scale(sa_scale);
    eqn.set_dynamic_viscosity(nsutils.dynamic_viscosity());
    eqn.set_constant_viscosity(true);
    eqn.set_initial_state(u_init);

    // set shock capturing.  For this transonic case, shock capturing is actually not required if the entropy variables are used.
    eqn.set_shock_capturing(false);
    //eqn.set_shock_viscosity_type(Euler::shock_viscosity_type::physical);

    // set boundary conditions
    std::vector<double> ub = arma::conv_to<std::vector<double>>::from(u_init);
    eqn.set_n_boundaries(4);

    eqn.set_boundary_type(2,NSBoundaryType::full_state);
    eqn.set_boundary_parameters(2,ub);
    // wing
    eqn.set_boundary_type(1,NSBoundaryType::adiabatic_wall);
    // symmetry 
    eqn.set_boundary_type(3,NSBoundaryType::symmetry_plane);
    
    // far field boundaries
    eqn.fs_chi_0 = 3.0;
    eqn.far_bnds.resize(1);
    eqn.far_bnds[0] = 2;

    eqn.foil_bnds.resize(1);
    eqn.foil_bnds[0] = 1;

    // output is drag on the foil.  Note that this is the directional force in the direction of the flow.
    eqn.set_n_outputs(1);
    eqn.set_output_type(0,NSOutputType::directional_force);
    std::vector<unsigned int> bids = {1};
    eqn.set_output_boundary_ids(0,bids);
    const double sca = 2.0/1.15315084119231;
    const double qinf = nsutils.dynamic_pressure();
    std::vector<double> bparams = {sca*cos(alpha)/qinf,0.0,sca*sin(alpha)/qinf};
    eqn.set_output_parameters(0,bparams);

    // extract surface mesh; the surface mesh is needed for RANS wall-distance calculation
    std::vector<bool> wall_marker = {false,true,false,false};

    // extract surface mesh
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
    
    Gmsh gmsh;
    mesh.set_enforce_one_regular_hanging_node(true);
    if (comm_rank == 0) {
      gmsh.read_msh_file(gmsh_file);
    }
    gmsh.load_mesh(mesh, mesh_geom);

  };
  
  void setup_eqp() {
    // DG-EQP evaluator and contructor
    dg_eqp_c.set_spatio_parameter_adaptivity(true);
    //dg_eqp_c.set_spatial_adapt_on_first_iteration_only(true);
    dg_eqp_c.adapt()->set_adaptation_type(Adaptation<double>::type::anisotropic_h);

    // use non-lifted state
    //dg_eqp_c.adapt()->primal_discretization()->set_source_gradient_type(DG<double>::SourceGradientType::raw);
    //dg_eqp_c.adapt()->adjoint_discretization()->set_source_gradient_type(DG<double>::SourceGradientType::raw);

    // RAE 9e-5 for 1%; 3e-5 for old
    // MDA 3e-4 for 1%
    // FP 5e-5 for 1%
    dg_eqp_c.adapt()->set_target_error(1e-4);
    dg_eqp_c.adapt()->set_adaptation_target_type(Adaptation<double>::target_type::output);
    dg_eqp_c.adapt()->set_max_iterations(20);
  
    dg_eqp_c.set_initial_polynomial_degree(poly_degree);
    dg_eqp_c.set_eqp_tolerance(1e-5); // 1e-7
    dg_eqp_c.set_eqp_output_functional_tolerance(1e-6);
    dg_eqp_c.set_eqp_dwr_tolerance(1e-5); // 1e-7
    
    dg_eqp_c.set_n_max_reduced_basis(20);
    dg_eqp_c.set_weak_greedy_tolerance(5e-5);
    dg_eqp_c.set_pod_tolerance(1e-10);
  
    dg_eqp_c.set_n_eqp_smoothing_iterations(3);
    dg_eqp_c.set_eqp_verbosity(-1); 
    dg_eqp_c.set_eqp_target_type(DGEQPConstructor<double>::EQPTargetType::output);
    dg_eqp_c.set_eqp_norm(EQPNorm::l2);
    dg_eqp_c.set_greedy_target_type(DGEQPConstructor<double>::GreedyTargetType::output);

    dg_eqp_c.set_eqp_separate_dwr(true);
    dg_eqp_c.set_eqp_add_soln_squared(false);
    dg_eqp_c.set_eqp_unity_weights(false);   
    dg_eqp_c.set_eqp_element_constant_constraint(true);
    dg_eqp_c.set_eqp_facet_constant_constraint(true);
    dg_eqp_c.set_eqp_min_nnz_weights(0);

    dg_eqp_c.set_write_reduced_basis(false);
    dg_eqp_c.set_write_reduced_mesh(false);

    dg_eqp_c.set_primal_fe_solver(&ptc);
    dg_eqp_c.set_primal_rb_solver(&ptc_rb);
    dg_eqp_c.adapt()->set_adjoint_solver(&adjoint_solver);
    dg_eqp_c.set_adjoint_fe_solver(&adjoint_solver);
    dg_eqp_c.set_adjoint_rb_solver(&adjoint_rb_solver);
    
    // set training parameters
    unsigned int n1d;
    switch (eqn.n_parameters()) {
    case 1:
      n1d = 25;
      break;
    case 2:
      n1d = 5;
      break;
    case 3:
    case 4:
      n1d = 3;
      break;
    default:
      Error("unsupported number of parameters");
    }
    arma::arma_rng::set_seed(10);    
    dg_eqp_c.generate_structured_parameter_set(10,Xi_train);
    //arma::Mat<double> Xi_train_2;
    //eqn.generate_random_Xi(10, Xi_train_2);
    //Xi_train = arma::join_rows(Xi_train, Xi_train_2);
    dg_eqp_c.set_training_parameters(Xi_train);
    
    // set test parameters
    arma::arma_rng::set_seed(0);
    unsigned int n_test = 20;
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
    ptc_rb.set_linear_solver(&dense_linsolver);
    ptc_rb.set_max_iter(100);
    ptc_rb.set_initial_timestep(1e-2); 
    ptc_rb.set_verbosity(-1); 
    ptc_rb.set_abs_tol(1e-8);  
    ptc_rb.set_timestep_increase_multiplier(3.0);
    ptc_rb.set_line_search_type(PTCSolver<double>::LineSearchType::unsteady);
    ptc_rb.set_line_search_maximum_physical_change(0.2); 
    ptc_rb.set_line_search_maximum_residual_increase(1.1);
    //ptc_rb.set_maximum_timestep(1e2);

    // set adjoint solver
    adjoint_rb_solver.set_linear_solver(&dense_linsolver);
    adjoint_rb_solver.set_residual_tolerance(1e-8);
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
    gmres.set_max_inner_iterations(250);
    gmres.set_max_outer_iterations(2);
    gmres.set_verbosity(-1);

    ptc.set_verbosity(1);
    ptc.set_linear_solver(&gmres);
    ptc.set_max_iter(150);
    ptc.set_initial_timestep(1e-2);
    ptc.set_maximum_timestep(1e3);
    ptc.set_timestep_increase_multiplier(3.0);
    ptc.set_abs_tol(2e-8);
    ptc.set_line_search_type(PTCSolver<double>::LineSearchType::unsteady);
    ptc.set_line_search_maximum_physical_change(0.2); 
    ptc.set_line_search_maximum_residual_increase(1.02);
    ptc.set_linear_solver_decrease_multiplier(1e-5);

    // set adjoint solver
    adjoint_solver.set_linear_solver(&gmres);
    adjoint_solver.set_residual_tolerance(2e-8);
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
  //   const unsigned int n_param = eqn.n_params();
    
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

  const unsigned int poly_degree = 2;
  const MeshEntityType entity_type = MeshEntityType::quad;
  const FiniteElementType fe_type = FiniteElementType::Legendre;

  unsigned int comm_rank;
  unsigned int dim;
  unsigned int geom_degree;
  NSVariableType nstype;
  NSUtils nsutils;

  FESet fe_set;
  QuadratureSet quad_set;

  ParametrizedRANSSA eqn;

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
    std::string mesh_file = "mesh.dat";
    std::string train_file = "train.dat";
    std::string test_file = "test.dat";
  
    EQPDriver eqpd;
    DGEQPConstructor<double>& dg_eqp_c = eqpd.dg_eqp_c;
  
    //eqpd.load_fp();
    //eqpd.load_rae();
    //eqpd.load_mda();
    eqpd.load_onera();
    if (Utilities::MPI::mpi_comm_rank() == 0) {
      printf("param mode = %d\n", eqpd.eqn.param_mode);
    }
  
#ifdef WITH_MPI
    eqpd.mesh.prepare_parallelization();
    eqpd.mesh.execute_repartition();
    eqpd.mesh_geom.execute_repartition();
    eqpd.mesh.finalize_repartition();
#endif
    
    // set wall geometry and initialize wall distance
    eqpd.mesh_geom.set_wall_geometry(eqpd.surf_mesh, eqpd.surf_geom);
    eqpd.mesh_geom.init_wall_distance();
  
    eqpd.set_fe_solver();
    eqpd.set_rb_solver();
    eqpd.setup_eqp();

    if (Utilities::MPI::mpi_comm_rank() == 0 &&
        (eqpd.Xi_train.n_elem == eqpd.Xi_test.n_elem) && 
        (arma::norm(arma::vectorise(eqpd.Xi_train - eqpd.Xi_test)) == 0)) {
      printf("setting test = train\n");
      test_file = train_file;
    }
  

    bool pod_test = false;
    if (pod_test) {
      bool restart = true;
  
  
      if (restart) {
        eqpd.load_mesh(mesh_file);
        // eqpd.load_training_data(train_file);
        eqpd.load_test_data(test_file);    
        dg_eqp_c.init();

        std::cout << "n_dofs = " <<  dg_eqp_c.fe_field()->n_dofs() << std::endl;
        std::cout << "n_train = " << eqpd.Xi_train.n_cols << std::endl;
        std::cout << "n_test = " << eqpd.Xi_test.n_cols << std::endl;

      } else {
        eqpd.eqn.set_parameters(eqpd.Xi_train.col(0));
        eqpd.run_adaptive_fe_solve();
        const BlockVector<double>* fe_init_vec = dg_eqp_c.adapt()->state_vector();
        dg_eqp_c.set_test_init_fe_state(fe_init_vec);

        std::cout << "solving for training states" << std::endl;
        dg_eqp_c.solve_training_states();

        std::cout << "solving for test states" << std::endl;
        dg_eqp_c.solve_test_states();
    
        eqpd.save_mesh(mesh_file);
        // eqpd.save_training_data(train_file);
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
      // dg_eqp_c.set_high_dim_training(true);
      eqpd.run_weak_greedy();
    }
  
#ifdef WITH_MPI
    Utilities::MPI::mpi_finalize();
#endif

    return 0;
  }

  
