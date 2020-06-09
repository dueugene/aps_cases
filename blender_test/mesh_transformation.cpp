#include <vector>
#include <string>
#include <cmath>
#include <armadillo>



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

#include "mesh/ffd.h"

#include "disc/dg_adaptation.h"

void evalutate_coord_change(const MeshGeometry& mesh_geom, BlockVector<double>* xvec, BlockVector<double>* yvec, BlockVector<double>* bvec);
  
void write_mesh_ffd(FEField& fe_field, MeshGeometry& mesh_geom, BlockVector<double>* bvec, FFD& ffd, std::string fname);

FFD* PaddedFFD(unsigned int dim, arma::Mat<double> bounds, std::vector<unsigned int> n_points, int continuity, std::vector<unsigned int> n_pads, std::vector<unsigned int>& ffd_mapping);

struct TransInfo {
  std::vector<unsigned int> inds;
  arma::Mat<double> deltas;
  unsigned int trans_mode;
  std::string name;
};

int main(int argc, char *argv[])
{
  FESet fe_set;
  const int dim = 2;
  Mesh mesh(dim);
  MeshGeometry mesh_geom(&mesh,&fe_set,2);
  NacaMeshGenerator naca_gen; // the mesh generator will construct a C mesh
  naca_gen.set_naca_digits("0012");
  naca_gen.set_n_foil_elements(7); // # elements on foil along flow direction
  naca_gen.set_n_tail_elements(3); // # elements after foil along flow direction
  naca_gen.set_n_radial_elements(5); // # elements in radial direction
  naca_gen.set_first_element_radial_spacing(0.1/4); 
  naca_gen.set_sqrt_fitting(true); // use sqrt relation for wall-normal spacing
  naca_gen.generate_mesh(mesh, mesh_geom);
  FEField fe_field(&mesh,&fe_set,1,FiniteElementType::DiscontinuousLagrange,2);

  // get initial coordinates of mesh
  BlockVector<double>* xvec = fe_field.allocate_element_wise_block_vector();
  BlockVector<double>* yvec = fe_field.allocate_element_wise_block_vector();
  BlockVector<double>* bvec = fe_field.allocate_element_wise_block_vector();
  for (typename MeshGeometry::iterator elem = mesh_geom.begin_element(); elem != mesh_geom.end_element(); ++elem) {
    arma::Mat<double> coords = elem->node_coordinates();
    if (elem->owner() >= 0) {
      for (unsigned int i = 0; i < coords.n_cols; ++i) {
        const double x = coords(0,i);
        const double y = coords(1,i);
        xvec->value(elem->index(),i) = x;
        yvec->value(elem->index(),i) = y;
        bvec->value(elem->index(),i) = 0;
      }
    }
  }
  
  // setup ffd here
  unsigned int continuity = 0;
  arma::Mat<double> bounds = {{-0.5,1.5},{-0.3,0.3}};
  std::vector<unsigned int> n_points = {6,4};
  // unsigned int continuity = -1;
  // arma::Mat<double> bounds = {{-5.0,5.0},{-5.0,5.0}};
  // std::vector<unsigned int> n_points = {4,4};
  FFD* ffd = new FFD(dim, bounds, n_points, continuity);
  std::vector<unsigned int> ffd_inds = {7, 8, 13, 14}; // manually set ffd inds
  // std::vector<unsigned int> ffd_inds = {5, 6, 9, 10}; // manually set ffd inds  

  // create a second transformation
  std::vector<unsigned int> n_pads = {2, 2};
  std::vector<unsigned int> ffd_map;
  FFD* ffd2 = PaddedFFD(dim, bounds, n_points, continuity, n_pads, ffd_map);

  // save initial mesh
  std::string fname = "original";
  write_mesh_ffd(fe_field, mesh_geom, bvec, *ffd, fname);

  // define a set of transformations to perform
  std::vector<struct TransInfo> n_transforms(2);
  n_transforms[0].inds = {7, 8, 13, 14};
  n_transforms[0].deltas = arma::Mat<double>(2, 4, arma::fill::ones);
  n_transforms[0].deltas *= 0.05;
  n_transforms[0].trans_mode = 1;
  n_transforms[0].name = "translate";

  n_transforms[1].inds = {7, 8, 9, 10, 13, 14, 15, 16};
  n_transforms[1].deltas = arma::Mat<double>(2, 8, arma::fill::randu);
  n_transforms[1].deltas -= 0.5;
  n_transforms[1].deltas /= 5;
  n_transforms[1].trans_mode = 1;
  n_transforms[1].name = "random";  
  
  for (int i = 0; i < n_transforms.size(); ++i) {
    struct TransInfo trans = n_transforms[i];
    
    // set transformed point locations
    for (int j = 0; j < trans.inds.size(); ++j) {
      printf("%d, %d\n",trans.inds[j], ffd_map[trans.inds[j]]);
      ffd->set_transformed_control_point(trans.inds[j], trans.deltas.col(j), trans.trans_mode);
      // set transformation for padded ffd
      ffd2->set_transformed_control_point(ffd_map[trans.inds[j]], trans.deltas.col(j), trans.trans_mode);
    }
    
    // attach to geometry and transform
    mesh_geom.set_ffd(ffd);
    mesh_geom.transform();
    // meassure change in position
    evalutate_coord_change(mesh_geom, xvec, yvec, bvec);
    write_mesh_ffd(fe_field, mesh_geom, bvec, *ffd, trans.name);

    // attach padded ffd and transform
    mesh_geom.set_ffd(ffd2);
    mesh_geom.transform();
    // meassure change in position
    evalutate_coord_change(mesh_geom, xvec, yvec, bvec);
    write_mesh_ffd(fe_field, mesh_geom, bvec, *ffd2, trans.name + "pad");
  }  

  return 0;
}

void evalutate_coord_change(const MeshGeometry& mesh_geom, BlockVector<double>* xvec, BlockVector<double>* yvec, BlockVector<double>* bvec)
{
  for (typename MeshGeometry::iterator elem = mesh_geom.begin_element(); elem != mesh_geom.end_element(); ++elem) {
    arma::Mat<double> coords = elem->node_coordinates();
    if (elem->owner() >= 0) {
      for (unsigned int i = 0; i < coords.n_cols; ++i) {
        const double x = xvec->value(elem->index(),i) - coords(0,i);
        const double y = yvec->value(elem->index(),i) - coords(1,i);
        bvec->value(elem->index(),i) = sqrt(x*x + y*y);
      }
    }
  }
}

FFD* PaddedFFD(unsigned int dim, arma::Mat<double> bounds, std::vector<unsigned int> n_points, int continuity, std::vector<unsigned int> n_pads, std::vector<unsigned int>& ffd_mapping)
{
  // remap padded ffd nodes to previous ffd nodes
  ffd_mapping.resize(n_points[0]*n_points[1]);
  unsigned int cnt = 0;
  for (unsigned int i1 = 0; i1 < n_points[1]; ++i1) {
    for (unsigned int i0 = 0; i0 < n_points[0]; ++i0) {
      // convert sub2ind
      unsigned int ind = (i1+n_pads[0])*(n_points[0] + n_pads[0]*2) + i0 + n_pads[0];
      ffd_mapping[cnt] = ind;
      ++cnt;
    }
  }
    
  // calculate padding in each direction
  double delta_x = (bounds(0,1) - bounds(0,0))/(n_points[0]-1);
  double delta_y = (bounds(1,1) - bounds(1,0))/(n_points[1]-1);

  bounds(0,0) -= delta_x * n_pads[0];
  bounds(0,1) += delta_x * n_pads[0];
  bounds(1,0) -= delta_y * n_pads[1];
  bounds(1,1) += delta_y * n_pads[1];

  n_points[0] += n_pads[0]*2;
  n_points[1] += n_pads[1]*2;
    
  // initialize underlying ffd
  FFD* ffd_ptr = new FFD(dim, bounds, n_points, continuity);
  printf("yoyo\n");
  printf("%d\n", ffd_ptr->dimension());

  return ffd_ptr;
}

void write_mesh_ffd(FEField& fe_field, MeshGeometry& mesh_geom, BlockVector<double>* bvec, FFD& ffd, std::string fname)
{
  VtkIO vtkio(&fe_field, &mesh_geom, bvec);
  vtkio.write_volume_data(fname + "_mesh");
  vtkio.write_ffd(ffd, fname + "_ffd");
}
