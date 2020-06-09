#include <fstream>
#include <string>

#include "base/error.h"
#include "io/gmsh.h"
#include "mesh/fe_field.h"
#include "mesh/ffd.h"
#include "mesh/sfe_field.h"
#include "mesh/mesh.h"
#include "mesh/mesh_geometry.h"
#include "mesh/mesh_generator.h"
#include "solver/block_vector.h"
#include "io/vtk_io.h"

#include "fe/fe_definition.h"
#include "fe/fe_set.h"

// blender files
#include "BKE_main.h"
#include "BKE_lattice.h" /* for lattice*/

int main(int argc, char *argv[])
{
  FESet fe_set;
  const int dim = 2;
  Mesh mesh(dim);
  MeshGenerator::create_hypercube_mesh(mesh,MeshEntityType::quad,2,1);
  MeshGeometry mesh_geom(&mesh,&fe_set,2);
  FEField fe_field(&mesh,&fe_set,1,FiniteElementType::DiscontinuousLagrange,1);

  BlockVector<double>* bvec = fe_field.allocate_element_wise_block_vector();
  for (typename FEField::active_iterator elem = fe_field.begin_active_element(); elem != fe_field.end_element(); ++elem) {
    if (elem->owner() >= 0) {
      for (unsigned int i = 0; i < 4; ++i) {
        const double x = elem->vertex(i)->coordinate(0);
        const double y = elem->vertex(i)->coordinate(1);
        bvec->value(elem->index(),i) = sqrt(x*x + y*y);
      }
    }
  }

  VtkIO vtkio(&fe_field, &mesh_geom, bvec);
  vtkio.write_volume_data("original");


  // define an ffd transformation
  unsigned int continuity = 1;
  arma::Mat<double> bounds = {{-0.75,1.75},{-0.5,1.5}};
  std::vector<unsigned int> n_points = {6,5};
  FFD ffd(2, bounds, n_points, continuity);
  mesh_geom.set_ffd(&ffd);
  arma::Col<double> delta(ffd.get_n_dof(), arma::fill::randn);
  ffd.set_transformed_control_point(delta, 1);
  mesh_geom.transform();
  vtkio.write_volume_data("ffd_transform");

  // test blender transformation engine
  Main* main_scene = BKE_main_new();
  char lattice_name[] = "test_name";
  Lattice* blen_lattice = BKE_lattice_add(main_scene, lattice_name);

  delete bvec;

  return 0;
}


