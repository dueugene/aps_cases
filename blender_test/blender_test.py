import sys, os
sys.path.append(os.path.expandvars('$BLENDER_PY'))

# retrieve blender modules
import bpy # blender python module
import bmesh # mesh module?
from mathutils import Vector, Euler

verts = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)]  # 4 verts made with XYZ coords
mesh = bpy.data.meshes.new("mesh")  # add a new mesh
mesh_obj = bpy.data.objects.new("MyMesh", mesh)  # add a new object using the mesh

bm = bmesh.new()
for v in verts:
    bm.verts.new(v)  # add a new vert

# make the bmesh the object's mesh
bm.to_mesh(mesh)  
bm.free()  # always do this when finished

bpy.context.collection.objects.link(mesh_obj) # put the object into the scene (link)


lattice = bpy.data.lattices.new("Lattice") # add a new lattice
lattice_ob = bpy.data.objects.new("Lattice", lattice) # create a new object with the lattice
# modifiy lattice properties
# location, scale, and rotation_euler is enough to define the bounding box in 3 space
lattice_ob.location = (1, 1, 0) # location of centroid of lattice?
lattice_ob.scale = (3, 2, 3) # make the lattice larger
lattice_ob.rotation_euler = (0, 0, 0)
print(lattice_ob.rotation_euler)
print(lattice_ob.location)

# points_u/v/w, interpolation_type_u/v/w controls the transformation behaviour within bounding box
lattice.points_u = 4
lattice.points_w = 1
lattice.interpolation_type_u = 'KEY_CATMULL_ROM' # KEY_BSPLine, KEY_LINEAR, KEY_BSPLINE, KEY_CARDINAL
print(lattice.interpolation_type_u)
print(lattice.points_u)
print(lattice.points)

# attach lattice as modifier to all mesh objects in scene
# scene = bpy.context.scene
# for ob in scene.objects:
#     if ob.type == 'MESH':
#         mod = ob.modifiers.new("Lattice", 'LATTICE')
#         mod.object = lattice_ob

# attach lattice as modifier to mesh
mod = mesh_obj.modifiers.new("Lattice", 'LATTICE')
mod.object = lattice_ob;

bpy.context.collection.objects.link(lattice_ob) # put the lattice into the scene (link)


# deform a few point(s)
lattice_ob.data.points[0].co_deform = Vector(tuple((-1.5, -1.7, 0)))
lattice_ob.data.points[7].co_deform = Vector(tuple((1.5, 1.7, 0)))


# print deformed points
depsgraph = bpy.context.evaluated_depsgraph_get() # get the dependecny graph
obj_eval = mesh_obj.evaluated_get(depsgraph)
mesh_eval = obj_eval.data;
transformed_verts = [v.co for v in obj_eval.data.vertices]

print(transformed_verts)
# coordinates as tuples
plain_verts = [v.to_tuple() for v in transformed_verts]
print(plain_verts)

