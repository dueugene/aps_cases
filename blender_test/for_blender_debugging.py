import sys, os
sys.path.append(os.path.expandvars('$BLENDER_PY'))

# retrieve blender modules
import bpy # blender python module
import bmesh # mesh module?
from mathutils import Vector, Euler, Matrix, Quaternion

# reset the scene by removing all objects
objs = bpy.data.objects
for ob in objs:
    print(ob)
    bpy.data.objects.remove(ob, do_unlink=True)

# for each 'area' change the viewpoint to top view
for area in bpy.context.screen.areas:
    if area.type == 'VIEW_3D':
        print(area.spaces[0].region_3d.view_rotation)
        area.spaces[0].region_3d.view_rotation = Quaternion((1.0, 0.0, 0.0, 0.0))
        # break
# create a linearly spaced set of points as a "mesh"    
n_x = 4
n_y = 3
scale_x = 3.0
scale_y = 3.0
mesh_origin = (3, 7, 0)
verts = []
for i in range(n_x):
    for j in range(n_y):
        point = (mesh_origin[0] + i/(n_x-1)*scale_x - scale_x/2, mesh_origin[1] + j/(n_y-1)*scale_y - scale_y/2, mesh_origin[2])
        verts.append(point)
        print(point)

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
lattice_ob.location = (3, 7, 0) # location of centroqid of lattice?
lattice_ob.scale = (2, 1.5, 1) # make the lattice larger
lattice_ob.rotation_euler = (0, 0, 0)

# points_u/v/w, interpolation_type_u/v/w controls the transformation behaviour within bounding box
lattice.points_u = 3
lattice.points_v = 2
lattice.points_w = 1
lattice.interpolation_type_u = 'KEY_BSPLINE' # KEY_BSPLINE, KEY_LINEAR, KEY_BSPLINE, KEY_CARDINAL
lattice.interpolation_type_v = 'KEY_BSPLINE'

# attach lattice as modifier to all mesh objects in scene
scene = bpy.context.scene
for ob in scene.objects:
    if ob.type == 'MESH':
        ob.display_type = 'WIRE' # set display of the mesh to wireframe
        mod = ob.modifiers.new("Lattice", 'LATTICE')
        mod.object = lattice_ob

# attach lattice as modifier to mesh
# mod = mesh_obj.modifiers.new("Lattice", 'LATTICE')
# mod.object = lattice_ob;

bpy.context.collection.objects.link(lattice_ob) # put the lattice into the scene (link)


# deform a few point(s)
# note deformation is with respect to local coordinates (orientation of bounding box) not sure about scaling yet
lattice_ob.data.points[0].co_deform =  lattice_ob.data.points[0].co + Vector(tuple((-0.2, -0.2, 0)))
lattice_ob.data.points[1].co_deform =  lattice_ob.data.points[1].co + Vector(tuple((-0.5, -0.2, 0)))
lattice_ob.data.points[2].co_deform =  lattice_ob.data.points[2].co + Vector(tuple((0.5, -0.3, 0)))
lattice_ob.data.points[3].co_deform =  lattice_ob.data.points[3].co + Vector(tuple((-0.2, 0.2, 0)))
lattice_ob.data.points[4].co_deform =  lattice_ob.data.points[4].co + Vector(tuple((0.5, -0.7, 0)))
lattice_ob.data.points[5].co_deform =  lattice_ob.data.points[5].co + Vector(tuple((0.4, 0.3, 0)))

# print deformed points
depsgraph = bpy.context.evaluated_depsgraph_get() # get the dependecny graph
obj_eval = mesh_obj.evaluated_get(depsgraph)
mesh_eval = obj_eval.data;
transformed_verts = [v.co for v in obj_eval.data.vertices]

# print control point locations
print(lattice)
print(lattice_ob.data)
for points in lattice_ob.data.points:
    print(points.co_deform)
# coordinates as tuples
plain_verts = [v.to_tuple() for v in transformed_verts]
for v in plain_verts:
    print(v)
