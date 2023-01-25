from nutils import mesh, topology, elementseq, transformseq
import numpy

#######################
# get background mesh #
#######################
def get_background_mesh(ttopo, topo):

  # define empty lists
  baseitopo = topo

  # loop over faces
  references = []
  for itransform, ireference in zip(baseitopo.transforms, baseitopo.references):
    try:
      ttopo.transforms.index_with_tail(itransform)
    except ValueError:
      references.append(ireference.empty)
    else:
      references.append(ireference)

  return topology.SubsetTopology(baseitopo, references)

#####################
# get skeleton mesh #
#####################
def get_skeleton_mesh(ttopo, topo):

  # define empty lists
  baseitopo = topo.interfaces

  # loop over faces
  references = []
  for iface_transform, iface_reference, iface_opposite in zip(baseitopo.transforms, baseitopo.references, baseitopo.opposites):
    try:
      ttopo.transforms.index_with_tail(iface_transform)
      ttopo.transforms.index_with_tail(iface_opposite)
    except ValueError:
      references.append(iface_reference.empty)
    else:
      references.append(iface_reference)

  return topology.SubsetTopology(baseitopo, references)

##################
# get ghost mesh #
##################
def get_ghost_mesh(ttopo, topo, skeleton):

  # define empty lists
  ghost_references = []
  ghost_transforms = []
  ghost_opposites  = []

  # loop over faces
  for iface_transform, iface_reference, iface_opposite in zip(skeleton.transforms, skeleton.references, skeleton.opposites):
    for trans in iface_transform, iface_opposite:

      # find the corresponding element in the background mesh
      ielemb, tailb = topo.transforms.index_with_tail(trans)

      # find the corresponding element in the trimmed mesh
      ielemt, tailt = ttopo.transforms.index_with_tail(trans)

      if topo.references[ielemb] != ttopo.references[ielemt]:
        assert topo.transforms[ielemb] == ttopo.transforms[ielemt]
        assert topo.opposites[ielemb]  == ttopo.opposites[ielemt]
        ghost_references.append(iface_reference)
        ghost_transforms.append(iface_transform)
        ghost_opposites.append(iface_opposite)
        break
  ghost_references = elementseq.References.from_iter(ghost_references, topo.ndims-1)
  ghost_opposites  = transformseq.PlainTransforms(ghost_opposites, fromdims=topo.ndims-1, todims=topo.ndims)
  ghost_transforms = transformseq.PlainTransforms(ghost_transforms, fromdims=topo.ndims-1, todims=topo.ndims)

  # spaces of the topology
  space, = ttopo.spaces

  return topology.TransformChainsTopology(space, ghost_references, ghost_transforms, ghost_opposites)

#################
# mesh topology #
#################
def mesh_topo(points, connectivity):

  meshdomain, geom0 = mesh.rectilinear([numpy.linspace(0, 1, connectivity.shape[0]+1)], space='1D')
  dbasis   = meshdomain.basis('discont',degree=1).vector(points.shape[1])
  meshgeom = dbasis.dot(numpy.ravel(points[connectivity]))

  return meshdomain, meshgeom

###########################
# get fragmented topology #
###########################
def get_fragment_topo(self):
  references = []
  transforms = []
  opposites  = []
  for elemrefs, elemtrans, elemopp in zip(self.references,self.transforms, self.opposites):
    for trans, refs in elemrefs.simplices:
      references.append(refs)
      transforms.append(elemtrans+(trans,))
      opposites.append(elemopp+(trans,))
  references = elementseq.References.from_iter(references, self.ndims)
  opposites  = transformseq.PlainTransforms(opposites , fromdims=self.ndims, todims=self.ndims)
  transforms = transformseq.PlainTransforms(transforms, fromdims=self.ndims, todims=self.ndims)
  # spaces of the topology
  space, = self.spaces
  return topology.TransformChainsTopology(space, references, transforms, opposites)