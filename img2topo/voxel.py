from nutils import function, types, cache, mesh, transform
import numpy, pathlib, treelog
import warnings

class VoxelFunc(function.Array):

  @types.apply_annotations
  def __init__(self, data:types.frozenarray, eval_topo, eval_unit_geom, shape=()):

    assert data.ndim==eval_topo.ndims+len(shape) and data.shape[eval_topo.ndims:]==shape, 'Data shape mismatch'

    #Internal data storage
    self.__data    = data
    self.eval_topo = eval_topo

    function.Array.__init__(self, args=[eval_unit_geom,function.TRANS], shape=shape, dtype=float)

  def evalf( self, x, trans ):
    assert self.eval_topo.transforms.contains_with_tail(trans), 'Evaluated on wrong topology'

    shp    = numpy.array(self.__data.shape)[:self.eval_topo.ndims]
    indf   = x*shp
    indi   = numpy.maximum(numpy.minimum(numpy.floor(indf),shp-1),0).astype(int)

    return self.__data[list(indi.T)]

  def _edit( self, op ):
    return self

  def _derivative(self, var, seen):
    return function.Zeros(self.shape + var.shape, dtype=self.dtype)

class VoxelData(types.Immutable):

  __cache__ = 'mesh',

  @types.apply_annotations
  def __init__(self, data:types.frozenarray, bounding_box:tuple, name='voxeldata', squeeze=True):

    if squeeze:
      indi = filter( lambda i : data.shape[i]==1 , range(data.ndim) )
      data = numpy.squeeze( data)
      bounding_box = tuple(bounding_box[i] for i in range(data.ndim) if not i in indi)

    self.name         = name
    self.__data       = data.astype(float)
    self.bounding_box = bounding_box
    self.ndim         = data.ndim
    self.shape        = data.shape

  def __getitem__ ( self, Slice ):

    if Slice is Ellipsis:
      return self

    if isinstance(Slice,slice):
      if Slice==slice(None):
        return self
      Slice = (Slice,)*self.ndim

    if isinstance(Slice,tuple):
      assert len(Slice)==self.ndim

      bounding_box = []
      for d in range(self.ndim):
        left_verts = numpy.linspace(self.bounding_box[d][0],self.bounding_box[d][1]-self.spacing[d],self.shape[d])[Slice[d]]
        bounding_box.append( (left_verts[0],left_verts[-1]+self.spacing[d]) )

      sliced = VoxelData( self.__data[Slice], bounding_box, '.'.join([self.name,'sliced']) )

      return sliced

    raise Exception('Unsupported slicing operation')

  def log(self):
    with treelog.context(self.name):
      treelog.user('domain size = {}'  .format(' × '.join(str(d) for d in self.lengths)))
      treelog.user('domain shape = {}' .format(' × '.join(str(d) for d in self.shape))  )
      treelog.user('voxel spacing = {}'.format(' × '.join(str(d) for d in self.spacing)))
      treelog.user('intensity range = [{},{}]'.format(*self.rng))

  def func(self, eval_topo, eval_unit_geom, discontinuous=False):
    if discontinuous:
      return function.elemwise( eval_topo.transforms, types.frozenarray(eval_topo.sample('uniform', 1).eval(self.func(eval_topo,eval_unit_geom))) )
    return VoxelFunc(data=self.__data, eval_topo=eval_topo, eval_unit_geom=eval_unit_geom)

  @property
  def ndims(self):
    return self.topo.ndims

  @property
  def lengths(self):
    return tuple(bb[1]-bb[0] for bb in self.bounding_box)

  @property
  def center(self):
    return tuple(0.5*(bb[1]+bb[0]) for bb in self.bounding_box)

  @property
  def rng(self):
    return (numpy.amin(self.__data), numpy.amax(self.__data))

  @property
  def volume(self):
    return numpy.prod( self.lengths )

  def count_voxels(self):
    return self.__data.size

  def count_porous_voxels(self, threshold):
    return (self.__data<threshold).sum()

  def count_solid_voxels(self, threshold):
    return (~(self.__data<threshold)).sum()

  @property
  def spacing(self):
    return tuple(l/float(sh) for l, sh in zip(self.lengths, self.shape))

  @property
  def mesh(self):
    return mesh.rectilinear([numpy.linspace(b[0], b[1], sh+1) for b,sh in zip(self.bounding_box,self.shape)], name=self.name)

  @property
  def topo(self):
    return self.mesh[0]

  @property
  def geom(self):
    return self.mesh[1]

  @property
  def unit_geom(self):
    return (self.geom-numpy.array(self.center))/numpy.array(self.lengths)+1/2

  def get_threshold(self, porosity, **kwargs):
    return get_threshold( self.__data, porosity, **kwargs)

  def coarsegrain(self, ncoarsegrain=1):

    if ncoarsegrain < 1:
      return self

    cg_shape = [d//(2**ncoarsegrain) for d in self.__data.shape]

    treelog.info( 'Coarse grained data shape: (%s)' % ','.join(map(str,cg_shape)) )

    #Create the coarse grain domain
    cg_name = '.'.join([self.name,'coarsegrained'])
    cg_topo, cg_unit_geom = mesh.rectilinear([numpy.linspace(0,1,sh+1) for sh in cg_shape], name=cg_name)
    cg_data = cg_topo.elem_mean(self.func(eval_topo=cg_topo,eval_unit_geom=cg_unit_geom), geometry=cg_unit_geom, ischeme='uniform%i'%(2**ncoarsegrain))

    return VoxelData(cg_data.reshape(cg_shape), self.bounding_box, cg_name)

  def remove_disconnected ( self, threshold, seed ):

    material = self.__data < threshold
    tagged   = ~material[seed]

    connected_material = numpy.zeros( material.shape, dtype=bool )

    #Loop over the cells in the seed
    for i in treelog.iter('Cell',numpy.ndindex(tagged.shape)):

      if tagged[i]:
        continue

      tagged[i] = True

      #Initialize the cellpool
      grouped   = numpy.zeros_like( material )
      cellpool  = [(0,)+i]  #TO DO: Generalize for seed

      while cellpool:

        j = cellpool.pop()
        
        grouped[j] = True

        #Expand the cellpool
        iNeighbors = [ j[:idim] + (j[idim]+d,) + j[idim+1:] for idim in range(material.ndim) for d in (-1,+1) if 0 <= j[idim]+d < material.shape[idim] ]

        for iNeighbor in iNeighbors:
          if material[iNeighbor] and not grouped[iNeighbor]:
            cellpool.append( iNeighbor )

      tagged |= grouped[0,:]  #TO DO: Generalize for seed

      connected_material |= grouped

    assert connected_material.sum()>0, 'Material completely unconnected'

    treelog.info('Connected %d percent' % (100.*connected_material.sum()/material.view(numpy.ndarray).sum()  ) )

    newdata = self.__data.copy()

    removed_material = material & ~connected_material

    newdata[removed_material] = numpy.mean(  newdata[~material] )

    return VoxelData(newdata, self.bounding_box, '.'.join([self.name,'connected']))


def voxread ( fname ):

  assert fname.endswith('.vox'), 'Expected a vox files'

  #Reading the data from the vox file
  with open( fname ) as fin:
    fin = open( fname )
    title = fin.readline()
    spacing = tuple(float(fin.readline().strip()) for i in range(3))
    shape = tuple(int(fin.readline().strip()) for i in range(3))
    sdata = fin.readline().strip()
    assert len(sdata)==numpy.prod(shape), 'Incorrect data size'

  #Convert to numpy array
  data = (numpy.fromstring( sdata, dtype=numpy.uint8 )==83).astype(float).reshape( shape )-0.5

  treelog.info( 'Original data shape: (%s)' % ','.join(map(str,data.shape)) )

  #Construct the domain and geometry
  bb = [ [0,sh*sp] for sp,sh in zip(spacing,shape)]

  return VoxelData( data, bb )

def imageread( fname, name='imagedata' ):

  from PIL import Image

  #Read the data
  data = numpy.array( Image.open( fname ) ).astype(int)

  assert data.ndim==2, 'Expected two-dimensional image'

  return VoxelData( data, [[0.,shp] for shp in data.shape], name=name )

def npzread( fname, name='npzdata' ):

  data=numpy.load('aorta.npz')

  #Assert that the voxels are all of the same size
  for key in ['x','y','z']:
    spacing = data[key][1:]-data[key][:-1]
    numpy.testing.assert_allclose( spacing, spacing[0], err_msg='Grid is not equidistantly spaced' )

  #Bounding box
  bb = [ [data[key][0],data[key][-1]] for key in ['x','y','z'] ]

  return VoxelData( data['array3d'], bb, name=name )

def jsonread(fpath):

  import json, pathlib

  assert isinstance(fpath, pathlib.Path), 'expected pathlib.Path object'
  assert fpath.is_file(), 'file {} does not exist'.format(fpath.name)

  with fpath.open() as jsonfile:    
    jsondict = json.load(jsonfile)

  fname   = jsondict['FNAME']
  shape   = jsondict['DIMS']
  spacing = jsondict['SIZE']
  order   = jsondict['ORDER']
  dtype   = jsondict['DTYPE'] # e.g., '<i2' for two byte integer little endian

  fpath = fpath.with_name(pathlib.Path(fname).name)

  assert fpath.is_file(), 'File {} does not exist'.format(fpath.name)

  with fpath.open('rb') as datafile:
    data = numpy.fromfile(file=datafile, dtype=dtype, count=numpy.prod(shape)).reshape(shape, order=order)

  # read additional properties
  props = {}
  if 'POROSITY' in jsondict:
    props['porosity'] = jsondict['POROSITY']

  # construct the domain and geometry
  bb = tuple((0,sh*sp) for sp,sh in zip(spacing,data.shape))

  return VoxelData(data, bb, name=fpath.stem), props


def get_threshold(data, porosity, rtol=1e-8, maxiter=1000):

  assert isinstance( data, numpy.ndarray ), 'Data should be numpy array'
  assert 0 < porosity < 1
  target = porosity*data.size
  lower = numpy.amin(data)
  upper = numpy.amax(data)
  atol = rtol*(upper-lower)

  for i in treelog.range('threshold bisectioning',maxiter):

    threshold = (lower+upper)/2

    porous_count = (data<threshold).sum()

    if porous_count > target:
      upper = threshold
    elif porous_count < target:
      lower = threshold
    else:
      upper = lower

    if upper-lower <= atol:
      treelog.info('threshold = {}'.format(str(threshold)))
      return threshold

  raise RuntimeError('threshold not found in {} iterations'.format(maxiter))

# vim:shiftwidth=2:softtabstop=2:expandtab:foldmethod=indent:foldnestmax=2
