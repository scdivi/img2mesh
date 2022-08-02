import numpy
from nutils import function, topology, element, points, sample, util, transform, types, elementseq, transformseq

class TopoMap(function.Array):
  """Topology mapping Array wrapper

  Array wrapper for function evaluation on an arbitrary structured topology.

  Parameters
  ----------
  func : Array
    Array to be wrapped
  func_topo
    Topology on which func is defined (evaluable)
  eval_topo
    Topology on which func is to be evaluated
  eval_unit_geom
    Unit geometry function on eval_topo

  Returns
  -------
  func : Array
    :class:`TopoMap`

  """

  def __init__(self, func, func_topo, eval_topo, eval_unit_geom):

    assert isinstance(func_topo, topology.StructuredTopology)
    assert isinstance(eval_topo, topology.StructuredTopology)
    assert func_topo.ndims==eval_topo.ndims

    #Internal data storage
    self.func           = func
    self.func_topo      = func_topo
    self.eval_topo      = eval_topo
    self.eval_unit_geom = eval_unit_geom
    self.jacobian       = numpy.array(func_topo.shape)/numpy.array(eval_topo.shape)

    function.Array.__init__(self, args=[eval_unit_geom,function.TRANS], shape=func.shape, dtype=float)

  def evalf(self, x, trans):
    assert self.eval_topo.transforms.contains_with_tail(trans), 'Evaluated on wrong topology'

    shp    = numpy.array(self.func_topo.shape)
    indf   = x*shp
    indi   = numpy.maximum(numpy.minimum(numpy.floor(indf),shp-1),0).astype(int)
    points = indf-indi

    transforms = [self.func_topo[tuple(slice(ii,ii+1) for ii in ind)].transforms[0] for ind in indi]
    transforms = numpy.array(transforms,dtype=numpy.dtype(','.join(['object']*len(transforms[0]))))

    values = numpy.zeros( shape=(points.shape[:1]+self.func.shape) )
    for trans in set(types.frozenarray(transforms)):
      mask = (transforms==trans)
      values[mask] = self.func.eval( _transforms=[tuple(trans)], _points=points[mask] )

    return values

  def _derivative( self, var, seen ):

    #Local gradient of 'func' on the 'eval_topo'
    assert isinstance(var, function.LocalCoords)
    func_deriv = self.jacobian * function.derivative(self.func, var, seen)

    #Wrap the local gradient in the UniformTopoMap
    eval_deriv = TopoMap(func=func_deriv, func_topo=self.func_topo, eval_topo=self.eval_topo, eval_unit_geom=self.eval_unit_geom)

    #Apply roottransform to accomodate evaluations on refinements of eval_topo
    roottrans = function.LinearFrom(function.PopHead(self.func_topo.ndims),self.func_topo.ndims)

    return ( eval_deriv[...,:,numpy.newaxis] * roottrans ).sum(-2)

  def _edit( self, op ):
    return self

def _stableInv(npmat, cons, tol, symmetric):

  _eig = numpy.linalg.eig if not symmetric else numpy.linalg.eigh

  while sum(~cons.where) > 0:
    #Constrain the local matrix
    consmat = npmat[~cons.where,:][:,~cons.where]

    #Eigen decomposition
    evals, evecs = _eig(consmat)

    if numpy.all( abs(evals)>tol*numpy.amax(abs(evals)) ):
      assert consmat.shape[0]==consmat.shape[1]==sum((~cons).where)
      return numpy.linalg.inv(consmat), cons
  
    imin = numpy.argmin(abs(evals)) #Index of smallest eigenvalue
    evecabs = numpy.zeros(shape=(npmat.shape[0],))
    evecabs[~cons.where] = abs(evecs[:,imin])
    imax = numpy.argmax(evecabs) #Index of largest basis function mode contribution

    #Supplement constraints
    cons[imax] = 0
  else:
    #Everything is constrained
    raise ValueError('All indices in local matrix appear to be constrained')


def getcbasprecon(matrix, groups=[], constrain=None, lconstrain=None, rconstrain=None, tol=numpy.spacing(100), symmetric=False ):
 
  """Connectivity-based Additive Schwarz (CbAS) preconditioner

  Parameters
  ----------
  matrix : nutils.matrix.SciPyMatrix
      Square sparse matrix for which to compute the preconditioner


  Keyword Arguments
  -----------------
  constrain : nutils.util.NanVec
      Constraint vector
  groups    : list(list(int)), optional
      List containing lists of group indices for which to assemble local inverses.
      Indices not seen by the groups are inverted per index (diagonal scaling).
      Defaults to empty list.
  symmetric : bool, optional
      Should be set to :obj:`True` when matrix is symmetric

  Returns
  -------
  scipy.sparse.linalg.LinearOperator 
      Preconditioner for :obj:`matrix` in linear operator form
  """

  if not lconstrain is None or not rconstrain is None:
    raise NotImplementedError('Current implementation of CbAS preconditioner does not facilitated left and right constraints')

  assert isinstance(constrain, util.NanVec), 'Expected a NanVec for the constrain argument'

  core = matrix.core
  assert core.shape[0]==core.shape[1], 'Matrix is not square'

  import scipy.sparse.linalg

  data  = numpy.zeros(shape=(0,), dtype=core.dtype)
  index = numpy.zeros(shape=(2,0), dtype=int)
  seen  = numpy.zeros( core.shape[0], dtype=bool )
  for indices in groups:

    #Skip groups with only one index (treated as unseen)
    if not len(indices) > 1:
      continue

    #Mark indices as seen
    seen[indices] = True

    #Compute the local inverse
    ij = numpy.broadcast_to(indices, (len(indices),)*2).T
    lmat = core[ij,ij.T]
    lcons = constrain[indices]
    linv, lcons = _stableInv(lmat.toarray(),lcons,tol=tol,symmetric=symmetric)

    #Update the constraints
    constrain[indices] = lcons
    ij = ij[~lcons.where,:][:,~lcons.where]

    #Append the sparsity data
    data  = numpy.append(data, linv.ravel())
    index = numpy.append(index, [ij.ravel(),ij.T.ravel()], axis=-1)

  #Diagonal scaling for unseen indices
  ii    = numpy.where(~seen)[0]
  data  = numpy.append(data, numpy.reciprocal(core.diagonal()[~seen]))
  index = numpy.append(index, [ii,ii], axis=-1)

  #Construct the preconditioner
  precon = scipy.sparse.coo_matrix((data,tuple(index)), shape=core.shape).tocsr().dot

  return scipy.sparse.linalg.LinearOperator(core.shape, precon, dtype=float)

def getprecon( matrix, name, *args, **kwargs ):
  name = name.lower()
  if name=='cbas':
    precon = getcbasprecon(matrix, *args, **kwargs)
  else:
    precon = matrix.getprecon(name, *args, **kwargs)
  return precon

def getmlpoints(ref, arg):
  ischeme, degree = arg
  if isinstance(ref, element.WithChildrenReference):
    return points.ConcatPoints(points.TransformPoints(getmlpoints(cref, (ischeme, degree[1:])), ctrans) for ctrans, cref in ref.children if cref)
  elif isinstance(ref,element.MosaicReference):
    assert len(degree)==2
    return ref.getpoints(ischeme, degree[1])
  else:
    return ref.getpoints(ischeme, degree[0])

def getfragments(ref, trans=()):
  if isinstance(ref, element.WithChildrenReference):
    for ctrans, cref in ref.children:
      yield from getfragments(cref, trans + (ctrans,))
  elif isinstance(ref, element.MosaicReference):
    ctrans = transform.Identity(trans[-1].todims)
    for cref in ref.subrefs:
      yield from getfragments(cref, trans + (ctrans,))
  elif not isinstance(ref, element.EmptyLike):
    yield ref, trans

@types.apply_annotations
def multilevel_ischeme(ref, arg:types.tuple):

  #unpack the arguments
  ischeme, degree = arg

  assert isinstance(ischeme,str)
  assert isinstance(degree,(int,tuple)) # int: uniform degree, tuple: per level degree

  #initialize coordinates and weights lists to be concatenated
  coords  = []
  weights = []

  #loop over the fragments
  for ref, trans in getfragments(ref):

    #get the points object for the fragment
    if isinstance(degree,int):
      pts = ref.getpoints(ischeme, degree)
    elif isinstance(degree,tuple):
      assert len(trans)<len(degree)
      pts = ref.getpoints(ischeme, degree[len(trans)])

    #accumulate the coordinates and weights
    coords .append(transform.apply(trans, pts.coords))
    weights.append(pts.weights * numpy.prod(list(abs(float(tr.det)) for tr in trans)))

  return points.CoordsWeightsPoints(numpy.concatenate(coords), numpy.concatenate(weights))

def integrate_subcellwise(self, func, ischeme, degree):
  # loop over element
  val = []
  for references, transforms in zip(self.references,self.transforms):
    for ref, trans in getfragments(references, transforms):
      points = ref.getpoints(ischeme, degree)
      values = func.eval(_points=points.coords, _transforms=(trans,))
      integral = values.dot(points.weights)
      val.append( integral )
  return val

# multilevel integral
def mlintegral(self, func, ischeme='gauss', degree=None, edit=None):
  assert isinstance(degree, tuple)
  ischeme, degree = element.parse_legacy_ischeme(ischeme if degree is None else ischeme + str(degree))
  if edit is not None:
    funcs = edit(func)
  return self.sample(getmlpoints, (ischeme, degree)).integral(func)

def slope_triangle(fig, ax, x, y):
    i, j = (-2, -1) if x[-1] < x[-2] else (-1, -2)
    if not all(numpy.isfinite(x[-2:])) or not all(numpy.isfinite(y[-2:])):
      treelog.warning('Not plotting slope triangle for +/-inf or nan values')
      return

    from matplotlib import transforms
    shifttrans = ax.transData + transforms.ScaledTranslation(0, -0.1, fig.dpi_scale_trans)
    xscale, yscale = ax.get_xscale(), ax.get_yscale()

    # delta() checks if either axis is log or lin scaled
    delta = lambda a, b, scale: numpy.log10(float(a) / b) if scale=='log' else float(a - b) if scale=='linear' else None
    slope = delta(y[-2], y[-1], yscale) / delta(x[-2], x[-1], xscale)
    if slope in (numpy.nan, numpy.inf, -numpy.inf):
      warnings.warn('Cannot draw slope triangle with slope: {}, drawing nothing'.format(str(slope)))
      return slope

    # handle positive and negative slopes correctly
    xtup, ytup = ((x[i], x[j], x[i]), (y[j], y[j], y[i]))
    a, b = (2/3., 1/3.)
    xval = a*x[i] + b*x[j]
    yval = b*y[i] + a*y[j]

    ax.fill(xtup, ytup, color='0.9', edgecolor='k', transform=shifttrans)
    slopefmt='{0:.1f}'
    ax.text(xval, yval,
      slopefmt.format(slope),
      horizontalalignment='center',
      verticalalignment='center',
      transform=shifttrans)

# get fragmented topology
def getfragmentTopo(self):
  references = []
  transforms = []
  opposites  = []
  for elemrefs, elemtrans, elemopp in zip(self.references,self.transforms, self.opposites):
    for trans, refs in elemrefs.simplices:
      references.append(refs)
      transforms.append(elemtrans+(trans,))
      opposites.append(elemopp+(trans,))
  references = elementseq.asreferences(references, self.ndims)
  opposites  = transformseq.PlainTransforms(opposites, self.ndims)
  transforms = transformseq.PlainTransforms(transforms, self.ndims)
  return topology.Topology(references, transforms, opposites)
