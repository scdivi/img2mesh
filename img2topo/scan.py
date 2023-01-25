import numpy, pathlib, os
from nutils import topology, function, mesh
from skimage.measure import regionprops, label

######################################
# get nutils topology data structure #
######################################

def nutils_topo(fname        = 'walle_70' ,
                nelems       = 7          ,
                maxrefine    = 2          ,
                topopreserve = False        
                ):

    # Read voxel data
    voxeltopo, voxelgeom, voxelfunc, levelset = get_scandata(fname, degree=2, topopreserve=topopreserve)

    # number of voxels
    nvoxels = voxelgeom.shape[0]
  
    # create the ambient domain grid
    voxelvertices = voxeltopo.sample('bezier',2).eval(voxelgeom)
    lengths = numpy.max(voxelvertices,axis=0)-numpy.min(voxelvertices,axis=0)

    # create the ambient domain grid
    topo, geom = mesh.rectilinear( [numpy.linspace(0,length,nelems+1) for length in lengths] )

    # topomap
    levelset = TopoMap(levelset - .5, voxeltopo, voxelgeom, topo, geom, 2**maxrefine+1)

    # trim levelset
    ttopo = topo.trim(levelset, maxrefine=maxrefine)

    return topo, ttopo, geom

###################
# read json files #
###################

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
      voxeldata = numpy.fromfile(file=datafile, dtype=dtype, count=numpy.prod(shape)).reshape(shape, order=order)

    # construct the domain and geometry
    bb = tuple((0,sh*sp) for sp,sh in zip(spacing,shape))

    voxeltopo, voxelgeom = mesh.rectilinear([numpy.linspace(b[0], b[1], sh+1) for b,sh in zip(bb,shape)])
    voxelfunc = function.get(voxeldata.ravel(), 0, voxeltopo.f_index)

    return voxeldata, voxeltopo, voxelgeom, voxelfunc

################################################
# mapping function from func_topo to eval_topo #
################################################

def TopoMap(func, func_topo, func_geom, eval_topo, eval_geom, npts):

    eval_sample = eval_topo.sample('bezier', npts)
    eval_coords = eval_sample.eval(eval_geom)
    func_sample = func_topo.locate(func_geom, eval_coords, tol=1)
    func_values, func_coords = func_sample.eval([func, func_geom])
    eval_func   = eval_sample.basis().dot(func_values)

    # locate check
    diff_coords = eval_coords-func_coords
    diff_tol    = numpy.max(numpy.sqrt(diff_coords * diff_coords))
    assert diff_tol <= 1e-12, 'Located with maximum distance = {:.4e}'.format(diff_tol)

    return eval_func

###################################
# get scan data with topopreserve #
###################################

def get_scandata(fname, degree=2, topopreserve=False):

    # set the json file name
    fjson = f'{fname}.json'

    assert os.path.exists(fjson), f"Json file {fjson} is not found"

    if not topopreserve:
      # read json
      voxeldata, voxeltopo, voxelgeom, voxelfunc = jsonread( pathlib.Path(fjson) )
    
      # construct basis function
      voxelbasis = voxeltopo.basis('th-spline',degree=degree)

      # construct levelset
      levelset = voxeltopo.projection(voxelfunc, onto=voxelbasis, geometry=voxelgeom, ptype='convolute', ischeme='gauss{}'.format(degree))

    else:
      # input parameters
      rings     = 1
      nsub      = 2
      degree    = degree 
      threshold = 0

      # read voxeldata with topology preservation
      voxeltopo, voxelgeom, voxelfunc, levelset = topopreserve2D(fjson, rings, nsub, degree, threshold)

    # call the main function
    return voxeltopo, voxelgeom, voxelfunc, levelset

def topopreserve2D(fname     = 'linearelasticity_32.json',
                  rings     = 1,
                  nsub      = 2,
                  degree    = 2,
                  threshold = 0):

    # read the voxel data
    voxeldata, voxeltopo, voxelgeom, voxelfunc = jsonread( pathlib.Path(fname) )

    # subdivision
    nsub = 2**nsub

    # sampling voxel function
    im_orig_gray = voxeltopo.sample('uniform',1).eval(voxelfunc - threshold).reshape(voxeldata.shape[0],voxeldata.shape[1])
    im_orig_gray = numpy.rot90(im_orig_gray)

    # subdivide
    im_subd_gray = numpy.empty(shape=tuple(nsub*s for s in im_orig_gray.shape),dtype=float)
    for i in range(im_orig_gray.shape[0]):
        for j in range(im_orig_gray.shape[1]):
            im_subd_gray[nsub*i:nsub*(i+1),nsub*j:nsub*(j+1)] = im_orig_gray[i,j]

    # segment voxel function
    im_subd_segm = ( im_subd_gray > 0 ).astype(int)

    ###############
    # convolution #
    ###############

    # construct basis function
    voxelbasis = voxeltopo.basis('th-spline',degree=degree)

    # construct levelset function
    levelset = voxeltopo.projection(voxelfunc, onto=voxelbasis, geometry=voxelgeom, ptype='convolute', ischeme='gauss{}'.format(degree))


    # sampling levelset
    im_sample = voxeltopo.sample('uniform',nsub).eval(levelset - threshold)
    im_smooth_gray = numpy.empty(shape=tuple(nsub*s for s in voxeldata.shape),dtype=float)
    ind = 0
    for i in range(voxeldata.shape[0]*nsub):
        indices = []
        if i>0:
            ind+=(voxeldata.shape[1]*nsub-(nsub-1))*nsub if i%nsub==0 else nsub
        for j in range(voxeldata.shape[1]):
            indices = numpy.append(indices,numpy.arange(j*nsub**2+ind, nsub*(j*nsub+1)+ind), axis=None).astype(int)
        im_smooth_gray[i] = im_sample[indices]
    im_smooth_gray = numpy.rot90(im_smooth_gray)
    im_smooth_segm = ( im_smooth_gray > 0 ).astype(int)


    def compare_window(window, plots=False, save=False):

      # get the window
      imin = max( (window[0]-rings)*nsub, 0 )
      imax = min( (window[0]+rings+1)*nsub, im_subd_segm.shape[0] )

      jmin = max( (window[1]-rings)*nsub, 0 )
      jmax = min( (window[1]+rings+1)*nsub, im_subd_segm.shape[1] )

      # get images
      A = im_subd_segm[imin:imax,jmin:jmax]
      B = im_smooth_segm[imin:imax,jmin:jmax]

      # symmetric difference
      AB = A*(1-B)+B*(1-A)

      # define the mask M
      M = numpy.zeros_like(AB)
      AB_label   = label(AB, connectivity = 2)
      AB_regions = regionprops(AB_label)
      for AB_reg in ((AB_label==AB_reg.label) for AB_reg in AB_regions):
        if AB_reg[nsub:-nsub,nsub:-nsub].sum()==0:
          M += AB_reg

      # filtered B image
      Bf = M*A+(1-M)*B

      # compare A and Bf
      eulernumbers = {}
      for reg in regionprops(label(A, connectivity = 2)):
        ereg = reg.euler_number
        if ereg in eulernumbers:
          eulernumbers[ereg]+=1
        else:
          eulernumbers[ereg]=1

      for reg in regionprops(label(Bf, connectivity = 2)):
        ereg = reg.euler_number
        if ereg in eulernumbers:
          eulernumbers[ereg]-=1
          if eulernumbers[ereg]==0:
            del eulernumbers[ereg]
        else:
          return False

      if len(eulernumbers)!=0:
        return False

      # compare A' and Bf'
      eulernumbers = {}
      for reg in regionprops(label(1-A, connectivity = 2)):
        ereg = reg.euler_number
        if ereg in eulernumbers:
          eulernumbers[ereg]+=1
        else:
          eulernumbers[ereg]=1

      for reg in regionprops(label(1-Bf, connectivity = 2)):
        ereg = reg.euler_number
        if ereg in eulernumbers:
          eulernumbers[ereg]-=1
          if eulernumbers[ereg]==0:
            del eulernumbers[ereg]
        else:
          return False

      return len(eulernumbers)==0

    # looping over the image
    tag_elem = []
    compare = numpy.empty(shape=im_orig_gray.shape, dtype=int)
    for i in range(im_orig_gray.shape[0]):
      for j in range(im_orig_gray.shape[1]):
        compare[i,j] = compare_window( (i,j) )
        if compare[i,j] == False:
          #tag_elem+=[j*im_orig_gray.shape[0]+i-1]
          tag_elem+=[j*im_orig_gray.shape[1]+(im_orig_gray.shape[0]-i)]
          print('Change found in ({},{})'.format(i,j))
    tag_elem = list(set(tag_elem))


    ################
    # refine basis #
    ################
    x = voxeltopo.elem_mean(voxelgeom,degree=1)
    select = numpy.zeros(x.shape[0], dtype=bool)
    # supported basis function refinement
    refdofs  = [voxelbasis.get_dofs(i) for i in tag_elem]
    refelems = [voxelbasis.get_support(i) for i in refdofs]
    for i in refelems:
        select[i] = True
    transforms = voxeltopo.transforms[select]
    voxeltopo  = voxeltopo.refined_by(transforms)

    # construct basis function
    voxelbasis = voxeltopo.basis('th-spline',degree=degree)

    # convolute voxel function
    levelset   = voxeltopo.projection(voxelfunc - threshold, onto=voxelbasis, geometry=voxelgeom, ptype='convolute', ischeme='gauss{}'.format(degree))

    return voxeltopo, voxelgeom, voxelfunc, levelset