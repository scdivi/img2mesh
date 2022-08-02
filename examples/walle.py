from nutils import *
import numpy, pathlib
from matplotlib import collections, colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from img2topo import voxel, util

def main ( fname  = 'walle_70.json' ,
           degree = 2               ,
           nelems = 7               ,
           mref   = 2               ,
           ncg    = 0               ):

  # read the voxel data
  voxeldata, props = voxel.jsonread( pathlib.Path('../data/' + fname) )
  voxeltopo, voxelgeom = voxeldata.mesh

  # voxel info
  voxeldata.log()

  # coarse grain the voxel data file
  voxeldata = voxeldata.coarsegrain(ncoarsegrain=ncg)

  # number of voxels
  nvoxels = voxeldata.shape[0]

  # number of elements
  nelems = (nelems,)*voxeldata.ndims

  # threshold
  threshold = 0

  # create the ambient domain grid
  topo, geom = mesh.rectilinear( [numpy.linspace(0,length,nelem+1) for nelem,length in zip(nelems,voxeldata.lengths)] )

  voxeltopo_func = voxeldata.func(eval_topo=voxeltopo,eval_unit_geom=voxeldata.unit_geom)

  bezier = voxeltopo.sample('bezier', 2**mref+1)
  points, vals = bezier.eval([voxelgeom, voxeltopo_func])

  levelset = voxeltopo.projection(voxeltopo_func - threshold, onto=voxeltopo.basis('th-spline',degree=degree), geometry=voxelgeom, ptype='convolute', ischeme='gauss{}'.format(degree))

  bezier = voxeltopo.sample('bezier', 2**mref+1)
  points, vf, lvl = bezier.eval([voxelgeom, voxeltopo_func, levelset])
  with export.mplfigure('voxelfunc.png') as fig:
    ax = fig.add_subplot(111, aspect=1)
    ax.autoscale(enable=True, axis='both', tight=True)
    im = ax.tripcolor(points[:,0], points[:,1], bezier.tri, vf, shading='gouraud', cmap='viridis')
    ax.add_collection(collections.LineCollection(points[bezier.hull], colors='k', linewidths=.5,     alpha=.1))
    ax.axis('off')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="5%", pad=0.2)
    fig.colorbar(im, orientation='horizontal', cax=cax)
  with export.mplfigure('levelset.png') as fig:
    ax = fig.add_subplot(111, aspect=1)
    ax.autoscale(enable=True, axis='both', tight=True)
    im = ax.tripcolor(points[:,0], points[:,1], bezier.tri, lvl, shading='gouraud', cmap='viridis')
    ax.add_collection(collections.LineCollection(points[bezier.hull], colors='k', linewidths=.5,     alpha=.1))
    ax.axis('off')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="5%", pad=0.2)
    fig.colorbar(im, orientation='horizontal', cax=cax)

  levelset = util.TopoMap(levelset, func_topo=voxeltopo, eval_topo=topo, eval_unit_geom=geom/voxeldata.lengths)
  ttopo = topo.trim( levelset, maxrefine=mref )

  bezier = ttopo.sample('bezier', 2**mref+1)
  points, vals = bezier.eval([geom, 0])
  with export.mplfigure('trimmed.png') as fig:
    ax = fig.add_subplot(111, aspect=1)
    ax.autoscale(enable=True, axis='both', tight=True)
    cmap = colors.ListedColormap("darkgray")
    im = ax.tripcolor(points[:,0], points[:,1], bezier.tri, vals, shading='gouraud', cmap=cmap)
    ax.axis('off')

cli.run(main)
