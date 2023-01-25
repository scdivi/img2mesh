from nutils import *
import numpy
from matplotlib import collections, colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from img2topo.scan import get_scandata, TopoMap, nutils_topo

def main ( fname        = 'walle_70' ,
           nelems       = 7          ,
           maxrefine    = 3          ,
           topopreserve = False        
          ):

    ttopo, geom = nutils_topo(fname, nelems, maxrefine, topopreserve)

    bezier = ttopo.sample('bezier', 2**maxrefine+1)
    points, vals = bezier.eval([geom, 0])
    with export.mplfigure('trimmed.png') as fig:
      ax = fig.add_subplot(111, aspect=1)
      ax.autoscale(enable=True, axis='both', tight=True)
      cmap = colors.ListedColormap("darkgray")
      im = ax.tripcolor(points[:,0], points[:,1], bezier.tri, vals, shading='gouraud', cmap=cmap)
      ax.axis('off')

cli.run(main)
